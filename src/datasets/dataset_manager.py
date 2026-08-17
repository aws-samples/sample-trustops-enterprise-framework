"""
Dataset Manager - Orchestrates all dataset operations.

This module provides a high-level DatasetManager class that integrates all
dataset components including format detection, parsing, validation, quality
analysis, PII detection, splitting, synthetic generation, and versioning.

Requirements: 2.1-2.19
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import boto3
import pyarrow.parquet as pq

from config.aws_config import config
from src.data_models.dataset import (
    DatasetFormat,
    DatasetMetadata,
    DatasetQualityReport,
    DatasetTaskType,
    TokenStatistics,
)
from src.datasets.format_converter import FormatConverter
from src.datasets.format_detector import FormatDetector
from src.datasets.model_validator import ModelValidator
from src.datasets.parsers.csv_parser import CSVParser
from src.datasets.parsers.jsonl_parser import JSONLParser
from src.datasets.parsers.parquet_parser import ParquetParser
from src.datasets.pii_detector import (
    MaskingStrategy,
    PIIDetectionResult,
    PIIDetector,
    PIIType,
)
from src.datasets.quality_analyzer import QualityAnalyzer
from src.datasets.splitter import DatasetSplitter
from src.datasets.synthetic_generator import SyntheticDataGenerator
from src.datasets.version_manager import DatasetVersionManager
from src.utils.retry_utils import retry_with_exponential_backoff


class DatasetManager:
    """
    Manages dataset lifecycle including upload, validation, and versioning.
    
    This class orchestrates all dataset operations:
    - Upload datasets with automatic format detection
    - Validate datasets against quality thresholds and templates
    - Apply transformations (PII masking, splitting, synthetic generation)
    - Convert between formats
    - Create and manage versions
    - Track lineage and usage
    
    Requirements: 2.1-2.19
    """

    def __init__(
        self,
        region: Optional[str] = None,
        datasets_bucket: Optional[str] = None,
        metadata_table: Optional[str] = None,
        inference_client: Optional[Any] = None,
        max_dataset_size_gb: float = 10.0,
    ):
        """
        Initialize the dataset manager.

        Args:
            region: AWS region (defaults to config.region)
            datasets_bucket: S3 bucket for datasets (defaults to config.datasets_bucket)
            metadata_table: DynamoDB table for metadata. Defaults to the
                'trustops-dataset-metadata' table in the current account; DynamoDB
                names are account-scoped, so this is not a squattable name.
            inference_client: InferenceClient for synthetic generation (optional)
            max_dataset_size_gb: Maximum dataset size in GB (default: 10.0)
        """
        self.region = region or config.region
        self.datasets_bucket = datasets_bucket or config.datasets_bucket
        self.metadata_table = metadata_table or "trustops-dataset-metadata"
        self.inference_client = inference_client
        self.max_dataset_size_bytes = int(max_dataset_size_gb * 1024 * 1024 * 1024)

        # Initialize AWS clients
        session_kwargs = config.get_boto3_session_kwargs()
        if region:
            session_kwargs["region_name"] = region

        session = boto3.Session(**session_kwargs)
        self.s3_client = session.client("s3")
        self.dynamodb_client = session.client("dynamodb")

        # Initialize component classes
        self.format_detector = FormatDetector()
        self.quality_analyzer = QualityAnalyzer()
        self.pii_detector = PIIDetector()
        self.splitter = DatasetSplitter()
        self.version_manager = DatasetVersionManager(
            region=region,
            datasets_bucket=datasets_bucket,
        )
        self.format_converter = FormatConverter()
        self.model_validator = ModelValidator()

        # Initialize parsers
        self.jsonl_parser = JSONLParser()
        self.csv_parser = CSVParser()
        self.parquet_parser = ParquetParser()

    async def upload_dataset(
            self,
            file_path: Union[str, Path],
            name: str,
            task_type: Optional[DatasetTaskType] = None,
            description: Optional[str] = None,
            progress_callback: Optional[callable] = None,
        ) -> DatasetMetadata:
            """
            Upload and register a new dataset with auto-detection.

            This method:
            1. Validates file size
            2. Auto-detects format and task type
            3. Parses and validates the dataset
            4. Analyzes quality
            5. Uploads to S3 (uses multipart upload for files >100MB)
            6. Creates initial version
            7. Stores metadata in DynamoDB

            Args:
                file_path: Path to the dataset file
                name: Name for the dataset
                task_type: Optional task type (auto-detected if not provided)
                description: Optional description
                progress_callback: Optional callback function(bytes_uploaded, total_bytes)
                    for tracking upload progress

            Returns:
                DatasetMetadata with complete information

            Raises:
                ValueError: If file doesn't exist, is too large, or invalid
                RuntimeError: If upload fails

            Requirements: 2.1, 2.2, 2.3, 2.5, 2.14, 2.15, 2.16
            """
            file_path = Path(file_path)

            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")

            # Validate file size (Requirement 2.15)
            file_size = file_path.stat().st_size
            if file_size > self.max_dataset_size_bytes:
                max_size_gb = self.max_dataset_size_bytes / (1024 * 1024 * 1024)
                actual_size_gb = file_size / (1024 * 1024 * 1024)
                raise ValueError(
                    f"Dataset size ({actual_size_gb:.2f} GB) exceeds maximum "
                    f"allowed size ({max_size_gb:.2f} GB). "
                    f"Please split the dataset into smaller files or contact "
                    f"support to increase the limit."
                )

            # Auto-detect format (Requirement 2.2)
            detected_format = self.format_detector.detect_format(file_path)

            # Auto-detect task type if not provided (Requirement 2.2)
            if task_type is None:
                task_type = self.format_detector.detect_task_type(file_path)

            # Parse dataset (Requirement 2.3)
            examples = self._parse_dataset(file_path, detected_format, task_type)

            if not examples:
                raise ValueError(f"No valid examples found in {file_path}")

            # Analyze quality (Requirement 2.5)
            quality_report = self.quality_analyzer.analyze(
                examples, task_type, detected_format
            )

            # Calculate token statistics
            token_stats = quality_report.token_stats

            # Generate unique dataset ID
            dataset_id = str(uuid.uuid4())

            # Read file content for upload
            with open(file_path, "rb") as f:
                content = f.read()

            # Create temporary metadata for version creation
            temp_metadata = DatasetMetadata(
                id=dataset_id,
                name=name,
                description=description,
                format=detected_format,
                task_type=task_type,
                version="v1",
                s3_uri="s3://placeholder",  # Temporary placeholder
                row_count=len(examples),
                token_stats=token_stats,
                created_at=datetime.utcnow(),
                checksum="placeholder",  # Temporary placeholder
                lineage=None,
                quality_report=quality_report,
            )

            # Create initial version with progress tracking (Requirement 2.9, 2.16, 2.17)
            version_info = self.version_manager.create_version(
                dataset_id=dataset_id,
                content=content,
                metadata=temp_metadata,
                description="Initial upload",
                progress_callback=progress_callback,
            )

            # Create final metadata with actual s3_uri and checksum
            metadata = DatasetMetadata(
                id=dataset_id,
                name=name,
                description=description,
                format=detected_format,
                task_type=task_type,
                version="v1",
                s3_uri=version_info["s3_uri"],
                row_count=len(examples),
                token_stats=token_stats,
                created_at=datetime.utcnow(),
                checksum=version_info["checksum"],
                lineage=None,
                quality_report=quality_report,
            )

            # Store metadata in DynamoDB
            self._store_metadata(metadata)

            return metadata

    def detect_format(self, file_path: Union[str, Path]) -> DatasetFormat:
        """
        Auto-detect dataset format from file content.

        Args:
            file_path: Path to the dataset file

        Returns:
            DatasetFormat enum value

        Raises:
            ValueError: If format cannot be detected

        Requirement 2.2
        """
        return self.format_detector.detect_format(file_path)

    def detect_task_type(self, file_path: Union[str, Path]) -> DatasetTaskType:
        """
        Auto-detect task type from dataset structure.

        Args:
            file_path: Path to the dataset file

        Returns:
            DatasetTaskType enum value

        Raises:
            ValueError: If task type cannot be detected

        Requirement 2.2
        """
        return self.format_detector.detect_task_type(file_path)

    async def validate_dataset(
        self,
        dataset_id: str,
        model_id: Optional[str] = None,
    ) -> DatasetQualityReport:
        """
        Validate dataset quality and model compatibility.

        Args:
            dataset_id: Dataset identifier
            model_id: Optional model ID for compatibility validation

        Returns:
            DatasetQualityReport with quality metrics and recommendations

        Raises:
            ValueError: If dataset not found
            RuntimeError: If validation fails

        Requirements: 2.5, 2.6, 2.12
        """
        # Get dataset metadata
        metadata = await self.get_dataset(dataset_id)

        if not metadata:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Load dataset content
        version_info = self.version_manager.get_version(
            dataset_id, metadata.version
        )
        content = version_info["content"]

        # Parse examples
        examples = self._parse_content(content, metadata.format)

        # Analyze quality
        quality_report = self.quality_analyzer.analyze(
            examples, metadata.task_type, metadata.format
        )

        # Validate model compatibility if model_id provided (Requirement 2.12)
        if model_id:
            validation_result = self.model_validator.validate(
                examples, model_id, metadata.task_type
            )

            if not validation_result.is_valid:
                # Add model compatibility issues to quality report
                for issue in validation_result.issues:
                    quality_report.issues.append(issue)

                # Add recommendations
                for recommendation in validation_result.recommendations:
                    quality_report.recommendations.append(recommendation)

        return quality_report

    async def convert_format(
        self,
        dataset_id: str,
        target_format: DatasetFormat,
    ) -> DatasetMetadata:
        """
        Convert dataset to a different format.

        Args:
            dataset_id: Dataset identifier
            target_format: Target format to convert to

        Returns:
            DatasetMetadata for the new converted dataset

        Raises:
            ValueError: If dataset not found or conversion fails
            RuntimeError: If conversion fails

        Requirement 2.4, 2.9
        """
        # Get dataset metadata
        metadata = await self.get_dataset(dataset_id)

        if not metadata:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Check if already in target format
        if metadata.format == target_format:
            return metadata

        # Load dataset content
        version_info = self.version_manager.get_version(
            dataset_id, metadata.version
        )
        content = version_info["content"]

        # Parse examples
        examples = self._parse_content(content, metadata.format)

        # Convert to target format
        converted_content = self.format_converter.convert(
            examples, metadata.format, target_format
        )

        # Create new version with converted format
        new_version_info = self.version_manager.create_version(
            dataset_id=dataset_id,
            content=converted_content,
            metadata=metadata,
            description=f"Converted from {metadata.format.value} to {target_format.value}",
            parent_version_id=metadata.version,
        )

        # Record transformation
        self.version_manager.record_transformation(
            dataset_id=dataset_id,
            version_id=new_version_info["version_id"],
            transformation_type="format_conversion",
            transformation_details={
                "source_format": metadata.format.value,
                "target_format": target_format.value,
            },
        )

        # Update metadata
        metadata.format = target_format
        metadata.version = new_version_info["version_id"]
        metadata.s3_uri = new_version_info["s3_uri"]
        metadata.checksum = new_version_info["checksum"]

        # Update metadata in DynamoDB
        self._store_metadata(metadata)

        return metadata

    async def analyze_quality(self, dataset_id: str) -> DatasetQualityReport:
        """
        Analyze dataset quality metrics.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DatasetQualityReport with quality metrics

        Raises:
            ValueError: If dataset not found

        Requirement 2.5, 2.6
        """
        # Get dataset metadata
        metadata = await self.get_dataset(dataset_id)

        if not metadata:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Load dataset content
        version_info = self.version_manager.get_version(
            dataset_id, metadata.version
        )
        content = version_info["content"]

        # Parse examples
        examples = self._parse_content(content, metadata.format)

        # Analyze quality
        quality_report = self.quality_analyzer.analyze(
            examples, metadata.task_type, metadata.format
        )

        return quality_report

    async def detect_pii(self, dataset_id: str) -> PIIDetectionResult:
        """
        Detect PII in dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            PIIDetectionResult with PII detection information

        Raises:
            ValueError: If dataset not found

        Requirement 2.7, 2.13
        """
        # Get dataset metadata
        metadata = await self.get_dataset(dataset_id)

        if not metadata:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Load dataset content
        version_info = self.version_manager.get_version(
            dataset_id, metadata.version
        )
        content = version_info["content"]

        # Parse examples
        examples = self._parse_content(content, metadata.format)

        # Detect PII
        pii_result = self.pii_detector.detect(examples)

        return pii_result

    async def mask_pii(
        self,
        dataset_id: str,
        strategy: str = "mask",
        selective_types: Optional[list[PIIType]] = None,
    ) -> DatasetMetadata:
        """
        Mask or redact PII in dataset.

        Args:
            dataset_id: Dataset identifier
            strategy: Masking strategy ("mask", "redact", or "remove")
            selective_types: Optional list of PII types to mask

        Returns:
            DatasetMetadata for the new masked dataset

        Raises:
            ValueError: If dataset not found or invalid strategy

        Requirement 2.7, 2.14
        """
        # Validate strategy
        try:
            masking_strategy = MaskingStrategy(strategy)
        except ValueError:
            raise ValueError(
                f"Invalid masking strategy: {strategy}. "
                f"Must be one of: mask, redact, remove"
            )

        # Get dataset metadata
        metadata = await self.get_dataset(dataset_id)

        if not metadata:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Load dataset content
        version_info = self.version_manager.get_version(
            dataset_id, metadata.version
        )
        content = version_info["content"]

        # Parse examples
        examples = self._parse_content(content, metadata.format)

        # Mask PII
        masked_examples, masking_report = self.pii_detector.mask(
            examples, masking_strategy, selective_types
        )

        # Serialize masked examples
        masked_content = self._serialize_examples(
            masked_examples, metadata.format
        )

        # Create new version with masked data
        new_version_info = self.version_manager.create_version(
            dataset_id=dataset_id,
            content=masked_content,
            metadata=metadata,
            description=f"PII {strategy} applied",
            parent_version_id=metadata.version,
        )

        # Record transformation
        self.version_manager.record_transformation(
            dataset_id=dataset_id,
            version_id=new_version_info["version_id"],
            transformation_type="pii_masking",
            transformation_details={
                "strategy": strategy,
                "selective_types": [t.value for t in selective_types] if selective_types else None,
                "pii_instances_masked": masking_report.pii_instances_masked,
                "rows_removed": masking_report.rows_removed,
            },
        )

        # Update metadata
        metadata.version = new_version_info["version_id"]
        metadata.s3_uri = new_version_info["s3_uri"]
        metadata.checksum = new_version_info["checksum"]
        metadata.row_count = masking_report.masked_row_count

        # Update metadata in DynamoDB
        self._store_metadata(metadata)

        return metadata

    async def split_dataset(
        self,
        dataset_id: str,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by: Optional[str] = None,
    ) -> tuple[DatasetMetadata, DatasetMetadata, DatasetMetadata]:
        """
        Split dataset into train/validation/test sets.

        Args:
            dataset_id: Dataset identifier
            train_ratio: Ratio for training set (default: 0.8)
            validation_ratio: Ratio for validation set (default: 0.1)
            test_ratio: Ratio for test set (default: 0.1)
            stratify_by: Optional field name to stratify by

        Returns:
            Tuple of (train_metadata, validation_metadata, test_metadata)

        Raises:
            ValueError: If dataset not found or ratios invalid

        Requirement 2.13, 2.15
        """
        # Get dataset metadata
        metadata = await self.get_dataset(dataset_id)

        if not metadata:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Load dataset content
        version_info = self.version_manager.get_version(
            dataset_id, metadata.version
        )
        content = version_info["content"]

        # Parse examples
        examples = self._parse_content(content, metadata.format)

        # Split dataset
        train_examples, val_examples, test_examples = self.splitter.split(
            examples, train_ratio, validation_ratio, test_ratio, stratify_by
        )

        # Create three new datasets
        train_metadata = await self._create_split_dataset(
            dataset_id,
            metadata,
            train_examples,
            "train",
            train_ratio,
            stratify_by,
        )

        val_metadata = await self._create_split_dataset(
            dataset_id,
            metadata,
            val_examples,
            "validation",
            validation_ratio,
            stratify_by,
        )

        test_metadata = await self._create_split_dataset(
            dataset_id,
            metadata,
            test_examples,
            "test",
            test_ratio,
            stratify_by,
        )

        return train_metadata, val_metadata, test_metadata

    async def generate_synthetic(
        self,
        dataset_id: str,
        augmentation_factor: float = 1.5,
        strategy: str = "paraphrase",
    ) -> DatasetMetadata:
        """
        Generate synthetic data for augmentation.

        Args:
            dataset_id: Dataset identifier
            augmentation_factor: Multiplier for dataset size (e.g., 1.5 = 50% more)
            strategy: Augmentation strategy ("paraphrase", "diverse", "creative")

        Returns:
            DatasetMetadata for the augmented dataset

        Raises:
            ValueError: If dataset not found or inference client not configured
            RuntimeError: If generation fails

        Requirement 2.8, 2.16
        """
        if not self.inference_client:
            raise ValueError(
                "InferenceClient not configured. Cannot generate synthetic data."
            )

        # Get dataset metadata
        metadata = await self.get_dataset(dataset_id)

        if not metadata:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Load dataset content
        version_info = self.version_manager.get_version(
            dataset_id, metadata.version
        )
        content = version_info["content"]

        # Parse examples
        examples = self._parse_content(content, metadata.format)

        # Generate synthetic examples
        generator = SyntheticDataGenerator(self.inference_client)
        synthetic_examples = await generator.generate_synthetic(
            examples, augmentation_factor, strategy, metadata.task_type
        )

        # Combine original and synthetic examples
        augmented_examples = examples + synthetic_examples

        # Serialize augmented examples
        augmented_content = self._serialize_examples(
            augmented_examples, metadata.format
        )

        # Create new version with augmented data
        new_version_info = self.version_manager.create_version(
            dataset_id=dataset_id,
            content=augmented_content,
            metadata=metadata,
            description=f"Synthetic augmentation ({strategy}, {augmentation_factor}x)",
            parent_version_id=metadata.version,
        )

        # Record transformation
        self.version_manager.record_transformation(
            dataset_id=dataset_id,
            version_id=new_version_info["version_id"],
            transformation_type="synthetic_generation",
            transformation_details={
                "strategy": strategy,
                "augmentation_factor": augmentation_factor,
                "original_count": len(examples),
                "synthetic_count": len(synthetic_examples),
                "total_count": len(augmented_examples),
            },
        )

        # Update metadata
        metadata.version = new_version_info["version_id"]
        metadata.s3_uri = new_version_info["s3_uri"]
        metadata.checksum = new_version_info["checksum"]
        metadata.row_count = len(augmented_examples)

        # Recalculate token statistics
        quality_report = self.quality_analyzer.analyze(
            augmented_examples, metadata.task_type, metadata.format
        )
        metadata.token_stats = quality_report.token_stats

        # Update metadata in DynamoDB
        self._store_metadata(metadata)

        return metadata

    async def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """
        Get dataset metadata by ID.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DatasetMetadata or None if not found

        Raises:
            RuntimeError: If retrieval fails
        """
        try:
            response = self.dynamodb_client.get_item(
                TableName=self.metadata_table,
                Key={"dataset_id": {"S": dataset_id}},
            )

            if "Item" not in response:
                return None

            return self._dynamodb_item_to_metadata(response["Item"])

        except Exception as e:
            raise RuntimeError(f"Failed to retrieve dataset metadata: {e}") from e

    async def list_datasets(
        self,
        task_type: Optional[DatasetTaskType] = None,
        format: Optional[DatasetFormat] = None,
    ) -> list[DatasetMetadata]:
        """
        List datasets with optional filtering.

        Args:
            task_type: Optional task type filter
            format: Optional format filter

        Returns:
            List of DatasetMetadata

        Raises:
            RuntimeError: If listing fails
        """
        try:
            # Scan DynamoDB table
            response = self.dynamodb_client.scan(TableName=self.metadata_table)

            items = response.get("Items", [])
            datasets = []

            for item in items:
                metadata = self._dynamodb_item_to_metadata(item)

                # Apply filters
                if task_type and metadata.task_type != task_type:
                    continue

                if format and metadata.format != format:
                    continue

                datasets.append(metadata)

            # Sort by creation date (newest first)
            datasets.sort(key=lambda x: x.created_at, reverse=True)

            return datasets

        except Exception as e:
            raise RuntimeError(f"Failed to list datasets: {e}") from e

    async def get_version_history(
        self, dataset_id: str
    ) -> list[DatasetMetadata]:
        """
        Get version history for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            List of DatasetMetadata for all versions

        Raises:
            ValueError: If dataset not found
            RuntimeError: If retrieval fails

        Requirement 2.9, 2.17
        """
        # Get all versions from version manager
        versions = self.version_manager.list_versions(dataset_id)

        if not versions:
            raise ValueError(f"No versions found for dataset: {dataset_id}")

        # Get base metadata
        base_metadata = await self.get_dataset(dataset_id)

        if not base_metadata:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Create metadata for each version
        version_metadatas = []

        for version in versions:
            version_metadata = DatasetMetadata(
                id=dataset_id,
                name=base_metadata.name,
                description=version.get("description", ""),
                format=DatasetFormat(version["format"]),
                task_type=DatasetTaskType(version["task_type"]),
                version=version["version_id"],
                s3_uri=version["s3_uri"],
                row_count=version["row_count"],
                token_stats=base_metadata.token_stats,  # Use base token stats
                created_at=datetime.fromisoformat(version["created_at"]),
                checksum=version["checksum"],
                lineage=None,
                quality_report=None,
            )
            version_metadatas.append(version_metadata)

        return version_metadatas

    # Helper methods

    def _parse_dataset(
        self, file_path: Path, format_type: DatasetFormat, task_type: DatasetTaskType
    ) -> list[dict[str, Any]]:
        """Parse dataset file based on format."""
        if format_type == DatasetFormat.JSONL:
            return self.jsonl_parser.parse(str(file_path), task_type, validate=False)
        elif format_type == DatasetFormat.CSV:
            return self.csv_parser.parse(str(file_path), task_type, validate=False)
        elif format_type == DatasetFormat.PARQUET:
            return self.parquet_parser.parse(str(file_path), task_type, validate=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _parse_content(
        self, content: bytes, format_type: DatasetFormat
    ) -> list[dict[str, Any]]:
        """Parse dataset content based on format."""
        if format_type == DatasetFormat.JSONL:
            lines = content.decode("utf-8").strip().split("\n")
            examples = []
            for line in lines:
                if line.strip():
                    examples.append(json.loads(line))
            return examples
        elif format_type == DatasetFormat.CSV:
            import csv
            import io

            reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
            return list(reader)
        elif format_type == DatasetFormat.PARQUET:
            import io

            table = pq.read_table(io.BytesIO(content))
            return table.to_pylist()
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _serialize_examples(
        self, examples: list[dict[str, Any]], format_type: DatasetFormat
    ) -> bytes:
        """Serialize examples to bytes based on format."""
        if format_type == DatasetFormat.JSONL:
            lines = [json.dumps(ex) for ex in examples]
            return "\n".join(lines).encode("utf-8")
        elif format_type == DatasetFormat.CSV:
            import csv
            import io

            if not examples:
                return b""

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=examples[0].keys())
            writer.writeheader()
            writer.writerows(examples)
            return output.getvalue().encode("utf-8")
        elif format_type == DatasetFormat.PARQUET:
            import io
            import pyarrow as pa

            table = pa.Table.from_pylist(examples)
            output = io.BytesIO()
            pq.write_table(table, output)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    async def _create_split_dataset(
        self,
        parent_dataset_id: str,
        parent_metadata: DatasetMetadata,
        examples: list[dict[str, Any]],
        split_name: str,
        split_ratio: float,
        stratify_by: Optional[str],
    ) -> DatasetMetadata:
        """Create a new dataset from a split."""
        # Generate new dataset ID
        split_dataset_id = str(uuid.uuid4())

        # Serialize examples
        content = self._serialize_examples(examples, parent_metadata.format)

        # Calculate token statistics
        quality_report = self.quality_analyzer.analyze(
            examples, parent_metadata.task_type, parent_metadata.format
        )

        # Create version first to get s3_uri and checksum
        # We need a temporary metadata object for the version manager
        temp_metadata = DatasetMetadata(
            id=split_dataset_id,
            name=f"{parent_metadata.name}_{split_name}",
            description=f"{split_name.capitalize()} split from {parent_metadata.name}",
            format=parent_metadata.format,
            task_type=parent_metadata.task_type,
            version="v1",
            s3_uri="s3://placeholder",  # Temporary placeholder
            row_count=len(examples),
            token_stats=quality_report.token_stats,
            created_at=datetime.utcnow(),
            checksum="placeholder",  # Temporary placeholder
            lineage=None,
            quality_report=quality_report,
        )

        # Create version
        version_info = self.version_manager.create_version(
            dataset_id=split_dataset_id,
            content=content,
            metadata=temp_metadata,
            description=f"{split_name.capitalize()} split",
            parent_version_id=parent_metadata.version,
        )

        # Record transformation
        self.version_manager.record_transformation(
            dataset_id=split_dataset_id,
            version_id=version_info["version_id"],
            transformation_type="dataset_split",
            transformation_details={
                "parent_dataset_id": parent_dataset_id,
                "split_name": split_name,
                "split_ratio": split_ratio,
                "stratify_by": stratify_by,
                "row_count": len(examples),
            },
        )

        # Create final metadata with actual s3_uri and checksum
        split_metadata = DatasetMetadata(
            id=split_dataset_id,
            name=f"{parent_metadata.name}_{split_name}",
            description=f"{split_name.capitalize()} split from {parent_metadata.name}",
            format=parent_metadata.format,
            task_type=parent_metadata.task_type,
            version="v1",
            s3_uri=version_info["s3_uri"],
            row_count=len(examples),
            token_stats=quality_report.token_stats,
            created_at=datetime.utcnow(),
            checksum=version_info["checksum"],
            lineage=None,
            quality_report=quality_report,
        )

        # Store metadata
        self._store_metadata(split_metadata)

        return split_metadata

    @retry_with_exponential_backoff(
        max_retries=3, initial_delay=1.0, backoff_multiplier=2.0
    )
    def _store_metadata(self, metadata: DatasetMetadata) -> None:
        """Store dataset metadata in DynamoDB."""
        try:
            item = {
                "dataset_id": {"S": metadata.id},
                "name": {"S": metadata.name},
                "description": {"S": metadata.description or ""},
                "format": {"S": metadata.format.value},
                "task_type": {"S": metadata.task_type.value},
                "version": {"S": metadata.version},
                "s3_uri": {"S": metadata.s3_uri},
                "row_count": {"N": str(metadata.row_count)},
                "created_at": {"S": metadata.created_at.isoformat()},
                "checksum": {"S": metadata.checksum},
                "total_tokens": {"N": str(metadata.token_stats.total_tokens)},
                "min_tokens": {"N": str(metadata.token_stats.min_tokens)},
                "max_tokens": {"N": str(metadata.token_stats.max_tokens)},
                "avg_tokens": {"N": str(metadata.token_stats.avg_tokens)},
                "p95_tokens": {"N": str(metadata.token_stats.p95_tokens)},
                "prompt_tokens": {"N": str(metadata.token_stats.prompt_tokens)},
                "completion_tokens": {"N": str(metadata.token_stats.completion_tokens)},
            }

            # Add quality report if available
            if metadata.quality_report:
                item["completeness_score"] = {
                    "N": str(metadata.quality_report.completeness_score)
                }
                item["diversity_score"] = {
                    "N": str(metadata.quality_report.diversity_score)
                }
                item["balance_score"] = {
                    "N": str(metadata.quality_report.balance_score)
                }

            self.dynamodb_client.put_item(
                TableName=self.metadata_table,
                Item=item,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to store dataset metadata: {e}") from e

    def _dynamodb_item_to_metadata(self, item: dict[str, Any]) -> DatasetMetadata:
        """Convert DynamoDB item to DatasetMetadata."""
        token_stats = TokenStatistics(
            total_tokens=int(item["total_tokens"]["N"]),
            min_tokens=int(item["min_tokens"]["N"]),
            max_tokens=int(item["max_tokens"]["N"]),
            avg_tokens=float(item["avg_tokens"]["N"]),
            p95_tokens=int(item["p95_tokens"]["N"]),
            prompt_tokens=int(item["prompt_tokens"]["N"]),
            completion_tokens=int(item["completion_tokens"]["N"]),
        )

        # Create quality report if scores are available
        quality_report = None
        if "completeness_score" in item:
            quality_report = DatasetQualityReport(
                completeness_score=float(item["completeness_score"]["N"]),
                diversity_score=float(item["diversity_score"]["N"]),
                balance_score=float(item["balance_score"]["N"]),
                token_stats=token_stats,
                issues=[],
                recommendations=[],
            )

        return DatasetMetadata(
            id=item["dataset_id"]["S"],
            name=item["name"]["S"],
            description=item["description"]["S"] or None,
            format=DatasetFormat(item["format"]["S"]),
            task_type=DatasetTaskType(item["task_type"]["S"]),
            version=item["version"]["S"],
            s3_uri=item["s3_uri"]["S"],
            row_count=int(item["row_count"]["N"]),
            token_stats=token_stats,
            created_at=datetime.fromisoformat(item["created_at"]["S"]),
            checksum=item["checksum"]["S"],
            lineage=None,
            quality_report=quality_report,
        )
