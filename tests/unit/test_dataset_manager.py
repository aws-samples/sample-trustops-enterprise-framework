"""
Unit tests for DatasetManager.

Tests the high-level dataset orchestration including upload, validation,
transformation, and versioning operations.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.data_models.dataset import (
    DatasetFormat,
    DatasetMetadata,
    DatasetQualityReport,
    DatasetTaskType,
    QualityIssue,
    TokenStatistics,
)
from src.datasets.dataset_manager import DatasetManager
from src.datasets.pii_detector import PIIDetectionResult, PIIType


@pytest.fixture
def mock_aws_clients():
    """Mock AWS clients."""
    with patch("boto3.Session") as mock_session:
        mock_s3 = MagicMock()
        mock_dynamodb = MagicMock()

        mock_session_instance = MagicMock()
        mock_session_instance.client.side_effect = lambda service: (
            mock_s3 if service == "s3" else mock_dynamodb
        )
        mock_session.return_value = mock_session_instance

        yield mock_s3, mock_dynamodb


@pytest.fixture
def sample_jsonl_file():
    """Create a sample JSONL file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        examples = [
            {"prompt": "What is AI?", "completion": "Artificial Intelligence"},
            {"prompt": "What is ML?", "completion": "Machine Learning"},
            {"prompt": "What is DL?", "completion": "Deep Learning"},
        ]
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def dataset_manager(mock_aws_clients):
    """Create DatasetManager instance with mocked AWS clients."""
    return DatasetManager(
        region="us-east-1",
        datasets_bucket="test-bucket",
        metadata_table="test-table",
    )


class TestDatasetManagerUpload:
    """Test dataset upload functionality."""

    @pytest.mark.asyncio
    async def test_upload_dataset_success(
        self, dataset_manager, sample_jsonl_file, mock_aws_clients
    ):
        """Test successful dataset upload."""
        mock_s3, mock_dynamodb = mock_aws_clients

        # Mock version manager
        with patch.object(
            dataset_manager.version_manager, "create_version"
        ) as mock_create_version:
            mock_create_version.return_value = {
                "version_id": "v1",
                "s3_uri": "s3://test-bucket/datasets/test-id/versions/v1/data",
                "checksum": "abc123",
                "size_bytes": 1024,
                "created_at": datetime.utcnow().isoformat(),
            }

            # Upload dataset
            metadata = await dataset_manager.upload_dataset(
                file_path=sample_jsonl_file,
                name="Test Dataset",
                description="Test description",
            )

            # Verify metadata
            assert metadata.name == "Test Dataset"
            assert metadata.description == "Test description"
            assert metadata.format == DatasetFormat.JSONL
            assert metadata.row_count == 3
            assert metadata.checksum == "abc123"

            # Verify version was created
            mock_create_version.assert_called_once()

            # Verify metadata was stored
            mock_dynamodb.put_item.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_dataset_file_not_found(self, dataset_manager):
        """Test upload with non-existent file."""
        with pytest.raises(ValueError, match="File not found"):
            await dataset_manager.upload_dataset(
                file_path="/nonexistent/file.jsonl",
                name="Test Dataset",
            )

    @pytest.mark.asyncio
    async def test_upload_dataset_exceeds_size_limit(
        self, dataset_manager, sample_jsonl_file
    ):
        """Test upload with file exceeding size limit."""
        # Set very small size limit
        dataset_manager.max_dataset_size_bytes = 10

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            await dataset_manager.upload_dataset(
                file_path=sample_jsonl_file,
                name="Test Dataset",
            )

    @pytest.mark.asyncio
    async def test_upload_dataset_size_validation_default_limit(self):
        """Test that default size limit is 10GB."""
        manager = DatasetManager(
            region="us-east-1",
            datasets_bucket="test-bucket",
        )
        
        # Verify default is 10GB
        expected_bytes = 10 * 1024 * 1024 * 1024
        assert manager.max_dataset_size_bytes == expected_bytes

    @pytest.mark.asyncio
    async def test_upload_dataset_size_validation_custom_limit(self):
        """Test configurable size limit."""
        # Create manager with custom 5GB limit
        manager = DatasetManager(
            region="us-east-1",
            datasets_bucket="test-bucket",
            max_dataset_size_gb=5.0,
        )
        
        # Verify custom limit is set
        expected_bytes = 5 * 1024 * 1024 * 1024
        assert manager.max_dataset_size_bytes == expected_bytes

    @pytest.mark.asyncio
    async def test_upload_dataset_size_error_message_format(
        self, dataset_manager, sample_jsonl_file
    ):
        """Test that size error message includes actual vs max size and remediation."""
        # Get actual file size and set limit below it
        file_size = Path(sample_jsonl_file).stat().st_size
        dataset_manager.max_dataset_size_bytes = file_size - 1
        
        try:
            await dataset_manager.upload_dataset(
                file_path=sample_jsonl_file,
                name="Test Dataset",
            )
            pytest.fail("Expected ValueError to be raised")
        except ValueError as e:
            error_msg = str(e)
            # Verify error message contains key information
            assert "exceeds maximum allowed size" in error_msg
            assert "GB" in error_msg  # Size in GB
            assert "split" in error_msg.lower()  # Remediation suggestion
            assert "contact support" in error_msg.lower()  # Alternative remediation

    @pytest.mark.asyncio
    async def test_upload_dataset_size_at_limit(
        self, dataset_manager, sample_jsonl_file, mock_aws_clients
    ):
        """Test upload with file exactly at size limit."""
        mock_s3, mock_dynamodb = mock_aws_clients
        
        # Get actual file size
        file_size = Path(sample_jsonl_file).stat().st_size
        
        # Set limit to exactly the file size
        dataset_manager.max_dataset_size_bytes = file_size
        
        with patch.object(
            dataset_manager.version_manager, "create_version"
        ) as mock_create_version:
            mock_create_version.return_value = {
                "version_id": "v1",
                "s3_uri": "s3://test-bucket/test",
                "checksum": "abc123",
                "size_bytes": file_size,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            # Should succeed when exactly at limit
            metadata = await dataset_manager.upload_dataset(
                file_path=sample_jsonl_file,
                name="Test Dataset",
            )
            
            assert metadata is not None
            assert metadata.name == "Test Dataset"

    @pytest.mark.asyncio
    async def test_upload_dataset_size_just_over_limit(
        self, dataset_manager, sample_jsonl_file
    ):
        """Test upload with file just 1 byte over limit."""
        # Get actual file size
        file_size = Path(sample_jsonl_file).stat().st_size
        
        # Set limit to 1 byte less than file size
        dataset_manager.max_dataset_size_bytes = file_size - 1
        
        # Should fail when over limit
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            await dataset_manager.upload_dataset(
                file_path=sample_jsonl_file,
                name="Test Dataset",
            )

    @pytest.mark.asyncio
    async def test_upload_dataset_size_validation_before_parsing(
        self, dataset_manager, sample_jsonl_file
    ):
        """Test that size validation happens before expensive parsing operations."""
        # Set very small limit
        dataset_manager.max_dataset_size_bytes = 10
        
        # Mock the parser to track if it was called
        with patch.object(
            dataset_manager, "_parse_dataset"
        ) as mock_parse:
            try:
                await dataset_manager.upload_dataset(
                    file_path=sample_jsonl_file,
                    name="Test Dataset",
                )
            except ValueError:
                pass  # Expected
            
            # Parser should NOT have been called (fail fast)
            mock_parse.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_dataset_auto_detect_format(
        self, dataset_manager, sample_jsonl_file, mock_aws_clients
    ):
        """Test automatic format detection during upload."""
        mock_s3, mock_dynamodb = mock_aws_clients

        with patch.object(
            dataset_manager.version_manager, "create_version"
        ) as mock_create_version:
            mock_create_version.return_value = {
                "version_id": "v1",
                "s3_uri": "s3://test-bucket/test",
                "checksum": "abc123",
                "size_bytes": 1024,
                "created_at": datetime.utcnow().isoformat(),
            }

            metadata = await dataset_manager.upload_dataset(
                file_path=sample_jsonl_file,
                name="Test Dataset",
            )

            # Format should be auto-detected as JSONL
            assert metadata.format == DatasetFormat.JSONL
            # Task type should be auto-detected
            assert metadata.task_type in [
                DatasetTaskType.TEXT_GENERATION,
                DatasetTaskType.QA,
            ]


class TestDatasetManagerValidation:
    """Test dataset validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_dataset_success(self, dataset_manager):
        """Test successful dataset validation."""
        # Mock get_dataset
        mock_metadata = DatasetMetadata(
            id="test-id",
            name="Test Dataset",
            description=None,
            format=DatasetFormat.JSONL,
            task_type=DatasetTaskType.TEXT_GENERATION,
            version="v1",
            s3_uri="s3://test-bucket/test",
            row_count=3,
            token_stats=TokenStatistics(
                total_tokens=100,
                min_tokens=10,
                max_tokens=50,
                avg_tokens=33.3,
                p95_tokens=45,
                prompt_tokens=60,
                completion_tokens=40,
            ),
            created_at=datetime.utcnow(),
            checksum="abc123",
        )

        with patch.object(
            dataset_manager, "get_dataset", return_value=mock_metadata
        ):
            with patch.object(
                dataset_manager.version_manager, "get_version"
            ) as mock_get_version:
                # Mock version content
                examples = [
                    {"prompt": "What is AI?", "completion": "Artificial Intelligence"},
                    {"prompt": "What is ML?", "completion": "Machine Learning"},
                ]
                content = "\n".join([json.dumps(ex) for ex in examples]).encode()

                mock_get_version.return_value = {
                    "content": content,
                    "metadata": {},
                    "checksum": "abc123",
                }

                # Validate dataset
                quality_report = await dataset_manager.validate_dataset("test-id")

                # Verify quality report
                assert isinstance(quality_report, DatasetQualityReport)
                assert 0.0 <= quality_report.completeness_score <= 1.0
                assert 0.0 <= quality_report.diversity_score <= 1.0
                assert 0.0 <= quality_report.balance_score <= 1.0

    @pytest.mark.asyncio
    async def test_validate_dataset_not_found(self, dataset_manager):
        """Test validation with non-existent dataset."""
        with patch.object(dataset_manager, "get_dataset", return_value=None):
            with pytest.raises(ValueError, match="Dataset not found"):
                await dataset_manager.validate_dataset("nonexistent-id")


class TestDatasetManagerTransformations:
    """Test dataset transformation operations."""

    @pytest.mark.asyncio
    async def test_detect_pii_success(self, dataset_manager):
        """Test PII detection."""
        # Mock get_dataset
        mock_metadata = DatasetMetadata(
            id="test-id",
            name="Test Dataset",
            description=None,
            format=DatasetFormat.JSONL,
            task_type=DatasetTaskType.TEXT_GENERATION,
            version="v1",
            s3_uri="s3://test-bucket/test",
            row_count=2,
            token_stats=TokenStatistics(
                total_tokens=100,
                min_tokens=10,
                max_tokens=50,
                avg_tokens=33.3,
                p95_tokens=45,
                prompt_tokens=60,
                completion_tokens=40,
            ),
            created_at=datetime.utcnow(),
            checksum="abc123",
        )

        with patch.object(
            dataset_manager, "get_dataset", return_value=mock_metadata
        ):
            with patch.object(
                dataset_manager.version_manager, "get_version"
            ) as mock_get_version:
                # Mock version content with PII
                examples = [
                    {
                        "prompt": "Contact me at john@example.com",
                        "completion": "OK",
                    },
                    {"prompt": "My phone is 555-123-4567", "completion": "OK"},
                ]
                content = "\n".join([json.dumps(ex) for ex in examples]).encode()

                mock_get_version.return_value = {
                    "content": content,
                    "metadata": {},
                    "checksum": "abc123",
                }

                # Detect PII
                pii_result = await dataset_manager.detect_pii("test-id")

                # Verify PII detection
                assert isinstance(pii_result, PIIDetectionResult)
                assert pii_result.total_rows == 2
                assert pii_result.rows_with_pii > 0

    @pytest.mark.asyncio
    async def test_mask_pii_success(self, dataset_manager, mock_aws_clients):
        """Test PII masking."""
        mock_s3, mock_dynamodb = mock_aws_clients

        # Mock get_dataset
        mock_metadata = DatasetMetadata(
            id="test-id",
            name="Test Dataset",
            description=None,
            format=DatasetFormat.JSONL,
            task_type=DatasetTaskType.TEXT_GENERATION,
            version="v1",
            s3_uri="s3://test-bucket/test",
            row_count=1,
            token_stats=TokenStatistics(
                total_tokens=100,
                min_tokens=10,
                max_tokens=50,
                avg_tokens=33.3,
                p95_tokens=45,
                prompt_tokens=60,
                completion_tokens=40,
            ),
            created_at=datetime.utcnow(),
            checksum="abc123",
        )

        with patch.object(
            dataset_manager, "get_dataset", return_value=mock_metadata
        ):
            with patch.object(
                dataset_manager.version_manager, "get_version"
            ) as mock_get_version:
                with patch.object(
                    dataset_manager.version_manager, "create_version"
                ) as mock_create_version:
                    with patch.object(
                        dataset_manager.version_manager, "record_transformation"
                    ):
                        # Mock version content with PII
                        examples = [
                            {
                                "prompt": "Contact me at john@example.com",
                                "completion": "OK",
                            }
                        ]
                        content = "\n".join([json.dumps(ex) for ex in examples]).encode()

                        mock_get_version.return_value = {
                            "content": content,
                            "metadata": {},
                            "checksum": "abc123",
                        }

                        mock_create_version.return_value = {
                            "version_id": "v2",
                            "s3_uri": "s3://test-bucket/test-v2",
                            "checksum": "def456",
                            "size_bytes": 1024,
                            "created_at": datetime.utcnow().isoformat(),
                        }

                        # Mask PII
                        masked_metadata = await dataset_manager.mask_pii(
                            "test-id", strategy="mask"
                        )

                        # Verify new version was created
                        assert masked_metadata.version == "v2"
                        assert masked_metadata.checksum == "def456"

                        # Verify transformation was recorded
                        mock_create_version.assert_called_once()


class TestDatasetManagerSplitting:
    """Test dataset splitting functionality."""

    @pytest.mark.asyncio
    async def test_split_dataset_success(self, dataset_manager, mock_aws_clients):
        """Test successful dataset splitting."""
        mock_s3, mock_dynamodb = mock_aws_clients

        # Mock get_dataset
        mock_metadata = DatasetMetadata(
            id="test-id",
            name="Test Dataset",
            description=None,
            format=DatasetFormat.JSONL,
            task_type=DatasetTaskType.TEXT_GENERATION,
            version="v1",
            s3_uri="s3://test-bucket/test",
            row_count=10,
            token_stats=TokenStatistics(
                total_tokens=100,
                min_tokens=10,
                max_tokens=50,
                avg_tokens=33.3,
                p95_tokens=45,
                prompt_tokens=60,
                completion_tokens=40,
            ),
            created_at=datetime.utcnow(),
            checksum="abc123",
        )

        with patch.object(
            dataset_manager, "get_dataset", return_value=mock_metadata
        ):
            with patch.object(
                dataset_manager.version_manager, "get_version"
            ) as mock_get_version:
                with patch.object(
                    dataset_manager.version_manager, "create_version"
                ) as mock_create_version:
                    with patch.object(
                        dataset_manager.version_manager, "record_transformation"
                    ):
                        # Mock version content
                        examples = [
                            {"prompt": f"Question {i}", "completion": f"Answer {i}"}
                            for i in range(10)
                        ]
                        content = "\n".join([json.dumps(ex) for ex in examples]).encode()

                        mock_get_version.return_value = {
                            "content": content,
                            "metadata": {},
                            "checksum": "abc123",
                        }

                        mock_create_version.return_value = {
                            "version_id": "v1",
                            "s3_uri": "s3://test-bucket/test-split",
                            "checksum": "split123",
                            "size_bytes": 1024,
                            "created_at": datetime.utcnow().isoformat(),
                        }

                        # Split dataset
                        train, val, test = await dataset_manager.split_dataset(
                            "test-id",
                            train_ratio=0.8,
                            validation_ratio=0.1,
                            test_ratio=0.1,
                        )

                        # Verify splits
                        assert train.name == "Test Dataset_train"
                        assert val.name == "Test Dataset_validation"
                        assert test.name == "Test Dataset_test"

                        # Verify row counts (approximately)
                        assert train.row_count == 8
                        assert val.row_count == 1
                        assert test.row_count == 1


class TestDatasetManagerQueries:
    """Test dataset query operations."""

    @pytest.mark.asyncio
    async def test_get_dataset_success(self, dataset_manager, mock_aws_clients):
        """Test successful dataset retrieval."""
        mock_s3, mock_dynamodb = mock_aws_clients

        # Mock DynamoDB response
        mock_dynamodb.get_item.return_value = {
            "Item": {
                "dataset_id": {"S": "test-id"},
                "name": {"S": "Test Dataset"},
                "description": {"S": "Test description"},
                "format": {"S": "jsonl"},
                "task_type": {"S": "text_generation"},
                "version": {"S": "v1"},
                "s3_uri": {"S": "s3://test-bucket/test"},
                "row_count": {"N": "10"},
                "created_at": {"S": datetime.utcnow().isoformat()},
                "checksum": {"S": "abc123"},
                "total_tokens": {"N": "100"},
                "min_tokens": {"N": "10"},
                "max_tokens": {"N": "50"},
                "avg_tokens": {"N": "33.3"},
                "p95_tokens": {"N": "45"},
                "prompt_tokens": {"N": "60"},
                "completion_tokens": {"N": "40"},
            }
        }

        # Get dataset
        metadata = await dataset_manager.get_dataset("test-id")

        # Verify metadata
        assert metadata is not None
        assert metadata.id == "test-id"
        assert metadata.name == "Test Dataset"
        assert metadata.format == DatasetFormat.JSONL

    @pytest.mark.asyncio
    async def test_get_dataset_not_found(self, dataset_manager, mock_aws_clients):
        """Test dataset retrieval when not found."""
        mock_s3, mock_dynamodb = mock_aws_clients

        # Mock DynamoDB response (no item)
        mock_dynamodb.get_item.return_value = {}

        # Get dataset
        metadata = await dataset_manager.get_dataset("nonexistent-id")

        # Verify None returned
        assert metadata is None

    @pytest.mark.asyncio
    async def test_list_datasets_success(self, dataset_manager, mock_aws_clients):
        """Test successful dataset listing."""
        mock_s3, mock_dynamodb = mock_aws_clients

        # Mock DynamoDB scan response
        mock_dynamodb.scan.return_value = {
            "Items": [
                {
                    "dataset_id": {"S": "test-id-1"},
                    "name": {"S": "Dataset 1"},
                    "description": {"S": ""},
                    "format": {"S": "jsonl"},
                    "task_type": {"S": "text_generation"},
                    "version": {"S": "v1"},
                    "s3_uri": {"S": "s3://test-bucket/test1"},
                    "row_count": {"N": "10"},
                    "created_at": {"S": datetime.utcnow().isoformat()},
                    "checksum": {"S": "abc123"},
                    "total_tokens": {"N": "100"},
                    "min_tokens": {"N": "10"},
                    "max_tokens": {"N": "50"},
                    "avg_tokens": {"N": "33.3"},
                    "p95_tokens": {"N": "45"},
                    "prompt_tokens": {"N": "60"},
                    "completion_tokens": {"N": "40"},
                },
                {
                    "dataset_id": {"S": "test-id-2"},
                    "name": {"S": "Dataset 2"},
                    "description": {"S": ""},
                    "format": {"S": "csv"},
                    "task_type": {"S": "qa"},
                    "version": {"S": "v1"},
                    "s3_uri": {"S": "s3://test-bucket/test2"},
                    "row_count": {"N": "20"},
                    "created_at": {"S": datetime.utcnow().isoformat()},
                    "checksum": {"S": "def456"},
                    "total_tokens": {"N": "200"},
                    "min_tokens": {"N": "20"},
                    "max_tokens": {"N": "60"},
                    "avg_tokens": {"N": "40.0"},
                    "p95_tokens": {"N": "55"},
                    "prompt_tokens": {"N": "120"},
                    "completion_tokens": {"N": "80"},
                },
            ]
        }

        # List all datasets
        datasets = await dataset_manager.list_datasets()

        # Verify results
        assert len(datasets) == 2
        assert datasets[0].name in ["Dataset 1", "Dataset 2"]

    @pytest.mark.asyncio
    async def test_list_datasets_with_filters(
        self, dataset_manager, mock_aws_clients
    ):
        """Test dataset listing with filters."""
        mock_s3, mock_dynamodb = mock_aws_clients

        # Mock DynamoDB scan response
        mock_dynamodb.scan.return_value = {
            "Items": [
                {
                    "dataset_id": {"S": "test-id-1"},
                    "name": {"S": "Dataset 1"},
                    "description": {"S": ""},
                    "format": {"S": "jsonl"},
                    "task_type": {"S": "text_generation"},
                    "version": {"S": "v1"},
                    "s3_uri": {"S": "s3://test-bucket/test1"},
                    "row_count": {"N": "10"},
                    "created_at": {"S": datetime.utcnow().isoformat()},
                    "checksum": {"S": "abc123"},
                    "total_tokens": {"N": "100"},
                    "min_tokens": {"N": "10"},
                    "max_tokens": {"N": "50"},
                    "avg_tokens": {"N": "33.3"},
                    "p95_tokens": {"N": "45"},
                    "prompt_tokens": {"N": "60"},
                    "completion_tokens": {"N": "40"},
                },
                {
                    "dataset_id": {"S": "test-id-2"},
                    "name": {"S": "Dataset 2"},
                    "description": {"S": ""},
                    "format": {"S": "csv"},
                    "task_type": {"S": "qa"},
                    "version": {"S": "v1"},
                    "s3_uri": {"S": "s3://test-bucket/test2"},
                    "row_count": {"N": "20"},
                    "created_at": {"S": datetime.utcnow().isoformat()},
                    "checksum": {"S": "def456"},
                    "total_tokens": {"N": "200"},
                    "min_tokens": {"N": "20"},
                    "max_tokens": {"N": "60"},
                    "avg_tokens": {"N": "40.0"},
                    "p95_tokens": {"N": "55"},
                    "prompt_tokens": {"N": "120"},
                    "completion_tokens": {"N": "80"},
                },
            ]
        }

        # List datasets with task_type filter
        datasets = await dataset_manager.list_datasets(
            task_type=DatasetTaskType.QA
        )

        # Verify only QA datasets returned
        assert len(datasets) == 1
        assert datasets[0].task_type == DatasetTaskType.QA


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
