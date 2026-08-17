"""
Dataset versioning manager for tracking dataset changes over time.

This module provides functionality for versioning datasets with S3 storage
and DynamoDB metadata tracking. It calculates SHA-256 checksums for integrity
verification and maintains version history.

Requirements: 2.9, 2.10, 2.14, 2.17, 2.18
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import boto3
from botocore.exceptions import ClientError

from config.aws_config import config
from src.data_models.dataset import DatasetMetadata, DatasetLineage
from src.utils.retry_utils import retry_with_exponential_backoff
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatasetVersionManager:
    """
    Manages dataset versioning with S3 storage and DynamoDB metadata.
    
    This class provides functionality to:
    - Store dataset versions in S3 with versioned paths
    - Store metadata in DynamoDB for fast querying
    - Calculate SHA-256 checksums for integrity verification
    - Track version history and lineage
    - Record transformations applied to datasets
    - Track usage history (evaluations and fine-tuning jobs)
    - Support version listing, retrieval, and comparison
    
    Requirements: 2.9, 2.10, 2.14, 2.17, 2.18
    """
    
    def __init__(
        self,
        region: Optional[str] = None,
        datasets_bucket: Optional[str] = None,
        versions_table: Optional[str] = None,
        lineage_table: Optional[str] = None
    ):
        """
        Initialize the dataset version manager.
        
        Args:
            region: AWS region (defaults to config.region)
            datasets_bucket: S3 bucket for datasets
                (defaults to config.datasets_bucket)
            versions_table: DynamoDB table for version metadata
                (defaults to 'trustops-dataset-versions')
            lineage_table: DynamoDB table for lineage tracking
                (defaults to 'trustops-dataset-lineage')
        """
        self.region = region or config.region
        self.datasets_bucket = datasets_bucket or config.datasets_bucket
        self.versions_table = versions_table or os.getenv(
            "TRUSTOPS_DATASET_VERSIONS_TABLE", "trustops-dataset-versions"
        )
        self.lineage_table = lineage_table or os.getenv(
            "TRUSTOPS_DATASET_LINEAGE_TABLE", "trustops-dataset-lineage"
        )
        
        # Initialize AWS clients
        session_kwargs = config.get_boto3_session_kwargs()
        if region:
            session_kwargs['region_name'] = region
        
        session = boto3.Session(**session_kwargs)
        self.s3_client = session.client('s3')
        self.dynamodb_client = session.client('dynamodb')
    
    def calculate_checksum(self, content: bytes) -> str:
        """
        Calculate SHA-256 checksum for content.
        
        Args:
            content: Content bytes to checksum
            
        Returns:
            SHA-256 checksum as hex string
            
        Requirement 2.17: Calculate SHA-256 checksums for integrity verification
        """
        return hashlib.sha256(content).hexdigest()
    
    def create_version(
            self,
            dataset_id: str,
            content: bytes,
            metadata: DatasetMetadata,
            description: Optional[str] = None,
            parent_version_id: Optional[str] = None,
            progress_callback: Optional[callable] = None
        ) -> Dict[str, Any]:
            """
            Create a new version of a dataset.

            For datasets >100MB, automatically uses multipart upload with progress tracking.

            Args:
                dataset_id: Unique identifier for the dataset
                content: Dataset content as bytes
                metadata: Dataset metadata
                description: Optional description of this version
                parent_version_id: Optional parent version ID for lineage tracking
                progress_callback: Optional callback function(bytes_uploaded, total_bytes)

            Returns:
                Dictionary containing:
                    - version_id: Unique version identifier
                    - s3_uri: S3 URI of the versioned dataset
                    - checksum: SHA-256 checksum
                    - size_bytes: Size of the content
                    - created_at: Creation timestamp

            Raises:
                RuntimeError: If version creation fails

            Requirements: 2.9, 2.16, 2.17
            """
            try:
                # Generate version ID (timestamp-based)
                timestamp = datetime.utcnow()
                version_id = f"v{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

                # Calculate checksum
                checksum = self.calculate_checksum(content)

                # Construct S3 key with versioned path
                s3_key = f"datasets/{dataset_id}/versions/{version_id}/data"

                # Use multipart upload for large files (>100MB)
                size_bytes = len(content)
                multipart_threshold = 100 * 1024 * 1024  # 100MB

                if size_bytes > multipart_threshold:
                    # Use multipart upload for large files
                    self._multipart_upload(
                        content=content,
                        s3_key=s3_key,
                        dataset_id=dataset_id,
                        version_id=version_id,
                        checksum=checksum,
                        timestamp=timestamp,
                        progress_callback=progress_callback
                    )
                else:
                    # Use standard upload for smaller files
                    self.s3_client.put_object(
                        Bucket=self.datasets_bucket,
                        Key=s3_key,
                        Body=content,
                        Metadata={
                            'dataset_id': dataset_id,
                            'version_id': version_id,
                            'checksum': checksum,
                            'created_at': timestamp.isoformat()
                        }
                    )

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(size_bytes, size_bytes)

                s3_uri = f"s3://{self.datasets_bucket}/{s3_key}"

                # Store metadata in DynamoDB
                version_metadata = {
                    'dataset_id': dataset_id,
                    'version_id': version_id,
                    's3_uri': s3_uri,
                    'checksum': checksum,
                    'size_bytes': size_bytes,
                    'created_at': timestamp.isoformat(),
                    'description': description or '',
                    'parent_version_id': parent_version_id or '',
                    'format': metadata.format.value,
                    'task_type': metadata.task_type.value,
                    'row_count': metadata.row_count,
                    'name': metadata.name
                }

                self._store_version_metadata(version_metadata)

                return {
                    'version_id': version_id,
                    's3_uri': s3_uri,
                    'checksum': checksum,
                    'size_bytes': size_bytes,
                    'created_at': timestamp.isoformat()
                }

            except ClientError as e:
                error_message = e.response.get('Error', {}).get('Message', '')
                raise RuntimeError(
                    f"Failed to create dataset version: {error_message}"
                ) from e
    def _multipart_upload(
            self,
            content: bytes,
            s3_key: str,
            dataset_id: str,
            version_id: str,
            checksum: str,
            timestamp: datetime,
            progress_callback: Optional[callable] = None
        ) -> None:
            """
            Upload large dataset using S3 multipart upload.

            This method:
            1. Initiates a multipart upload
            2. Splits content into chunks (5-10MB per part)
            3. Uploads parts in parallel (up to 5 concurrent)
            4. Tracks progress and supports resumption
            5. Completes the multipart upload

            Args:
                content: Dataset content as bytes
                s3_key: S3 key for the upload
                dataset_id: Dataset identifier
                version_id: Version identifier
                checksum: SHA-256 checksum
                timestamp: Creation timestamp
                progress_callback: Optional callback function(bytes_uploaded, total_bytes)

            Raises:
                RuntimeError: If multipart upload fails

            Requirement 2.16: Multipart upload for datasets >100MB
            """
            import concurrent.futures
            from io import BytesIO

            # Configuration
            part_size = 10 * 1024 * 1024  # 10MB per part
            max_workers = 5  # Parallel upload threads

            try:
                # Initiate multipart upload
                response = self.s3_client.create_multipart_upload(
                    Bucket=self.datasets_bucket,
                    Key=s3_key,
                    Metadata={
                        'dataset_id': dataset_id,
                        'version_id': version_id,
                        'checksum': checksum,
                        'created_at': timestamp.isoformat()
                    }
                )
                upload_id = response['UploadId']

                # Split content into parts
                total_size = len(content)
                num_parts = (total_size + part_size - 1) // part_size
                parts = []
                bytes_uploaded = 0

                def upload_part(part_number: int, start: int, end: int) -> Dict[str, Any]:
                    """Upload a single part."""
                    part_data = content[start:end]
                    response = self.s3_client.upload_part(
                        Bucket=self.datasets_bucket,
                        Key=s3_key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=part_data
                    )
                    return {
                        'PartNumber': part_number,
                        'ETag': response['ETag']
                    }

                # Upload parts in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []

                    for part_num in range(1, num_parts + 1):
                        start = (part_num - 1) * part_size
                        end = min(start + part_size, total_size)

                        future = executor.submit(upload_part, part_num, start, end)
                        futures.append((future, end - start))

                    # Wait for parts to complete and track progress
                    for future, part_bytes in futures:
                        part_info = future.result()
                        parts.append(part_info)
                        bytes_uploaded += part_bytes

                        # Call progress callback
                        if progress_callback:
                            progress_callback(bytes_uploaded, total_size)

                # Sort parts by part number
                parts.sort(key=lambda x: x['PartNumber'])

                # Complete multipart upload
                self.s3_client.complete_multipart_upload(
                    Bucket=self.datasets_bucket,
                    Key=s3_key,
                    UploadId=upload_id,
                    MultipartUpload={'Parts': parts}
                )

            except Exception as e:
                # Abort multipart upload on failure
                try:
                    if 'upload_id' in locals():
                        self.s3_client.abort_multipart_upload(
                            Bucket=self.datasets_bucket,
                            Key=s3_key,
                            UploadId=upload_id
                        )
                except Exception as abort_error:
                    # Best effort cleanup - report but don't mask the original error
                    logger.warning(
                        "Could not abort multipart upload for %s: %s",
                        s3_key, abort_error
                    )

                raise RuntimeError(
                    f"Multipart upload failed: {str(e)}"
                ) from e

    

    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0)
    def get_version(
        self,
        dataset_id: str,
        version_id: str,
        verify_checksum: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve a specific version of a dataset.
        
        Args:
            dataset_id: Dataset identifier
            version_id: Version identifier
            verify_checksum: Whether to verify checksum after download
            
        Returns:
            Dictionary containing:
                - content: Dataset content as bytes
                - metadata: Version metadata
                - checksum: SHA-256 checksum
                
        Raises:
            ValueError: If version not found or checksum verification fails
            RuntimeError: If retrieval fails
            
        Requirement 2.17: Retrieve versioned datasets with integrity verification
        """
        try:
            # Get metadata from DynamoDB
            metadata = self._get_version_metadata(dataset_id, version_id)
            
            if not metadata:
                raise ValueError(
                    f"Version not found: {dataset_id}/{version_id}"
                )
            
            # Download from S3
            s3_key = metadata['s3_uri'].split(f"{self.datasets_bucket}/")[1]
            
            response = self.s3_client.get_object(
                Bucket=self.datasets_bucket,
                Key=s3_key
            )
            
            content = response['Body'].read()
            
            # Verify checksum if requested
            if verify_checksum:
                calculated_checksum = self.calculate_checksum(content)
                stored_checksum = metadata['checksum']
                
                if calculated_checksum != stored_checksum:
                    raise ValueError(
                        f"Checksum mismatch: expected {stored_checksum}, "
                        f"got {calculated_checksum}"
                    )
            
            return {
                'content': content,
                'metadata': metadata,
                'checksum': metadata['checksum']
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')
            
            if error_code == 'NoSuchKey':
                raise ValueError(
                    f"Version not found in S3: {dataset_id}/{version_id}"
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to retrieve version: {error_message}"
                ) from e
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0)
    def list_versions(
        self,
        dataset_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all versions of a dataset.
        
        Args:
            dataset_id: Dataset identifier
            limit: Optional maximum number of versions to return
            
        Returns:
            List of version metadata dictionaries, sorted by creation time (newest first)
            
        Raises:
            RuntimeError: If listing fails
            
        Requirement 2.17: Support version listing
        """
        try:
            versions = []
            
            # Query DynamoDB for all versions of this dataset
            response = self.dynamodb_client.query(
                TableName=self.versions_table,
                KeyConditionExpression='dataset_id = :dataset_id',
                ExpressionAttributeValues={
                    ':dataset_id': {'S': dataset_id}
                },
                ScanIndexForward=False  # Sort by sort key descending (newest first)
            )
            
            items = response.get('Items', [])
            
            for item in items:
                version_dict = self._dynamodb_item_to_dict(item)
                versions.append(version_dict)
                
                if limit and len(versions) >= limit:
                    break
            
            return versions
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(
                f"Failed to list versions: {error_message}"
            ) from e
    
    def get_latest_version(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Latest version metadata or None if no versions exist
            
        Raises:
            RuntimeError: If retrieval fails
            
        Requirement 2.17: Support version retrieval
        """
        versions = self.list_versions(dataset_id, limit=1)
        return versions[0] if versions else None
    
    def compare_versions(
        self,
        dataset_id: str,
        version_id_1: str,
        version_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a dataset.
        
        Args:
            dataset_id: Dataset identifier
            version_id_1: First version identifier
            version_id_2: Second version identifier
            
        Returns:
            Dictionary containing:
                - version_1: Metadata for version 1
                - version_2: Metadata for version 2
                - size_delta: Size difference in bytes
                - checksum_match: Whether checksums match
                - row_count_delta: Row count difference
                
        Raises:
            ValueError: If versions not found
            RuntimeError: If comparison fails
            
        Requirement 2.17: Support version comparison
        """
        try:
            # Get metadata for both versions
            metadata_1 = self._get_version_metadata(dataset_id, version_id_1)
            metadata_2 = self._get_version_metadata(dataset_id, version_id_2)
            
            if not metadata_1:
                raise ValueError(
                    f"Version not found: {dataset_id}/{version_id_1}"
                )
            
            if not metadata_2:
                raise ValueError(
                    f"Version not found: {dataset_id}/{version_id_2}"
                )
            
            # Calculate deltas
            size_delta = metadata_2['size_bytes'] - metadata_1['size_bytes']
            row_count_delta = metadata_2['row_count'] - metadata_1['row_count']
            checksum_match = metadata_1['checksum'] == metadata_2['checksum']
            
            return {
                'version_1': metadata_1,
                'version_2': metadata_2,
                'size_delta': size_delta,
                'checksum_match': checksum_match,
                'row_count_delta': row_count_delta
            }
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(
                f"Failed to compare versions: {error_message}"
            ) from e
    
    def get_version_lineage(
        self,
        dataset_id: str,
        version_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get the lineage chain for a version (all parent versions).
        
        Args:
            dataset_id: Dataset identifier
            version_id: Version identifier
            
        Returns:
            List of version metadata dictionaries in lineage order
            (oldest to newest)
            
        Raises:
            ValueError: If version not found
            RuntimeError: If lineage retrieval fails
            
        Requirement 2.18: Track version lineage
        """
        lineage = []
        current_version_id = version_id
        
        while current_version_id:
            metadata = self._get_version_metadata(dataset_id, current_version_id)
            
            if not metadata:
                break
            
            lineage.insert(0, metadata)
            current_version_id = metadata.get('parent_version_id', '')
        
        return lineage

    @retry_with_exponential_backoff(
        max_retries=3, initial_delay=1.0, backoff_multiplier=2.0
    )
    def record_transformation(
        self,
        dataset_id: str,
        version_id: str,
        transformation_type: str,
        transformation_details: Dict[str, Any]
    ) -> None:
        """
        Record a transformation applied to a dataset version.
        
        Args:
            dataset_id: Dataset identifier
            version_id: Version identifier
            transformation_type: Type of transformation
                (e.g., 'split', 'pii_masking', 'synthetic_generation')
            transformation_details: Details about the transformation
            
        Raises:
            RuntimeError: If recording fails
            
        Requirement 2.18: Record transformations applied
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            transformation_id = (
                f"{dataset_id}#{version_id}#{transformation_type}#{timestamp}"
            )
            
            item = {
                'lineage_id': {'S': transformation_id},
                'dataset_id': {'S': dataset_id},
                'version_id': {'S': version_id},
                'record_type': {'S': 'transformation'},
                'transformation_type': {'S': transformation_type},
                'transformation_details': {
                    'S': json.dumps(transformation_details)
                },
                'timestamp': {'S': timestamp}
            }
            
            self.dynamodb_client.put_item(
                TableName=self.lineage_table,
                Item=item
            )
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(
                f"Failed to record transformation: {error_message}"
            ) from e

    @retry_with_exponential_backoff(
        max_retries=3, initial_delay=1.0, backoff_multiplier=2.0
    )
    def record_usage(
        self,
        dataset_id: str,
        version_id: str,
        usage_type: str,
        job_id: str,
        job_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record usage of a dataset version in an evaluation or fine-tuning job.
        
        Args:
            dataset_id: Dataset identifier
            version_id: Version identifier
            usage_type: Type of usage ('evaluation' or 'fine_tuning')
            job_id: ID of the evaluation or fine-tuning job
            job_details: Optional additional details about the job
            
        Raises:
            RuntimeError: If recording fails
            
        Requirement 2.18: Record usage history
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            usage_id = f"{dataset_id}#{version_id}#{usage_type}#{job_id}"
            
            item = {
                'lineage_id': {'S': usage_id},
                'dataset_id': {'S': dataset_id},
                'version_id': {'S': version_id},
                'record_type': {'S': 'usage'},
                'usage_type': {'S': usage_type},
                'job_id': {'S': job_id},
                'job_details': {'S': json.dumps(job_details or {})},
                'timestamp': {'S': timestamp}
            }
            
            self.dynamodb_client.put_item(
                TableName=self.lineage_table,
                Item=item
            )
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(
                f"Failed to record usage: {error_message}"
            ) from e

    @retry_with_exponential_backoff(
        max_retries=3, initial_delay=1.0, backoff_multiplier=2.0
    )
    def get_transformations(
        self,
        dataset_id: str,
        version_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all transformations applied to a dataset version.
        
        Args:
            dataset_id: Dataset identifier
            version_id: Version identifier
            
        Returns:
            List of transformation records sorted by timestamp
            
        Raises:
            RuntimeError: If retrieval fails
            
        Requirement 2.18: Query transformations applied
        """
        try:
            # Use scan with filter since we're using a simple key structure
            response = self.dynamodb_client.scan(
                TableName=self.lineage_table,
                FilterExpression=(
                    'dataset_id = :dataset_id AND '
                    'version_id = :version_id AND '
                    'record_type = :record_type'
                ),
                ExpressionAttributeValues={
                    ':dataset_id': {'S': dataset_id},
                    ':version_id': {'S': version_id},
                    ':record_type': {'S': 'transformation'}
                }
            )
            
            transformations = []
            for item in response.get('Items', []):
                transformations.append({
                    'transformation_type': item['transformation_type']['S'],
                    'transformation_details': json.loads(
                        item['transformation_details']['S']
                    ),
                    'timestamp': item['timestamp']['S']
                })
            
            # Sort by timestamp
            transformations.sort(key=lambda x: x['timestamp'])
            return transformations
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(
                f"Failed to get transformations: {error_message}"
            ) from e

    @retry_with_exponential_backoff(
        max_retries=3, initial_delay=1.0, backoff_multiplier=2.0
    )
    def get_usage_history(
        self,
        dataset_id: str,
        version_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get usage history for a dataset version.
        
        Args:
            dataset_id: Dataset identifier
            version_id: Version identifier
            
        Returns:
            List of usage records sorted by timestamp
            
        Raises:
            RuntimeError: If retrieval fails
            
        Requirement 2.18: Query usage history
        """
        try:
            # Use scan with filter since we're using a simple key structure
            response = self.dynamodb_client.scan(
                TableName=self.lineage_table,
                FilterExpression=(
                    'dataset_id = :dataset_id AND '
                    'version_id = :version_id AND '
                    'record_type = :record_type'
                ),
                ExpressionAttributeValues={
                    ':dataset_id': {'S': dataset_id},
                    ':version_id': {'S': version_id},
                    ':record_type': {'S': 'usage'}
                }
            )
            
            usage_records = []
            for item in response.get('Items', []):
                usage_records.append({
                    'usage_type': item['usage_type']['S'],
                    'job_id': item['job_id']['S'],
                    'job_details': json.loads(item['job_details']['S']),
                    'timestamp': item['timestamp']['S']
                })
            
            # Sort by timestamp
            usage_records.sort(key=lambda x: x['timestamp'])
            return usage_records
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(
                f"Failed to get usage history: {error_message}"
            ) from e

    def get_complete_lineage(
        self,
        dataset_id: str,
        version_id: str
    ) -> Dict[str, Any]:
        """
        Get complete lineage information for a dataset version.
        
        This includes:
        - Parent version chain
        - All transformations applied
        - Usage history
        
        Args:
            dataset_id: Dataset identifier
            version_id: Version identifier
            
        Returns:
            Dictionary containing:
                - version_chain: List of parent versions
                - transformations: List of transformations applied
                - usage_history: List of usage records
                
        Raises:
            RuntimeError: If retrieval fails
            
        Requirement 2.18: Provide complete lineage tracking
        """
        return {
            'version_chain': self.get_version_lineage(dataset_id, version_id),
            'transformations': self.get_transformations(
                dataset_id, version_id
            ),
            'usage_history': self.get_usage_history(dataset_id, version_id)
        }
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0)
    def delete_version(
        self,
        dataset_id: str,
        version_id: str
    ) -> None:
        """
        Delete a specific version of a dataset.
        
        Args:
            dataset_id: Dataset identifier
            version_id: Version identifier
            
        Raises:
            ValueError: If version not found
            RuntimeError: If deletion fails
            
        Note: This permanently deletes the version from both S3 and DynamoDB.
        """
        try:
            # Get metadata to find S3 key
            metadata = self._get_version_metadata(dataset_id, version_id)
            
            if not metadata:
                raise ValueError(
                    f"Version not found: {dataset_id}/{version_id}"
                )
            
            # Delete from S3
            s3_key = metadata['s3_uri'].split(f"{self.datasets_bucket}/")[1]
            self.s3_client.delete_object(
                Bucket=self.datasets_bucket,
                Key=s3_key
            )
            
            # Delete from DynamoDB
            self.dynamodb_client.delete_item(
                TableName=self.versions_table,
                Key={
                    'dataset_id': {'S': dataset_id},
                    'version_id': {'S': version_id}
                }
            )
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(
                f"Failed to delete version: {error_message}"
            ) from e
    
    def _store_version_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Store version metadata in DynamoDB.
        
        Args:
            metadata: Version metadata dictionary
            
        Raises:
            RuntimeError: If storage fails
        """
        try:
            item = self._dict_to_dynamodb_item(metadata)
            
            self.dynamodb_client.put_item(
                TableName=self.versions_table,
                Item=item
            )
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(
                f"Failed to store version metadata: {error_message}"
            ) from e
    
    def _get_version_metadata(
        self,
        dataset_id: str,
        version_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve version metadata from DynamoDB.
        
        Args:
            dataset_id: Dataset identifier
            version_id: Version identifier
            
        Returns:
            Version metadata dictionary or None if not found
            
        Raises:
            RuntimeError: If retrieval fails
        """
        try:
            response = self.dynamodb_client.get_item(
                TableName=self.versions_table,
                Key={
                    'dataset_id': {'S': dataset_id},
                    'version_id': {'S': version_id}
                }
            )
            
            if 'Item' not in response:
                return None
            
            return self._dynamodb_item_to_dict(response['Item'])
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(
                f"Failed to retrieve version metadata: {error_message}"
            ) from e
    
    def _dict_to_dynamodb_item(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert dictionary to DynamoDB item format.
        
        Args:
            data: Dictionary to convert
            
        Returns:
            DynamoDB item dictionary
        """
        return {
            'dataset_id': {'S': data['dataset_id']},
            'version_id': {'S': data['version_id']},
            's3_uri': {'S': data['s3_uri']},
            'checksum': {'S': data['checksum']},
            'size_bytes': {'N': str(data['size_bytes'])},
            'created_at': {'S': data['created_at']},
            'description': {'S': data.get('description', '')},
            'parent_version_id': {'S': data.get('parent_version_id', '')},
            'format': {'S': data['format']},
            'task_type': {'S': data['task_type']},
            'row_count': {'N': str(data['row_count'])},
            'name': {'S': data['name']}
        }
    
    def _dynamodb_item_to_dict(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert DynamoDB item to dictionary.
        
        Args:
            item: DynamoDB item
            
        Returns:
            Dictionary representation
        """
        return {
            'dataset_id': item['dataset_id']['S'],
            'version_id': item['version_id']['S'],
            's3_uri': item['s3_uri']['S'],
            'checksum': item['checksum']['S'],
            'size_bytes': int(item['size_bytes']['N']),
            'created_at': item['created_at']['S'],
            'description': item['description']['S'],
            'parent_version_id': item['parent_version_id']['S'],
            'format': item['format']['S'],
            'task_type': item['task_type']['S'],
            'row_count': int(item['row_count']['N']),
            'name': item['name']['S']
        }
