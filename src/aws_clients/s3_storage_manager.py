"""
AWS S3 storage manager for datasets, results, and artifacts.
"""
import json
import gzip
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

from config.aws_config import config
from src.utils.retry_utils import retry_with_exponential_backoff


class S3StorageManager:
    """Manager for S3 storage operations with versioning and compression."""
    
    def __init__(self, region: Optional[str] = None):
        """
        Initialize S3 storage manager.
        
        Args:
            region: AWS region (defaults to config.region)
        """
        self.region = region or config.region
        session_kwargs = config.get_boto3_session_kwargs()
        if region:
            session_kwargs['region_name'] = region
        
        session = boto3.Session(**session_kwargs)
        self.s3_client = session.client('s3')
        
        # Bucket configuration
        self.datasets_bucket = config.datasets_bucket
        self.results_bucket = config.results_bucket
        self.artifacts_bucket = config.artifacts_bucket
        
        # Compression threshold (bytes) - compress files larger than 1MB
        self.compression_threshold = 1024 * 1024
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0)
    def upload_dataset(
        self,
        dataset_content: str,
        dataset_name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload dataset to S3 with versioning.
        
        Args:
            dataset_content: Dataset content as string (JSON/JSONL)
            dataset_name: Name/key for the dataset
            metadata: Optional metadata to attach to the object
            
        Returns:
            Dictionary containing:
                - s3_uri: S3 URI of uploaded dataset
                - version_id: S3 version ID
                - checksum: SHA256 checksum of content
                - size_bytes: Size of uploaded content
                
        Raises:
            RuntimeError: If upload fails
        """
        try:
            # Generate checksum
            checksum = self._generate_checksum(dataset_content)
            
            # Prepare metadata
            upload_metadata = metadata or {}
            upload_metadata['checksum'] = checksum
            upload_metadata['uploaded_at'] = datetime.utcnow().isoformat()
            
            # Upload to S3
            put_response = self.s3_client.put_object(
                Bucket=self.datasets_bucket,
                Key=dataset_name,
                Body=dataset_content.encode('utf-8'),
                Metadata=upload_metadata,
                ContentType='application/json'
            )
            
            s3_uri = f"s3://{self.datasets_bucket}/{dataset_name}"
            
            return {
                's3_uri': s3_uri,
                'version_id': put_response.get('VersionId'),
                'checksum': checksum,
                'size_bytes': len(dataset_content.encode('utf-8'))
            }
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(f"Failed to upload dataset: {error_message}") from e

    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0)
    def download_dataset(
        self,
        s3_uri: str,
        version_id: Optional[str] = None,
        verify_checksum: bool = True
    ) -> Dict[str, Any]:
        """
        Download dataset from S3.
        
        Args:
            s3_uri: S3 URI of the dataset (s3://bucket/key)
            version_id: Optional specific version to download
            verify_checksum: Whether to verify checksum after download
            
        Returns:
            Dictionary containing:
                - content: Dataset content as string
                - checksum: SHA256 checksum of content
                - metadata: Object metadata
                - version_id: Version ID of downloaded object
                
        Raises:
            ValueError: If S3 URI is invalid or checksum verification fails
            RuntimeError: If download fails
        """
        try:
            # Parse S3 URI
            bucket, key = self._parse_s3_uri(s3_uri)
            
            # Download from S3
            get_params = {
                'Bucket': bucket,
                'Key': key
            }
            if version_id:
                get_params['VersionId'] = version_id
            
            response = self.s3_client.get_object(**get_params)
            
            # Read content
            content = response['Body'].read().decode('utf-8')
            
            # Get metadata
            metadata = response.get('Metadata', {})
            stored_checksum = metadata.get('checksum')
            
            # Verify checksum if requested
            if verify_checksum and stored_checksum:
                calculated_checksum = self._verify_checksum(
                    content, stored_checksum
                )
                if calculated_checksum != stored_checksum:
                    raise ValueError(
                        f"Checksum mismatch: expected "
                        f"{stored_checksum}, got {calculated_checksum}"
                    )
            
            return {
                'content': content,
                'checksum': self._generate_checksum(content),
                'metadata': metadata,
                'version_id': response.get('VersionId')
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')
            
            if error_code == 'NoSuchKey':
                raise ValueError(f"Dataset not found: {s3_uri}") from e
            elif error_code == 'NoSuchBucket':
                raise ValueError(f"Bucket not found: {bucket}") from e
            else:
                error_msg = f"Failed to download dataset: {error_message}"
                raise RuntimeError(error_msg) from e
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0)
    def store_results(
        self,
        results: Dict[str, Any],
        workflow_id: str,
        model_id: str,
        compress: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Store evaluation results with organized paths.
        Path format: workflow_id/timestamp/model_id
        
        Args:
            results: Results dictionary to store
            workflow_id: Workflow identifier
            model_id: Model identifier
            compress: Whether to compress (auto-detects if None)
            
        Returns:
            Dictionary containing:
                - s3_uri: S3 URI of stored results
                - checksum: SHA256 checksum of content
                - compressed: Whether content was compressed
                - size_bytes: Size of stored content
                
        Raises:
            RuntimeError: If storage fails
        """
        try:
            # Generate organized path
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            key = f"{workflow_id}/{timestamp}/{model_id}/results.json"
            
            # Serialize results
            results_json = json.dumps(results, indent=2)
            content = results_json.encode('utf-8')
            
            # Determine if compression is needed
            should_compress = (
                compress if compress is not None
                else len(content) > self.compression_threshold
            )
            
            if should_compress:
                content = gzip.compress(content)
                key += '.gz'
                content_type = 'application/gzip'
            else:
                content_type = 'application/json'
            
            # Generate checksum (of uncompressed content for consistency)
            checksum = self._generate_checksum(results_json)
            
            # Prepare metadata
            metadata = {
                'checksum': checksum,
                'workflow_id': workflow_id,
                'model_id': model_id,
                'timestamp': timestamp,
                'compressed': str(should_compress).lower()
            }
            
            # Upload to S3
            put_response = self.s3_client.put_object(
                Bucket=self.results_bucket,
                Key=key,
                Body=content,
                Metadata=metadata,
                ContentType=content_type
            )
            
            s3_uri = f"s3://{self.results_bucket}/{key}"
            
            return {
                's3_uri': s3_uri,
                'checksum': checksum,
                'compressed': should_compress,
                'size_bytes': len(content),
                'version_id': put_response.get('VersionId')
            }
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            error_msg = f"Failed to store results: {error_message}"
            raise RuntimeError(error_msg) from e

    def retrieve_results(
        self,
        workflow_id: Optional[str] = None,
        model_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results with filtering.
        
        Args:
            workflow_id: Optional workflow ID filter
            model_id: Optional model ID filter
            start_date: Optional start date filter (YYYYMMDD format)
            end_date: Optional end date filter (YYYYMMDD format)
            
        Returns:
            List of result dictionaries, each containing:
                - s3_uri: S3 URI of the result
                - workflow_id: Workflow identifier
                - model_id: Model identifier
                - timestamp: Result timestamp
                - metadata: Object metadata
                
        Raises:
            RuntimeError: If retrieval fails
        """
        try:
            results = []
            
            # Determine prefix based on workflow_id
            prefix = f"{workflow_id}/" if workflow_id else ""
            
            # List objects with pagination
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.results_bucket,
                Prefix=prefix
            )
            
            for page in pages:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    
                    # Parse key to extract metadata
                    parts = key.split('/')
                    if len(parts) < 4:
                        continue
                    
                    obj_workflow_id = parts[0]
                    obj_timestamp = parts[1]
                    obj_model_id = parts[2]
                    
                    # Apply filters
                    if model_id and obj_model_id != model_id:
                        continue
                    
                    # Extract date from timestamp (YYYYMMDD)
                    obj_date = obj_timestamp.split('_')[0]
                    
                    if start_date and obj_date < start_date:
                        continue
                    
                    if end_date and obj_date > end_date:
                        continue
                    
                    # Get object metadata
                    head_response = self.s3_client.head_object(
                        Bucket=self.results_bucket,
                        Key=key
                    )
                    
                    results.append({
                        's3_uri': f"s3://{self.results_bucket}/{key}",
                        'workflow_id': obj_workflow_id,
                        'model_id': obj_model_id,
                        'timestamp': obj_timestamp,
                        'metadata': head_response.get('Metadata', {}),
                        'size_bytes': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })
            
            return results
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', '')
            error_msg = f"Failed to retrieve results: {error_message}"
            raise RuntimeError(error_msg) from e
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0)
    def load_results(self, s3_uri: str) -> Dict[str, Any]:
        """
        Load results from S3 URI, handling compression automatically.
        
        Args:
            s3_uri: S3 URI of the results
            
        Returns:
            Deserialized results dictionary
            
        Raises:
            ValueError: If S3 URI is invalid
            RuntimeError: If loading fails
        """
        try:
            # Parse S3 URI
            bucket, key = self._parse_s3_uri(s3_uri)
            
            # Download from S3
            response = self.s3_client.get_object(
                Bucket=bucket,
                Key=key
            )
            
            # Read content
            content = response['Body'].read()
            
            # Decompress if needed
            if key.endswith('.gz'):
                content = gzip.decompress(content)
            
            # Deserialize JSON
            results = json.loads(content.decode('utf-8'))
            
            return results
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')
            
            if error_code == 'NoSuchKey':
                raise ValueError(
                    f"Results not found: {s3_uri}"
                ) from e
            else:
                error_msg = f"Failed to load results: {error_message}"
                raise RuntimeError(error_msg) from e
    
    def _generate_checksum(self, content: str) -> str:
        """
        Generate SHA256 checksum for content.

        Args:
            content: Content string

        Returns:
            SHA256 checksum as hex string
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _verify_checksum(self, content: str, stored_checksum: str) -> str:
        """
        Recompute a checksum using the same algorithm as the stored value.

        Objects uploaded before the switch to SHA256 carry a 32-character MD5
        digest. Those are re-verified with MD5 so existing data stays readable;
        everything written from now on uses SHA256.

        Args:
            content: Content string
            stored_checksum: Checksum recorded in the object metadata

        Returns:
            Checksum of content in the same format as stored_checksum
        """
        if len(stored_checksum) == 32:
            # Legacy MD5 digest - comparison only, not a security control.
            return hashlib.md5(
                content.encode('utf-8'), usedforsecurity=False
            ).hexdigest()
        return self._generate_checksum(content)
    
    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """
        Parse S3 URI into bucket and key.
        
        Args:
            s3_uri: S3 URI (s3://bucket/key)
            
        Returns:
            Tuple of (bucket, key)
            
        Raises:
            ValueError: If URI format is invalid
        """
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        
        parts = s3_uri[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        
        return parts[0], parts[1]
    
    def upload_json(
        self,
        data: Dict[str, Any],
        key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload JSON data to S3.
        
        Args:
            data: Dictionary to upload as JSON
            key: S3 key (path) for the object
            metadata: Optional metadata
            
        Returns:
            S3 URI of uploaded object
        """
        json_content = json.dumps(data, indent=2)
        result = self.upload_dataset(json_content, key, metadata)
        return result['s3_uri']
    
    def upload_jsonl(
        self,
        data: List[Dict[str, Any]],
        key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload JSONL data to S3.
        
        Args:
            data: List of dictionaries to upload as JSONL
            key: S3 key (path) for the object
            metadata: Optional metadata
            
        Returns:
            S3 URI of uploaded object
        """
        jsonl_content = '\n'.join(json.dumps(item) for item in data)
        result = self.upload_dataset(jsonl_content, key, metadata)
        return result['s3_uri']
