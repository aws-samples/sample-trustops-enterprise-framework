"""
Unit tests for S3StorageManager.
"""
import pytest
import json
import gzip
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError
from datetime import datetime

from src.aws_clients.s3_storage_manager import S3StorageManager


@pytest.fixture
def s3_manager():
    """Create an S3StorageManager instance with mocked boto3 client."""
    with patch('src.aws_clients.s3_storage_manager.boto3.Session') as mock_s:
        mock_s3_client = Mock()
        
        mock_session_instance = Mock()
        mock_session_instance.client.return_value = mock_s3_client
        mock_s.return_value = mock_session_instance
        
        manager = S3StorageManager()
        manager.s3_client = mock_s3_client
        
        return manager


class TestUploadDataset:
    """Tests for upload_dataset method."""
    
    def test_upload_dataset_success(self, s3_manager):
        """Test successful dataset upload."""
        dataset_content = '{"prompt": "test", "response": "test"}'
        dataset_name = 'test_dataset.json'
        
        s3_manager.s3_client.put_object.return_value = {
            'VersionId': 'v123'
        }
        
        result = s3_manager.upload_dataset(
            dataset_content=dataset_content,
            dataset_name=dataset_name
        )
        
        assert 's3_uri' in result
        assert result['s3_uri'].endswith(dataset_name)
        assert result['version_id'] == 'v123'
        assert 'checksum' in result
        assert result['size_bytes'] == len(dataset_content.encode('utf-8'))
        
        # Verify S3 client was called correctly
        s3_manager.s3_client.put_object.assert_called_once()
        call_kwargs = s3_manager.s3_client.put_object.call_args[1]
        assert call_kwargs['Key'] == dataset_name
        assert 'checksum' in call_kwargs['Metadata']
    
    def test_upload_dataset_with_metadata(self, s3_manager):
        """Test dataset upload with custom metadata."""
        dataset_content = '{"data": "test"}'
        dataset_name = 'test.json'
        metadata = {'custom_field': 'custom_value'}
        
        s3_manager.s3_client.put_object.return_value = {'VersionId': 'v1'}
        
        result = s3_manager.upload_dataset(
            dataset_content=dataset_content,
            dataset_name=dataset_name,
            metadata=metadata
        )
        
        assert result['s3_uri']
        call_kwargs = s3_manager.s3_client.put_object.call_args[1]
        assert call_kwargs['Metadata']['custom_field'] == 'custom_value'
        assert 'checksum' in call_kwargs['Metadata']
    
    def test_upload_dataset_error(self, s3_manager):
        """Test handling of upload errors."""
        error_response = {
            'Error': {
                'Code': 'AccessDenied',
                'Message': 'Access denied'
            }
        }
        s3_manager.s3_client.put_object.side_effect = ClientError(
            error_response, 'PutObject'
        )
        
        with pytest.raises(RuntimeError, match="Failed to upload dataset"):
            s3_manager.upload_dataset('content', 'test.json')


class TestDownloadDataset:
    """Tests for download_dataset method."""
    
    def test_download_dataset_success(self, s3_manager):
        """Test successful dataset download."""
        s3_uri = 's3://test-bucket/test.json'
        content = '{"data": "test"}'
        checksum = s3_manager._generate_checksum(content)
        
        mock_body = Mock()
        mock_body.read.return_value = content.encode('utf-8')
        
        s3_manager.s3_client.get_object.return_value = {
            'Body': mock_body,
            'Metadata': {'checksum': checksum},
            'VersionId': 'v1'
        }
        
        result = s3_manager.download_dataset(s3_uri)
        
        assert result['content'] == content
        assert result['checksum'] == checksum
        assert result['version_id'] == 'v1'
        
        s3_manager.s3_client.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test.json'
        )
    
    def test_download_dataset_with_version(self, s3_manager):
        """Test downloading specific version."""
        s3_uri = 's3://test-bucket/test.json'
        version_id = 'v123'
        content = '{"data": "test"}'
        
        mock_body = Mock()
        mock_body.read.return_value = content.encode('utf-8')
        
        s3_manager.s3_client.get_object.return_value = {
            'Body': mock_body,
            'Metadata': {},
            'VersionId': version_id
        }
        
        result = s3_manager.download_dataset(
            s3_uri,
            version_id=version_id,
            verify_checksum=False
        )
        
        assert result['content'] == content
        
        s3_manager.s3_client.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test.json',
            VersionId=version_id
        )
    
    def test_download_dataset_checksum_mismatch(self, s3_manager):
        """Test checksum verification failure."""
        s3_uri = 's3://test-bucket/test.json'
        content = '{"data": "test"}'
        
        mock_body = Mock()
        mock_body.read.return_value = content.encode('utf-8')
        
        s3_manager.s3_client.get_object.return_value = {
            'Body': mock_body,
            'Metadata': {'checksum': 'wrong_checksum'},
            'VersionId': 'v1'
        }
        
        with pytest.raises(ValueError, match="Checksum mismatch"):
            s3_manager.download_dataset(s3_uri, verify_checksum=True)
    
    def test_download_dataset_not_found(self, s3_manager):
        """Test handling of missing dataset."""
        error_response = {
            'Error': {
                'Code': 'NoSuchKey',
                'Message': 'Key not found'
            }
        }
        s3_manager.s3_client.get_object.side_effect = ClientError(
            error_response, 'GetObject'
        )
        
        with pytest.raises(ValueError, match="Dataset not found"):
            s3_manager.download_dataset('s3://bucket/missing.json')
    
    def test_download_dataset_invalid_uri(self, s3_manager):
        """Test handling of invalid S3 URI."""
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            s3_manager.download_dataset('invalid-uri')


class TestStoreResults:
    """Tests for store_results method."""
    
    def test_store_results_uncompressed(self, s3_manager):
        """Test storing results without compression."""
        results = {'score': 0.95, 'data': 'test'}
        workflow_id = 'wf-123'
        model_id = 'model-abc'
        
        s3_manager.s3_client.put_object.return_value = {'VersionId': 'v1'}
        
        result = s3_manager.store_results(
            results=results,
            workflow_id=workflow_id,
            model_id=model_id,
            compress=False
        )
        
        assert 's3_uri' in result
        assert workflow_id in result['s3_uri']
        assert model_id in result['s3_uri']
        assert result['compressed'] is False
        assert 'checksum' in result
        
        call_kwargs = s3_manager.s3_client.put_object.call_args[1]
        assert call_kwargs['ContentType'] == 'application/json'
        assert not call_kwargs['Key'].endswith('.gz')
    
    def test_store_results_compressed(self, s3_manager):
        """Test storing results with compression."""
        results = {'data': 'x' * 2000000}  # Large data
        workflow_id = 'wf-123'
        model_id = 'model-abc'
        
        s3_manager.s3_client.put_object.return_value = {'VersionId': 'v1'}
        
        result = s3_manager.store_results(
            results=results,
            workflow_id=workflow_id,
            model_id=model_id,
            compress=True
        )
        
        assert result['compressed'] is True
        
        call_kwargs = s3_manager.s3_client.put_object.call_args[1]
        assert call_kwargs['ContentType'] == 'application/gzip'
        assert call_kwargs['Key'].endswith('.gz')
    
    def test_store_results_auto_compress(self, s3_manager):
        """Test automatic compression for large results."""
        # Create large results that exceed compression threshold
        large_results = {'data': 'x' * 2000000}
        
        s3_manager.s3_client.put_object.return_value = {'VersionId': 'v1'}
        
        result = s3_manager.store_results(
            results=large_results,
            workflow_id='wf-123',
            model_id='model-abc'
        )
        
        # Should auto-compress
        assert result['compressed'] is True
    
    def test_store_results_path_organization(self, s3_manager):
        """Test results are stored with correct path structure."""
        results = {'test': 'data'}
        workflow_id = 'wf-123'
        model_id = 'model-abc'
        
        s3_manager.s3_client.put_object.return_value = {'VersionId': 'v1'}
        
        result = s3_manager.store_results(
            results=results,
            workflow_id=workflow_id,
            model_id=model_id
        )
        
        # Verify path structure: workflow_id/timestamp/model_id/results.json
        call_kwargs = s3_manager.s3_client.put_object.call_args[1]
        key_parts = call_kwargs['Key'].split('/')
        assert key_parts[0] == workflow_id
        assert key_parts[2] == model_id
        assert 'results.json' in key_parts[3]


class TestRetrieveResults:
    """Tests for retrieve_results method."""
    
    def test_retrieve_results_all(self, s3_manager):
        """Test retrieving all results."""
        mock_paginator = Mock()
        s3_manager.s3_client.get_paginator.return_value = mock_paginator
        
        mock_paginator.paginate.return_value = [
            {
                'Contents': [
                    {
                        'Key': 'wf-1/20240101_120000/model-a/results.json',
                        'Size': 1024,
                        'LastModified': datetime(2024, 1, 1, 12, 0, 0)
                    }
                ]
            }
        ]
        
        s3_manager.s3_client.head_object.return_value = {
            'Metadata': {'checksum': 'abc123'}
        }
        
        results = s3_manager.retrieve_results()
        
        assert len(results) == 1
        assert results[0]['workflow_id'] == 'wf-1'
        assert results[0]['model_id'] == 'model-a'
        assert results[0]['timestamp'] == '20240101_120000'
    
    def test_retrieve_results_filtered_by_workflow(self, s3_manager):
        """Test filtering results by workflow ID."""
        mock_paginator = Mock()
        s3_manager.s3_client.get_paginator.return_value = mock_paginator
        
        mock_paginator.paginate.return_value = [
            {
                'Contents': [
                    {
                        'Key': 'wf-123/20240101_120000/model-a/results.json',
                        'Size': 1024,
                        'LastModified': datetime(2024, 1, 1, 12, 0, 0)
                    }
                ]
            }
        ]
        
        s3_manager.s3_client.head_object.return_value = {'Metadata': {}}
        
        results = s3_manager.retrieve_results(workflow_id='wf-123')
        
        # Verify prefix was used
        call_kwargs = mock_paginator.paginate.call_args[1]
        assert call_kwargs['Prefix'] == 'wf-123/'
    
    def test_retrieve_results_filtered_by_model(self, s3_manager):
        """Test filtering results by model ID."""
        mock_paginator = Mock()
        s3_manager.s3_client.get_paginator.return_value = mock_paginator
        
        mock_paginator.paginate.return_value = [
            {
                'Contents': [
                    {
                        'Key': 'wf-1/20240101_120000/model-a/results.json',
                        'Size': 1024,
                        'LastModified': datetime(2024, 1, 1, 12, 0, 0)
                    },
                    {
                        'Key': 'wf-1/20240101_130000/model-b/results.json',
                        'Size': 2048,
                        'LastModified': datetime(2024, 1, 1, 13, 0, 0)
                    }
                ]
            }
        ]
        
        s3_manager.s3_client.head_object.return_value = {'Metadata': {}}
        
        results = s3_manager.retrieve_results(model_id='model-a')
        
        # Should only return model-a results
        assert len(results) == 1
        assert results[0]['model_id'] == 'model-a'
    
    def test_retrieve_results_filtered_by_date(self, s3_manager):
        """Test filtering results by date range."""
        mock_paginator = Mock()
        s3_manager.s3_client.get_paginator.return_value = mock_paginator
        
        mock_paginator.paginate.return_value = [
            {
                'Contents': [
                    {
                        'Key': 'wf-1/20240101_120000/model-a/results.json',
                        'Size': 1024,
                        'LastModified': datetime(2024, 1, 1, 12, 0, 0)
                    },
                    {
                        'Key': 'wf-1/20240105_120000/model-a/results.json',
                        'Size': 2048,
                        'LastModified': datetime(2024, 1, 5, 12, 0, 0)
                    }
                ]
            }
        ]
        
        s3_manager.s3_client.head_object.return_value = {'Metadata': {}}
        
        results = s3_manager.retrieve_results(
            start_date='20240101',
            end_date='20240103'
        )
        
        # Should only return results within date range
        assert len(results) == 1
        assert '20240101' in results[0]['timestamp']


class TestLoadResults:
    """Tests for load_results method."""
    
    def test_load_results_uncompressed(self, s3_manager):
        """Test loading uncompressed results."""
        s3_uri = 's3://bucket/wf-1/20240101_120000/model-a/results.json'
        results_data = {'score': 0.95}
        
        mock_body = Mock()
        mock_body.read.return_value = json.dumps(results_data).encode('utf-8')
        
        s3_manager.s3_client.get_object.return_value = {
            'Body': mock_body
        }
        
        results = s3_manager.load_results(s3_uri)
        
        assert results == results_data
    
    def test_load_results_compressed(self, s3_manager):
        """Test loading compressed results."""
        s3_uri = 's3://bucket/wf-1/20240101_120000/model-a/results.json.gz'
        results_data = {'score': 0.95}
        
        compressed = gzip.compress(json.dumps(results_data).encode('utf-8'))
        
        mock_body = Mock()
        mock_body.read.return_value = compressed
        
        s3_manager.s3_client.get_object.return_value = {
            'Body': mock_body
        }
        
        results = s3_manager.load_results(s3_uri)
        
        assert results == results_data
    
    def test_load_results_not_found(self, s3_manager):
        """Test handling of missing results."""
        error_response = {
            'Error': {
                'Code': 'NoSuchKey',
                'Message': 'Key not found'
            }
        }
        s3_manager.s3_client.get_object.side_effect = ClientError(
            error_response, 'GetObject'
        )
        
        with pytest.raises(ValueError, match="Results not found"):
            s3_manager.load_results('s3://bucket/missing.json')


class TestHelperMethods:
    """Tests for helper methods."""
    
    def test_generate_checksum(self, s3_manager):
        """Test checksum generation."""
        content = "test content"
        checksum1 = s3_manager._generate_checksum(content)
        checksum2 = s3_manager._generate_checksum(content)
        
        # Same content should produce same checksum
        assert checksum1 == checksum2
        
        # Different content should produce different checksum
        checksum3 = s3_manager._generate_checksum("different content")
        assert checksum1 != checksum3

    def test_generate_checksum_uses_sha256(self, s3_manager):
        """Checksums are SHA256 (64 hex chars), not MD5."""
        import hashlib
        content = "test content"

        checksum = s3_manager._generate_checksum(content)

        assert len(checksum) == 64
        assert checksum == hashlib.sha256(content.encode('utf-8')).hexdigest()

    def test_verify_checksum_sha256(self, s3_manager):
        """A stored SHA256 digest is re-verified with SHA256."""
        content = "test content"
        stored = s3_manager._generate_checksum(content)

        assert s3_manager._verify_checksum(content, stored) == stored

    def test_verify_checksum_accepts_legacy_md5(self, s3_manager):
        """Objects stored with the old 32-char MD5 digest still verify."""
        import hashlib
        content = "test content"
        legacy = hashlib.md5(
            content.encode('utf-8'), usedforsecurity=False
        ).hexdigest()

        assert s3_manager._verify_checksum(content, legacy) == legacy

    def test_verify_checksum_detects_legacy_mismatch(self, s3_manager):
        """A corrupted object with a legacy digest still fails verification."""
        import hashlib
        legacy = hashlib.md5(
            b"original content", usedforsecurity=False
        ).hexdigest()

        assert s3_manager._verify_checksum("tampered content", legacy) != legacy


    def test_parse_s3_uri_valid(self, s3_manager):
        """Test parsing valid S3 URI."""
        bucket, key = s3_manager._parse_s3_uri('s3://my-bucket/my/key.json')
        assert bucket == 'my-bucket'
        assert key == 'my/key.json'
    
    def test_parse_s3_uri_invalid(self, s3_manager):
        """Test parsing invalid S3 URIs."""
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            s3_manager._parse_s3_uri('http://bucket/key')
        
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            s3_manager._parse_s3_uri('s3://bucket-only')
