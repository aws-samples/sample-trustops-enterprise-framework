"""
Unit tests for KnowledgeBaseClient.
"""
import json
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from src.aws_clients.knowledge_base_client import KnowledgeBaseClient


@pytest.fixture
def kb_client():
    """Create a KnowledgeBaseClient with mocked boto3 clients."""
    with patch('src.aws_clients.knowledge_base_client.boto3.Session') as mock_session, \
         patch('src.aws_clients.knowledge_base_client.config') as mock_config:

        mock_config.knowledge_base_id = 'kb-123'
        mock_config.knowledge_base_data_source_id = 'ds-456'
        mock_config.datasets_bucket = 'trustops-test-datasets-123456789012-us-east-1'
        mock_config.get_boto3_session_kwargs.return_value = {'region_name': 'us-east-1'}

        clients = {
            'bedrock-agent-runtime': Mock(),
            'bedrock-agent': Mock(),
            's3': Mock(),
        }
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = lambda service: clients[service]
        mock_session.return_value = mock_session_instance

        client = KnowledgeBaseClient()
        return client


@pytest.fixture
def mock_kb_client():
    """Create a KnowledgeBaseClient in mock mode."""
    with patch('src.aws_clients.knowledge_base_client.config') as mock_config:
        mock_config.knowledge_base_id = None
        mock_config.knowledge_base_data_source_id = None
        mock_config.datasets_bucket = 'trustops-test-datasets-123456789012-us-east-1'
        return KnowledgeBaseClient()


class TestInitialization:
    """Tests for client initialization."""

    def test_mock_mode_without_knowledge_base_id(self, mock_kb_client):
        """No Knowledge Base configured means the client falls back to mock mode."""
        assert mock_kb_client.mock_mode is True
        assert mock_kb_client._bedrock_agent_runtime is None

    def test_live_mode_with_knowledge_base_id(self, kb_client):
        """A configured Knowledge Base ID enables live mode."""
        assert kb_client.mock_mode is False
        assert kb_client.knowledge_base_id == 'kb-123'
        assert kb_client.data_source_id == 'ds-456'

    def test_explicit_mock_mode_overrides_config(self):
        """mock_mode=True forces mock mode even with a configured KB."""
        with patch('src.aws_clients.knowledge_base_client.config') as mock_config:
            mock_config.knowledge_base_id = 'kb-123'
            mock_config.knowledge_base_data_source_id = 'ds-456'
            mock_config.datasets_bucket = 'trustops-test-datasets-123456789012-us-east-1'

            client = KnowledgeBaseClient(mock_mode=True)
            assert client.mock_mode is True


class TestStoreDocument:
    """Tests for store_document method."""

    def test_store_document_uploads_to_s3(self, kb_client):
        """Documents are written to S3 under the knowledge-base/ prefix."""
        result = kb_client.store_document(
            document_id='doc-1',
            text='Policy text',
            metadata={'category': 'policy'},
            sync=False,
        )

        assert result == 'doc-1'
        kb_client._s3.put_object.assert_called_once()
        call_kwargs = kb_client._s3.put_object.call_args[1]
        assert call_kwargs['Bucket'] == 'trustops-test-datasets-123456789012-us-east-1'
        assert call_kwargs['Key'] == 'knowledge-base/doc-1.json'

        payload = json.loads(call_kwargs['Body'])
        assert payload['document_id'] == 'doc-1'
        assert payload['text'] == 'Policy text'
        assert payload['metadata'] == {'category': 'policy'}

    def test_store_document_triggers_sync(self, kb_client):
        """sync=True starts a Knowledge Base ingestion job."""
        kb_client._bedrock_agent.start_ingestion_job.return_value = {
            'ingestionJob': {'ingestionJobId': 'job-1'}
        }

        kb_client.store_document('doc-1', 'text', sync=True)

        kb_client._bedrock_agent.start_ingestion_job.assert_called_once_with(
            knowledgeBaseId='kb-123',
            dataSourceId='ds-456',
        )

    def test_store_document_skips_sync_when_disabled(self, kb_client):
        """sync=False leaves ingestion to a later explicit sync."""
        kb_client.store_document('doc-1', 'text', sync=False)
        kb_client._bedrock_agent.start_ingestion_job.assert_not_called()

    def test_store_document_wraps_s3_error(self, kb_client):
        """An S3 failure is surfaced as a RuntimeError."""
        kb_client._s3.put_object.side_effect = ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'denied'}}, 'PutObject'
        )

        with pytest.raises(RuntimeError, match="Failed to upload document 'doc-1'"):
            kb_client.store_document('doc-1', 'text', sync=False)

    def test_store_document_in_mock_mode(self, mock_kb_client):
        """Mock mode stores documents in memory."""
        result = mock_kb_client.store_document('doc-1', 'text', {'k': 'v'})

        assert result == 'doc-1'
        assert mock_kb_client.get_document('doc-1') == {
            'text': 'text',
            'metadata': {'k': 'v'},
        }


class TestSearchSimilar:
    """Tests for search_similar method."""

    def test_search_similar_parses_results(self, kb_client):
        """Retrieval results map to (document_id, score, text) tuples."""
        kb_client._bedrock_agent_runtime.retrieve.return_value = {
            'retrievalResults': [
                {
                    'score': 0.91,
                    'content': {'text': 'Refunds are processed in 5 days.'},
                    'location': {
                        's3Location': {
                            'uri': 's3://trustops-test-datasets-123456789012-us-east-1/knowledge-base/doc-1.json'
                        }
                    },
                },
            ]
        }

        results = kb_client.search_similar('refund policy', top_k=3)

        assert results == [('doc-1', 0.91, 'Refunds are processed in 5 days.')]
        call_kwargs = kb_client._bedrock_agent_runtime.retrieve.call_args[1]
        assert call_kwargs['knowledgeBaseId'] == 'kb-123'
        assert call_kwargs['retrievalQuery'] == {'text': 'refund policy'}
        vector_config = call_kwargs['retrievalConfiguration']['vectorSearchConfiguration']
        assert vector_config['numberOfResults'] == 3

    def test_search_similar_filters_below_min_score(self, kb_client):
        """Results scoring below min_score are dropped."""
        kb_client._bedrock_agent_runtime.retrieve.return_value = {
            'retrievalResults': [
                {
                    'score': 0.95,
                    'content': {'text': 'high'},
                    'location': {
                        's3Location': {'uri': 's3://b/knowledge-base/high.json'}
                    },
                },
                {
                    'score': 0.40,
                    'content': {'text': 'low'},
                    'location': {
                        's3Location': {'uri': 's3://b/knowledge-base/low.json'}
                    },
                },
            ]
        }

        results = kb_client.search_similar('query', min_score=0.8)

        assert len(results) == 1
        assert results[0][0] == 'high'

    def test_search_similar_wraps_error(self, kb_client):
        """A retrieve failure is surfaced as a RuntimeError."""
        kb_client._bedrock_agent_runtime.retrieve.side_effect = ClientError(
            {'Error': {'Code': 'ValidationException', 'Message': 'bad'}}, 'Retrieve'
        )

        with pytest.raises(RuntimeError, match="Failed to retrieve from Knowledge Base"):
            kb_client.search_similar('query')

    def test_search_similar_respects_top_k_in_mock_mode(self, mock_kb_client):
        """Mock mode returns at most top_k stored documents."""
        for i in range(5):
            mock_kb_client.store_document(f'doc-{i}', f'text-{i}')

        results = mock_kb_client.search_similar('query', top_k=2)
        assert len(results) == 2


class TestExtractDocumentId:
    """Tests for _extract_document_id_from_uri."""

    @pytest.mark.parametrize("uri,expected", [
        ('s3://bucket/knowledge-base/doc-1.json', 'doc-1'),
        ('s3://bucket/knowledge-base/nested/doc-2.json', 'nested/doc-2'),
        # Only the knowledge-base/ prefix is stripped; other prefixes remain.
        ('s3://bucket/other/doc-3.json', 'other/doc-3'),
    ])
    def test_extract_document_id(self, kb_client, uri, expected):
        assert kb_client._extract_document_id_from_uri(uri) == expected


class TestGetAndDeleteDocument:
    """Tests for get_document and delete_document."""

    def test_get_document_returns_payload(self, kb_client):
        """A stored document is parsed from its S3 body."""
        body = Mock()
        body.read.return_value = json.dumps({'document_id': 'doc-1'}).encode('utf-8')
        kb_client._s3.get_object.return_value = {'Body': body}

        assert kb_client.get_document('doc-1') == {'document_id': 'doc-1'}

    def test_get_document_missing_returns_none(self, kb_client):
        """A missing key yields None rather than raising."""
        kb_client._s3.get_object.side_effect = ClientError(
            {'Error': {'Code': 'NoSuchKey', 'Message': 'missing'}}, 'GetObject'
        )

        assert kb_client.get_document('doc-1') is None

    def test_delete_document_success(self, kb_client):
        """An existing document is deleted from S3."""
        assert kb_client.delete_document('doc-1') is True
        kb_client._s3.delete_object.assert_called_once_with(
            Bucket='trustops-test-datasets-123456789012-us-east-1',
            Key='knowledge-base/doc-1.json',
        )

    def test_delete_document_missing_returns_false(self, kb_client):
        """Deleting an absent document reports False."""
        kb_client._s3.head_object.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'missing'}}, 'HeadObject'
        )

        assert kb_client.delete_document('doc-1') is False
        kb_client._s3.delete_object.assert_not_called()

    def test_delete_document_in_mock_mode(self, mock_kb_client):
        """Mock mode deletes from in-memory storage."""
        mock_kb_client.store_document('doc-1', 'text')

        assert mock_kb_client.delete_document('doc-1') is True
        assert mock_kb_client.delete_document('doc-1') is False


class TestBulkStoreDocuments:
    """Tests for bulk_store_documents."""

    def test_bulk_store_defers_sync_to_single_job(self, kb_client):
        """All documents upload, then one ingestion job runs."""
        documents = [
            {'document_id': f'doc-{i}', 'text': f'text-{i}'} for i in range(3)
        ]

        result = kb_client.bulk_store_documents(documents, sync=True)

        assert result == {'success': 3, 'failed': 0}
        assert kb_client._s3.put_object.call_count == 3
        kb_client._bedrock_agent.start_ingestion_job.assert_called_once()

    def test_bulk_store_counts_failures(self, kb_client):
        """A failed upload is counted without aborting the batch."""
        kb_client._s3.put_object.side_effect = [
            None,
            ClientError({'Error': {'Code': 'AccessDenied', 'Message': 'no'}}, 'PutObject'),
            None,
        ]
        documents = [
            {'document_id': f'doc-{i}', 'text': f'text-{i}'} for i in range(3)
        ]

        result = kb_client.bulk_store_documents(documents, sync=False)

        assert result == {'success': 2, 'failed': 1}
