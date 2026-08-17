"""
Amazon Bedrock Knowledge Bases client for document storage and semantic retrieval.
"""
import json
from typing import List, Tuple, Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError

from config.aws_config import config


class KnowledgeBaseClient:
    """Client wrapper for Amazon Bedrock Knowledge Bases operations."""

    def __init__(
        self,
        knowledge_base_id: Optional[str] = None,
        data_source_id: Optional[str] = None,
        mock_mode: bool = False
    ):
        """
        Initialize Bedrock Knowledge Bases client.

        Args:
            knowledge_base_id: Knowledge Base ID (defaults to config.knowledge_base_id)
            data_source_id: Data source ID for ingestion sync (defaults to config.knowledge_base_data_source_id)
            mock_mode: If True, run in mock mode without actual AWS connections
        """
        self.knowledge_base_id = knowledge_base_id or config.knowledge_base_id
        self.data_source_id = data_source_id or config.knowledge_base_data_source_id
        self.datasets_bucket = config.datasets_bucket
        self.mock_mode = mock_mode or not self.knowledge_base_id

        if self.mock_mode:
            # Mock mode - no actual AWS connections
            self._bedrock_agent_runtime = None
            self._bedrock_agent = None
            self._s3 = None
            self._mock_storage: Dict[str, Dict[str, Any]] = {}
            return

        # Set up AWS session
        session_kwargs = config.get_boto3_session_kwargs()
        session = boto3.Session(**session_kwargs)

        # Initialize clients
        self._bedrock_agent_runtime = session.client('bedrock-agent-runtime')
        self._bedrock_agent = session.client('bedrock-agent')
        self._s3 = session.client('s3')

    def _s3_key_for_document(self, document_id: str) -> str:
        """
        Build the S3 object key for a document.

        Args:
            document_id: Unique document identifier

        Returns:
            S3 key under the knowledge-base/ prefix
        """
        return f"knowledge-base/{document_id}.json"

    def store_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = True
    ) -> str:
        """
        Store a document in S3 for Knowledge Base ingestion.

        Writes the document to the datasets bucket under the knowledge-base/ prefix,
        then optionally triggers a data source sync so the Knowledge Base ingests it.

        Args:
            document_id: Unique document identifier
            text: Document text content
            metadata: Optional metadata dictionary
            sync: If True, trigger a data source ingestion job after upload

        Returns:
            Document ID of stored document
        """
        if self.mock_mode:
            self._mock_storage[document_id] = {
                'text': text,
                'metadata': metadata or {}
            }
            return document_id

        # Build document payload
        document_payload = {
            'document_id': document_id,
            'text': text,
            'metadata': metadata or {}
        }

        # Upload to S3
        s3_key = self._s3_key_for_document(document_id)
        try:
            self._s3.put_object(
                Bucket=self.datasets_bucket,
                Key=s3_key,
                Body=json.dumps(document_payload, indent=2),
                ContentType='application/json'
            )
        except ClientError as e:
            raise RuntimeError(
                f"Failed to upload document '{document_id}' to S3: {e}"
            ) from e

        # Optionally trigger data source sync
        if sync and self.data_source_id:
            self._start_ingestion_job()

        return document_id

    def _start_ingestion_job(self) -> Optional[str]:
        """
        Trigger a data source ingestion job to sync new documents into the Knowledge Base.

        Returns:
            Ingestion job ID if successful, None otherwise
        """
        if self.mock_mode or not self.data_source_id:
            return None

        try:
            response = self._bedrock_agent.start_ingestion_job(
                knowledgeBaseId=self.knowledge_base_id,
                dataSourceId=self.data_source_id
            )
            return response.get('ingestionJob', {}).get('ingestionJobId')
        except ClientError as e:
            # Log but don't fail - document is already in S3
            print(f"Warning: Failed to start ingestion job: {e}")
            return None

    def search_similar(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Tuple[str, float, str]]:
        """
        Search for similar documents using the Knowledge Base retrieve API.

        The Knowledge Base handles embedding the query text internally.

        Args:
            query_text: Natural language query text
            top_k: Number of results to return
            min_score: Minimum relevance score threshold (optional)

        Returns:
            List of (document_id, score, text) tuples
        """
        if self.mock_mode:
            import random
            results = []
            for doc_id, doc_data in list(self._mock_storage.items())[:top_k]:
                # Synthetic score for local mock mode only - never a
                # security control.
                score = random.uniform(0.75, 0.95)  # nosec B311
                if min_score is None or score >= min_score:
                    results.append((doc_id, score, doc_data['text']))
            return results

        # Build retrieval configuration
        retrieval_config: Dict[str, Any] = {
            'vectorSearchConfiguration': {
                'numberOfResults': top_k
            }
        }

        # Add score filter if specified
        if min_score is not None:
            retrieval_config['vectorSearchConfiguration']['overrideSearchType'] = 'HYBRID'

        try:
            response = self._bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={
                    'text': query_text
                },
                retrievalConfiguration=retrieval_config
            )
        except ClientError as e:
            raise RuntimeError(
                f"Failed to retrieve from Knowledge Base: {e}"
            ) from e

        # Parse results
        results = []
        for result in response.get('retrievalResults', []):
            score = result.get('score', 0.0)

            # Filter by minimum score if specified
            if min_score is not None and score < min_score:
                continue

            # Extract document ID from metadata or location URI
            location = result.get('location', {})
            uri = location.get('s3Location', {}).get('uri', '')
            document_id = self._extract_document_id_from_uri(uri)

            # Extract text content
            text = result.get('content', {}).get('text', '')

            results.append((document_id, score, text))

        return results

    def _extract_document_id_from_uri(self, uri: str) -> str:
        """
        Extract a document ID from an S3 URI.

        Expects URIs like: s3://bucket/knowledge-base/document_id.json

        Args:
            uri: S3 URI string

        Returns:
            Extracted document ID or the raw URI if parsing fails
        """
        try:
            # Strip s3://bucket/ prefix and knowledge-base/ prefix
            path = uri.split('/', 3)[-1] if '/' in uri else uri
            if path.startswith('knowledge-base/'):
                path = path[len('knowledge-base/'):]
            # Remove .json extension
            if path.endswith('.json'):
                path = path[:-5]
            return path
        except (IndexError, ValueError):
            return uri

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID from S3.

        Args:
            document_id: Document identifier

        Returns:
            Document dictionary or None if not found
        """
        if self.mock_mode:
            return self._mock_storage.get(document_id)

        s3_key = self._s3_key_for_document(document_id)
        try:
            response = self._s3.get_object(
                Bucket=self.datasets_bucket,
                Key=s3_key
            )
            body = response['Body'].read().decode('utf-8')
            return json.loads(body)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID from S3.

        Note: The document will be removed from the Knowledge Base on the next
        data source sync.

        Args:
            document_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        if self.mock_mode:
            if document_id in self._mock_storage:
                del self._mock_storage[document_id]
                return True
            return False

        s3_key = self._s3_key_for_document(document_id)
        try:
            # Check if object exists first
            self._s3.head_object(Bucket=self.datasets_bucket, Key=s3_key)
            # Delete the object
            self._s3.delete_object(Bucket=self.datasets_bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] in ('404', 'NoSuchKey'):
                return False
            raise

    def bulk_store_documents(
        self,
        documents: List[Dict[str, Any]],
        sync: bool = True
    ) -> Dict[str, int]:
        """
        Store multiple documents in bulk.

        Args:
            documents: List of document dictionaries with keys:
                - document_id: Unique identifier
                - text: Document text content
                - metadata: Optional metadata
            sync: If True, trigger a single ingestion job after all uploads

        Returns:
            Dictionary with 'success' and 'failed' counts
        """
        success_count = 0
        failed_count = 0

        for doc in documents:
            try:
                self.store_document(
                    document_id=doc['document_id'],
                    text=doc['text'],
                    metadata=doc.get('metadata'),
                    sync=False  # Defer sync until all documents are uploaded
                )
                success_count += 1
            except (RuntimeError, ClientError):
                failed_count += 1

        # Trigger a single ingestion job for all uploaded documents
        if sync and self.data_source_id and success_count > 0:
            self._start_ingestion_job()

        return {
            'success': success_count,
            'failed': failed_count
        }
