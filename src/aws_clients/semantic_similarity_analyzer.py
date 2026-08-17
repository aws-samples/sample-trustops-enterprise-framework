"""
Semantic similarity analyzer using embeddings and Bedrock Knowledge Bases.
"""
import json
import numpy as np
from typing import List, Tuple, Optional
import boto3
from botocore.exceptions import ClientError

from config.aws_config import config
from src.aws_clients.knowledge_base_client import KnowledgeBaseClient


class SemanticSimilarityAnalyzer:
    """Analyzer for calculating semantic similarity between texts."""

    def __init__(
        self,
        embedding_model_id: Optional[str] = None,
        knowledge_base_client: Optional[KnowledgeBaseClient] = None,
        mock_mode: bool = False
    ):
        """
        Initialize semantic similarity analyzer.

        Args:
            embedding_model_id: Bedrock embedding model ID (defaults to config)
            knowledge_base_client: Knowledge Base client instance (creates new if None)
            mock_mode: If True, run in mock mode without actual AWS calls
        """
        self.embedding_model_id = embedding_model_id or config.default_embedding_model
        self.mock_mode = mock_mode or not config.knowledge_base_id

        if not self.mock_mode:
            # Initialize Bedrock runtime client
            session_kwargs = config.get_boto3_session_kwargs()
            session = boto3.Session(**session_kwargs)
            self.bedrock_runtime = session.client('bedrock-runtime')
        else:
            self.bedrock_runtime = None

        # Initialize or use provided Knowledge Base client
        self.kb_client = knowledge_base_client or KnowledgeBaseClient(mock_mode=self.mock_mode)
    
    def generate_embedding(self, text: str, model_id: Optional[str] = None) -> List[float]:
        """
        Generate embedding vector for text using Bedrock.
        
        Args:
            text: Input text to embed
            model_id: Bedrock embedding model identifier (defaults to instance default)
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If text is empty or model is invalid
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if self.mock_mode:
            # Return mock embedding (1536 dimensions for Titan)
            import random
            # Deterministic synthetic embedding for local mock mode only -
            # never a security control.
            random.seed(hash(text) % (2**32))
            return [random.uniform(-1, 1) for _ in range(1536)]  # nosec B311
        
        model = model_id or self.embedding_model_id
        
        try:
            # Format request based on model
            if model.startswith('amazon.titan-embed'):
                body = json.dumps({
                    'inputText': text
                })
            elif model.startswith('cohere.embed'):
                body = json.dumps({
                    'texts': [text],
                    'input_type': 'search_document'
                })
            else:
                raise ValueError(f"Unsupported embedding model: {model}")
            
            # Invoke embedding model
            response = self.bedrock_runtime.invoke_model(
                modelId=model,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract embedding based on model
            if model.startswith('amazon.titan-embed'):
                embedding = response_body.get('embedding', [])
            elif model.startswith('cohere.embed'):
                embeddings = response_body.get('embeddings', [[]])
                embedding = embeddings[0] if embeddings else []
            else:
                embedding = []
            
            if not embedding:
                raise RuntimeError(f"Failed to generate embedding: empty response from {model}")
            
            return embedding
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')
            
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid embedding request: {error_message}") from e
            elif error_code == 'ResourceNotFoundException':
                raise ValueError(f"Embedding model not found: {model}") from e
            elif error_code == 'ThrottlingException':
                raise RuntimeError(f"Rate limit exceeded for embedding model {model}") from e
            else:
                raise RuntimeError(f"Embedding generation failed: {error_message}") from e
    
    def store_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Store a document in the Knowledge Base for later retrieval.

        Args:
            document_id: Unique document identifier
            text: Text to store
            metadata: Optional metadata dictionary

        Returns:
            Document ID of stored document
        """
        return self.kb_client.store_document(
            document_id=document_id,
            text=text,
            metadata=metadata,
            sync=False,
        )
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using cosine similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Generate embeddings for both texts
        embedding1 = self.generate_embedding(text1)
        embedding2 = self.generate_embedding(text2)
        
        # Calculate cosine similarity
        return self._cosine_similarity(embedding1, embedding2)
    
    def find_similar_documents(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Find most similar documents to query text.

        Args:
            query_text: Query text
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (optional)

        Returns:
            List of (document_text, similarity_score) tuples
        """
        results = self.kb_client.search_similar(
            query_text=query_text,
            top_k=top_k,
            min_score=min_score,
        )

        return [(text, score) for _, score, text in results]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Clamp to [0, 1] range (cosine similarity is in [-1, 1])
        # For embeddings, we typically expect positive similarity
        return float(max(0.0, min(1.0, (similarity + 1.0) / 2.0)))
