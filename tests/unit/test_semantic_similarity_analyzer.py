"""
Unit tests for SemanticSimilarityAnalyzer.
"""
import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

from src.aws_clients.semantic_similarity_analyzer import SemanticSimilarityAnalyzer


@pytest.fixture
def mock_bedrock_runtime():
    """Create a mock Bedrock runtime client."""
    with patch('src.aws_clients.semantic_similarity_analyzer.boto3.Session') as mock_session:
        mock_runtime = Mock()
        
        mock_session_instance = Mock()
        mock_session_instance.client.return_value = mock_runtime
        mock_session.return_value = mock_session_instance
        
        yield mock_runtime


@pytest.fixture
def mock_kb_client():
    """Create a mock Knowledge Base client."""
    return Mock()


@pytest.fixture
def analyzer(mock_bedrock_runtime, mock_kb_client):
    """Create a SemanticSimilarityAnalyzer instance with mocked dependencies."""
    with patch('src.aws_clients.semantic_similarity_analyzer.config') as mock_config, \
         patch('src.aws_clients.semantic_similarity_analyzer.KnowledgeBaseClient') as mock_kb_class:

        mock_config.default_embedding_model = 'amazon.titan-embed-text-v1'
        mock_config.get_boto3_session_kwargs.return_value = {'region_name': 'us-east-1'}

        mock_kb_class.return_value = mock_kb_client

        analyzer = SemanticSimilarityAnalyzer()
        analyzer.bedrock_runtime = mock_bedrock_runtime
        analyzer.kb_client = mock_kb_client

        return analyzer


class TestGenerateEmbedding:
    """Tests for generate_embedding method."""
    
    def test_generate_embedding_titan_model(self, analyzer, mock_bedrock_runtime):
        """Test generating embedding with Titan model."""
        text = "Test document for embedding"
        embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            'embedding': embedding_vector
        }).encode('utf-8')
        
        mock_bedrock_runtime.invoke_model.return_value = {
            'body': mock_response
        }
        
        result = analyzer.generate_embedding(text)
        
        assert result == embedding_vector
        
        # Verify the call
        call_args = mock_bedrock_runtime.invoke_model.call_args
        assert call_args[1]['modelId'] == 'amazon.titan-embed-text-v1'
        
        body = json.loads(call_args[1]['body'])
        assert body['inputText'] == text
    
    def test_generate_embedding_cohere_model(self, analyzer, mock_bedrock_runtime):
        """Test generating embedding with Cohere model."""
        text = "Test document"
        embedding_vector = [0.1, 0.2, 0.3]
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            'embeddings': [embedding_vector]
        }).encode('utf-8')
        
        mock_bedrock_runtime.invoke_model.return_value = {
            'body': mock_response
        }
        
        result = analyzer.generate_embedding(text, model_id='cohere.embed-english-v3')
        
        assert result == embedding_vector
    
    def test_generate_embedding_empty_text(self, analyzer):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyzer.generate_embedding("")
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyzer.generate_embedding("   ")
    
    def test_generate_embedding_unsupported_model(self, analyzer):
        """Test that unsupported model raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported embedding model"):
            analyzer.generate_embedding("test", model_id="unsupported.model")
    
    def test_generate_embedding_api_error(self, analyzer, mock_bedrock_runtime):
        """Test handling of API errors."""
        error_response = {
            'Error': {
                'Code': 'ThrottlingException',
                'Message': 'Rate limit exceeded'
            }
        }
        mock_bedrock_runtime.invoke_model.side_effect = ClientError(
            error_response, 'InvokeModel'
        )
        
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            analyzer.generate_embedding("test")
    
    def test_generate_embedding_empty_response(self, analyzer, mock_bedrock_runtime):
        """Test handling of empty embedding response."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            'embedding': []
        }).encode('utf-8')
        
        mock_bedrock_runtime.invoke_model.return_value = {
            'body': mock_response
        }
        
        with pytest.raises(RuntimeError, match="Failed to generate embedding"):
            analyzer.generate_embedding("test")


class TestStoreDocument:
    """Tests for store_document method."""

    def test_store_document(self, analyzer, mock_kb_client):
        """Test storing a document in the Knowledge Base."""
        document_id = 'doc-123'
        text = 'Test document'
        metadata = {'category': 'test'}

        mock_kb_client.store_document.return_value = document_id

        result = analyzer.store_document(document_id, text, metadata)

        assert result == document_id

        # The Knowledge Base embeds the document itself, so the analyzer
        # forwards the raw text without generating an embedding locally.
        mock_kb_client.store_document.assert_called_once_with(
            document_id=document_id,
            text=text,
            metadata=metadata,
            sync=False,
        )


class TestCalculateSimilarity:
    """Tests for calculate_similarity method."""
    
    def test_calculate_similarity_identical_texts(self, analyzer):
        """Test similarity between identical texts."""
        text = "This is a test document"
        embedding = [0.5, 0.5, 0.5, 0.5]
        
        with patch.object(analyzer, 'generate_embedding', return_value=embedding):
            similarity = analyzer.calculate_similarity(text, text)
            
            # Identical embeddings should have similarity close to 1.0
            assert 0.99 <= similarity <= 1.0
    
    def test_calculate_similarity_different_texts(self, analyzer):
        """Test similarity between different texts."""
        text1 = "First document"
        text2 = "Second document"
        
        embedding1 = [1.0, 0.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0, 0.0]
        
        with patch.object(analyzer, 'generate_embedding', side_effect=[embedding1, embedding2]):
            similarity = analyzer.calculate_similarity(text1, text2)
            
            # Orthogonal vectors should have low similarity
            assert 0.0 <= similarity <= 1.0
            assert similarity < 0.6  # Should be relatively low
    
    def test_calculate_similarity_similar_texts(self, analyzer):
        """Test similarity between similar texts."""
        text1 = "Document about cats"
        text2 = "Document about dogs"
        
        # Similar but not identical embeddings
        embedding1 = [0.8, 0.2, 0.1, 0.1]
        embedding2 = [0.7, 0.3, 0.15, 0.05]
        
        with patch.object(analyzer, 'generate_embedding', side_effect=[embedding1, embedding2]):
            similarity = analyzer.calculate_similarity(text1, text2)
            
            # Should have moderate to high similarity
            assert 0.0 <= similarity <= 1.0
            assert similarity > 0.5


class TestFindSimilarDocuments:
    """Tests for find_similar_documents method."""
    
    def test_find_similar_documents(self, analyzer, mock_kb_client):
        """Test finding similar documents."""
        query_text = "Search query"

        mock_kb_client.search_similar.return_value = [
            ('doc-1', 0.95, 'Similar document 1'),
            ('doc-2', 0.85, 'Similar document 2'),
            ('doc-3', 0.75, 'Similar document 3')
        ]

        results = analyzer.find_similar_documents(query_text, top_k=3)

        assert len(results) == 3
        assert results[0] == ('Similar document 1', 0.95)
        assert results[1] == ('Similar document 2', 0.85)
        assert results[2] == ('Similar document 3', 0.75)

        # The Knowledge Base embeds the query internally.
        mock_kb_client.search_similar.assert_called_once_with(
            query_text=query_text,
            top_k=3,
            min_score=None,
        )

    def test_find_similar_documents_with_min_score(self, analyzer, mock_kb_client):
        """Test finding similar documents with minimum score."""
        query_text = "Search query"
        min_score = 0.8

        mock_kb_client.search_similar.return_value = [
            ('doc-1', 0.95, 'High score document')
        ]

        results = analyzer.find_similar_documents(
            query_text,
            top_k=5,
            min_score=min_score
        )

        assert len(results) == 1
        assert results[0][1] >= min_score

        mock_kb_client.search_similar.assert_called_once_with(
            query_text=query_text,
            top_k=5,
            min_score=min_score,
        )


class TestCosineSimilarity:
    """Tests for _cosine_similarity method."""
    
    def test_cosine_similarity_identical_vectors(self, analyzer):
        """Test cosine similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0, 4.0]
        
        similarity = analyzer._cosine_similarity(vec, vec)
        
        assert 0.99 <= similarity <= 1.0
    
    def test_cosine_similarity_orthogonal_vectors(self, analyzer):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        similarity = analyzer._cosine_similarity(vec1, vec2)
        
        # Orthogonal vectors should have similarity around 0.5 after normalization
        assert 0.4 <= similarity <= 0.6
    
    def test_cosine_similarity_opposite_vectors(self, analyzer):
        """Test cosine similarity of opposite vectors."""
        vec1 = [1.0, 1.0, 1.0]
        vec2 = [-1.0, -1.0, -1.0]
        
        similarity = analyzer._cosine_similarity(vec1, vec2)
        
        # Opposite vectors should have low similarity (close to 0)
        assert 0.0 <= similarity <= 0.1
    
    def test_cosine_similarity_zero_vector(self, analyzer):
        """Test cosine similarity with zero vector."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        
        similarity = analyzer._cosine_similarity(vec1, vec2)
        
        assert similarity == 0.0
    
    def test_cosine_similarity_range(self, analyzer):
        """Test that cosine similarity is always in [0, 1] range."""
        # Test various vector combinations
        test_cases = [
            ([1.0, 0.0], [0.0, 1.0]),
            ([1.0, 1.0], [1.0, 1.0]),
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
            ([-1.0, -2.0], [3.0, 4.0])
        ]
        
        for vec1, vec2 in test_cases:
            similarity = analyzer._cosine_similarity(vec1, vec2)
            assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} out of range for {vec1}, {vec2}"
