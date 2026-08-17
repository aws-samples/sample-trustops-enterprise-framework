"""AWS service client wrappers."""
from .bedrock_client import BedrockClient
from .s3_storage_manager import S3StorageManager
from .knowledge_base_client import KnowledgeBaseClient
from .semantic_similarity_analyzer import SemanticSimilarityAnalyzer

__all__ = [
    'BedrockClient',
    'S3StorageManager',
    'KnowledgeBaseClient',
    'SemanticSimilarityAnalyzer'
]
