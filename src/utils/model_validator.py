"""
Model validation utilities for foundation model comparison.

This module provides validation functions for AWS Bedrock model IDs,
ensuring models are valid, different, and retrieving model metadata.
"""
import re
from typing import Dict, Any
from botocore.exceptions import ClientError


# Supported AWS Bedrock model families and their ID patterns.
# Note: Amazon has two current text families — Titan and Nova — so the amazon
# pattern accepts both.
SUPPORTED_MODEL_PATTERNS = {
    'anthropic': r'^anthropic\.claude-.*',
    'amazon': r'^amazon\.(titan|nova)-.*',
    'meta': r'^meta\.llama.*'
}

# Model family information. Amazon spans Titan and Nova; the family label is
# refined per model in get_model_metadata().
MODEL_FAMILIES = {
    'anthropic': 'Anthropic Claude',
    'amazon': 'Amazon Titan / Nova',
    'meta': 'Meta Llama'
}


def validate_bedrock_model_id(model_id: str) -> bool:
    """
    Validate that a model ID is a valid AWS Bedrock foundation model identifier.
    
    This function checks:
    1. Model ID format matches supported patterns (Anthropic Claude, Amazon Titan, Meta Llama)
    2. Model ID is not empty or None
    
    Args:
        model_id: AWS Bedrock model identifier string
        
    Returns:
        True if model ID is valid, False otherwise
        
    Examples:
        >>> validate_bedrock_model_id("anthropic.claude-3-haiku-20240307-v1:0")
        True
        >>> validate_bedrock_model_id("amazon.titan-text-express-v1")
        True
        >>> validate_bedrock_model_id("invalid-model")
        False
    """
    if not model_id or not isinstance(model_id, str):
        return False
    
    # Check if model ID matches any supported pattern
    for family, pattern in SUPPORTED_MODEL_PATTERNS.items():
        if re.match(pattern, model_id):
            return True
    
    return False


def validate_models_different(model_id_1: str, model_id_2: str) -> None:
    """
    Validate that two model IDs are different.
    
    This function ensures that users don't attempt to compare a model with itself,
    which would be meaningless for comparison purposes.
    
    Args:
        model_id_1: First model identifier
        model_id_2: Second model identifier
        
    Raises:
        ValueError: If model IDs are identical
        
    Examples:
        >>> validate_models_different("anthropic.claude-3-haiku-20240307-v1:0", 
        ...                          "anthropic.claude-3-sonnet-20240229-v1:0")
        # No exception raised
        
        >>> validate_models_different("anthropic.claude-3-haiku-20240307-v1:0",
        ...                          "anthropic.claude-3-haiku-20240307-v1:0")
        Traceback (most recent call last):
        ...
        ValueError: Cannot compare identical models...
    """
    if model_id_1 == model_id_2:
        raise ValueError(
            f"Cannot compare identical models. Please provide two different model IDs. "
            f"Both model_id_1 and model_id_2 are set to: {model_id_1}"
        )


def get_model_metadata(model_id: str) -> Dict[str, Any]:
    """
    Retrieve model metadata including pricing and model family information.
    
    This function extracts metadata from the model ID without making AWS API calls.
    It provides pricing information and model family classification.
    
    Args:
        model_id: AWS Bedrock model identifier
        
    Returns:
        Dictionary containing:
            - model_id: The model identifier
            - model_family: Model family name (e.g., "Anthropic Claude")
            - provider: Provider prefix (e.g., "anthropic")
            - pricing: Dictionary with 'input' and 'output' costs per 1000 tokens
            
    Raises:
        ValueError: If model_id is invalid or not supported
        
    Examples:
        >>> metadata = get_model_metadata("anthropic.claude-3-haiku-20240307-v1:0")
        >>> metadata['model_family']
        'Anthropic Claude'
        >>> metadata['pricing']['input']
        0.00025
    """
    # First validate the model ID format
    if not validate_bedrock_model_id(model_id):
        raise ValueError(
            f"Invalid model ID: {model_id}. Must be a valid AWS Bedrock foundation model "
            f"identifier from supported families: {', '.join(MODEL_FAMILIES.values())}"
        )
    
    # Extract provider from model ID
    provider = model_id.split('.')[0]

    # Get model family. Refine the Amazon label to distinguish Titan vs Nova.
    model_family = MODEL_FAMILIES.get(provider, 'Unknown')
    if provider == 'amazon':
        if model_id.startswith('amazon.nova'):
            model_family = 'Amazon Nova'
        elif model_id.startswith('amazon.titan'):
            model_family = 'Amazon Titan'

    # Pricing information (USD per 1000 tokens)
    # This matches the pricing in BedrockClient
    pricing_map = {
        'anthropic.claude-v2': {'input': 0.008, 'output': 0.024},
        'anthropic.claude-v2:1': {'input': 0.008, 'output': 0.024},
        'anthropic.claude-instant-v1': {'input': 0.0008, 'output': 0.0024},
        'anthropic.claude-3-haiku-20240307-v1:0': {'input': 0.00025, 'output': 0.00125},
        'anthropic.claude-3-sonnet-20240229-v1:0': {'input': 0.003, 'output': 0.015},
        'anthropic.claude-3-opus-20240229-v1:0': {'input': 0.015, 'output': 0.075},
        'amazon.titan-text-express-v1': {'input': 0.0002, 'output': 0.0006},
        'amazon.titan-text-lite-v1': {'input': 0.00015, 'output': 0.0002},
        # Amazon Nova (current) — USD per 1000 tokens.
        'amazon.nova-micro-v1:0': {'input': 0.000035, 'output': 0.00014},
        'amazon.nova-lite-v1:0': {'input': 0.00006, 'output': 0.00024},
        'amazon.nova-pro-v1:0': {'input': 0.0008, 'output': 0.0032},
        'meta.llama2-13b-chat-v1': {'input': 0.00075, 'output': 0.001},
        'meta.llama2-70b-chat-v1': {'input': 0.00195, 'output': 0.00256}
    }
    
    # Get pricing, default to 0.0 if not found
    pricing = pricing_map.get(model_id, {'input': 0.0, 'output': 0.0})
    
    return {
        'model_id': model_id,
        'model_family': model_family,
        'provider': provider,
        'pricing': pricing
    }
