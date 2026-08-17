"""Input validation utilities for CLI."""
import re
from typing import Optional


def validate_s3_uri(uri: str) -> None:
    """
    Validate S3 URI format.
    
    Args:
        uri: S3 URI to validate
        
    Raises:
        ValueError: If URI format is invalid
    """
    if not uri:
        raise ValueError("S3 URI cannot be empty")
    
    if not uri.startswith('s3://'):
        raise ValueError(
            f"Invalid S3 URI format: {uri}. "
            "Must start with 's3://'"
        )
    
    # Check for bucket and key
    parts = uri.replace('s3://', '').split('/', 1)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid S3 URI format: {uri}. "
            "Must be 's3://bucket/key'"
        )
    
    # Validate bucket name (basic validation)
    bucket = parts[0]
    if not re.match(r'^[a-z0-9][a-z0-9.-]*[a-z0-9]$', bucket):
        raise ValueError(
            f"Invalid S3 bucket name: {bucket}. "
            "Bucket names must be lowercase alphanumeric with hyphens/dots"
        )


def validate_model_id(model_id: str) -> None:
    """
    Validate AWS Bedrock model ID format.
    
    Args:
        model_id: Model ID to validate
        
    Raises:
        ValueError: If model ID format is invalid
    """
    if not model_id:
        raise ValueError("Model ID cannot be empty")
    
    # Check for valid formats:
    # 1. Provider.model-name (e.g., anthropic.claude-v2)
    # 2. ARN format (e.g., arn:aws:bedrock:...)
    
    if model_id.startswith('arn:aws:bedrock:'):
        # ARN format - basic validation
        parts = model_id.split(':')
        if len(parts) < 6:
            raise ValueError(
                f"Invalid model ARN format: {model_id}"
            )
    elif '.' in model_id:
        # Provider.model format
        parts = model_id.split('.')
        if len(parts) < 2:
            raise ValueError(
                f"Invalid model ID format: {model_id}. "
                "Expected format: provider.model-name"
            )
    else:
        raise ValueError(
            f"Invalid model ID format: {model_id}. "
            "Expected format: provider.model-name or ARN"
        )


def validate_workflow_id(workflow_id: str) -> None:
    """
    Validate workflow ID format.
    
    Args:
        workflow_id: Workflow ID to validate
        
    Raises:
        ValueError: If workflow ID format is invalid
    """
    if not workflow_id:
        raise ValueError("Workflow ID cannot be empty")
    
    # Basic validation - alphanumeric with hyphens
    if not re.match(r'^[a-z0-9-]+$', workflow_id):
        raise ValueError(
            f"Invalid workflow ID format: {workflow_id}. "
            "Must be lowercase alphanumeric with hyphens"
        )


def validate_positive_int(value: int, name: str) -> None:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        
    Raises:
        ValueError: If value is not a positive integer
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer, got: {value}"
        )


def validate_positive_float(value: float, name: str) -> None:
    """
    Validate that a value is a positive float.
    
    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        
    Raises:
        ValueError: If value is not a positive float
    """
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(
            f"{name} must be a positive number, got: {value}"
        )


def validate_range(value: float, min_val: float, max_val: float, 
                   name: str) -> None:
    """
    Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the parameter (for error messages)
        
    Raises:
        ValueError: If value is outside the range
    """
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a number, got: {type(value).__name__}"
        )
    
    if value < min_val or value > max_val:
        raise ValueError(
            f"{name} must be between {min_val} and {max_val}, got: {value}"
        )
