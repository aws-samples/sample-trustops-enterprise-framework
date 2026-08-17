"""AWS utility functions for client initialization and common operations."""

import boto3
from typing import Optional
from config.aws_config import config


def get_boto3_session(profile: Optional[str] = None, region: Optional[str] = None) -> boto3.Session:
    """
    Create a boto3 session with optional profile and region.
    
    Args:
        profile: AWS profile name (uses config default if not provided)
        region: AWS region (uses config default if not provided)
        
    Returns:
        Configured boto3 Session
    """
    session_kwargs = {}
    
    if profile or config.profile:
        session_kwargs["profile_name"] = profile or config.profile
    
    if region or config.region:
        session_kwargs["region_name"] = region or config.region
    
    return boto3.Session(**session_kwargs)


def get_s3_client(session: Optional[boto3.Session] = None):
    """Get S3 client."""
    if session is None:
        session = get_boto3_session()
    return session.client("s3")


def get_dynamodb_client(session: Optional[boto3.Session] = None):
    """Get DynamoDB client."""
    if session is None:
        session = get_boto3_session()
    return session.client("dynamodb")


def get_dynamodb_resource(session: Optional[boto3.Session] = None):
    """Get DynamoDB resource."""
    if session is None:
        session = get_boto3_session()
    return session.resource("dynamodb")


def get_bedrock_client(session: Optional[boto3.Session] = None):
    """Get Bedrock client."""
    if session is None:
        session = get_boto3_session()
    return session.client("bedrock")


def get_bedrock_runtime_client(session: Optional[boto3.Session] = None):
    """Get Bedrock Runtime client for model inference."""
    if session is None:
        session = get_boto3_session()
    return session.client("bedrock-runtime")


def get_logs_client(session: Optional[boto3.Session] = None):
    """Get CloudWatch Logs client."""
    if session is None:
        session = get_boto3_session()
    return session.client("logs")


def get_stepfunctions_client(session: Optional[boto3.Session] = None):
    """Get Step Functions client."""
    if session is None:
        session = get_boto3_session()
    return session.client("stepfunctions")
