"""
S3 path convention for TrustOps Enterprise Framework.

Path structure: s3://{bucket}/{prefix}/{workflow_id}/{artifact_type}/{timestamp}/

This module provides path builder and parser functions for consistent
S3 path generation across all storage operations.

Requirements: 9.2
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from src.data_models.storage import StoragePathConfig


class ParsedPath(BaseModel):
    """Parsed components of a TrustOps S3 path."""

    bucket: str
    prefix: str
    workflow_id: str
    artifact_type: str
    timestamp: str
    filename: Optional[str] = None


TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

VALID_ARTIFACT_TYPES = {
    "baseline_evaluation",
    "comparison",
    "fine_tuning",
    "trust_score",
    "hallucination_analysis",
    "workflow_manifest",
    "dataset_quality",
    "model_metadata",
}


def build_s3_path(
    config: StoragePathConfig,
    workflow_id: str,
    artifact_type: str,
    timestamp: Optional[datetime] = None,
    filename: Optional[str] = None,
) -> str:
    """Build an S3 path following the TrustOps convention.

    Convention: s3://{bucket}/{prefix}/{workflow_id}/{artifact_type}/{timestamp}/

    Args:
        config: Storage path configuration with bucket and prefix.
        workflow_id: Unique workflow identifier.
        artifact_type: Type of artifact being stored.
        timestamp: Timestamp for the artifact. Defaults to current UTC time.
        filename: Optional filename to append to the path.

    Returns:
        Full S3 URI string.
    """
    if not workflow_id:
        raise ValueError("workflow_id cannot be empty")
    if not artifact_type:
        raise ValueError("artifact_type cannot be empty")

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    ts_str = timestamp.strftime(TIMESTAMP_FORMAT)
    path = f"s3://{config.bucket}/{config.prefix}/{workflow_id}/{artifact_type}/{ts_str}"

    if filename:
        path = f"{path}/{filename}"
    else:
        path = f"{path}/"

    return path


def build_s3_key(
    config: StoragePathConfig,
    workflow_id: str,
    artifact_type: str,
    timestamp: Optional[datetime] = None,
    filename: Optional[str] = None,
) -> str:
    """Build an S3 key (without bucket) following the TrustOps convention.

    Args:
        config: Storage path configuration with prefix.
        workflow_id: Unique workflow identifier.
        artifact_type: Type of artifact being stored.
        timestamp: Timestamp for the artifact. Defaults to current UTC time.
        filename: Optional filename to append to the key.

    Returns:
        S3 key string (without s3://bucket prefix).
    """
    if not workflow_id:
        raise ValueError("workflow_id cannot be empty")
    if not artifact_type:
        raise ValueError("artifact_type cannot be empty")

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    ts_str = timestamp.strftime(TIMESTAMP_FORMAT)
    key = f"{config.prefix}/{workflow_id}/{artifact_type}/{ts_str}"

    if filename:
        key = f"{key}/{filename}"
    else:
        key = f"{key}/"

    return key


def parse_s3_path(s3_uri: str) -> ParsedPath:
    """Parse a TrustOps S3 path into its components.

    Args:
        s3_uri: Full S3 URI to parse.

    Returns:
        ParsedPath with extracted components.

    Raises:
        ValueError: If the path does not follow the expected convention.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: must start with 's3://': {s3_uri}")

    path = s3_uri[5:]  # Remove 's3://'
    parts = path.rstrip("/").split("/")

    if len(parts) < 4:
        raise ValueError(
            f"Invalid TrustOps S3 path: expected at least "
            f"bucket/prefix/workflow_id/artifact_type/timestamp, got: {s3_uri}"
        )

    bucket = parts[0]
    prefix = parts[1]
    workflow_id = parts[2]
    artifact_type = parts[3]

    timestamp = parts[4] if len(parts) > 4 else ""
    filename = parts[5] if len(parts) > 5 else None

    return ParsedPath(
        bucket=bucket,
        prefix=prefix,
        workflow_id=workflow_id,
        artifact_type=artifact_type,
        timestamp=timestamp,
        filename=filename,
    )


def build_result_path(
    config: StoragePathConfig,
    result_id: str,
    workflow_id: str,
    artifact_type: str,
    timestamp: Optional[datetime] = None,
) -> str:
    """Build an S3 path for a specific result.

    Args:
        config: Storage path configuration.
        result_id: Unique result identifier.
        workflow_id: Workflow identifier.
        artifact_type: Type of artifact.
        timestamp: Optional timestamp. Defaults to current UTC time.

    Returns:
        Full S3 URI for the result.
    """
    filename = f"{result_id}.json.gz"
    return build_s3_path(config, workflow_id, artifact_type, timestamp, filename)
