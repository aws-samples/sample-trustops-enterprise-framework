"""
S3-based results store for TrustOps Enterprise Framework.

Stores evaluation results in S3 with versioning enabled and
compressed JSON format (gzip) to minimize storage costs.

Requirements: 9.1, 9.6
"""

import gzip
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.data_models.storage import StoragePathConfig
from src.orchestration.checksum_calculator import calculate_checksum
from src.storage.path_convention import build_s3_key

logger = logging.getLogger(__name__)


class S3ResultsStore:
    """Stores and retrieves results from S3 with gzip compression.

    Results are stored as compressed JSON (gzip) with S3 versioning enabled.
    """

    def __init__(self, s3_client: Any, config: StoragePathConfig):
        """Initialize S3ResultsStore.

        Args:
            s3_client: boto3 S3 client.
            config: Storage path configuration.
        """
        self._s3 = s3_client
        self._config = config

    def store(
        self,
        data: Any,
        workflow_id: str,
        artifact_type: str,
        result_id: str,
        timestamp: Optional[datetime] = None,
    ) -> tuple[str, str, int]:
        """Store data in S3 as compressed JSON.

        Args:
            data: JSON-serializable data to store.
            workflow_id: Workflow identifier.
            artifact_type: Type of artifact.
            result_id: Unique result identifier.
            timestamp: Optional timestamp. Defaults to current UTC time.

        Returns:
            Tuple of (s3_uri, checksum, size_bytes).
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        filename = f"{result_id}.json.gz"
        key = build_s3_key(
            self._config, workflow_id, artifact_type, timestamp, filename
        )

        serialized = json.dumps(data, sort_keys=True, default=str)
        compressed = gzip.compress(serialized.encode("utf-8"))
        checksum = calculate_checksum(compressed)

        self._s3.put_object(
            Bucket=self._config.bucket,
            Key=key,
            Body=compressed,
            ContentType="application/gzip",
            Metadata={
                "checksum": checksum,
                "result_id": result_id,
                "artifact_type": artifact_type,
                "workflow_id": workflow_id,
            },
        )

        s3_uri = f"s3://{self._config.bucket}/{key}"
        logger.info("Stored result %s at %s (%d bytes)", result_id, s3_uri, len(compressed))
        return s3_uri, checksum, len(compressed)

    def get(self, s3_uri: str) -> tuple[Any, str]:
        """Retrieve data from S3 and decompress.

        Args:
            s3_uri: Full S3 URI of the stored result.

        Returns:
            Tuple of (deserialized data, checksum).
        """
        bucket, key = self._parse_s3_uri(s3_uri)

        response = self._s3.get_object(Bucket=bucket, Key=key)
        compressed = response["Body"].read()
        checksum = calculate_checksum(compressed)

        decompressed = gzip.decompress(compressed)
        data = json.loads(decompressed.decode("utf-8"))

        logger.info("Retrieved result from %s", s3_uri)
        return data, checksum

    def get_version(self, s3_uri: str, version_id: str) -> tuple[Any, str]:
        """Retrieve a specific version of data from S3.

        Args:
            s3_uri: Full S3 URI of the stored result.
            version_id: S3 version ID.

        Returns:
            Tuple of (deserialized data, checksum).
        """
        bucket, key = self._parse_s3_uri(s3_uri)

        response = self._s3.get_object(
            Bucket=bucket, Key=key, VersionId=version_id
        )
        compressed = response["Body"].read()
        checksum = calculate_checksum(compressed)

        decompressed = gzip.decompress(compressed)
        data = json.loads(decompressed.decode("utf-8"))

        return data, checksum

    def list_versions(self, s3_uri: str) -> list[dict]:
        """List all versions of an object in S3.

        Args:
            s3_uri: Full S3 URI.

        Returns:
            List of version info dicts with version_id, last_modified, size.
        """
        bucket, key = self._parse_s3_uri(s3_uri)

        response = self._s3.list_object_versions(Bucket=bucket, Prefix=key)
        versions = []
        for v in response.get("Versions", []):
            versions.append({
                "version_id": v["VersionId"],
                "last_modified": v["LastModified"],
                "size": v["Size"],
                "is_latest": v["IsLatest"],
            })
        return versions

    def delete(self, s3_uri: str) -> None:
        """Delete an object from S3.

        Args:
            s3_uri: Full S3 URI.
        """
        bucket, key = self._parse_s3_uri(s3_uri)
        self._s3.delete_object(Bucket=bucket, Key=key)
        logger.info("Deleted %s", s3_uri)

    def change_storage_class(self, s3_uri: str, storage_class: str) -> None:
        """Change the storage class of an S3 object (e.g., for archival).

        Args:
            s3_uri: Full S3 URI.
            storage_class: Target storage class (e.g., 'GLACIER').
        """
        bucket, key = self._parse_s3_uri(s3_uri)
        self._s3.copy_object(
            Bucket=bucket,
            Key=key,
            CopySource={"Bucket": bucket, "Key": key},
            StorageClass=storage_class,
        )
        logger.info("Changed storage class of %s to %s", s3_uri, storage_class)

    @staticmethod
    def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        """Parse an S3 URI into bucket and key.

        Args:
            s3_uri: Full S3 URI.

        Returns:
            Tuple of (bucket, key).
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        path = s3_uri[5:]
        parts = path.split("/", 1)
        if len(parts) < 2:
            raise ValueError(f"Invalid S3 URI, missing key: {s3_uri}")
        return parts[0], parts[1]
