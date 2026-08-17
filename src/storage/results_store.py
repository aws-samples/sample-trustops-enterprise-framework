"""
Unified ResultsStore for TrustOps Enterprise Framework.

Integrates S3 and DynamoDB stores to provide a complete results
storage solution with versioning, checksums, comparison, export,
and access logging.

Requirements: 9.1-9.10
"""

import gzip
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.data_models.storage import (
    QueryFilter,
    ResultMetadata,
    RetentionPolicy,
    StoragePathConfig,
)
from src.storage.access_logger import AccessLogger
from src.storage.checksum_verifier import verify_data_checksum
from src.storage.comparison_api import ComparisonResult, compare_results
from src.storage.dynamodb_metadata_store import DynamoDBMetadataStore
from src.storage.export_api import export_result
from src.storage.query_api import PaginatedResults, ResultsQueryAPI
from src.storage.retention_manager import RetentionManager
from src.storage.s3_results_store import S3ResultsStore
from src.storage.version_manager import VersionManager

logger = logging.getLogger(__name__)


class ResultsStore:
    """Manages versioned storage of evaluation results.

    Integrates S3 for data storage and DynamoDB for metadata,
    with support for versioning, checksums, comparison, export,
    and audit logging.
    """

    def __init__(
        self,
        s3_client: Any,
        dynamodb_client: Any,
        path_config: StoragePathConfig,
    ):
        """Initialize ResultsStore.

        Args:
            s3_client: boto3 S3 client.
            dynamodb_client: boto3 DynamoDB client.
            path_config: Storage path configuration.
        """
        self._config = path_config
        self._s3_store = S3ResultsStore(s3_client, path_config)
        self._metadata_store = DynamoDBMetadataStore(dynamodb_client)
        self._query_api = ResultsQueryAPI(self._metadata_store)
        self._version_manager = VersionManager(self._metadata_store)
        self._retention_manager = RetentionManager(self._s3_store, self._metadata_store)
        self._access_logger = AccessLogger(self._metadata_store)

    def store_result(
        self,
        result: Any,
        result_type: str,
        model_id: str,
        workflow_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        user_id: str = "system",
    ) -> ResultMetadata:
        """Store a result with automatic versioning.

        Args:
            result: JSON-serializable result data.
            result_type: Type of result.
            model_id: Model identifier.
            workflow_id: Optional workflow identifier.
            dataset_id: Optional dataset identifier.
            tags: Optional tags for categorization.
            user_id: User performing the store operation.

        Returns:
            ResultMetadata for the stored result.
        """
        result_id = str(uuid.uuid4())
        version = self._version_manager.get_next_version(result_id)
        wf_id = workflow_id or "standalone"
        timestamp = datetime.now(timezone.utc)

        # Store in S3
        s3_uri, checksum, size_bytes = self._s3_store.store(
            result, wf_id, result_type, result_id, timestamp
        )

        # Create metadata
        metadata = ResultMetadata(
            result_id=result_id,
            result_type=result_type,
            workflow_id=workflow_id,
            model_id=model_id,
            dataset_id=dataset_id,
            s3_uri=s3_uri,
            checksum=checksum,
            size_bytes=size_bytes,
            created_at=timestamp,
            version=version,
            tags=tags or {},
        )

        # Store metadata in DynamoDB
        self._metadata_store.put_metadata(metadata)

        # Record version
        self._version_manager.record_version(metadata)

        # Log access
        self._access_logger.log(result_id, "write", user_id)

        logger.info("Stored result %s (v%d) at %s", result_id, version, s3_uri)
        return metadata

    def get_result(
        self,
        result_id: str,
        version: Optional[int] = None,
        user_id: str = "system",
    ) -> tuple[Any, ResultMetadata]:
        """Retrieve a result by ID, optionally a specific version.

        Args:
            result_id: Unique result identifier.
            version: Optional version number. Defaults to latest.
            user_id: User performing the retrieval.

        Returns:
            Tuple of (result data, ResultMetadata).

        Raises:
            ValueError: If result not found.
        """
        if version is not None:
            metadata = self._version_manager.get_specific_version(result_id, version)
        else:
            metadata = self._version_manager.get_latest_version(result_id)

        if metadata is None:
            raise ValueError(f"Result not found: {result_id} (version={version})")

        data, _ = self._s3_store.get(metadata.s3_uri)

        # Log access
        self._access_logger.log(result_id, "read", user_id)

        return data, metadata

    def query_results(
        self,
        query_filter: Optional[QueryFilter] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> PaginatedResults:
        """Query results with filtering and pagination.

        Args:
            query_filter: Optional filter criteria.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            PaginatedResults.
        """
        return self._query_api.query(query_filter, limit, offset)

    def compare_results(
        self,
        result_id_1: str,
        result_id_2: str,
        user_id: str = "system",
    ) -> ComparisonResult:
        """Compare two results side-by-side.

        Args:
            result_id_1: First result ID.
            result_id_2: Second result ID.
            user_id: User performing the comparison.

        Returns:
            ComparisonResult with metrics diff.
        """
        data_1, _ = self.get_result(result_id_1, user_id=user_id)
        data_2, _ = self.get_result(result_id_2, user_id=user_id)

        return compare_results(data_1, data_2, result_id_1, result_id_2)

    def export_result(
        self,
        result_id: str,
        format: str = "json",
        user_id: str = "system",
    ) -> bytes:
        """Export a result in the specified format.

        Args:
            result_id: Result identifier.
            format: Export format ('json', 'csv', 'pdf').
            user_id: User performing the export.

        Returns:
            Exported data as bytes.
        """
        data, _ = self.get_result(result_id, user_id=user_id)

        # Log export access
        self._access_logger.log(
            result_id, "export", user_id, {"format": format}
        )

        return export_result(data, format)

    def verify_checksum(
        self,
        result_id: str,
        user_id: str = "system",
    ) -> bool:
        """Verify data integrity using stored checksum.

        Args:
            result_id: Result identifier.
            user_id: User performing the verification.

        Returns:
            True if checksum matches, False otherwise.
        """
        metadata = self._version_manager.get_latest_version(result_id)
        if metadata is None:
            raise ValueError(f"Result not found: {result_id}")

        # Get raw compressed data from S3
        bucket, key = S3ResultsStore._parse_s3_uri(metadata.s3_uri)
        response = self._s3_store._s3.get_object(Bucket=bucket, Key=key)
        raw_data = response["Body"].read()

        verification = verify_data_checksum(
            raw_data, metadata.checksum, result_id
        )

        self._access_logger.log(
            result_id, "verify", user_id,
            {"is_valid": verification.is_valid},
        )

        return verification.is_valid

    def get_version_history(
        self,
        result_id: str,
    ) -> list[ResultMetadata]:
        """Get version history for a result.

        Args:
            result_id: Result identifier.

        Returns:
            List of ResultMetadata sorted by version.
        """
        return self._version_manager.get_version_history(result_id)

    def apply_retention_policy(
        self,
        policy: RetentionPolicy,
    ) -> int:
        """Apply retention policy, return count of archived/deleted items.

        Args:
            policy: Retention policy to apply.

        Returns:
            Total count of items archived or deleted.
        """
        # Get all items
        all_items = self._metadata_store._scan(limit=10000)
        counts = self._retention_manager.apply_policy(all_items, policy)

        total_affected = counts["archived"] + counts["deleted"]
        logger.info(
            "Retention policy applied: archived=%d, deleted=%d, kept=%d",
            counts["archived"], counts["deleted"], counts["kept"],
        )
        return total_affected

    def log_access(
        self,
        result_id: str,
        action: str,
        user_id: str,
    ) -> None:
        """Log data access for audit compliance.

        Args:
            result_id: Result identifier.
            action: Action performed.
            user_id: User performing the action.
        """
        self._access_logger.log(result_id, action, user_id)
