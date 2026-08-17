"""
Version preservation for stored results.

Preserves previous versions on modification with monotonically
increasing version numbers.

Requirements: 9.5
"""

import logging
from typing import Any, Optional

from src.data_models.storage import ResultMetadata

logger = logging.getLogger(__name__)


class VersionManager:
    """Manages version history for stored results."""

    def __init__(self, metadata_store: Any):
        """Initialize VersionManager.

        Args:
            metadata_store: DynamoDBMetadataStore instance.
        """
        self._store = metadata_store
        # In-memory version tracking (keyed by result_id)
        self._versions: dict[str, list[ResultMetadata]] = {}

    def get_next_version(self, result_id: str) -> int:
        """Get the next version number for a result.

        Args:
            result_id: Result identifier.

        Returns:
            Next monotonically increasing version number.
        """
        if result_id in self._versions and self._versions[result_id]:
            current_max = max(v.version for v in self._versions[result_id])
            return current_max + 1
        return 1

    def record_version(self, metadata: ResultMetadata) -> None:
        """Record a new version of a result.

        Args:
            metadata: ResultMetadata for the new version.
        """
        if metadata.result_id not in self._versions:
            self._versions[metadata.result_id] = []
        self._versions[metadata.result_id].append(metadata)
        logger.info(
            "Recorded version %d for result %s",
            metadata.version, metadata.result_id,
        )

    def get_version_history(self, result_id: str) -> list[ResultMetadata]:
        """Get all versions of a result.

        Args:
            result_id: Result identifier.

        Returns:
            List of ResultMetadata sorted by version (ascending).
        """
        versions = self._versions.get(result_id, [])
        return sorted(versions, key=lambda v: v.version)

    def get_specific_version(
        self, result_id: str, version: int
    ) -> Optional[ResultMetadata]:
        """Get a specific version of a result.

        Args:
            result_id: Result identifier.
            version: Version number to retrieve.

        Returns:
            ResultMetadata for the version, or None if not found.
        """
        versions = self._versions.get(result_id, [])
        for v in versions:
            if v.version == version:
                return v
        return None

    def get_latest_version(self, result_id: str) -> Optional[ResultMetadata]:
        """Get the latest version of a result.

        Args:
            result_id: Result identifier.

        Returns:
            Latest ResultMetadata, or None if no versions exist.
        """
        versions = self._versions.get(result_id, [])
        if not versions:
            return None
        return max(versions, key=lambda v: v.version)
