"""
Data retention policy manager for TrustOps Enterprise Framework.

Supports configurable archival to S3 Glacier after N days,
deletion after N days, and exclusion of tagged items.

Requirements: 9.7
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from src.data_models.storage import RetentionPolicy

logger = logging.getLogger(__name__)


class RetentionAction:
    """Represents an action to take on a result based on retention policy."""

    ARCHIVE = "archive"
    DELETE = "delete"
    KEEP = "keep"

    def __init__(self, action: str, result_id: str, reason: str):
        self.action = action
        self.result_id = result_id
        self.reason = reason


class RetentionManager:
    """Manages data retention policies for stored results."""

    def __init__(self, s3_store: Any, metadata_store: Any):
        """Initialize RetentionManager.

        Args:
            s3_store: S3ResultsStore instance.
            metadata_store: DynamoDBMetadataStore instance.
        """
        self._s3_store = s3_store
        self._metadata_store = metadata_store

    def evaluate_item(
        self,
        item: dict,
        policy: RetentionPolicy,
        now: datetime | None = None,
    ) -> RetentionAction:
        """Evaluate what retention action to take on an item.

        Args:
            item: Metadata dict with created_at and tags.
            policy: Retention policy to apply.
            now: Current time (defaults to UTC now).

        Returns:
            RetentionAction indicating what to do.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        result_id = item.get("result_id", "unknown")
        tags = item.get("tags", {})

        # Check if item is excluded from retention
        for tag_key in tags:
            if tag_key in policy.exclude_tags:
                return RetentionAction(
                    RetentionAction.KEEP,
                    result_id,
                    f"Excluded by tag: {tag_key}",
                )

        # Check tag values too
        for tag_value in tags.values():
            if tag_value in policy.exclude_tags:
                return RetentionAction(
                    RetentionAction.KEEP,
                    result_id,
                    f"Excluded by tag value: {tag_value}",
                )

        # Calculate age
        created_at_str = item.get("created_at", "")
        try:
            if isinstance(created_at_str, datetime):
                created_at = created_at_str
            else:
                created_at = datetime.fromisoformat(created_at_str)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return RetentionAction(
                RetentionAction.KEEP,
                result_id,
                "Cannot parse created_at date",
            )

        age_days = (now - created_at).days

        if age_days >= policy.delete_after_days:
            return RetentionAction(
                RetentionAction.DELETE,
                result_id,
                f"Age {age_days} days >= delete threshold {policy.delete_after_days}",
            )
        elif age_days >= policy.archive_after_days:
            return RetentionAction(
                RetentionAction.ARCHIVE,
                result_id,
                f"Age {age_days} days >= archive threshold {policy.archive_after_days}",
            )
        else:
            return RetentionAction(
                RetentionAction.KEEP,
                result_id,
                f"Age {age_days} days < archive threshold {policy.archive_after_days}",
            )

    def apply_policy(
        self,
        items: list[dict],
        policy: RetentionPolicy,
    ) -> dict[str, int]:
        """Apply retention policy to a list of items.

        Args:
            items: List of metadata dicts.
            policy: Retention policy to apply.

        Returns:
            Dict with counts: archived, deleted, kept.
        """
        counts = {"archived": 0, "deleted": 0, "kept": 0}

        for item in items:
            action = self.evaluate_item(item, policy)

            if action.action == RetentionAction.ARCHIVE:
                s3_uri = item.get("s3_uri", "")
                if s3_uri:
                    try:
                        self._s3_store.change_storage_class(s3_uri, "GLACIER")
                        counts["archived"] += 1
                        logger.info("Archived %s: %s", action.result_id, action.reason)
                    except Exception as e:
                        logger.error("Failed to archive %s: %s", action.result_id, e)
                        counts["kept"] += 1
                else:
                    counts["kept"] += 1

            elif action.action == RetentionAction.DELETE:
                s3_uri = item.get("s3_uri", "")
                result_id = item.get("result_id", "")
                try:
                    if s3_uri:
                        self._s3_store.delete(s3_uri)
                    if result_id:
                        self._metadata_store.delete_metadata(result_id)
                    counts["deleted"] += 1
                    logger.info("Deleted %s: %s", action.result_id, action.reason)
                except Exception as e:
                    logger.error("Failed to delete %s: %s", action.result_id, e)
                    counts["kept"] += 1

            else:
                counts["kept"] += 1

        return counts
