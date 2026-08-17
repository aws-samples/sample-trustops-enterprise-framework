"""
DynamoDB-based metadata store for TrustOps Enterprise Framework.

Stores result metadata in DynamoDB for fast querying with GSI support
for common query patterns.

Requirements: 9.3
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.data_models.storage import QueryFilter, ResultMetadata

logger = logging.getLogger(__name__)

# DynamoDB table names are scoped to the account, so these are not exposed to
# the name-squatting risk that S3 bucket names are. They remain overridable so
# a deployment whose tables carry an environment suffix can point at them.
TABLE_NAME = os.getenv("TRUSTOPS_RESULTS_TABLE", "trustops-evaluation-results")
ACCESS_LOG_TABLE = os.getenv("TRUSTOPS_ACCESS_LOG_TABLE", "trustops-access-logs")


class DynamoDBMetadataStore:
    """Stores and queries result metadata in DynamoDB."""

    def __init__(self, dynamodb_client: Any, table_name: str = TABLE_NAME):
        """Initialize DynamoDBMetadataStore.

        Args:
            dynamodb_client: boto3 DynamoDB client (resource-level Table).
            table_name: Name of the DynamoDB table.
        """
        self._table_name = table_name
        self._client = dynamodb_client

    def put_metadata(self, metadata: ResultMetadata) -> None:
        """Store result metadata in DynamoDB.

        Args:
            metadata: ResultMetadata to store.
        """
        item = {
            "result_id": metadata.result_id,
            "result_type": metadata.result_type,
            "model_id": metadata.model_id,
            "s3_uri": metadata.s3_uri,
            "checksum": metadata.checksum,
            "size_bytes": metadata.size_bytes,
            "created_at": metadata.created_at.isoformat(),
            "version": metadata.version,
            "tags": metadata.tags,
        }
        if metadata.workflow_id:
            item["workflow_id"] = metadata.workflow_id
        if metadata.dataset_id:
            item["dataset_id"] = metadata.dataset_id

        self._client.put_item(TableName=self._table_name, Item=item)
        logger.info("Stored metadata for result %s (v%d)", metadata.result_id, metadata.version)

    def get_metadata(self, result_id: str) -> Optional[dict]:
        """Get metadata for a result by ID.

        Args:
            result_id: Unique result identifier.

        Returns:
            Metadata dict or None if not found.
        """
        response = self._client.get_item(
            TableName=self._table_name,
            Key={"result_id": result_id},
        )
        return response.get("Item")

    def query_by_result_type(
        self,
        result_type: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query results by type using GSI.

        Args:
            result_type: Type of result to query.
            date_from: Optional start date (ISO format).
            date_to: Optional end date (ISO format).
            limit: Maximum number of results.

        Returns:
            List of metadata dicts.
        """
        params: dict[str, Any] = {
            "TableName": self._table_name,
            "IndexName": "result-type-index",
            "KeyConditionExpression": "result_type = :rt",
            "ExpressionAttributeValues": {":rt": result_type},
            "Limit": limit,
            "ScanIndexForward": False,
        }

        if date_from and date_to:
            params["KeyConditionExpression"] += " AND created_at BETWEEN :df AND :dt"
            params["ExpressionAttributeValues"][":df"] = date_from
            params["ExpressionAttributeValues"][":dt"] = date_to
        elif date_from:
            params["KeyConditionExpression"] += " AND created_at >= :df"
            params["ExpressionAttributeValues"][":df"] = date_from
        elif date_to:
            params["KeyConditionExpression"] += " AND created_at <= :dt"
            params["ExpressionAttributeValues"][":dt"] = date_to

        response = self._client.query(**params)
        return response.get("Items", [])

    def query_by_model_id(
        self,
        model_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Query results by model ID using GSI.

        Args:
            model_id: Model identifier.
            limit: Maximum number of results.

        Returns:
            List of metadata dicts.
        """
        response = self._client.query(
            TableName=self._table_name,
            IndexName="model-id-index",
            KeyConditionExpression="model_id = :mid",
            ExpressionAttributeValues={":mid": model_id},
            Limit=limit,
            ScanIndexForward=False,
        )
        return response.get("Items", [])

    def query_by_workflow_id(
        self,
        workflow_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Query results by workflow ID using GSI.

        Args:
            workflow_id: Workflow identifier.
            limit: Maximum number of results.

        Returns:
            List of metadata dicts.
        """
        response = self._client.query(
            TableName=self._table_name,
            IndexName="workflow-id-index",
            KeyConditionExpression="workflow_id = :wid",
            ExpressionAttributeValues={":wid": workflow_id},
            Limit=limit,
            ScanIndexForward=False,
        )
        return response.get("Items", [])

    def query_with_filter(
        self,
        query_filter: QueryFilter,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Query results using a composite filter.

        Uses the most selective GSI available, then applies remaining
        filters as DynamoDB filter expressions.

        Args:
            query_filter: Filter criteria.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of metadata dicts.
        """
        # Choose the best GSI based on available filters
        if query_filter.workflow_id:
            items = self.query_by_workflow_id(query_filter.workflow_id, limit=limit + offset)
        elif query_filter.model_id:
            items = self.query_by_model_id(query_filter.model_id, limit=limit + offset)
        elif query_filter.result_type:
            date_from = query_filter.date_from.isoformat() if query_filter.date_from else None
            date_to = query_filter.date_to.isoformat() if query_filter.date_to else None
            items = self.query_by_result_type(
                query_filter.result_type, date_from, date_to, limit=limit + offset
            )
        else:
            items = self._scan(limit=limit + offset)

        # Apply remaining filters in-memory
        filtered = self._apply_filters(items, query_filter)

        # Apply pagination
        return filtered[offset: offset + limit]

    def delete_metadata(self, result_id: str) -> None:
        """Delete metadata for a result.

        Args:
            result_id: Unique result identifier.
        """
        self._client.delete_item(
            TableName=self._table_name,
            Key={"result_id": result_id},
        )
        logger.info("Deleted metadata for result %s", result_id)

    def log_access(
        self,
        result_id: str,
        action: str,
        user_id: str,
    ) -> None:
        """Log a data access event for audit compliance.

        Args:
            result_id: ID of the result accessed.
            action: Action performed (e.g., 'read', 'write', 'delete').
            user_id: ID of the user performing the action.
        """
        log_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        self._client.put_item(
            TableName=ACCESS_LOG_TABLE,
            Item={
                "log_id": log_id,
                "result_id": result_id,
                "action": action,
                "user_id": user_id,
                "timestamp": timestamp,
            },
        )
        logger.info(
            "Access log: user=%s action=%s result=%s",
            user_id, action, result_id,
        )

    def _scan(self, limit: int = 100) -> list[dict]:
        """Scan the table (fallback when no GSI is applicable).

        Args:
            limit: Maximum number of results.

        Returns:
            List of metadata dicts.
        """
        response = self._client.scan(
            TableName=self._table_name,
            Limit=limit,
        )
        return response.get("Items", [])

    @staticmethod
    def _apply_filters(items: list[dict], query_filter: QueryFilter) -> list[dict]:
        """Apply in-memory filters to a list of items.

        Args:
            items: List of DynamoDB items.
            query_filter: Filter criteria.

        Returns:
            Filtered list of items.
        """
        result = []
        for item in items:
            if query_filter.result_type and item.get("result_type") != query_filter.result_type:
                continue
            if query_filter.model_id and item.get("model_id") != query_filter.model_id:
                continue
            if query_filter.dataset_id and item.get("dataset_id") != query_filter.dataset_id:
                continue
            if query_filter.workflow_id and item.get("workflow_id") != query_filter.workflow_id:
                continue
            if query_filter.date_from:
                created = item.get("created_at", "")
                if created < query_filter.date_from.isoformat():
                    continue
            if query_filter.date_to:
                created = item.get("created_at", "")
                if created > query_filter.date_to.isoformat():
                    continue
            if query_filter.tags:
                item_tags = item.get("tags", {})
                if not all(item_tags.get(k) == v for k, v in query_filter.tags.items()):
                    continue
            result.append(item)
        return result
