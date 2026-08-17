"""
Results query API for TrustOps Enterprise Framework.

Supports filtering by date range, model, evaluation type, workflow ID, tags,
and pagination with limit and offset.

Requirements: 9.4
"""

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.data_models.storage import QueryFilter
from src.storage.dynamodb_metadata_store import DynamoDBMetadataStore

logger = logging.getLogger(__name__)


class PaginatedResults(BaseModel):
    """Paginated query results."""

    items: list[dict] = Field(default_factory=list)
    total_count: int = 0
    limit: int = 100
    offset: int = 0
    has_more: bool = False


class ResultsQueryAPI:
    """API for querying stored results with filtering and pagination."""

    def __init__(self, metadata_store: DynamoDBMetadataStore):
        """Initialize ResultsQueryAPI.

        Args:
            metadata_store: DynamoDB metadata store instance.
        """
        self._store = metadata_store

    def query(
        self,
        query_filter: Optional[QueryFilter] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> PaginatedResults:
        """Query results with filtering and pagination.

        Args:
            query_filter: Optional filter criteria.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            PaginatedResults with items and pagination info.
        """
        if query_filter is None:
            query_filter = QueryFilter()

        if limit < 1:
            limit = 1
        if limit > 1000:
            limit = 1000
        if offset < 0:
            offset = 0

        # Fetch enough items to determine total and apply pagination
        all_items = self._store.query_with_filter(
            query_filter, limit=limit + offset + 1, offset=0
        )

        total_count = len(all_items)
        paginated = all_items[offset: offset + limit]
        has_more = total_count > offset + limit

        logger.info(
            "Query returned %d items (offset=%d, limit=%d, has_more=%s)",
            len(paginated), offset, limit, has_more,
        )

        return PaginatedResults(
            items=paginated,
            total_count=total_count,
            limit=limit,
            offset=offset,
            has_more=has_more,
        )

    def get_by_id(self, result_id: str) -> Optional[dict]:
        """Get a single result by ID.

        Args:
            result_id: Unique result identifier.

        Returns:
            Metadata dict or None if not found.
        """
        return self._store.get_metadata(result_id)

    def query_by_model(
        self,
        model_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> PaginatedResults:
        """Query results for a specific model.

        Args:
            model_id: Model identifier.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            PaginatedResults.
        """
        qf = QueryFilter(model_id=model_id)
        return self.query(qf, limit=limit, offset=offset)

    def query_by_workflow(
        self,
        workflow_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> PaginatedResults:
        """Query results for a specific workflow.

        Args:
            workflow_id: Workflow identifier.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            PaginatedResults.
        """
        qf = QueryFilter(workflow_id=workflow_id)
        return self.query(qf, limit=limit, offset=offset)

    def query_by_type(
        self,
        result_type: str,
        limit: int = 100,
        offset: int = 0,
    ) -> PaginatedResults:
        """Query results of a specific type.

        Args:
            result_type: Type of result.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            PaginatedResults.
        """
        qf = QueryFilter(result_type=result_type)
        return self.query(qf, limit=limit, offset=offset)
