"""
Evaluation results storage for the TrustOps Enterprise Framework.

Stores evaluation results (baseline reports and comparison reports) in S3
with metadata in DynamoDB for fast querying.

S3 path convention:
    s3://{bucket}/evaluations/{workflow_id}/{model_id}/{timestamp}/

Requirements: 3.7, 7.8
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import boto3
from pydantic import BaseModel

from src.data_models.evaluation import (
    BaselineEvaluationReport,
    ComparisonReport,
)
from src.data_models.storage import (
    ResultMetadata,
    StoragePathConfig,
)


class EvaluationResultsStorage:
    """Stores and retrieves evaluation results using S3 and DynamoDB.

    S3 is used for the full JSON payloads; DynamoDB holds lightweight
    metadata records that enable fast querying by evaluation_id,
    model_id, or workflow_id.

    Requirements: 3.7, 7.8
    """

    _BASELINE_FILENAME = "baseline_report.json"
    _COMPARISON_FILENAME = "comparison_report.json"
    _DYNAMODB_TABLE = "trustops_evaluation_results"

    def __init__(
        self,
        path_config: StoragePathConfig,
        s3_client: Any | None = None,
        dynamodb_resource: Any | None = None,
    ) -> None:
        self._path_config = path_config
        self._s3 = s3_client or boto3.client("s3")
        self._dynamodb = dynamodb_resource or boto3.resource("dynamodb")
        self._table = self._dynamodb.Table(self._DYNAMODB_TABLE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_baseline_report(
        self,
        report: BaselineEvaluationReport,
        workflow_id: str = "default",
    ) -> ResultMetadata:
        """Persist a baseline evaluation report to S3
        and record metadata in DynamoDB.

        Args:
            report: The baseline evaluation report to store.
            workflow_id: Workflow that produced this evaluation.

        Returns:
            ResultMetadata describing the stored artifact.
        """
        body = self._serialize(report)
        timestamp = self._iso_now()
        s3_key = self._build_s3_key(
            workflow_id=workflow_id,
            model_id=report.model_id,
            timestamp=timestamp,
            filename=self._BASELINE_FILENAME,
        )
        s3_uri = f"s3://{self._path_config.bucket}/{s3_key}"

        self._put_s3_object(s3_key, body)

        metadata = ResultMetadata(
            result_id=report.evaluation_id,
            result_type="baseline_evaluation",
            workflow_id=workflow_id,
            model_id=report.model_id,
            dataset_id=report.dataset_id,
            s3_uri=s3_uri,
            checksum=self._sha256(body),
            size_bytes=len(body.encode("utf-8")),
            created_at=datetime.now(timezone.utc),
            version=1,
            tags={},
        )
        self._put_dynamodb_metadata(metadata)
        return metadata

    def store_comparison_report(
        self,
        report: ComparisonReport,
        workflow_id: str = "default",
    ) -> ResultMetadata:
        """Persist a comparison report to S3 and record metadata in DynamoDB.

        Args:
            report: The comparison report to store.
            workflow_id: Workflow that produced this comparison.

        Returns:
            ResultMetadata describing the stored artifact.
        """
        body = self._serialize(report)
        timestamp = self._iso_now()
        model_id = f"{report.model_1_id}_vs_{report.model_2_id}"
        s3_key = self._build_s3_key(
            workflow_id=workflow_id,
            model_id=model_id,
            timestamp=timestamp,
            filename=self._COMPARISON_FILENAME,
        )
        s3_uri = f"s3://{self._path_config.bucket}/{s3_key}"

        self._put_s3_object(s3_key, body)

        metadata = ResultMetadata(
            result_id=report.comparison_id,
            result_type="comparison",
            workflow_id=workflow_id,
            model_id=model_id,
            dataset_id=report.dataset_id,
            s3_uri=s3_uri,
            checksum=self._sha256(body),
            size_bytes=len(body.encode("utf-8")),
            created_at=datetime.now(timezone.utc),
            version=1,
            tags={},
        )
        self._put_dynamodb_metadata(metadata)
        return metadata

    def get_result(self, evaluation_id: str) -> tuple[dict, ResultMetadata]:
        """Retrieve a stored result by its evaluation/comparison ID.

        Args:
            evaluation_id: The unique ID of the evaluation or comparison.

        Returns:
            A tuple of (deserialized report dict, metadata).

        Raises:
            KeyError: If no result with the given ID exists.
        """
        metadata = self._get_dynamodb_metadata(evaluation_id)
        body = self._get_s3_object(metadata.s3_uri)
        return json.loads(body), metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_s3_key(
        self,
        workflow_id: str,
        model_id: str,
        timestamp: str,
        filename: str,
    ) -> str:
        """Build a consistent S3 key following the path convention.

        Pattern: evaluations/{workflow_id}/{model_id}/{timestamp}/{filename}
        """
        return f"evaluations/{workflow_id}/{model_id}/{timestamp}/{filename}"

    @staticmethod
    def _serialize(model: BaseModel) -> str:
        """Serialize a Pydantic model to a JSON string."""
        return model.model_dump_json(indent=2)

    @staticmethod
    def _sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _put_s3_object(self, key: str, body: str) -> None:
        self._s3.put_object(
            Bucket=self._path_config.bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )

    def _get_s3_object(self, s3_uri: str) -> str:
        """Download an object from S3 given its full s3:// URI."""
        # Parse s3://bucket/key
        without_scheme = s3_uri[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        response = self._s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    def _put_dynamodb_metadata(self, metadata: ResultMetadata) -> None:
        item = json.loads(metadata.model_dump_json())
        # DynamoDB requires string-typed datetime values
        self._table.put_item(Item=item)

    def _get_dynamodb_metadata(self, result_id: str) -> ResultMetadata:
        response = self._table.get_item(Key={"result_id": result_id})
        item = response.get("Item")
        if not item:
            raise KeyError(f"No result found with id: {result_id}")
        return ResultMetadata(**item)
