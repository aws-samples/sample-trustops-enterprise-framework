"""Tests for EvaluationResultsStorage.

All AWS interactions (S3, DynamoDB) are mocked so tests run
without real infrastructure.

Requirements: 3.7, 7.8
"""

from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from src.data_models.evaluation import (
    AggregateMetrics,
    BaselineEvaluationReport,
    ComparisonReport,
    CostPerformanceAnalysis,
    CostSummary,
    DeploymentRecommendation,
    EvaluationConfig,
    ImprovementMetrics,
)
from src.data_models.model import InferenceRequest
from src.data_models.storage import (
    ResultMetadata,
    StoragePathConfig,
)
from src.evaluation.results_storage import (
    EvaluationResultsStorage,
)


# ---------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------

def _path_config() -> StoragePathConfig:
    return StoragePathConfig(bucket="test-bucket")


def _aggregate_metrics() -> AggregateMetrics:
    return AggregateMetrics(
        total_examples=10,
        successful_examples=9,
        failed_examples=1,
        mean_trust_score=0.75,
        median_trust_score=0.76,
        trust_score_std=0.05,
        mean_hallucination_rate=0.1,
        latency_p50_ms=100.0,
        latency_p95_ms=200.0,
        latency_p99_ms=300.0,
        total_input_tokens=5000,
        total_output_tokens=3000,
        total_cost=1.50,
        cost_per_query=0.15,
    )


def _cost_summary() -> CostSummary:
    return CostSummary(
        total_cost=1.50,
        inference_cost=1.40,
        storage_cost=0.10,
        cost_per_example=0.15,
    )


def _baseline_report() -> BaselineEvaluationReport:
    return BaselineEvaluationReport(
        evaluation_id="eval-001",
        model_id="model-a",
        dataset_id="ds-001",
        config=EvaluationConfig(
            model_id="model-a",
            dataset_id="ds-001",
            inference_params=InferenceRequest(prompt="hi"),
        ),
        total_examples=10,
        aggregate_metrics=_aggregate_metrics(),
        per_category_metrics={},
        cost_summary=_cost_summary(),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        status="completed",
        s3_results_uri="s3://test-bucket/placeholder",
    )


def _improvement_metrics() -> ImprovementMetrics:
    return ImprovementMetrics(
        trust_score_delta=0.05,
        trust_score_delta_percent=6.67,
        hallucination_reduction=0.1,
        hallucination_reduction_percent=50.0,
        latency_delta_ms=-10.0,
        latency_delta_percent=-5.0,
        cost_delta_per_query=0.01,
        cost_delta_percent=6.67,
        statistical_significance=0.97,
        p_value=0.03,
        confidence_interval=(0.02, 0.08),
    )


def _comparison_report() -> ComparisonReport:
    return ComparisonReport(
        comparison_id="cmp-001",
        model_1_id="model-a",
        model_2_id="model-b",
        dataset_id="ds-001",
        model_1_metrics=_aggregate_metrics(),
        model_2_metrics=_aggregate_metrics(),
        improvement_metrics=_improvement_metrics(),
        recommendation=DeploymentRecommendation.DEPLOY,
        recommendation_justification="All thresholds met.",
        per_category_breakdown={},
        cost_performance_analysis=CostPerformanceAnalysis(
            model_1_cost_per_trust_point=0.2,
            model_2_cost_per_trust_point=0.18,
            quality_gain_justifies_cost=True,
        ),
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        s3_report_uri="s3://test-bucket/placeholder",
    )


def _mock_clients():
    """Return (mock_s3, mock_dynamodb_resource, mock_table)."""
    s3 = MagicMock()
    dynamodb = MagicMock()
    table = MagicMock()
    dynamodb.Table.return_value = table
    return s3, dynamodb, table


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestStoreBaselineReport:
    """Storing a baseline report writes to S3 and DynamoDB."""

    def test_stores_json_in_s3(self):
        s3, dynamodb, table = _mock_clients()
        storage = EvaluationResultsStorage(
            path_config=_path_config(),
            s3_client=s3,
            dynamodb_resource=dynamodb,
        )

        storage.store_baseline_report(
            _baseline_report(), workflow_id="wf-1"
        )

        s3.put_object.assert_called_once()
        call_kwargs = s3.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "test-bucket"
        assert "wf-1" in call_kwargs["Key"]
        assert "model-a" in call_kwargs["Key"]
        assert call_kwargs["ContentType"] == "application/json"

    def test_stores_metadata_in_dynamodb(self):
        s3, dynamodb, table = _mock_clients()
        storage = EvaluationResultsStorage(
            path_config=_path_config(),
            s3_client=s3,
            dynamodb_resource=dynamodb,
        )

        storage.store_baseline_report(
            _baseline_report(), workflow_id="wf-1"
        )

        table.put_item.assert_called_once()
        item = table.put_item.call_args.kwargs["Item"]
        assert item["result_id"] == "eval-001"
        assert item["result_type"] == "baseline_evaluation"
        assert item["model_id"] == "model-a"

    def test_returns_valid_result_metadata(self):
        s3, dynamodb, table = _mock_clients()
        storage = EvaluationResultsStorage(
            path_config=_path_config(),
            s3_client=s3,
            dynamodb_resource=dynamodb,
        )

        metadata = storage.store_baseline_report(
            _baseline_report(), workflow_id="wf-1"
        )

        assert isinstance(metadata, ResultMetadata)
        assert metadata.result_id == "eval-001"
        assert metadata.result_type == "baseline_evaluation"
        assert metadata.s3_uri.startswith("s3://test-bucket/")
        assert metadata.checksum  # non-empty SHA-256
        assert metadata.size_bytes > 0
        assert metadata.version == 1

    def test_s3_key_follows_path_convention(self):
        s3, dynamodb, table = _mock_clients()
        storage = EvaluationResultsStorage(
            path_config=_path_config(),
            s3_client=s3,
            dynamodb_resource=dynamodb,
        )

        storage.store_baseline_report(
            _baseline_report(), workflow_id="wf-1"
        )

        key = s3.put_object.call_args.kwargs["Key"]
        parts = key.split("/")
        assert parts[0] == "evaluations"
        assert parts[1] == "wf-1"
        assert parts[2] == "model-a"
        # parts[3] is timestamp, parts[4] is filename
        assert parts[4] == "baseline_report.json"


class TestStoreComparisonReport:
    """Storing a comparison report writes to S3 and DynamoDB."""

    def test_stores_comparison_in_s3(self):
        s3, dynamodb, table = _mock_clients()
        storage = EvaluationResultsStorage(
            path_config=_path_config(),
            s3_client=s3,
            dynamodb_resource=dynamodb,
        )

        storage.store_comparison_report(
            _comparison_report(), workflow_id="wf-2"
        )

        s3.put_object.assert_called_once()
        key = s3.put_object.call_args.kwargs["Key"]
        assert "model-a_vs_model-b" in key
        assert "comparison_report.json" in key

    def test_comparison_metadata_type(self):
        s3, dynamodb, table = _mock_clients()
        storage = EvaluationResultsStorage(
            path_config=_path_config(),
            s3_client=s3,
            dynamodb_resource=dynamodb,
        )

        metadata = storage.store_comparison_report(
            _comparison_report(), workflow_id="wf-2"
        )

        assert metadata.result_type == "comparison"
        assert metadata.result_id == "cmp-001"
        assert metadata.dataset_id == "ds-001"


class TestGetResult:
    """Retrieving a stored result by evaluation_id."""

    def test_returns_deserialized_report_and_metadata(self):
        s3, dynamodb, table = _mock_clients()
        storage = EvaluationResultsStorage(
            path_config=_path_config(),
            s3_client=s3,
            dynamodb_resource=dynamodb,
        )

        report = _baseline_report()
        body = report.model_dump_json(indent=2)

        # Mock DynamoDB get_item
        table.get_item.return_value = {
            "Item": {
                "result_id": "eval-001",
                "result_type": "baseline_evaluation",
                "model_id": "model-a",
                "dataset_id": "ds-001",
                "s3_uri": "s3://test-bucket/evaluations/k/v",
                "checksum": "abc123",
                "size_bytes": len(body.encode()),
                "created_at": "2024-01-01T00:00:00Z",
                "version": 1,
                "workflow_id": "wf-1",
            }
        }

        # Mock S3 get_object
        s3.get_object.return_value = {
            "Body": BytesIO(body.encode("utf-8")),
        }

        data, meta = storage.get_result("eval-001")

        assert isinstance(data, dict)
        assert data["evaluation_id"] == "eval-001"
        assert isinstance(meta, ResultMetadata)
        assert meta.result_id == "eval-001"

    def test_raises_key_error_when_not_found(self):
        s3, dynamodb, table = _mock_clients()
        storage = EvaluationResultsStorage(
            path_config=_path_config(),
            s3_client=s3,
            dynamodb_resource=dynamodb,
        )

        table.get_item.return_value = {}

        with pytest.raises(KeyError, match="eval-999"):
            storage.get_result("eval-999")


class TestDefaultWorkflowId:
    """When no workflow_id is provided, 'default' is used."""

    def test_default_workflow_id_in_path(self):
        s3, dynamodb, table = _mock_clients()
        storage = EvaluationResultsStorage(
            path_config=_path_config(),
            s3_client=s3,
            dynamodb_resource=dynamodb,
        )

        storage.store_baseline_report(_baseline_report())

        key = s3.put_object.call_args.kwargs["Key"]
        assert key.startswith("evaluations/default/")
