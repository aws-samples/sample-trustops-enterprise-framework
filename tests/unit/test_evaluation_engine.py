"""
Unit tests for the EvaluationEngine facade class.

All external dependencies (InferenceClient, TrustScoringEngine,
HallucinationDetector) are mocked so tests run without AWS services.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.inference_client import InferenceClient
from src.data_models.evaluation import (
    ComparisonConfig,
    ComparisonReport,
    DeploymentThresholds,
    EvaluationConfig,
)
from src.data_models.hallucination import (
    HallucinationResult as HallResult,
    SensitivityLevel,
)
from src.data_models.model import (
    InferenceRequest,
    InferenceResponse,
    ModelPricing,
)
from src.data_models.trust_score import TrustScoreResult
from src.evaluation.evaluation_engine import EvaluationEngine
from src.hallucination.hallucination_detector import HallucinationDetector
from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_inference_response(
    text: str = "response",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> InferenceResponse:
    return InferenceResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=50.0,
        model_id="test-model",
        finish_reason="stop",
    )


def _make_trust_score(score: float = 0.8) -> TrustScoreResult:
    return TrustScoreResult(
        overall_score=score,
        dimension_scores={},
        confidence_level=0.9,
        flagged_for_review=False,
        explanation="ok",
    )


def _make_hall_result(rate: float = 0.1) -> HallResult:
    return HallResult(
        has_hallucinations=rate > 0,
        hallucination_rate=rate,
        total_claims=10,
        factual_claims=10,
        supported_claims=int(10 * (1 - rate)),
        unsupported_claims=int(10 * rate),
        flagged_spans=[],
        overall_grounding_score=1.0 - rate,
        claim_evidence=[],
        sensitivity_level=SensitivityLevel.MODERATE,
    )


@pytest.fixture
def mock_client():
    client = MagicMock(spec=InferenceClient)
    client.invoke = AsyncMock(return_value=_make_inference_response())
    return client


@pytest.fixture
def mock_trust_scorer():
    scorer = MagicMock(spec=TrustScoringEngine)
    scorer.score_response = AsyncMock(
        return_value=_make_trust_score(0.8)
    )
    return scorer


@pytest.fixture
def mock_hall_detector():
    detector = MagicMock(spec=HallucinationDetector)
    detector.detect = MagicMock(
        return_value=_make_hall_result(0.1)
    )
    return detector


@pytest.fixture
def engine(mock_client, mock_trust_scorer, mock_hall_detector):
    return EvaluationEngine(
        inference_client=mock_client,
        trust_scorer=mock_trust_scorer,
        hallucination_detector=mock_hall_detector,
    )


@pytest.fixture
def sample_items():
    return [
        {"prompt": "What is AI?", "expected_response": "AI is...", "category": "tech"},
        {"prompt": "What is ML?", "expected_response": "ML is...", "category": "tech"},
        {"prompt": "What is NLP?", "expected_response": "NLP is...", "category": "nlp"},
    ]


@pytest.fixture
def eval_config():
    return EvaluationConfig(
        model_id="test-model",
        dataset_id="ds-001",
        inference_params=InferenceRequest(prompt="placeholder"),
    )


@pytest.fixture
def comparison_config():
    return ComparisonConfig(
        model_id_1="model-a",
        model_id_2="model-b",
        dataset_id="ds-001",
        inference_params=InferenceRequest(prompt="placeholder"),
        deploy_thresholds=DeploymentThresholds(),
    )


# ---------------------------------------------------------------------------
# Baseline evaluation tests
# ---------------------------------------------------------------------------

class TestBaselineEvaluation:
    @pytest.mark.asyncio
    async def test_baseline_returns_report(
        self, engine, eval_config, sample_items
    ):
        report = await engine.run_baseline_evaluation(
            config=eval_config,
            dataset_items=sample_items,
        )
        assert report.model_id == "test-model"
        assert report.dataset_id == "ds-001"
        assert report.status == "completed"
        assert report.total_examples == len(sample_items)

    @pytest.mark.asyncio
    async def test_baseline_empty_dataset(self, engine, eval_config):
        report = await engine.run_baseline_evaluation(
            config=eval_config,
            dataset_items=[],
        )
        assert report.total_examples == 0

    @pytest.mark.asyncio
    async def test_baseline_with_pricing(
        self, engine, eval_config, sample_items
    ):
        pricing = ModelPricing(
            input_price_per_1k_tokens=0.003,
            output_price_per_1k_tokens=0.015,
        )
        report = await engine.run_baseline_evaluation(
            config=eval_config,
            dataset_items=sample_items,
            pricing=pricing,
        )
        assert report.cost_summary.total_cost >= 0

    @pytest.mark.asyncio
    async def test_baseline_progress_callback(
        self, engine, eval_config, sample_items
    ):
        progress_calls = []

        def on_progress(completed, total):
            progress_calls.append((completed, total))

        await engine.run_baseline_evaluation(
            config=eval_config,
            dataset_items=sample_items,
            progress_callback=on_progress,
        )
        assert len(progress_calls) == len(sample_items)

    @pytest.mark.asyncio
    async def test_baseline_per_category_metrics(
        self, engine, eval_config, sample_items
    ):
        report = await engine.run_baseline_evaluation(
            config=eval_config,
            dataset_items=sample_items,
        )
        # sample_items have "tech" and "nlp" categories
        assert len(report.per_category_metrics) >= 1


# ---------------------------------------------------------------------------
# Comparative evaluation tests
# ---------------------------------------------------------------------------

class TestComparativeEvaluation:
    @pytest.mark.asyncio
    async def test_comparative_returns_report(
        self, engine, comparison_config, sample_items
    ):
        report = await engine.run_comparative_evaluation(
            config=comparison_config,
            dataset_items=sample_items,
        )
        assert isinstance(report, ComparisonReport)
        assert report.model_1_id == "model-a"
        assert report.model_2_id == "model-b"
        assert report.dataset_id == "ds-001"

    @pytest.mark.asyncio
    async def test_comparative_improvement_metrics(
        self, engine, comparison_config, sample_items
    ):
        report = await engine.run_comparative_evaluation(
            config=comparison_config,
            dataset_items=sample_items,
        )
        # Both models get same mock scores, so delta should be ~0
        assert report.improvement_metrics.trust_score_delta == pytest.approx(
            0.0, abs=0.01
        )

    @pytest.mark.asyncio
    async def test_comparative_has_recommendation(
        self, engine, comparison_config, sample_items
    ):
        report = await engine.run_comparative_evaluation(
            config=comparison_config,
            dataset_items=sample_items,
        )
        assert report.recommendation is not None
        assert report.recommendation_justification != ""

    @pytest.mark.asyncio
    async def test_comparative_has_cost_performance(
        self, engine, comparison_config, sample_items
    ):
        report = await engine.run_comparative_evaluation(
            config=comparison_config,
            dataset_items=sample_items,
        )
        assert report.cost_performance_analysis is not None

    @pytest.mark.asyncio
    async def test_comparative_empty_dataset(
        self, engine, comparison_config
    ):
        report = await engine.run_comparative_evaluation(
            config=comparison_config,
            dataset_items=[],
        )
        assert isinstance(report, ComparisonReport)

    @pytest.mark.asyncio
    async def test_comparative_with_pricing(
        self, engine, comparison_config, sample_items
    ):
        pricing = ModelPricing(
            input_price_per_1k_tokens=0.003,
            output_price_per_1k_tokens=0.015,
        )
        report = await engine.run_comparative_evaluation(
            config=comparison_config,
            dataset_items=sample_items,
            pricing_1=pricing,
            pricing_2=pricing,
        )
        assert report.model_1_metrics.total_cost >= 0
        assert report.model_2_metrics.total_cost >= 0


# ---------------------------------------------------------------------------
# Status query tests
# ---------------------------------------------------------------------------

class TestEvaluationStatus:
    def test_unknown_evaluation_id(self, engine):
        status = engine.get_evaluation_status("nonexistent")
        assert status["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_status_after_baseline(
        self, engine, eval_config, sample_items
    ):
        report = await engine.run_baseline_evaluation(
            config=eval_config,
            dataset_items=sample_items,
        )
        # The engine stores status keyed by internal evaluation_id,
        # so we verify at least one completed entry exists
        completed = [
            v
            for v in engine._status_store.values()
            if v["status"] == "completed"
        ]
        assert len(completed) >= 1


# ---------------------------------------------------------------------------
# Compatibility check integration
# ---------------------------------------------------------------------------

class TestCompatibilityCheck:
    @pytest.mark.asyncio
    async def test_incompatible_raises(
        self, engine, eval_config, sample_items
    ):
        from src.data_models.dataset import (
            DatasetFormat,
            DatasetMetadata,
            DatasetTaskType,
            TokenStatistics,
        )
        from src.data_models.model import (
            ModelCapability,
            ModelMetadata,
            ModelStatus,
        )

        model_meta = ModelMetadata(
            id="test-model",
            provider="bedrock",
            name="Test",
            capabilities=[ModelCapability.EMBEDDING],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            input_modalities=["text"],
            output_modalities=["text"],
            max_tokens=1024,
            region="us-east-1",
        )
        dataset_meta = DatasetMetadata(
            id="ds-001",
            name="test",
            format=DatasetFormat.JSONL,
            task_type=DatasetTaskType.CHAT,
            version="1",
            s3_uri="s3://bucket/data",
            row_count=10,
            token_stats=TokenStatistics(
                total_tokens=100,
                min_tokens=5,
                max_tokens=50,
                avg_tokens=10.0,
                p95_tokens=45,
                prompt_tokens=50,
                completion_tokens=50,
            ),
            created_at="2024-01-01T00:00:00Z",
            checksum="abc123",
        )

        with pytest.raises(ValueError, match="incompatible"):
            await engine.run_baseline_evaluation(
                config=eval_config,
                dataset_items=sample_items,
                model_metadata=model_meta,
                dataset_metadata=dataset_meta,
            )
