"""
Unit tests for trust scoring data models.

Tests Requirements: 5.1, 5.2, 5.3, 5.6
"""
import pytest
from pydantic import ValidationError

from src.data_models.trust_score import (
    TrustDimension,
    TrustScoreWeights,
    CustomMetricConfig,
    TrustScoreConfig,
    DimensionScore,
    TrustScoreResult,
)


class TestTrustDimension:
    """Test TrustDimension enum."""

    def test_trust_dimension_enum_values(self):
        """Test TrustDimension enum has all required values."""
        assert TrustDimension.ACCURACY == "accuracy"
        assert TrustDimension.CONSISTENCY == "consistency"
        assert TrustDimension.SAFETY == "safety"
        assert TrustDimension.BIAS == "bias"
        assert TrustDimension.CONTEXT_GROUNDING == "context_grounding"


class TestTrustScoreWeights:
    """Test TrustScoreWeights data model."""

    def test_trust_score_weights_default_values(self):
        """Test TrustScoreWeights with default values sum to 1.0."""
        weights = TrustScoreWeights()

        assert weights.accuracy == 0.25
        assert weights.consistency == 0.20
        assert weights.safety == 0.20
        assert weights.bias == 0.15
        assert weights.context_grounding == 0.20

        # Verify weights sum to 1.0
        assert weights.validate_weights() is True

    def test_trust_score_weights_custom_values(self):
        """Test TrustScoreWeights with custom values."""
        weights = TrustScoreWeights(
            accuracy=0.3,
            consistency=0.2,
            safety=0.2,
            bias=0.1,
            context_grounding=0.2,
        )

        assert weights.accuracy == 0.3
        assert weights.consistency == 0.2
        assert weights.safety == 0.2
        assert weights.bias == 0.1
        assert weights.context_grounding == 0.2
        assert weights.validate_weights() is True

    def test_trust_score_weights_validation_method(self):
        """Test validate_weights method correctly checks sum."""
        # Valid weights that sum to 1.0
        weights = TrustScoreWeights(
            accuracy=0.2,
            consistency=0.2,
            safety=0.2,
            bias=0.2,
            context_grounding=0.2,
        )
        assert weights.validate_weights() is True

        # Invalid weights that don't sum to 1.0
        weights_invalid = TrustScoreWeights(
            accuracy=0.5,
            consistency=0.2,
            safety=0.2,
            bias=0.2,
            context_grounding=0.2,
        )
        assert weights_invalid.validate_weights() is False

    def test_trust_score_weights_out_of_range_rejected(self):
        """Test that weights outside [0, 1] are rejected."""
        with pytest.raises(ValidationError):
            TrustScoreWeights(
                accuracy=1.5,
                consistency=0.2,
                safety=0.2,
                bias=0.1,
                context_grounding=0.2,
            )

        with pytest.raises(ValidationError):
            TrustScoreWeights(
                accuracy=0.25,
                consistency=-0.1,
                safety=0.2,
                bias=0.15,
                context_grounding=0.2,
            )

    def test_trust_score_weights_boundary_values(self):
        """Test TrustScoreWeights with boundary values."""
        # All zeros (valid range but won't sum to 1.0)
        weights_zeros = TrustScoreWeights(
            accuracy=0.0,
            consistency=0.0,
            safety=0.0,
            bias=0.0,
            context_grounding=0.0,
        )
        assert weights_zeros.accuracy == 0.0
        assert weights_zeros.validate_weights() is False

        # One dimension gets all weight
        weights_single = TrustScoreWeights(
            accuracy=1.0,
            consistency=0.0,
            safety=0.0,
            bias=0.0,
            context_grounding=0.0,
        )
        assert weights_single.accuracy == 1.0
        assert weights_single.validate_weights() is True


class TestCustomMetricConfig:
    """Test CustomMetricConfig data model."""

    def test_custom_metric_config_valid_creation(self):
        """Test creating CustomMetricConfig with valid data."""
        config = CustomMetricConfig(
            name="domain_relevance",
            weight=0.15,
            scorer_class="custom.scorers.DomainRelevanceScorer",
            config={"threshold": 0.7, "model": "embedding-model"},
        )

        assert config.name == "domain_relevance"
        assert config.weight == 0.15
        assert config.scorer_class == "custom.scorers.DomainRelevanceScorer"
        assert config.config == {"threshold": 0.7, "model": "embedding-model"}

    def test_custom_metric_config_empty_config(self):
        """Test CustomMetricConfig with empty config dict."""
        config = CustomMetricConfig(
            name="simple_metric",
            weight=0.1,
            scorer_class="custom.scorers.SimpleScorer",
        )

        assert config.config == {}

    def test_custom_metric_config_invalid_weight_rejected(self):
        """Test that invalid weight is rejected."""
        with pytest.raises(ValidationError):
            CustomMetricConfig(
                name="invalid_metric",
                weight=1.5,
                scorer_class="custom.scorers.InvalidScorer",
            )

        with pytest.raises(ValidationError):
            CustomMetricConfig(
                name="invalid_metric",
                weight=-0.1,
                scorer_class="custom.scorers.InvalidScorer",
            )

    def test_custom_metric_config_boundary_weights(self):
        """Test CustomMetricConfig with boundary weight values."""
        config_zero = CustomMetricConfig(
            name="zero_weight",
            weight=0.0,
            scorer_class="custom.scorers.ZeroScorer",
        )
        assert config_zero.weight == 0.0

        config_one = CustomMetricConfig(
            name="full_weight",
            weight=1.0,
            scorer_class="custom.scorers.FullScorer",
        )
        assert config_one.weight == 1.0


class TestTrustScoreConfig:
    """Test TrustScoreConfig data model."""

    def test_trust_score_config_default_values(self):
        """Test TrustScoreConfig with default values."""
        config = TrustScoreConfig()

        assert isinstance(config.weights, TrustScoreWeights)
        assert config.review_threshold == 0.6
        assert config.consistency_samples == 3
        assert config.custom_metrics == []

    def test_trust_score_config_custom_values(self):
        """Test TrustScoreConfig with custom values."""
        weights = TrustScoreWeights(
            accuracy=0.3,
            consistency=0.25,
            safety=0.2,
            bias=0.15,
            context_grounding=0.1,
        )
        custom_metric = CustomMetricConfig(
            name="domain_score",
            weight=0.2,
            scorer_class="custom.DomainScorer",
        )

        config = TrustScoreConfig(
            weights=weights,
            review_threshold=0.7,
            consistency_samples=5,
            custom_metrics=[custom_metric],
        )

        assert config.weights == weights
        assert config.review_threshold == 0.7
        assert config.consistency_samples == 5
        assert len(config.custom_metrics) == 1
        assert config.custom_metrics[0] == custom_metric

    def test_trust_score_config_invalid_threshold_rejected(self):
        """Test that invalid review threshold is rejected."""
        with pytest.raises(ValidationError):
            TrustScoreConfig(review_threshold=1.5)

        with pytest.raises(ValidationError):
            TrustScoreConfig(review_threshold=-0.1)

    def test_trust_score_config_invalid_consistency_samples_rejected(self):
        """Test that invalid consistency samples are rejected."""
        with pytest.raises(ValidationError):
            TrustScoreConfig(consistency_samples=0)

        with pytest.raises(ValidationError):
            TrustScoreConfig(consistency_samples=11)

        with pytest.raises(ValidationError):
            TrustScoreConfig(consistency_samples=-1)

    def test_trust_score_config_boundary_values(self):
        """Test TrustScoreConfig with boundary values."""
        config_min = TrustScoreConfig(
            review_threshold=0.0,
            consistency_samples=1,
        )
        assert config_min.review_threshold == 0.0
        assert config_min.consistency_samples == 1

        config_max = TrustScoreConfig(
            review_threshold=1.0,
            consistency_samples=10,
        )
        assert config_max.review_threshold == 1.0
        assert config_max.consistency_samples == 10

    def test_trust_score_config_multiple_custom_metrics(self):
        """Test TrustScoreConfig with multiple custom metrics."""
        metric1 = CustomMetricConfig(
            name="metric1",
            weight=0.1,
            scorer_class="custom.Metric1",
        )
        metric2 = CustomMetricConfig(
            name="metric2",
            weight=0.15,
            scorer_class="custom.Metric2",
        )

        config = TrustScoreConfig(custom_metrics=[metric1, metric2])

        assert len(config.custom_metrics) == 2
        assert config.custom_metrics[0].name == "metric1"
        assert config.custom_metrics[1].name == "metric2"


class TestDimensionScore:
    """Test DimensionScore data model."""

    def test_dimension_score_valid_creation(self):
        """Test creating DimensionScore with valid data."""
        score = DimensionScore(
            dimension=TrustDimension.ACCURACY,
            score=0.85,
            confidence=0.9,
            details={"exact_match": True, "semantic_similarity": 0.92},
            checks_passed=["exact_match", "fuzzy_match"],
            checks_failed=["keyword_match"],
        )

        assert score.dimension == TrustDimension.ACCURACY
        assert score.score == 0.85
        assert score.confidence == 0.9
        assert score.details == {
            "exact_match": True,
            "semantic_similarity": 0.92
        }
        assert score.checks_passed == ["exact_match", "fuzzy_match"]
        assert score.checks_failed == ["keyword_match"]

    def test_dimension_score_minimal_creation(self):
        """Test creating DimensionScore with minimal data."""
        score = DimensionScore(
            dimension=TrustDimension.SAFETY,
            score=0.95,
            confidence=1.0,
        )

        assert score.dimension == TrustDimension.SAFETY
        assert score.score == 0.95
        assert score.confidence == 1.0
        assert score.details == {}
        assert score.checks_passed == []
        assert score.checks_failed == []

    def test_dimension_score_invalid_score_rejected(self):
        """Test that invalid scores are rejected."""
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=TrustDimension.BIAS,
                score=1.5,
                confidence=0.9,
            )

        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=TrustDimension.BIAS,
                score=-0.1,
                confidence=0.9,
            )

    def test_dimension_score_invalid_confidence_rejected(self):
        """Test that invalid confidence is rejected."""
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.8,
                confidence=1.2,
            )

        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.8,
                confidence=-0.1,
            )

    def test_dimension_score_boundary_values(self):
        """Test DimensionScore with boundary values."""
        score_min = DimensionScore(
            dimension=TrustDimension.CONTEXT_GROUNDING,
            score=0.0,
            confidence=0.0,
        )
        assert score_min.score == 0.0
        assert score_min.confidence == 0.0

        score_max = DimensionScore(
            dimension=TrustDimension.CONTEXT_GROUNDING,
            score=1.0,
            confidence=1.0,
        )
        assert score_max.score == 1.0
        assert score_max.confidence == 1.0

    def test_dimension_score_all_dimensions(self):
        """Test DimensionScore can be created for all dimensions."""
        for dimension in TrustDimension:
            score = DimensionScore(
                dimension=dimension,
                score=0.8,
                confidence=0.9,
            )
            assert score.dimension == dimension


class TestTrustScoreResult:
    """Test TrustScoreResult data model."""

    def test_trust_score_result_valid_creation(self):
        """Test creating TrustScoreResult with valid data."""
        dim_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.9,
                confidence=0.95,
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=0.85,
                confidence=0.9,
            ),
        }

        result = TrustScoreResult(
            overall_score=0.87,
            dimension_scores=dim_scores,
            confidence_level=0.92,
            flagged_for_review=False,
            explanation="High trust score across all dimensions",
            component_details=[
                {"component": "accuracy", "value": 0.9},
                {"component": "safety", "value": 0.85},
            ],
        )

        assert result.overall_score == 0.87
        assert len(result.dimension_scores) == 2
        assert result.confidence_level == 0.92
        assert result.flagged_for_review is False
        assert result.explanation == "High trust score across all dimensions"
        assert len(result.component_details) == 2

    def test_trust_score_result_minimal_creation(self):
        """Test creating TrustScoreResult with minimal data."""
        result = TrustScoreResult(
            overall_score=0.75,
            confidence_level=0.8,
        )

        assert result.overall_score == 0.75
        assert result.dimension_scores == {}
        assert result.confidence_level == 0.8
        assert result.flagged_for_review is False
        assert result.explanation == ""
        assert result.component_details == []

    def test_trust_score_result_flagged_for_review(self):
        """Test TrustScoreResult flagged for review."""
        result = TrustScoreResult(
            overall_score=0.55,
            confidence_level=0.7,
            flagged_for_review=True,
            explanation="Score below review threshold",
        )

        assert result.overall_score == 0.55
        assert result.flagged_for_review is True
        assert result.explanation == "Score below review threshold"

    def test_trust_score_result_invalid_overall_score_rejected(self):
        """Test that invalid overall score is rejected."""
        with pytest.raises(ValidationError):
            TrustScoreResult(
                overall_score=1.5,
                confidence_level=0.9,
            )

        with pytest.raises(ValidationError):
            TrustScoreResult(
                overall_score=-0.1,
                confidence_level=0.9,
            )

    def test_trust_score_result_invalid_confidence_rejected(self):
        """Test that invalid confidence level is rejected."""
        with pytest.raises(ValidationError):
            TrustScoreResult(
                overall_score=0.8,
                confidence_level=1.2,
            )

        with pytest.raises(ValidationError):
            TrustScoreResult(
                overall_score=0.8,
                confidence_level=-0.1,
            )

    def test_trust_score_result_boundary_values(self):
        """Test TrustScoreResult with boundary values."""
        result_min = TrustScoreResult(
            overall_score=0.0,
            confidence_level=0.0,
        )
        assert result_min.overall_score == 0.0
        assert result_min.confidence_level == 0.0

        result_max = TrustScoreResult(
            overall_score=1.0,
            confidence_level=1.0,
        )
        assert result_max.overall_score == 1.0
        assert result_max.confidence_level == 1.0

    def test_trust_score_result_all_dimensions(self):
        """Test TrustScoreResult with all trust dimensions."""
        dim_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.9,
                confidence=0.95,
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.85,
                confidence=0.9,
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=0.95,
                confidence=0.98,
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=0.88,
                confidence=0.92,
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.82,
                confidence=0.87,
            ),
        }

        result = TrustScoreResult(
            overall_score=0.88,
            dimension_scores=dim_scores,
            confidence_level=0.92,
        )

        assert len(result.dimension_scores) == 5
        assert all(
            dim in result.dimension_scores for dim in TrustDimension
        )


class TestDataModelSerialization:
    """Test serialization of trust scoring data models."""

    def test_trust_score_weights_serialization(self):
        """Test TrustScoreWeights JSON serialization."""
        weights = TrustScoreWeights(
            accuracy=0.3,
            consistency=0.2,
            safety=0.2,
            bias=0.15,
            context_grounding=0.15,
        )

        # Serialize to dict
        data = weights.model_dump()
        assert data["accuracy"] == 0.3
        assert data["consistency"] == 0.2

        # Serialize to JSON
        json_str = weights.model_dump_json()
        assert "0.3" in json_str
        assert "accuracy" in json_str

    def test_trust_score_config_serialization(self):
        """Test TrustScoreConfig JSON serialization."""
        config = TrustScoreConfig(
            review_threshold=0.7,
            consistency_samples=5,
        )

        # Serialize to dict
        data = config.model_dump()
        assert data["review_threshold"] == 0.7
        assert data["consistency_samples"] == 5

        # Serialize to JSON
        json_str = config.model_dump_json()
        assert "0.7" in json_str
        assert "review_threshold" in json_str

    def test_dimension_score_serialization(self):
        """Test DimensionScore JSON serialization."""
        score = DimensionScore(
            dimension=TrustDimension.ACCURACY,
            score=0.85,
            confidence=0.9,
            details={"method": "semantic"},
        )

        # Serialize to dict
        data = score.model_dump()
        assert data["dimension"] == "accuracy"
        assert data["score"] == 0.85
        assert data["confidence"] == 0.9

        # Serialize to JSON
        json_str = score.model_dump_json()
        assert "accuracy" in json_str
        assert "0.85" in json_str

    def test_trust_score_result_serialization(self):
        """Test TrustScoreResult JSON serialization."""
        dim_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.9,
                confidence=0.95,
            ),
        }

        result = TrustScoreResult(
            overall_score=0.87,
            dimension_scores=dim_scores,
            confidence_level=0.92,
            flagged_for_review=False,
        )

        # Serialize to dict
        data = result.model_dump()
        assert data["overall_score"] == 0.87
        assert data["confidence_level"] == 0.92
        assert data["flagged_for_review"] is False

        # Serialize to JSON
        json_str = result.model_dump_json()
        assert "0.87" in json_str
        assert "overall_score" in json_str
