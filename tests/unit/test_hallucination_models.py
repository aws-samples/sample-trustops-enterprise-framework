"""
Unit tests for hallucination detection data models.

Tests Requirements: 6.1, 6.2, 6.3, 6.4, 6.8
"""
import pytest
from pydantic import ValidationError

from src.data_models.hallucination import (
    SensitivityLevel,
    HallucinationConfig,
    Claim,
    EvidenceSource,
    ClaimEvidence,
    FlaggedSpan,
    HallucinationResult,
    HallucinationComparison,
)


class TestSensitivityLevel:
    """Test SensitivityLevel enum."""

    def test_sensitivity_level_enum_values(self):
        """Test SensitivityLevel enum has all required values."""
        assert SensitivityLevel.STRICT == "strict"
        assert SensitivityLevel.MODERATE == "moderate"
        assert SensitivityLevel.LENIENT == "lenient"


class TestHallucinationConfig:
    """Test HallucinationConfig data model."""

    def test_hallucination_config_default_values(self):
        """Test HallucinationConfig with default values."""
        config = HallucinationConfig()

        assert config.similarity_threshold == 0.6
        assert config.sensitivity_level == SensitivityLevel.MODERATE
        assert config.max_claims_per_response == 50
        assert len(config.hedging_keywords) > 0
        assert "might" in config.hedging_keywords
        assert "possibly" in config.hedging_keywords
        assert config.embedding_model_id == "amazon.titan-embed-text-v2:0"

    def test_hallucination_config_custom_values(self):
        """Test HallucinationConfig with custom values."""
        config = HallucinationConfig(
            similarity_threshold=0.8,
            sensitivity_level=SensitivityLevel.STRICT,
            max_claims_per_response=100,
            hedging_keywords=["maybe", "perhaps"],
            embedding_model_id="custom-embedding-model",
        )

        assert config.similarity_threshold == 0.8
        assert config.sensitivity_level == SensitivityLevel.STRICT
        assert config.max_claims_per_response == 100
        assert config.hedging_keywords == ["maybe", "perhaps"]
        assert config.embedding_model_id == "custom-embedding-model"

    def test_hallucination_config_invalid_threshold_rejected(self):
        """Test that invalid similarity threshold is rejected."""
        with pytest.raises(ValidationError):
            HallucinationConfig(similarity_threshold=1.5)

        with pytest.raises(ValidationError):
            HallucinationConfig(similarity_threshold=-0.1)

    def test_hallucination_config_boundary_values(self):
        """Test HallucinationConfig with boundary values."""
        config_min = HallucinationConfig(
            similarity_threshold=0.0,
            max_claims_per_response=1,
        )
        assert config_min.similarity_threshold == 0.0
        assert config_min.max_claims_per_response == 1

        config_max = HallucinationConfig(
            similarity_threshold=1.0,
            max_claims_per_response=1000,
        )
        assert config_max.similarity_threshold == 1.0
        assert config_max.max_claims_per_response == 1000


class TestClaim:
    """Test Claim data model."""

    def test_claim_valid_creation(self):
        """Test creating Claim with valid data."""
        claim = Claim(
            text="The capital of France is Paris.",
            start_idx=0,
            end_idx=32,
            is_factual=True,
            is_opinion=False,
            is_hedged=False,
        )

        assert claim.text == "The capital of France is Paris."
        assert claim.start_idx == 0
        assert claim.end_idx == 32
        assert claim.is_factual is True
        assert claim.is_opinion is False
        assert claim.is_hedged is False

    def test_claim_hedged_statement(self):
        """Test Claim with hedged statement."""
        claim = Claim(
            text="It might rain tomorrow.",
            start_idx=0,
            end_idx=23,
            is_factual=False,
            is_opinion=True,
            is_hedged=True,
        )

        assert claim.is_hedged is True
        assert claim.is_opinion is True

    def test_claim_invalid_indices_rejected(self):
        """Test that invalid indices are rejected."""
        with pytest.raises(ValidationError):
            Claim(
                text="Invalid claim",
                start_idx=10,
                end_idx=5,  # end_idx <= start_idx
                is_factual=True,
                is_opinion=False,
            )

        with pytest.raises(ValidationError):
            Claim(
                text="Invalid claim",
                start_idx=10,
                end_idx=10,  # end_idx == start_idx
                is_factual=True,
                is_opinion=False,
            )


class TestEvidenceSource:
    """Test EvidenceSource data model."""

    def test_evidence_source_valid_creation(self):
        """Test creating EvidenceSource with valid data."""
        source = EvidenceSource(
            document_id="doc_123",
            span="Paris is the capital and largest city of France.",
            similarity_score=0.92,
        )

        assert source.document_id == "doc_123"
        assert source.span == (
            "Paris is the capital and largest city of France."
        )
        assert source.similarity_score == 0.92

    def test_evidence_source_invalid_score_rejected(self):
        """Test that invalid similarity score is rejected."""
        with pytest.raises(ValidationError):
            EvidenceSource(
                document_id="doc_123",
                span="Some text",
                similarity_score=1.5,
            )

        with pytest.raises(ValidationError):
            EvidenceSource(
                document_id="doc_123",
                span="Some text",
                similarity_score=-0.1,
            )

    def test_evidence_source_boundary_values(self):
        """Test EvidenceSource with boundary values."""
        source_min = EvidenceSource(
            document_id="doc_min",
            span="Text",
            similarity_score=0.0,
        )
        assert source_min.similarity_score == 0.0

        source_max = EvidenceSource(
            document_id="doc_max",
            span="Text",
            similarity_score=1.0,
        )
        assert source_max.similarity_score == 1.0


class TestClaimEvidence:
    """Test ClaimEvidence data model."""

    def test_claim_evidence_supported(self):
        """Test ClaimEvidence for a supported claim."""
        claim = Claim(
            text="Paris is the capital of France.",
            start_idx=0,
            end_idx=32,
            is_factual=True,
            is_opinion=False,
        )

        source = EvidenceSource(
            document_id="doc_1",
            span="Paris is the capital city of France.",
            similarity_score=0.95,
        )

        evidence = ClaimEvidence(
            claim=claim,
            is_supported=True,
            similarity_score=0.95,
            best_matching_document="doc_1",
            best_matching_span="Paris is the capital city of France.",
            evidence_sources=[source],
        )

        assert evidence.is_supported is True
        assert evidence.similarity_score == 0.95
        assert evidence.best_matching_document == "doc_1"
        assert len(evidence.evidence_sources) == 1

    def test_claim_evidence_unsupported(self):
        """Test ClaimEvidence for an unsupported claim."""
        claim = Claim(
            text="The moon is made of cheese.",
            start_idx=0,
            end_idx=28,
            is_factual=True,
            is_opinion=False,
        )

        evidence = ClaimEvidence(
            claim=claim,
            is_supported=False,
            similarity_score=0.15,
            best_matching_document=None,
            best_matching_span=None,
            evidence_sources=[],
        )

        assert evidence.is_supported is False
        assert evidence.similarity_score == 0.15
        assert evidence.best_matching_document is None
        assert len(evidence.evidence_sources) == 0

    def test_claim_evidence_multiple_sources(self):
        """Test ClaimEvidence with multiple evidence sources."""
        claim = Claim(
            text="Water boils at 100 degrees Celsius.",
            start_idx=0,
            end_idx=36,
            is_factual=True,
            is_opinion=False,
        )

        source1 = EvidenceSource(
            document_id="doc_1",
            span="Water boils at 100°C at sea level.",
            similarity_score=0.92,
        )

        source2 = EvidenceSource(
            document_id="doc_2",
            span="The boiling point of water is 100 degrees Celsius.",
            similarity_score=0.88,
        )

        evidence = ClaimEvidence(
            claim=claim,
            is_supported=True,
            similarity_score=0.92,
            best_matching_document="doc_1",
            best_matching_span="Water boils at 100°C at sea level.",
            evidence_sources=[source1, source2],
        )

        assert len(evidence.evidence_sources) == 2
        assert evidence.evidence_sources[0].similarity_score == 0.92
        assert evidence.evidence_sources[1].similarity_score == 0.88

    def test_claim_evidence_invalid_score_rejected(self):
        """Test that invalid similarity score is rejected."""
        claim = Claim(
            text="Test claim",
            start_idx=0,
            end_idx=10,
            is_factual=True,
            is_opinion=False,
        )

        with pytest.raises(ValidationError):
            ClaimEvidence(
                claim=claim,
                is_supported=True,
                similarity_score=1.5,
            )


class TestFlaggedSpan:
    """Test FlaggedSpan data model."""

    def test_flagged_span_valid_creation(self):
        """Test creating FlaggedSpan with valid data."""
        span = FlaggedSpan(
            text="The moon is made of cheese.",
            start_idx=0,
            end_idx=28,
            grounding_score=0.15,
            reason="No supporting evidence found in source documents",
        )

        assert span.text == "The moon is made of cheese."
        assert span.start_idx == 0
        assert span.end_idx == 28
        assert span.grounding_score == 0.15
        assert span.reason == (
            "No supporting evidence found in source documents"
        )

    def test_flagged_span_invalid_indices_rejected(self):
        """Test that invalid indices are rejected."""
        with pytest.raises(ValidationError):
            FlaggedSpan(
                text="Invalid span",
                start_idx=10,
                end_idx=5,
                grounding_score=0.2,
                reason="Test",
            )

    def test_flagged_span_invalid_score_rejected(self):
        """Test that invalid grounding score is rejected."""
        with pytest.raises(ValidationError):
            FlaggedSpan(
                text="Test span",
                start_idx=0,
                end_idx=9,
                grounding_score=1.5,
                reason="Test",
            )

        with pytest.raises(ValidationError):
            FlaggedSpan(
                text="Test span",
                start_idx=0,
                end_idx=9,
                grounding_score=-0.1,
                reason="Test",
            )

    def test_flagged_span_boundary_values(self):
        """Test FlaggedSpan with boundary values."""
        span_min = FlaggedSpan(
            text="Low score",
            start_idx=0,
            end_idx=9,
            grounding_score=0.0,
            reason="No evidence",
        )
        assert span_min.grounding_score == 0.0

        span_max = FlaggedSpan(
            text="High score",
            start_idx=0,
            end_idx=10,
            grounding_score=1.0,
            reason="Perfect match",
        )
        assert span_max.grounding_score == 1.0


class TestHallucinationResult:
    """Test HallucinationResult data model."""

    def test_hallucination_result_no_hallucinations(self):
        """Test HallucinationResult with no hallucinations."""
        result = HallucinationResult(
            has_hallucinations=False,
            hallucination_rate=0.0,
            total_claims=5,
            factual_claims=5,
            supported_claims=5,
            unsupported_claims=0,
            flagged_spans=[],
            overall_grounding_score=0.95,
            claim_evidence=[],
            sensitivity_level=SensitivityLevel.MODERATE,
        )

        assert result.has_hallucinations is False
        assert result.hallucination_rate == 0.0
        assert result.total_claims == 5
        assert result.factual_claims == 5
        assert result.supported_claims == 5
        assert result.unsupported_claims == 0
        assert len(result.flagged_spans) == 0
        assert result.overall_grounding_score == 0.95

    def test_hallucination_result_with_hallucinations(self):
        """Test HallucinationResult with hallucinations detected."""
        flagged = FlaggedSpan(
            text="Unsupported claim",
            start_idx=0,
            end_idx=17,
            grounding_score=0.2,
            reason="No evidence",
        )

        result = HallucinationResult(
            has_hallucinations=True,
            hallucination_rate=0.4,
            total_claims=10,
            factual_claims=5,
            supported_claims=3,
            unsupported_claims=2,
            flagged_spans=[flagged],
            overall_grounding_score=0.65,
            claim_evidence=[],
            sensitivity_level=SensitivityLevel.STRICT,
        )

        assert result.has_hallucinations is True
        assert result.hallucination_rate == 0.4
        assert result.total_claims == 10
        assert result.factual_claims == 5
        assert result.supported_claims == 3
        assert result.unsupported_claims == 2
        assert len(result.flagged_spans) == 1
        assert result.sensitivity_level == SensitivityLevel.STRICT

    def test_hallucination_result_invalid_rate_rejected(self):
        """Test that invalid hallucination rate is rejected."""
        with pytest.raises(ValidationError):
            HallucinationResult(
                has_hallucinations=True,
                hallucination_rate=1.5,
                total_claims=5,
                factual_claims=5,
                supported_claims=3,
                unsupported_claims=2,
                overall_grounding_score=0.7,
                sensitivity_level=SensitivityLevel.MODERATE,
            )

    def test_hallucination_result_invalid_claim_counts(self):
        """Test that invalid claim counts are rejected."""
        # factual_claims > total_claims
        with pytest.raises(ValidationError):
            HallucinationResult(
                has_hallucinations=False,
                hallucination_rate=0.0,
                total_claims=5,
                factual_claims=10,
                supported_claims=5,
                unsupported_claims=0,
                overall_grounding_score=0.9,
                sensitivity_level=SensitivityLevel.MODERATE,
            )

        # supported_claims > factual_claims
        with pytest.raises(ValidationError):
            HallucinationResult(
                has_hallucinations=False,
                hallucination_rate=0.0,
                total_claims=10,
                factual_claims=5,
                supported_claims=8,
                unsupported_claims=0,
                overall_grounding_score=0.9,
                sensitivity_level=SensitivityLevel.MODERATE,
            )

        # unsupported_claims > factual_claims
        with pytest.raises(ValidationError):
            HallucinationResult(
                has_hallucinations=True,
                hallucination_rate=0.5,
                total_claims=10,
                factual_claims=5,
                supported_claims=2,
                unsupported_claims=8,
                overall_grounding_score=0.5,
                sensitivity_level=SensitivityLevel.MODERATE,
            )

    def test_hallucination_result_boundary_values(self):
        """Test HallucinationResult with boundary values."""
        result_min = HallucinationResult(
            has_hallucinations=False,
            hallucination_rate=0.0,
            total_claims=0,
            factual_claims=0,
            supported_claims=0,
            unsupported_claims=0,
            overall_grounding_score=0.0,
            sensitivity_level=SensitivityLevel.LENIENT,
        )
        assert result_min.hallucination_rate == 0.0
        assert result_min.overall_grounding_score == 0.0

        result_max = HallucinationResult(
            has_hallucinations=True,
            hallucination_rate=1.0,
            total_claims=5,
            factual_claims=5,
            supported_claims=0,
            unsupported_claims=5,
            overall_grounding_score=1.0,
            sensitivity_level=SensitivityLevel.STRICT,
        )
        assert result_max.hallucination_rate == 1.0
        assert result_max.overall_grounding_score == 1.0


class TestHallucinationComparison:
    """Test HallucinationComparison data model."""

    def test_hallucination_comparison_improvement(self):
        """Test HallucinationComparison showing improvement."""
        comparison = HallucinationComparison(
            baseline_mean_rate=0.35,
            comparison_mean_rate=0.15,
            reduction=0.20,
            reduction_percent=57.14,
            statistical_significance=0.95,
            p_value=0.02,
        )

        assert comparison.baseline_mean_rate == 0.35
        assert comparison.comparison_mean_rate == 0.15
        assert comparison.reduction == 0.20
        assert comparison.reduction_percent == 57.14
        assert comparison.statistical_significance == 0.95
        assert comparison.p_value == 0.02

    def test_hallucination_comparison_no_improvement(self):
        """Test HallucinationComparison with no improvement."""
        comparison = HallucinationComparison(
            baseline_mean_rate=0.25,
            comparison_mean_rate=0.30,
            reduction=-0.05,
            reduction_percent=-20.0,
            statistical_significance=0.85,
            p_value=0.15,
        )

        assert comparison.baseline_mean_rate == 0.25
        assert comparison.comparison_mean_rate == 0.30
        assert comparison.reduction == -0.05
        assert comparison.reduction_percent == -20.0

    def test_hallucination_comparison_invalid_rates_rejected(self):
        """Test that invalid hallucination rates are rejected."""
        with pytest.raises(ValidationError):
            HallucinationComparison(
                baseline_mean_rate=1.5,
                comparison_mean_rate=0.2,
                reduction=0.1,
                reduction_percent=10.0,
                statistical_significance=0.95,
                p_value=0.05,
            )

        with pytest.raises(ValidationError):
            HallucinationComparison(
                baseline_mean_rate=0.3,
                comparison_mean_rate=-0.1,
                reduction=0.1,
                reduction_percent=10.0,
                statistical_significance=0.95,
                p_value=0.05,
            )

    def test_hallucination_comparison_invalid_significance_rejected(self):
        """Test that invalid significance values are rejected."""
        with pytest.raises(ValidationError):
            HallucinationComparison(
                baseline_mean_rate=0.3,
                comparison_mean_rate=0.2,
                reduction=0.1,
                reduction_percent=33.33,
                statistical_significance=1.5,
                p_value=0.05,
            )

        with pytest.raises(ValidationError):
            HallucinationComparison(
                baseline_mean_rate=0.3,
                comparison_mean_rate=0.2,
                reduction=0.1,
                reduction_percent=33.33,
                statistical_significance=0.95,
                p_value=1.2,
            )

    def test_hallucination_comparison_boundary_values(self):
        """Test HallucinationComparison with boundary values."""
        comparison_min = HallucinationComparison(
            baseline_mean_rate=0.0,
            comparison_mean_rate=0.0,
            reduction=0.0,
            reduction_percent=0.0,
            statistical_significance=0.0,
            p_value=0.0,
        )
        assert comparison_min.baseline_mean_rate == 0.0
        assert comparison_min.statistical_significance == 0.0

        comparison_max = HallucinationComparison(
            baseline_mean_rate=1.0,
            comparison_mean_rate=1.0,
            reduction=0.0,
            reduction_percent=0.0,
            statistical_significance=1.0,
            p_value=1.0,
        )
        assert comparison_max.baseline_mean_rate == 1.0
        assert comparison_max.statistical_significance == 1.0


class TestDataModelSerialization:
    """Test serialization of hallucination detection data models."""

    def test_hallucination_config_serialization(self):
        """Test HallucinationConfig JSON serialization."""
        config = HallucinationConfig(
            similarity_threshold=0.7,
            sensitivity_level=SensitivityLevel.STRICT,
        )

        # Serialize to dict
        data = config.model_dump()
        assert data["similarity_threshold"] == 0.7
        assert data["sensitivity_level"] == "strict"

        # Serialize to JSON
        json_str = config.model_dump_json()
        assert "0.7" in json_str
        assert "strict" in json_str

    def test_claim_serialization(self):
        """Test Claim JSON serialization."""
        claim = Claim(
            text="Test claim",
            start_idx=0,
            end_idx=10,
            is_factual=True,
            is_opinion=False,
        )

        # Serialize to dict
        data = claim.model_dump()
        assert data["text"] == "Test claim"
        assert data["is_factual"] is True

        # Serialize to JSON
        json_str = claim.model_dump_json()
        assert "Test claim" in json_str

    def test_hallucination_result_serialization(self):
        """Test HallucinationResult JSON serialization."""
        result = HallucinationResult(
            has_hallucinations=True,
            hallucination_rate=0.3,
            total_claims=10,
            factual_claims=5,
            supported_claims=3,
            unsupported_claims=2,
            overall_grounding_score=0.7,
            sensitivity_level=SensitivityLevel.MODERATE,
        )

        # Serialize to dict
        data = result.model_dump()
        assert data["has_hallucinations"] is True
        assert data["hallucination_rate"] == 0.3
        assert data["sensitivity_level"] == "moderate"

        # Serialize to JSON
        json_str = result.model_dump_json()
        assert "0.3" in json_str
        assert "moderate" in json_str

    def test_hallucination_comparison_serialization(self):
        """Test HallucinationComparison JSON serialization."""
        comparison = HallucinationComparison(
            baseline_mean_rate=0.4,
            comparison_mean_rate=0.2,
            reduction=0.2,
            reduction_percent=50.0,
            statistical_significance=0.95,
            p_value=0.03,
        )

        # Serialize to dict
        data = comparison.model_dump()
        assert data["baseline_mean_rate"] == 0.4
        assert data["reduction"] == 0.2

        # Serialize to JSON
        json_str = comparison.model_dump_json()
        assert "0.4" in json_str
        assert "0.2" in json_str
