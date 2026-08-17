"""
Data models for the Hallucination Detector.

This module defines the core data models for hallucination detection and
measurement, including claim extraction, evidence search, and grounding
analysis in the TrustOps Enterprise Framework.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.8
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SensitivityLevel(str, Enum):
    """Enumeration of hallucination detection sensitivity levels.

    Requirement 6.1: Define SensitivityLevel enum with configurable thresholds
    """

    STRICT = "strict"      # threshold = 0.8
    MODERATE = "moderate"  # threshold = 0.6
    LENIENT = "lenient"    # threshold = 0.4


class HallucinationConfig(BaseModel):
    """Configuration for hallucination detection.

    Requirement 6.1: Define HallucinationConfig schema
    """

    similarity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score for evidence to be considered supporting"
        )
    )
    sensitivity_level: SensitivityLevel = Field(
        default=SensitivityLevel.MODERATE,
        description="Sensitivity level for hallucination detection"
    )
    max_claims_per_response: int = Field(
        default=50,
        gt=0,
        description="Maximum number of claims to extract per response"
    )
    hedging_keywords: list[str] = Field(
        default_factory=lambda: [
            "might", "possibly", "perhaps", "I think", "could be",
            "may", "probably", "likely", "seems", "appears"
        ],
        description="Keywords that indicate hedging or uncertainty"
    )
    embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        min_length=1,
        description="Model ID for embedding-based evidence search"
    )

    @field_validator('similarity_threshold')
    @classmethod
    def validate_threshold_range(cls, v: float) -> float:
        """Ensure similarity threshold is within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError(
                "Similarity threshold must be between 0.0 and 1.0"
            )
        return v


class Claim(BaseModel):
    """A single claim extracted from a model response.

    Requirement 6.2: Define Claim schema for claim extraction
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The text of the claim"
    )
    start_idx: int = Field(
        ...,
        ge=0,
        description="Starting character index in the original response"
    )
    end_idx: int = Field(
        ...,
        ge=0,
        description="Ending character index in the original response"
    )
    is_factual: bool = Field(
        ...,
        description="Whether this claim is factual (vs opinion)"
    )
    is_opinion: bool = Field(
        ...,
        description="Whether this claim is an opinion"
    )
    is_hedged: bool = Field(
        default=False,
        description="Whether this claim contains hedging language"
    )

    @field_validator('end_idx')
    @classmethod
    def validate_indices(cls, v: int, info) -> int:
        """Ensure end_idx is greater than start_idx."""
        if 'start_idx' in info.data and v <= info.data['start_idx']:
            raise ValueError("end_idx must be greater than start_idx")
        return v


class EvidenceSource(BaseModel):
    """A source document that provides evidence for a claim.

    Requirement 6.2: Define EvidenceSource schema for evidence tracking
    """

    document_id: str = Field(
        ...,
        min_length=1,
        description="Identifier for the source document"
    )
    span: str = Field(
        ...,
        min_length=1,
        description="The text span from the document that provides evidence"
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Similarity score between claim and evidence (0-1)"
        )
    )

    @field_validator('similarity_score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure similarity score is within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")
        return v


class ClaimEvidence(BaseModel):
    """Evidence analysis for a single claim.

    Requirement 6.2: Define ClaimEvidence schema linking claims to evidence
    """

    claim: Claim = Field(
        ...,
        description="The claim being analyzed"
    )
    is_supported: bool = Field(
        ...,
        description="Whether the claim is supported by evidence"
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Best similarity score found for this claim"
    )
    best_matching_document: Optional[str] = Field(
        default=None,
        description="ID of the document with the best matching evidence"
    )
    best_matching_span: Optional[str] = Field(
        default=None,
        description="Text span with the best matching evidence"
    )
    evidence_sources: list[EvidenceSource] = Field(
        default_factory=list,
        description="List of evidence sources found for this claim"
    )

    @field_validator('similarity_score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure similarity score is within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")
        return v


class FlaggedSpan(BaseModel):
    """A text span flagged as a potential hallucination.

    Requirement 6.3: Define FlaggedSpan schema for highlighting unsupported
    text
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The flagged text span"
    )
    start_idx: int = Field(
        ...,
        ge=0,
        description="Starting character index in the original response"
    )
    end_idx: int = Field(
        ...,
        ge=0,
        description="Ending character index in the original response"
    )
    grounding_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Grounding score for this span (0-1)"
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Reason why this span was flagged"
    )

    @field_validator('end_idx')
    @classmethod
    def validate_indices(cls, v: int, info) -> int:
        """Ensure end_idx is greater than start_idx."""
        if 'start_idx' in info.data and v <= info.data['start_idx']:
            raise ValueError("end_idx must be greater than start_idx")
        return v

    @field_validator('grounding_score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure grounding score is within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Grounding score must be between 0.0 and 1.0")
        return v


class HallucinationResult(BaseModel):
    """Complete result of hallucination detection analysis.

    Requirement 6.3: Define HallucinationResult schema
    """

    has_hallucinations: bool = Field(
        ...,
        description="Whether hallucinations were detected"
    )
    hallucination_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of unsupported to total factual claims (0-1)"
    )
    total_claims: int = Field(
        ...,
        ge=0,
        description="Total number of claims extracted"
    )
    factual_claims: int = Field(
        ...,
        ge=0,
        description="Number of factual claims (excluding opinions)"
    )
    supported_claims: int = Field(
        ...,
        ge=0,
        description="Number of claims supported by evidence"
    )
    unsupported_claims: int = Field(
        ...,
        ge=0,
        description="Number of claims not supported by evidence"
    )
    flagged_spans: list[FlaggedSpan] = Field(
        default_factory=list,
        description="Text spans flagged as potential hallucinations"
    )
    overall_grounding_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall grounding score for the response (0-1)"
    )
    claim_evidence: list[ClaimEvidence] = Field(
        default_factory=list,
        description="Detailed evidence analysis for each claim"
    )
    sensitivity_level: SensitivityLevel = Field(
        ...,
        description="Sensitivity level used for this analysis"
    )

    @field_validator('hallucination_rate', 'overall_grounding_score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v

    @field_validator('factual_claims')
    @classmethod
    def validate_factual_claims(cls, v: int, info) -> int:
        """Ensure factual claims doesn't exceed total claims."""
        if 'total_claims' in info.data and v > info.data['total_claims']:
            raise ValueError("factual_claims cannot exceed total_claims")
        return v

    @field_validator('supported_claims')
    @classmethod
    def validate_supported_claims(cls, v: int, info) -> int:
        """Ensure supported claims doesn't exceed factual claims."""
        if 'factual_claims' in info.data and v > info.data['factual_claims']:
            raise ValueError("supported_claims cannot exceed factual_claims")
        return v

    @field_validator('unsupported_claims')
    @classmethod
    def validate_unsupported_claims(cls, v: int, info) -> int:
        """Ensure unsupported claims doesn't exceed factual claims."""
        if 'factual_claims' in info.data and v > info.data['factual_claims']:
            raise ValueError("unsupported_claims cannot exceed factual_claims")
        return v


class HallucinationComparison(BaseModel):
    """Comparison of hallucination rates between two models.

    Requirement 6.4: Define HallucinationComparison schema for model comparison
    Requirement 6.8: Calculate hallucination rate reduction between models
    """

    baseline_mean_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean hallucination rate for baseline model"
    )
    comparison_mean_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean hallucination rate for comparison model"
    )
    reduction: float = Field(
        ...,
        description="Absolute reduction in hallucination rate"
    )
    reduction_percent: float = Field(
        ...,
        description="Percentage reduction in hallucination rate"
    )
    statistical_significance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Statistical significance level of the difference"
    )
    p_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="P-value for statistical test"
    )

    @field_validator('baseline_mean_rate', 'comparison_mean_rate')
    @classmethod
    def validate_rate_range(cls, v: float) -> float:
        """Ensure hallucination rates are within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Hallucination rate must be between 0.0 and 1.0")
        return v

    @field_validator('statistical_significance', 'p_value')
    @classmethod
    def validate_significance_range(cls, v: float) -> float:
        """Ensure significance values are within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Significance value must be between 0.0 and 1.0")
        return v


# Re-export HallucinationAnalysis from model_response for backward compatibility
from .model_response import HallucinationAnalysis  # noqa: F401
