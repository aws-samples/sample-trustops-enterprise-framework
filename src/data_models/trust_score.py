"""
Data models for the Trust Scoring Engine.

This module defines the core data models for multi-dimensional trust scoring,
including trust dimensions, scoring configuration, and detailed results in the
TrustOps Enterprise Framework.

Requirements: 5.1, 5.2, 5.3, 5.6
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class TrustDimension(str, Enum):
    """Enumeration of trust scoring dimensions.

    Requirement 5.1: Define trust score dimensions
    """

    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    SAFETY = "safety"
    BIAS = "bias"
    CONTEXT_GROUNDING = "context_grounding"


class TrustScoreWeights(BaseModel):
    """Configurable weights for trust score dimensions.

    Requirement 5.1: Define TrustScoreWeights schema with configurable weights
    """

    accuracy: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for accuracy dimension"
    )
    consistency: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for consistency dimension"
    )
    safety: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for safety dimension"
    )
    bias: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for bias dimension"
    )
    context_grounding: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for context grounding dimension"
    )

    @field_validator('accuracy', 'consistency', 'safety', 'bias',
                     'context_grounding')
    @classmethod
    def validate_weight_range(cls, v: float) -> float:
        """Ensure weights are within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v

    def validate_weights(self) -> bool:
        """Ensure weights sum to approximately 1.0.

        Requirement 5.1: Weights must sum to 1.0 for proper combination
        """
        total = (self.accuracy + self.consistency + self.safety +
                 self.bias + self.context_grounding)
        return abs(total - 1.0) < 0.001


class CustomMetricConfig(BaseModel):
    """Configuration for custom user-defined trust metrics.

    Requirement 5.6: Support custom trust metrics
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Name of the custom metric"
    )
    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weight for this custom metric in overall score"
    )
    scorer_class: str = Field(
        ...,
        min_length=1,
        description="Fully qualified class name of the scorer implementation"
    )
    config: dict = Field(
        default_factory=dict,
        description="Configuration parameters for the custom scorer"
    )

    @field_validator('weight')
    @classmethod
    def validate_weight_range(cls, v: float) -> float:
        """Ensure weight is within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v


class TrustScoreConfig(BaseModel):
    """Configuration for trust score calculation.

    Requirement 5.1: Define TrustScoreConfig schema
    """

    weights: TrustScoreWeights = Field(
        default_factory=TrustScoreWeights,
        description="Weights for each trust dimension"
    )
    review_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold below which responses are flagged for review"
    )
    consistency_samples: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of samples to use for consistency scoring"
    )
    custom_metrics: list[CustomMetricConfig] = Field(
        default_factory=list,
        description="List of custom metrics to include in scoring"
    )

    @field_validator('review_threshold')
    @classmethod
    def validate_threshold_range(cls, v: float) -> float:
        """Ensure review threshold is within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Review threshold must be between 0.0 and 1.0")
        return v

    @field_validator('consistency_samples')
    @classmethod
    def validate_consistency_samples(cls, v: int) -> int:
        """Ensure consistency samples is within reasonable range."""
        if v < 1 or v > 10:
            raise ValueError("Consistency samples must be between 1 and 10")
        return v


class DimensionScore(BaseModel):
    """Score for a single trust dimension.

    Requirement 5.2: Define DimensionScore schema
    """

    dimension: TrustDimension = Field(
        ...,
        description="The trust dimension being scored"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score for this dimension (0-1)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this score (0-1)"
    )
    details: dict = Field(
        default_factory=dict,
        description="Detailed information about the scoring"
    )
    checks_passed: list[str] = Field(
        default_factory=list,
        description="List of checks that passed for this dimension"
    )
    checks_failed: list[str] = Field(
        default_factory=list,
        description="List of checks that failed for this dimension"
    )

    @field_validator('score', 'confidence')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v


class TrustScoreResult(BaseModel):
    """Complete result of trust score calculation.

    Requirement 5.3: Define TrustScoreResult schema
    """

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall weighted trust score (0-1)"
    )
    dimension_scores: dict[TrustDimension, DimensionScore] = Field(
        default_factory=dict,
        description="Scores for each trust dimension"
    )
    confidence_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the trust score (0-1)"
    )
    flagged_for_review: bool = Field(
        default=False,
        description="Whether this response is flagged for human review"
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the trust score"
    )
    component_details: list[dict] = Field(
        default_factory=list,
        description="Detailed breakdown of scoring components"
    )

    @field_validator('overall_score', 'confidence_level')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v
