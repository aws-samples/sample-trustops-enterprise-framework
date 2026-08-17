"""
Data models for the Evaluation Engine.

This module defines the core data models for baseline and comparative model
evaluations, trust scoring, and deployment recommendations in the TrustOps
Enterprise Framework.

Requirements: 3.1, 3.2, 3.3, 3.5, 7.1, 7.2, 7.4
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .model import InferenceRequest


class EvaluationConfig(BaseModel):
    """Configuration for running a model evaluation.

    Requirement 3.1: Define EvaluationConfig pydantic schema
    """

    model_id: str = Field(
        ...,
        min_length=1,
        description="ID of the model to evaluate"
    )
    dataset_id: str = Field(
        ...,
        min_length=1,
        description="ID of the dataset to use for evaluation"
    )
    inference_params: InferenceRequest = Field(
        ...,
        description="Inference parameters for model invocation"
    )
    trust_score_weights: Optional["TrustScoreWeights"] = Field(
        default=None,
        description="Weights for trust score dimensions"
    )
    batch_size: int = Field(
        default=10,
        gt=0,
        description="Number of examples to process in each batch"
    )
    timeout_per_request: float = Field(
        default=60.0,
        gt=0,
        description="Timeout in seconds for each inference request"
    )
    concurrency: int = Field(
        default=5,
        gt=0,
        description="Number of concurrent inference requests"
    )


class TrustScoreWeights(BaseModel):
    """Weights for trust score dimensions.

    Used in EvaluationConfig for configurable trust scoring.
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

    @field_validator('accuracy', 'consistency', 'safety', 'bias', 'context_grounding')
    @classmethod
    def validate_weight_range(cls, v: float) -> float:
        """Ensure weights are within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v


class TrustScoreResult(BaseModel):
    """Result of trust score calculation.

    Placeholder for trust scoring integration.
    """

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall trust score (0-1)"
    )
    dimension_scores: dict = Field(
        default_factory=dict,
        description="Scores for each trust dimension"
    )
    confidence_level: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the trust score"
    )
    flagged_for_review: bool = Field(
        default=False,
        description="Whether this response is flagged for human review"
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the trust score"
    )


class HallucinationResult(BaseModel):
    """Result of hallucination detection.

    Placeholder for hallucination detection integration.
    """

    has_hallucinations: bool = Field(
        default=False,
        description="Whether hallucinations were detected"
    )
    hallucination_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of unsupported to total factual claims"
    )
    flagged_spans: list = Field(
        default_factory=list,
        description="Text spans flagged as potential hallucinations"
    )
    overall_grounding_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall grounding score for the response"
    )


class EvaluationResult(BaseModel):
    """Result of evaluating a single example.

    Requirement 3.2: Define EvaluationResult schema
    """

    example_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the example"
    )
    prompt: str = Field(
        ...,
        description="The input prompt"
    )
    expected_response: Optional[str] = Field(
        default=None,
        description="Expected response if available"
    )
    actual_response: str = Field(
        ...,
        description="Actual response from the model"
    )
    model_id: str = Field(
        ...,
        min_length=1,
        description="ID of the model that generated the response"
    )
    trust_score: TrustScoreResult = Field(
        ...,
        description="Trust score for this response"
    )
    hallucination_analysis: HallucinationResult = Field(
        ...,
        description="Hallucination detection results"
    )
    latency_ms: float = Field(
        ...,
        ge=0,
        description="Response latency in milliseconds"
    )
    input_tokens: int = Field(
        ...,
        ge=0,
        description="Number of input tokens"
    )
    output_tokens: int = Field(
        ...,
        ge=0,
        description="Number of output tokens"
    )
    cost: float = Field(
        ...,
        ge=0,
        description="Cost of this inference"
    )
    category: Optional[str] = Field(
        default=None,
        description="Category of the example"
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp when the evaluation was performed"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if evaluation failed"
    )


class AggregateMetrics(BaseModel):
    """Aggregate metrics for a set of evaluation results.

    Requirement 3.3: Define AggregateMetrics schema
    """

    total_examples: int = Field(
        ...,
        ge=0,
        description="Total number of examples evaluated"
    )
    successful_examples: int = Field(
        ...,
        ge=0,
        description="Number of successfully evaluated examples"
    )
    failed_examples: int = Field(
        ...,
        ge=0,
        description="Number of failed examples"
    )
    mean_trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean trust score across all examples"
    )
    median_trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Median trust score"
    )
    trust_score_std: float = Field(
        ...,
        ge=0.0,
        description="Standard deviation of trust scores"
    )
    mean_hallucination_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean hallucination rate"
    )
    latency_p50_ms: float = Field(
        ...,
        ge=0,
        description="50th percentile latency in milliseconds"
    )
    latency_p95_ms: float = Field(
        ...,
        ge=0,
        description="95th percentile latency in milliseconds"
    )
    latency_p99_ms: float = Field(
        ...,
        ge=0,
        description="99th percentile latency in milliseconds"
    )
    total_input_tokens: int = Field(
        ...,
        ge=0,
        description="Total input tokens across all examples"
    )
    total_output_tokens: int = Field(
        ...,
        ge=0,
        description="Total output tokens across all examples"
    )
    total_cost: float = Field(
        ...,
        ge=0,
        description="Total cost for all inferences"
    )
    cost_per_query: float = Field(
        ...,
        ge=0,
        description="Average cost per query"
    )


class CostSummary(BaseModel):
    """Summary of costs for an evaluation.

    Requirement 3.5: Define CostSummary schema
    """

    total_cost: float = Field(
        ...,
        ge=0,
        description="Total cost for the evaluation"
    )
    inference_cost: float = Field(
        ...,
        ge=0,
        description="Cost of model inference"
    )
    storage_cost: float = Field(
        ...,
        ge=0,
        description="Cost of storing results"
    )
    cost_per_example: float = Field(
        ...,
        ge=0,
        description="Average cost per example"
    )
    currency: str = Field(
        default="USD",
        description="Currency for cost values"
    )


class BaselineEvaluationReport(BaseModel):
    """Report for a baseline model evaluation.

    Requirement 3.3: Define BaselineEvaluationReport schema
    """

    evaluation_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this evaluation"
    )
    model_id: str = Field(
        ...,
        min_length=1,
        description="ID of the evaluated model"
    )
    dataset_id: str = Field(
        ...,
        min_length=1,
        description="ID of the dataset used"
    )
    config: EvaluationConfig = Field(
        ...,
        description="Configuration used for this evaluation"
    )
    total_examples: int = Field(
        ...,
        ge=0,
        description="Total number of examples in the evaluation"
    )
    aggregate_metrics: AggregateMetrics = Field(
        ...,
        description="Aggregate metrics across all examples"
    )
    per_category_metrics: dict[str, AggregateMetrics] = Field(
        default_factory=dict,
        description="Metrics broken down by category"
    )
    cost_summary: CostSummary = Field(
        ...,
        description="Cost summary for the evaluation"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when evaluation was created"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when evaluation completed"
    )
    status: str = Field(
        ...,
        description="Status of the evaluation (running, completed, failed)"
    )
    s3_results_uri: str = Field(
        ...,
        min_length=1,
        description="S3 URI where detailed results are stored"
    )

    @field_validator('s3_results_uri')
    @classmethod
    def validate_s3_uri(cls, v: str) -> str:
        """Ensure s3_results_uri has valid S3 URI format."""
        if not v.startswith('s3://'):
            raise ValueError("s3_results_uri must start with 's3://'")
        return v


class DeploymentRecommendation(str, Enum):
    """Recommendation for model deployment.

    Requirement 7.1: Define DeploymentRecommendation enum
    """

    DEPLOY = "deploy"
    ITERATE = "iterate"
    REJECT = "reject"


class ComparisonConfig(BaseModel):
    """Configuration for comparing two models.

    Requirement 7.1: Define ComparisonConfig schema
    """

    model_id_1: str = Field(
        ...,
        min_length=1,
        description="ID of the first model (baseline)"
    )
    model_id_2: str = Field(
        ...,
        min_length=1,
        description="ID of the second model (fine-tuned or comparison)"
    )
    dataset_id: str = Field(
        ...,
        min_length=1,
        description="ID of the dataset to use for comparison"
    )
    inference_params: InferenceRequest = Field(
        ...,
        description="Inference parameters for model invocation"
    )
    trust_score_config: Optional[TrustScoreWeights] = Field(
        default=None,
        description="Trust score configuration"
    )
    deploy_thresholds: "DeploymentThresholds" = Field(
        ...,
        description="Thresholds for deployment recommendation"
    )


class DeploymentThresholds(BaseModel):
    """Thresholds for deployment recommendation.

    Requirement 7.1: Define DeploymentThresholds schema
    """

    min_trust_score_improvement: float = Field(
        default=0.05,
        ge=0.0,
        description="Minimum trust score improvement required"
    )
    max_cost_increase_percent: float = Field(
        default=20.0,
        ge=0.0,
        description="Maximum acceptable cost increase percentage"
    )
    min_hallucination_reduction: float = Field(
        default=0.1,
        ge=0.0,
        description="Minimum hallucination rate reduction required"
    )
    min_statistical_significance: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum statistical significance level"
    )


class ImprovementMetrics(BaseModel):
    """Metrics comparing two models.

    Requirement 7.2: Define ImprovementMetrics schema
    """

    trust_score_delta: float = Field(
        ...,
        description="Absolute change in trust score"
    )
    trust_score_delta_percent: float = Field(
        ...,
        description="Percentage change in trust score"
    )
    hallucination_reduction: float = Field(
        ...,
        description="Absolute reduction in hallucination rate"
    )
    hallucination_reduction_percent: float = Field(
        ...,
        description="Percentage reduction in hallucination rate"
    )
    latency_delta_ms: float = Field(
        ...,
        description="Change in latency in milliseconds"
    )
    latency_delta_percent: float = Field(
        ...,
        description="Percentage change in latency"
    )
    cost_delta_per_query: float = Field(
        ...,
        description="Change in cost per query"
    )
    cost_delta_percent: float = Field(
        ...,
        description="Percentage change in cost"
    )
    statistical_significance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Statistical significance level"
    )
    p_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="P-value for statistical test"
    )
    confidence_interval: tuple[float, float] = Field(
        ...,
        description="Confidence interval for the improvement"
    )


class CostPerformanceAnalysis(BaseModel):
    """Analysis of cost vs performance trade-offs.

    Requirement 7.4: Define CostPerformanceAnalysis schema
    """

    model_1_cost_per_trust_point: float = Field(
        ...,
        ge=0,
        description="Cost per trust score point for model 1"
    )
    model_2_cost_per_trust_point: float = Field(
        ...,
        ge=0,
        description="Cost per trust score point for model 2"
    )
    quality_gain_justifies_cost: bool = Field(
        ...,
        description="Whether quality improvement justifies cost increase"
    )
    break_even_volume: Optional[int] = Field(
        default=None,
        ge=0,
        description="Query volume at which cost difference breaks even"
    )


class ComparisonReport(BaseModel):
    """Report comparing two models.

    Requirement 7.2: Define ComparisonReport schema
    """

    comparison_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this comparison"
    )
    model_1_id: str = Field(
        ...,
        min_length=1,
        description="ID of the first model"
    )
    model_2_id: str = Field(
        ...,
        min_length=1,
        description="ID of the second model"
    )
    dataset_id: str = Field(
        ...,
        min_length=1,
        description="ID of the dataset used"
    )
    model_1_metrics: AggregateMetrics = Field(
        ...,
        description="Aggregate metrics for model 1"
    )
    model_2_metrics: AggregateMetrics = Field(
        ...,
        description="Aggregate metrics for model 2"
    )
    improvement_metrics: ImprovementMetrics = Field(
        ...,
        description="Metrics showing improvement from model 1 to model 2"
    )
    recommendation: DeploymentRecommendation = Field(
        ...,
        description="Deployment recommendation"
    )
    recommendation_justification: str = Field(
        ...,
        description="Explanation for the recommendation"
    )
    per_category_breakdown: dict[str, ImprovementMetrics] = Field(
        default_factory=dict,
        description="Improvement metrics broken down by category"
    )
    cost_performance_analysis: CostPerformanceAnalysis = Field(
        ...,
        description="Analysis of cost vs performance trade-offs"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when comparison was created"
    )
    s3_report_uri: str = Field(
        ...,
        min_length=1,
        description="S3 URI where detailed comparison report is stored"
    )

    @field_validator('s3_report_uri')
    @classmethod
    def validate_s3_uri(cls, v: str) -> str:
        """Ensure s3_report_uri has valid S3 URI format."""
        if not v.startswith('s3://'):
            raise ValueError("s3_report_uri must start with 's3://'")
        return v


# --- Legacy dataclass models for backward compatibility ---
from dataclasses import dataclass, field as dc_field
import json as _json


@dataclass
class EvaluationExample:
    """A single evaluation example."""
    prompt: str
    expected_response: str = ""
    source_documents: list = dc_field(default_factory=list)
    category: str = "general"
    metadata: dict = dc_field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'prompt': self.prompt,
            'expected_response': self.expected_response,
            'source_documents': self.source_documents,
            'category': self.category,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EvaluationExample':
        return cls(
            prompt=data['prompt'],
            expected_response=data.get('expected_response', ''),
            source_documents=data.get('source_documents', []),
            category=data.get('category', 'general'),
            metadata=data.get('metadata', {}),
        )

    def to_json(self) -> str:
        return _json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'EvaluationExample':
        return cls.from_dict(_json.loads(json_str))


@dataclass
class EvaluationDataset:
    """A dataset of evaluation examples."""
    dataset_id: str
    name: str
    description: str = ""
    examples: list = dc_field(default_factory=list)
    created_at: Optional[datetime] = None
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            'dataset_id': self.dataset_id,
            'name': self.name,
            'description': self.description,
            'examples': [
                ex.to_dict() if hasattr(ex, 'to_dict') else ex
                for ex in self.examples
            ],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'version': self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EvaluationDataset':
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        examples = [
            EvaluationExample.from_dict(ex) if isinstance(ex, dict) else ex
            for ex in data.get('examples', [])
        ]
        return cls(
            dataset_id=data.get('dataset_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            examples=examples,
            created_at=created_at,
            version=data.get('version', '1.0'),
        )

    def to_json(self) -> str:
        return _json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'EvaluationDataset':
        return cls.from_dict(_json.loads(json_str))
