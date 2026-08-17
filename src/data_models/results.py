"""
Data models for evaluation results and metrics.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json

from .model_response import ModelResponse, TrustScore
from .model_response import HallucinationAnalysis


@dataclass
class EvaluationResult:
    """Result of evaluating a single example."""
    example_id: str
    model_response: ModelResponse
    trust_score: TrustScore
    hallucination_analysis: HallucinationAnalysis
    semantic_similarity: Optional[float]
    category: str
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'example_id': self.example_id,
            'model_response': self.model_response.to_dict(),
            'trust_score': self.trust_score.to_dict(),
            'hallucination_analysis': self.hallucination_analysis.to_dict(),
            'semantic_similarity': self.semantic_similarity,
            'category': self.category,
            'passed': self.passed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create instance from dictionary."""
        return cls(
            example_id=data['example_id'],
            model_response=ModelResponse.from_dict(data['model_response']),
            trust_score=TrustScore.from_dict(data['trust_score']),
            hallucination_analysis=HallucinationAnalysis.from_dict(
                data['hallucination_analysis']
            ),
            semantic_similarity=data.get('semantic_similarity'),
            category=data['category'],
            passed=data['passed']
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'EvaluationResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class BaselineMetrics:
    """Aggregate metrics from baseline evaluation."""
    model_id: str
    total_examples: int
    mean_trust_score: float
    median_trust_score: float
    trust_score_distribution: Dict[str, int]  # "high", "medium", "low" counts
    mean_latency_ms: float
    p95_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    hallucination_rate: float
    category_breakdown: Dict[str, float] = field(
        default_factory=dict
    )  # Category -> mean trust score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineMetrics':
        """Create instance from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'BaselineMetrics':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ImprovementMetrics:
    """Metrics comparing baseline and fine-tuned models."""
    baseline_model_id: str
    finetuned_model_id: str
    trust_score_improvement: float  # Percentage point change
    hallucination_reduction: float  # Percentage point change
    latency_delta_ms: float
    cost_delta_per_query: float
    cost_delta_percentage: float
    statistical_significance: bool
    recommendation: str  # "deploy", "iterate", "reject"
    justification: str
    p_value_ttest: float = 1.0
    p_value_wilcoxon: float = 1.0
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    test_type_used: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImprovementMetrics':
        """Create instance from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'ImprovementMetrics':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class CostPerformanceMetrics:
    """Cost-performance analysis metrics."""
    total_cost: float
    cost_per_query: float
    cost_per_high_trust_response: float
    cost_per_token: float
    mean_trust_score: float
    cost_efficiency_score: float  # Trust score per dollar
    projected_monthly_cost: Dict[int, float] = field(
        default_factory=dict
    )  # Query volume -> cost
    break_even_volume: Optional[int] = None
    remediation_cost_per_hallucination: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert integer keys to strings for JSON compatibility
        result['projected_monthly_cost'] = {
            str(k): v for k, v in self.projected_monthly_cost.items()
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CostPerformanceMetrics':
        """Create instance from dictionary."""
        # Convert string keys back to integers
        projected_monthly_cost = {
            int(k): v for k, v in data.get('projected_monthly_cost', {}).items()
        }
        data_copy = data.copy()
        data_copy['projected_monthly_cost'] = projected_monthly_cost
        # Handle optional fields with defaults
        data_copy.setdefault('break_even_volume', None)
        data_copy.setdefault('remediation_cost_per_hallucination', 50.0)
        return cls(**data_copy)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'CostPerformanceMetrics':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class BaselineEvaluationResult:
    """Result of a complete baseline evaluation."""
    workflow_id: str
    model_id: str
    evaluation_results: list[EvaluationResult]
    metrics: BaselineMetrics
    dataset_s3_uri: str
    results_s3_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'workflow_id': self.workflow_id,
            'model_id': self.model_id,
            'evaluation_results': [
                result.to_dict() for result in self.evaluation_results
            ],
            'metrics': self.metrics.to_dict(),
            'dataset_s3_uri': self.dataset_s3_uri,
            'results_s3_uri': self.results_s3_uri
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineEvaluationResult':
        """Create instance from dictionary."""
        return cls(
            workflow_id=data['workflow_id'],
            model_id=data['model_id'],
            evaluation_results=[
                EvaluationResult.from_dict(result)
                for result in data['evaluation_results']
            ],
            metrics=BaselineMetrics.from_dict(data['metrics']),
            dataset_s3_uri=data['dataset_s3_uri'],
            results_s3_uri=data.get('results_s3_uri')
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'BaselineEvaluationResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ComparativeEvaluationResult:
    """Result of a comparative evaluation between two models."""
    workflow_id: str
    baseline_model_id: str
    finetuned_model_id: str
    baseline_results: list[EvaluationResult]
    finetuned_results: list[EvaluationResult]
    baseline_metrics: BaselineMetrics
    finetuned_metrics: BaselineMetrics
    improvement_metrics: ImprovementMetrics
    dataset_s3_uri: str
    results_s3_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'workflow_id': self.workflow_id,
            'baseline_model_id': self.baseline_model_id,
            'finetuned_model_id': self.finetuned_model_id,
            'baseline_results': [
                result.to_dict() for result in self.baseline_results
            ],
            'finetuned_results': [
                result.to_dict() for result in self.finetuned_results
            ],
            'baseline_metrics': self.baseline_metrics.to_dict(),
            'finetuned_metrics': self.finetuned_metrics.to_dict(),
            'improvement_metrics': self.improvement_metrics.to_dict(),
            'dataset_s3_uri': self.dataset_s3_uri,
            'results_s3_uri': self.results_s3_uri
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComparativeEvaluationResult':
        """Create instance from dictionary."""
        return cls(
            workflow_id=data['workflow_id'],
            baseline_model_id=data['baseline_model_id'],
            finetuned_model_id=data['finetuned_model_id'],
            baseline_results=[
                EvaluationResult.from_dict(result)
                for result in data['baseline_results']
            ],
            finetuned_results=[
                EvaluationResult.from_dict(result)
                for result in data['finetuned_results']
            ],
            baseline_metrics=BaselineMetrics.from_dict(data['baseline_metrics']),
            finetuned_metrics=BaselineMetrics.from_dict(data['finetuned_metrics']),
            improvement_metrics=ImprovementMetrics.from_dict(data['improvement_metrics']),
            dataset_s3_uri=data['dataset_s3_uri'],
            results_s3_uri=data.get('results_s3_uri')
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'ComparativeEvaluationResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
