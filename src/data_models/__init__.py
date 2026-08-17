"""
TrustOps data models for evaluation, trust scoring, and workflow management.
"""

from .evaluation import (
    EvaluationConfig,
    TrustScoreWeights,
    TrustScoreResult,
    HallucinationResult,
    AggregateMetrics,
    CostSummary,
    BaselineEvaluationReport,
    DeploymentRecommendation,
    ComparisonConfig,
    DeploymentThresholds,
    CostPerformanceAnalysis,
    ComparisonReport,
    EvaluationExample,
    EvaluationDataset,
)
from .evaluation import ImprovementMetrics as _PydanticImprovementMetrics  # noqa: F401
from .evaluation import EvaluationResult as _PydanticEvaluationResult  # noqa: F401
from .model import (
    ModelProvider,
    ModelCapability,
    ModelStatus,
    ModelMetadata,
    ModelPricing,
    InferenceRequest,
    InferenceResponse
)
from .dataset import (
    DatasetFormat,
    DatasetTaskType,
    DatasetMetadata,
    TokenStatistics,
    DatasetLineage,
    DatasetQualityReport,
    QualityIssue
)
from .workflow import (
    DataValidationResult,
    ValidationError,
    FineTuningJob,
)
from .workflow import WorkflowManifest as _PydanticWorkflowManifest  # noqa: F401
from .workflow_legacy import WorkflowManifest
from .data_validator import DataValidator
from .model_response import (
    ModelResponse,
    TrustScore,
    TrustScoreComponents,
    HallucinationSpan,
    HallucinationAnalysis,
)
from .results import (
    BaselineMetrics,
    CostPerformanceMetrics,
    ImprovementMetrics,
    EvaluationResult as _DataclassEvaluationResult,
)
# Use the dataclass EvaluationResult for backward compatibility
EvaluationResult = _DataclassEvaluationResult

__all__ = [
    # Model Registry models
    'ModelProvider',
    'ModelCapability',
    'ModelStatus',
    'ModelMetadata',
    'ModelPricing',
    'InferenceRequest',
    'InferenceResponse',

    # Dataset models
    'DatasetFormat',
    'DatasetTaskType',
    'DatasetMetadata',
    'TokenStatistics',
    'DatasetLineage',
    'DatasetQualityReport',
    'QualityIssue',

    # Evaluation models
    'EvaluationConfig',
    'TrustScoreWeights',
    'TrustScoreResult',
    'HallucinationResult',
    'EvaluationResult',
    'AggregateMetrics',
    'CostSummary',
    'BaselineEvaluationReport',
    'DeploymentRecommendation',
    'ComparisonConfig',
    'DeploymentThresholds',
    'ImprovementMetrics',
    'CostPerformanceAnalysis',
    'ComparisonReport',
    'EvaluationExample',
    'EvaluationDataset',

    # Workflow models
    'DataValidationResult',
    'ValidationError',
    'FineTuningJob',
    'WorkflowManifest',

    # Model response models
    'ModelResponse',
    'TrustScore',
    'TrustScoreComponents',
    'HallucinationSpan',
    'HallucinationAnalysis',

    # Result models
    'BaselineMetrics',
    'CostPerformanceMetrics',
]
