"""
Data models for model responses and trust scores.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any
import json


@dataclass
class ModelResponse:
    """Response from a model inference."""
    response_id: str
    model_id: str
    prompt: str
    response_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'response_id': self.response_id,
            'model_id': self.model_id,
            'prompt': self.prompt,
            'response_text': self.response_text,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'latency_ms': self.latency_ms,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelResponse':
        """Create instance from dictionary."""
        return cls(
            response_id=data['response_id'],
            model_id=data['model_id'],
            prompt=data['prompt'],
            response_text=data['response_text'],
            input_tokens=data['input_tokens'],
            output_tokens=data['output_tokens'],
            latency_ms=data['latency_ms'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'ModelResponse':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TrustScoreComponents:
    """Breakdown of trust score components."""
    context_grounding: float  # 0-1
    output_structure: float   # 0-1
    uncertainty_indicators: float  # 0-1
    factual_consistency: float  # 0-1
    response_completeness: float  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrustScoreComponents':
        """Create instance from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'TrustScoreComponents':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TrustScore:
    """Trust score for a model response."""
    overall_score: float  # 0-1
    components: TrustScoreComponents
    confidence_level: str  # "high", "medium", "low"
    flagged_for_review: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'overall_score': self.overall_score,
            'components': self.components.to_dict(),
            'confidence_level': self.confidence_level,
            'flagged_for_review': self.flagged_for_review,
            'explanation': self.explanation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrustScore':
        """Create instance from dictionary."""
        return cls(
            overall_score=data['overall_score'],
            components=TrustScoreComponents.from_dict(data['components']),
            confidence_level=data['confidence_level'],
            flagged_for_review=data['flagged_for_review'],
            explanation=data['explanation']
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'TrustScore':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class HallucinationSpan:
    """Flagged text span with low grounding score."""
    text: str
    start_idx: int
    end_idx: int
    grounding_score: float
    evidence_documents: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HallucinationSpan':
        """Create instance from dictionary."""
        evidence = data.get('evidence_documents', [])
        # Convert lists back to tuples for evidence documents
        evidence = [tuple(e) if isinstance(e, list) else e for e in evidence]
        return cls(
            text=data['text'],
            start_idx=data['start_idx'],
            end_idx=data['end_idx'],
            grounding_score=data['grounding_score'],
            evidence_documents=evidence,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'HallucinationSpan':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class HallucinationAnalysis:
    """Analysis of potential hallucinations in model response."""
    has_hallucinations: bool
    hallucination_rate: float  # 0-1
    flagged_spans: list = field(default_factory=list)
    overall_grounding_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'has_hallucinations': self.has_hallucinations,
            'hallucination_rate': self.hallucination_rate,
            'flagged_spans': [span.to_dict() if hasattr(span, 'to_dict') else span for span in self.flagged_spans],
            'overall_grounding_score': self.overall_grounding_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HallucinationAnalysis':
        """Create instance from dictionary."""
        return cls(
            has_hallucinations=data['has_hallucinations'],
            hallucination_rate=data['hallucination_rate'],
            flagged_spans=[HallucinationSpan.from_dict(span) if isinstance(span, dict) else span for span in data.get('flagged_spans', [])],
            overall_grounding_score=data.get('overall_grounding_score', 0.0)
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'HallucinationAnalysis':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
