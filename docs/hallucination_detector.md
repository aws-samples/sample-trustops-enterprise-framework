# Hallucination Detector

## Overview

The HallucinationDetector is the main orchestrator for hallucination detection in the TrustOps Enterprise Framework. It integrates claim extraction, evidence search, claim classification, and rate calculation into a unified pipeline for measuring unsupported claims in model outputs.

**Requirements**:
- 6.1 - Extract claims from responses
- 6.2 - Search for supporting evidence
- 6.3 - Classify claims as supported/unsupported
- 6.4 - Calculate hallucination rate
- 6.5 - Identify flagged text spans
- 6.7 - Aggregate hallucination metrics
- 6.8 - Support configurable sensitivity levels
- 6.10 - Store detailed results
- 6.12 - Alert when hallucination rate exceeds threshold

## Features

- **Hallucination Rate Calculation**: Computes ratio of unsupported to total factual claims
- **Claim Extraction**: Splits responses into individual claims via ClaimExtractor
- **Claim Classification**: Classifies claims as supported/unsupported via ClaimClassifier
- **Opinion Filtering**: Excludes opinions and hedged statements from rate calculation
- **Edge Case Handling**: Returns 0.0 when no factual claims exist
- **Configurable Sensitivity**: Supports strict, moderate, and lenient detection modes
- **Aggregate Metrics**: Groups results by category, model, or time period with statistical summaries
- **Alert Trigger**: Fires alerts when hallucination rate exceeds a configurable threshold

## Installation

```bash
# No additional dependencies required beyond base TrustOps installation
pip install -r requirements.txt
```

## Usage

### Hallucination Rate Calculation

The core feature of the HallucinationDetector is computing the hallucination rate from classified claim evidence:

```python
from src.hallucination.hallucination_detector import HallucinationDetector
from src.data_models.hallucination import (
    Claim, ClaimEvidence, EvidenceSource
)

detector = HallucinationDetector()

# Claim evidence from the classification pipeline
claim_evidence = [
    ClaimEvidence(
        claim=Claim(
            text="The Earth orbits the Sun.",
            start_idx=0, end_idx=27,
            is_factual=True, is_opinion=False
        ),
        is_supported=True,
        similarity_score=0.9,
        evidence_sources=[]
    ),
    ClaimEvidence(
        claim=Claim(
            text="Jupiter has 200 moons.",
            start_idx=28, end_idx=50,
            is_factual=True, is_opinion=False
        ),
        is_supported=False,
        similarity_score=0.3,
        evidence_sources=[]
    )
]

rate = detector.calculate_hallucination_rate(claim_evidence)
print(f"Hallucination Rate: {rate:.2%}")  # 50.00%
```

### Edge Cases

```python
# No factual claims (all opinions) → rate = 0.0
opinion_evidence = [
    ClaimEvidence(
        claim=Claim(
            text="I think this is good.",
            start_idx=0, end_idx=21,
            is_factual=False, is_opinion=True
        ),
        is_supported=False,
        similarity_score=0.1,
        evidence_sources=[]
    )
]
rate = detector.calculate_hallucination_rate(opinion_evidence)
print(f"Rate (opinions only): {rate}")  # 0.0

# Empty list → rate = 0.0
rate = detector.calculate_hallucination_rate([])
print(f"Rate (empty): {rate}")  # 0.0
```

### Custom Configuration

```python
from src.data_models.hallucination import (
    HallucinationConfig, SensitivityLevel
)

config = HallucinationConfig(
    sensitivity_level=SensitivityLevel.STRICT,
    similarity_threshold=0.8
)
detector = HallucinationDetector(config)
```

### Sensitivity Level Configuration

The detector supports three sensitivity levels with corresponding thresholds:
- **STRICT** (0.8): High threshold — flags more claims as unsupported
- **MODERATE** (0.6): Balanced threshold (default)
- **LENIENT** (0.4): Low threshold — only flags clearly unsupported claims

```python
from src.hallucination.hallucination_detector import HallucinationDetector
from src.data_models.hallucination import SensitivityLevel

detector = HallucinationDetector()

# Update sensitivity level after initialization
detector.set_sensitivity_level(SensitivityLevel.STRICT)

# Query current sensitivity configuration
config = detector.get_sensitivity_config()
print(config)
# {'sensitivity_level': <SensitivityLevel.STRICT: 'strict'>,
#  'threshold': 0.8,
#  'all_thresholds': {'strict': 0.8, 'moderate': 0.6, 'lenient': 0.4}}

# Per-run sensitivity override (does not change detector's config)
result = detector.detect(
    "The sky is blue.",
    sensitivity_level=SensitivityLevel.LENIENT
)
print(result.sensitivity_level)  # SensitivityLevel.LENIENT
print(detector.get_sensitivity_config()["sensitivity_level"])  # still STRICT
```

### Text Span Highlighting

The `generate_flagged_spans` method maps unsupported factual claims to their character positions in the original response. Each flagged span includes a grounding score and a reason for flagging.

```python
from src.hallucination.hallucination_detector import HallucinationDetector
from src.data_models.hallucination import (
    Claim, ClaimEvidence, EvidenceSource
)

detector = HallucinationDetector()
response = "The Earth orbits the Sun. Jupiter has 200 moons."

claim_evidence = [
    ClaimEvidence(
        claim=Claim(
            text="The Earth orbits the Sun.",
            start_idx=0, end_idx=25,
            is_factual=True, is_opinion=False
        ),
        is_supported=True,
        similarity_score=0.95,
        evidence_sources=[]
    ),
    ClaimEvidence(
        claim=Claim(
            text="Jupiter has 200 moons.",
            start_idx=26, end_idx=48,
            is_factual=True, is_opinion=False
        ),
        is_supported=False,
        similarity_score=0.25,
        evidence_sources=[
            EvidenceSource(
                document_id="doc1",
                span="Jupiter has 95 known moons",
                similarity_score=0.25
            )
        ]
    )
]

spans = detector.generate_flagged_spans(claim_evidence, response)
for span in spans:
    print(f"[{span.start_idx}:{span.end_idx}] {span.text}")
    print(f"  Score: {span.grounding_score:.2f}, Reason: {span.reason}")
# Output:
# [26:48] Jupiter has 200 moons.
#   Score: 0.25, Reason: Low similarity score: 0.25
```

Flagging rules:
- Only unsupported factual claims are flagged (opinions excluded)
- Claims with no evidence get reason: "No supporting evidence found"
- Claims with low evidence get reason: "Low similarity score: X.XX"
- Claims with out-of-bounds indices are skipped

### Hallucination Rate Comparison

Compare hallucination rates between a baseline and fine-tuned model to measure improvement. The method calculates absolute reduction, percentage reduction, and statistical significance using a two-proportion z-test.

```python
from src.hallucination.hallucination_detector import HallucinationDetector
from src.data_models.hallucination import (
    HallucinationResult, SensitivityLevel
)

detector = HallucinationDetector()

# Results from evaluating each model
baseline_result = HallucinationResult(
    has_hallucinations=True,
    hallucination_rate=0.4,
    total_claims=10,
    factual_claims=10,
    supported_claims=6,
    unsupported_claims=4,
    flagged_spans=[],
    overall_grounding_score=0.6,
    claim_evidence=[],
    sensitivity_level=SensitivityLevel.MODERATE,
)

finetuned_result = HallucinationResult(
    has_hallucinations=True,
    hallucination_rate=0.1,
    total_claims=10,
    factual_claims=10,
    supported_claims=9,
    unsupported_claims=1,
    flagged_spans=[],
    overall_grounding_score=0.9,
    claim_evidence=[],
    sensitivity_level=SensitivityLevel.MODERATE,
)

comparison = detector.compare_hallucination_rates(
    baseline_result, finetuned_result
)

print(f"Baseline rate: {comparison.baseline_mean_rate:.2%}")
print(f"Fine-tuned rate: {comparison.comparison_mean_rate:.2%}")
print(f"Absolute reduction: {comparison.reduction:.2%}")
print(f"Percentage reduction: {comparison.reduction_percent:.1f}%")
print(f"P-value: {comparison.p_value:.4f}")
print(f"Statistically significant: {comparison.p_value < 0.05}")
```

Edge cases:
- Zero baseline rate: percentage reduction is 0.0 (avoids division by zero)
- No factual claims: p-value defaults to 1.0 (no significance)
- Identical rates: reduction is 0.0, p-value is 1.0

### Aggregate Hallucination Metrics

The `aggregate_metrics` method computes aggregate statistics across multiple `HallucinationResult` objects, optionally grouped by category, model, or time period.

```python
from src.hallucination.hallucination_detector import HallucinationDetector
from src.data_models.hallucination import (
    HallucinationResult, SensitivityLevel
)

detector = HallucinationDetector()

# Create results (e.g. from evaluating multiple responses)
results = [
    HallucinationResult(
        has_hallucinations=True, hallucination_rate=0.3,
        total_claims=10, factual_claims=10,
        supported_claims=7, unsupported_claims=3,
        flagged_spans=[], overall_grounding_score=0.7,
        claim_evidence=[], sensitivity_level=SensitivityLevel.MODERATE,
    ),
    HallucinationResult(
        has_hallucinations=True, hallucination_rate=0.5,
        total_claims=8, factual_claims=8,
        supported_claims=4, unsupported_claims=4,
        flagged_spans=[], overall_grounding_score=0.5,
        claim_evidence=[], sensitivity_level=SensitivityLevel.MODERATE,
    ),
]

# Aggregate without grouping
metrics = detector.aggregate_metrics(results)
print(f"Mean rate: {metrics['overall']['mean_rate']:.2%}")
print(f"Median rate: {metrics['overall']['median_rate']:.2%}")

# Aggregate by category
metrics = detector.aggregate_metrics(
    results,
    group_by="category",
    group_keys=["medical", "legal"],
)
for name, stats in metrics["groups"].items():
    print(f"{name}: mean={stats['mean_rate']:.2%}")

# Aggregate by model
metrics = detector.aggregate_metrics(
    results,
    group_by="model",
    group_keys=["model-a", "model-b"],
)

# Aggregate by time period
metrics = detector.aggregate_metrics(
    results,
    group_by="time_period",
    group_keys=["2024-01", "2024-02"],
)
```

Each group (and the overall) contains: `count`, `mean_rate`, `median_rate`, `std_rate`, `min_rate`, `max_rate`, `mean_grounding_score`, `total_claims`, `total_factual_claims`, `total_supported_claims`, `total_unsupported_claims`, and `total_flagged_spans`.

### Alert Trigger

The `check_alert_threshold` method evaluates a hallucination result against a configurable threshold and returns an alert dictionary. This enables automated monitoring and alerting when hallucination rates exceed acceptable levels.

```python
from src.hallucination.hallucination_detector import HallucinationDetector
from src.data_models.hallucination import (
    HallucinationResult, SensitivityLevel
)

detector = HallucinationDetector()

result = HallucinationResult(
    has_hallucinations=True,
    hallucination_rate=0.7,
    total_claims=10,
    factual_claims=10,
    supported_claims=3,
    unsupported_claims=7,
    flagged_spans=[],
    overall_grounding_score=0.3,
    claim_evidence=[],
    sensitivity_level=SensitivityLevel.MODERATE,
)

# Check with default threshold (0.5)
alert = detector.check_alert_threshold(result)
print(alert["triggered"])  # True
print(alert["message"])
# "ALERT: Hallucination rate 70.00% exceeds threshold 50.00%"

# Check with custom threshold
alert = detector.check_alert_threshold(result, alert_threshold=0.8)
print(alert["triggered"])  # False
print(alert["message"])
# "Hallucination rate 70.00% is within threshold 80.00%"
```

The returned dictionary contains:
- `triggered` (bool): Whether the rate exceeds the threshold
- `hallucination_rate` (float): The actual hallucination rate
- `threshold` (float): The configured alert threshold
- `message` (str): Descriptive message about the alert status

## Rate Calculation Logic

The hallucination rate is calculated as:

```
hallucination_rate = unsupported_factual_claims / total_factual_claims
```

1. **Filter**: Only factual claims are considered (opinions excluded)
2. **Count**: Unsupported factual claims are counted
3. **Divide**: Ratio of unsupported to total factual claims
4. **Edge case**: Returns 0.0 if no factual claims exist

The result is always in the range [0.0, 1.0].

## API Reference

### HallucinationDetector

```python
class HallucinationDetector:
    def __init__(self, config: Optional[HallucinationConfig] = None)
    def set_sensitivity_level(self, level: SensitivityLevel) -> None
    def get_sensitivity_config(self) -> dict
    def calculate_hallucination_rate(
        self, claim_evidence: list[ClaimEvidence]
    ) -> float
    def generate_flagged_spans(
        self, claim_evidence: list[ClaimEvidence],
        response: str
    ) -> list[FlaggedSpan]
    def compare_hallucination_rates(
        self, baseline_result: HallucinationResult,
        finetuned_result: HallucinationResult
    ) -> HallucinationComparison
    def aggregate_metrics(
        self, results: list[HallucinationResult],
        group_by: str = "none",
        group_keys: Optional[list[str]] = None
    ) -> dict
    def check_alert_threshold(
        self, result: HallucinationResult,
        alert_threshold: float = 0.5
    ) -> dict
    def detect(
        self, response: str,
        source_documents: Optional[list[str]] = None,
        sensitivity_level: Optional[SensitivityLevel] = None
    ) -> HallucinationResult
```

#### `set_sensitivity_level(level)`

Update the sensitivity level after initialization. Recreates the internal
ClaimClassifier with the new threshold.

**Parameters:**
- `level` (SensitivityLevel): The new sensitivity level (STRICT, MODERATE, or LENIENT)

#### `get_sensitivity_config()`

Get the current sensitivity configuration.

**Returns:**
- `dict`: Contains `sensitivity_level`, `threshold`, and `all_thresholds`

#### `detect(response, source_documents, sensitivity_level)`

Detect hallucinations in a model response. Supports per-run sensitivity
override via the `sensitivity_level` parameter without changing the
detector's persistent configuration.

**Parameters:**
- `response` (str): The model response to analyze
- `source_documents` (list[str], optional): Source documents for evidence search
- `sensitivity_level` (SensitivityLevel, optional): Per-run sensitivity override

**Returns:**
- `HallucinationResult`: Detection analysis with the effective sensitivity level

#### `calculate_hallucination_rate(claim_evidence)`

Calculate the ratio of unsupported to total factual claims.

**Parameters:**
- `claim_evidence` (list[ClaimEvidence]): Classified claim evidence

**Returns:**
- `float`: Hallucination rate in [0.0, 1.0]

#### `generate_flagged_spans(claim_evidence, response)`

Map unsupported factual claims to character positions in the original response.

**Parameters:**
- `claim_evidence` (list[ClaimEvidence]): Classified claim evidence
- `response` (str): The original model response text

**Returns:**
- `list[FlaggedSpan]`: Flagged spans with grounding scores and reasons

#### `compare_hallucination_rates(baseline_result, finetuned_result)`

Compare hallucination rates between baseline and fine-tuned models. Calculates
absolute reduction, percentage reduction, and statistical significance using a
two-proportion z-test.

**Parameters:**
- `baseline_result` (HallucinationResult): Result from the baseline model
- `finetuned_result` (HallucinationResult): Result from the fine-tuned model

**Returns:**
- `HallucinationComparison`: Comparison metrics including reduction, percentage reduction, p-value, and statistical significance

#### `aggregate_metrics(results, group_by, group_keys)`

Aggregate hallucination metrics across multiple results, optionally grouped
by category, model, or time period.

**Parameters:**
- `results` (list[HallucinationResult]): Results to aggregate
- `group_by` (str): Grouping strategy — "none", "category", "model", or "time_period"
- `group_keys` (list[str], optional): Parallel list of group keys for each result

**Returns:**
- `dict`: Aggregated metrics with `total_results`, `group_by`, `groups`, and `overall`

#### `check_alert_threshold(result, alert_threshold)`

Check if a hallucination result exceeds the alert threshold.

**Parameters:**
- `result` (HallucinationResult): A detection result to evaluate
- `alert_threshold` (float): Threshold that triggers an alert (default 0.5)

**Returns:**
- `dict`: Contains `triggered`, `hallucination_rate`, `threshold`, and `message`

## Testing

```bash
pytest tests/unit/test_hallucination_detector.py -v
```

## Related Components

- **ClaimExtractor**: Extracts claims from model responses
- **ClaimClassifier**: Classifies claims as supported/unsupported
- **EvidenceSearcher**: Searches source documents for evidence

## References

- Requirements Document: Sections 6.1-6.13
- Design Document: Hallucination Detector section
- Data Models: `src/data_models/hallucination.py`
