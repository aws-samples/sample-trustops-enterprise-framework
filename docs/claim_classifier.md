# Claim Classifier

## Overview

The Claim Classifier is a core component of the Hallucination Detector in the TrustOps Enterprise Framework. It determines whether claims are supported or unsupported based on evidence similarity scores, using configurable thresholds that adapt to different sensitivity levels.

**Requirements**: 
- 6.3 - Classify claims based on evidence
- 6.8 - Support sensitivity levels (strict, moderate, lenient)

## Features

- **Evidence-Based Classification**: Determines claim support based on similarity scores
- **Configurable Thresholds**: Adapts to different use cases and risk tolerances
- **Sensitivity Levels**: Three built-in levels (strict, moderate, lenient)
- **Batch Processing**: Classify multiple claims efficiently
- **Confidence Scoring**: Returns best similarity score for each claim
- **Multiple Evidence Sources**: Handles multiple evidence sources per claim

## Installation

The Claim Classifier is part of the hallucination detection module:

```bash
# No additional dependencies required beyond base TrustOps installation
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.hallucination.claim_classifier import ClaimClassifier
from src.data_models.hallucination import Claim, EvidenceSource

# Initialize the classifier
classifier = ClaimClassifier()

# Create a claim
claim = Claim(
    text="The Earth orbits the Sun.",
    start_idx=0,
    end_idx=27,
    is_factual=True,
    is_opinion=False
)

# Provide evidence sources with similarity scores
evidence_sources = [
    EvidenceSource(
        document_id="astronomy_doc",
        span="The Earth revolves around the Sun in an elliptical orbit.",
        similarity_score=0.85
    )
]

# Classify the claim
result = classifier.classify_claim(claim, evidence_sources)

print(f"Claim: {result.claim.text}")
print(f"Supported: {result.is_supported}")
print(f"Confidence: {result.similarity_score}")
print(f"Best Evidence: {result.best_matching_span}")
```

Output:
```
Claim: The Earth orbits the Sun.
Supported: True
Confidence: 0.85
Best Evidence: The Earth revolves around the Sun in an elliptical orbit.
```

### Sensitivity Levels

The classifier supports three sensitivity levels with different thresholds:

```python
from src.data_models.hallucination import (
    HallucinationConfig,
    SensitivityLevel
)

# Strict mode (threshold = 0.8)
strict_config = HallucinationConfig(
    sensitivity_level=SensitivityLevel.STRICT
)
strict_classifier = ClaimClassifier(strict_config)

# Moderate mode (threshold = 0.6) - default
moderate_config = HallucinationConfig(
    sensitivity_level=SensitivityLevel.MODERATE
)
moderate_classifier = ClaimClassifier(moderate_config)

# Lenient mode (threshold = 0.4)
lenient_config = HallucinationConfig(
    sensitivity_level=SensitivityLevel.LENIENT
)
lenient_classifier = ClaimClassifier(lenient_config)

# Same evidence, different results
evidence = [
    EvidenceSource(
        document_id="doc1",
        span="Supporting evidence.",
        similarity_score=0.5
    )
]

strict_result = strict_classifier.classify_claim(claim, evidence)
moderate_result = moderate_classifier.classify_claim(claim, evidence)
lenient_result = lenient_classifier.classify_claim(claim, evidence)

print(f"Strict (0.8): {strict_result.is_supported}")      # False
print(f"Moderate (0.6): {moderate_result.is_supported}")  # False
print(f"Lenient (0.4): {lenient_result.is_supported}")    # True
```

### Custom Threshold

You can override the sensitivity level with a custom threshold:

```python
# Custom threshold of 0.7
custom_config = HallucinationConfig(
    similarity_threshold=0.7
)
classifier = ClaimClassifier(custom_config)

print(f"Threshold: {classifier.get_threshold()}")  # 0.7
```

### Batch Classification

Classify multiple claims at once:

```python
claims = [
    Claim(text="Claim 1.", start_idx=0, end_idx=8, 
          is_factual=True, is_opinion=False),
    Claim(text="Claim 2.", start_idx=9, end_idx=17,
          is_factual=True, is_opinion=False),
    Claim(text="Claim 3.", start_idx=18, end_idx=26,
          is_factual=True, is_opinion=False)
]

# Evidence map: claim text -> evidence sources
evidence_map = {
    "Claim 1.": [
        EvidenceSource(
            document_id="doc1",
            span="Evidence for claim 1.",
            similarity_score=0.8
        )
    ],
    "Claim 2.": [
        EvidenceSource(
            document_id="doc2",
            span="Evidence for claim 2.",
            similarity_score=0.3
        )
    ],
    "Claim 3.": []  # No evidence found
}

# Classify all claims
results = classifier.classify_claims(claims, evidence_map)

for result in results:
    print(f"{result.claim.text}: {result.is_supported}")
```

Output:
```
Claim 1.: True
Claim 2.: False
Claim 3.: False
```

### Multiple Evidence Sources

When multiple evidence sources exist, the classifier uses the best one:

```python
claim = Claim(
    text="Python was created by Guido van Rossum.",
    start_idx=0,
    end_idx=40,
    is_factual=True,
    is_opinion=False
)

evidence_sources = [
    EvidenceSource(
        document_id="doc1",
        span="Python is a programming language.",
        similarity_score=0.4
    ),
    EvidenceSource(
        document_id="doc2",
        span="Guido van Rossum created Python in 1991.",
        similarity_score=0.9  # Best match
    ),
    EvidenceSource(
        document_id="doc3",
        span="Python has many features.",
        similarity_score=0.2
    )
]

result = classifier.classify_claim(claim, evidence_sources)

print(f"Best Score: {result.similarity_score}")           # 0.9
print(f"Best Document: {result.best_matching_document}")  # doc2
print(f"Total Evidence: {len(result.evidence_sources)}")  # 3
```

## Classification Logic

### Decision Process

1. **No Evidence**: If no evidence sources provided → `is_supported = False`
2. **Find Best Evidence**: Select evidence source with highest similarity score
3. **Apply Threshold**: Compare best score against threshold
   - `score >= threshold` → `is_supported = True`
   - `score < threshold` → `is_supported = False`
4. **Return Result**: Include best evidence details and all sources

### Threshold Selection

The classifier determines the threshold using this logic:

```python
if config.similarity_threshold != default_threshold:
    # User explicitly set custom threshold
    use config.similarity_threshold
else:
    # Use sensitivity level mapping
    use SENSITIVITY_THRESHOLDS[config.sensitivity_level]
```

### Sensitivity Level Mappings

| Sensitivity Level | Threshold | Use Case |
|-------------------|-----------|----------|
| STRICT | 0.8 | High-risk applications (medical, legal, financial) |
| MODERATE | 0.6 | General purpose (default) |
| LENIENT | 0.4 | Exploratory analysis, low-risk content |

## Configuration Options

The `HallucinationConfig` provides configuration for the classifier:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sensitivity_level` | SensitivityLevel | MODERATE | Detection sensitivity |
| `similarity_threshold` | float | 0.6 | Custom threshold (overrides sensitivity) |
| `embedding_model_id` | str | "amazon.titan-embed-text-v2:0" | Model for embeddings (used upstream) |

## Integration with Hallucination Detection

The Claim Classifier is the third step in the hallucination detection pipeline:

```python
from src.hallucination.claim_extractor import ClaimExtractor
from src.hallucination.claim_classifier import ClaimClassifier
from src.data_models.hallucination import HallucinationConfig

# Configuration
config = HallucinationConfig(sensitivity_level=SensitivityLevel.MODERATE)

# Step 1: Extract claims
extractor = ClaimExtractor(config)
claims = extractor.extract_claims(model_response)

# Step 2: Search for evidence (implemented in evidence_searcher.py)
# evidence_map = evidence_searcher.search_evidence(claims, source_documents)

# Step 3: Classify claims
classifier = ClaimClassifier(config)
claim_evidence = classifier.classify_claims(claims, evidence_map)

# Step 4: Calculate hallucination rate
factual_claims = [ce for ce in claim_evidence if ce.claim.is_factual]
unsupported = [ce for ce in factual_claims if not ce.is_supported]
hallucination_rate = len(unsupported) / len(factual_claims) if factual_claims else 0.0

print(f"Hallucination Rate: {hallucination_rate:.2%}")
```

## API Reference

### ClaimClassifier

```python
class ClaimClassifier:
    def __init__(self, config: Optional[HallucinationConfig] = None)
    def classify_claim(
        self,
        claim: Claim,
        evidence_sources: list[EvidenceSource]
    ) -> ClaimEvidence
    def classify_claims(
        self,
        claims: list[Claim],
        evidence_map: dict[str, list[EvidenceSource]]
    ) -> list[ClaimEvidence]
    def get_threshold(self) -> float
    def get_sensitivity_level(self) -> SensitivityLevel
```

#### `__init__(config)`

Initialize the claim classifier.

**Parameters:**
- `config` (HallucinationConfig, optional): Configuration for classification

#### `classify_claim(claim, evidence_sources)`

Classify a single claim based on evidence.

**Parameters:**
- `claim` (Claim): The claim to classify
- `evidence_sources` (list[EvidenceSource]): Evidence with similarity scores

**Returns:**
- `ClaimEvidence`: Classification result with confidence score

#### `classify_claims(claims, evidence_map)`

Classify multiple claims based on evidence map.

**Parameters:**
- `claims` (list[Claim]): Claims to classify
- `evidence_map` (dict): Mapping from claim text to evidence sources

**Returns:**
- `list[ClaimEvidence]`: Classification results for all claims

#### `get_threshold()`

Get the current similarity threshold.

**Returns:**
- `float`: Threshold value (0.0 to 1.0)

#### `get_sensitivity_level()`

Get the current sensitivity level.

**Returns:**
- `SensitivityLevel`: Current sensitivity level

## Examples

### Example 1: High-Risk Application (Strict Mode)

```python
# Medical/legal content requires high confidence
config = HallucinationConfig(sensitivity_level=SensitivityLevel.STRICT)
classifier = ClaimClassifier(config)

claim = Claim(
    text="This medication is FDA approved.",
    start_idx=0,
    end_idx=32,
    is_factual=True,
    is_opinion=False
)

evidence = [
    EvidenceSource(
        document_id="fda_database",
        span="The medication received FDA approval in 2020.",
        similarity_score=0.75  # Not high enough for strict mode
    )
]

result = classifier.classify_claim(claim, evidence)
print(f"Supported: {result.is_supported}")  # False (needs >= 0.8)
```

### Example 2: Exploratory Analysis (Lenient Mode)

```python
# Exploratory research can use lower threshold
config = HallucinationConfig(sensitivity_level=SensitivityLevel.LENIENT)
classifier = ClaimClassifier(config)

claim = Claim(
    text="This approach might be effective.",
    start_idx=0,
    end_idx=33,
    is_factual=False,
    is_opinion=True,
    is_hedged=True
)

evidence = [
    EvidenceSource(
        document_id="research_paper",
        span="The approach showed promising results.",
        similarity_score=0.45
    )
]

result = classifier.classify_claim(claim, evidence)
print(f"Supported: {result.is_supported}")  # True (>= 0.4)
```

### Example 3: No Evidence Found

```python
classifier = ClaimClassifier()

claim = Claim(
    text="The company will triple revenue next year.",
    start_idx=0,
    end_idx=43,
    is_factual=True,
    is_opinion=False
)

# No evidence found in source documents
result = classifier.classify_claim(claim, [])

print(f"Supported: {result.is_supported}")              # False
print(f"Confidence: {result.similarity_score}")         # 0.0
print(f"Best Document: {result.best_matching_document}") # None
```

### Example 4: Filtering Unsupported Claims

```python
# Classify all claims
results = classifier.classify_claims(claims, evidence_map)

# Filter to find hallucinations (unsupported factual claims)
hallucinations = [
    result for result in results
    if result.claim.is_factual and not result.is_supported
]

print(f"Found {len(hallucinations)} potential hallucinations:")
for h in hallucinations:
    print(f"  - {h.claim.text} (confidence: {h.similarity_score:.2f})")
```

## Performance Considerations

- **Time Complexity**: O(n × m) where n = claims, m = avg evidence sources per claim
- **Space Complexity**: O(n × m) for storing all evidence sources
- **Typical Performance**:
  - Single claim: <1ms
  - 100 claims: ~10-50ms
  - Batch processing recommended for large datasets

## Testing

Comprehensive test coverage includes:

- Basic classification (supported/unsupported)
- Sensitivity levels (strict, moderate, lenient)
- Threshold boundaries
- Multiple evidence sources
- Edge cases (no evidence, perfect scores)
- Batch processing
- Unicode handling

Run tests:
```bash
pytest tests/unit/test_claim_classifier.py -v
```

## Related Components

- **ClaimExtractor**: Extracts claims from model responses (step 1)
- **EvidenceSearcher**: Searches for supporting evidence (step 2)
- **HallucinationDetector**: Calculates overall hallucination metrics (step 4)

## References

- Requirements Document: Sections 6.3, 6.8
- Design Document: Hallucination Detector section
- Data Models: `src/data_models/hallucination.py`

## Changelog

### Version 1.0.0 (Initial Release)
- Implemented evidence-based claim classification
- Added three sensitivity levels (strict, moderate, lenient)
- Support for custom thresholds
- Batch classification support
- Comprehensive test coverage (30+ tests)
