# Context Grounding Scorer

## Overview

The Context Grounding Scorer is a component of the TrustOps Enterprise Framework's Trust Scoring Engine that evaluates whether model responses are properly grounded in provided source documents. It uses embedding-based semantic similarity to measure how well a response aligns with the context it should be based on.

## Purpose

Context grounding is critical for:
- **Factual Accuracy**: Ensuring responses are based on provided sources
- **Hallucination Prevention**: Detecting when models generate unsupported claims
- **Citation Verification**: Checking if responses reference source material
- **RAG Evaluation**: Assessing retrieval-augmented generation quality
- **Compliance**: Verifying responses stay within approved content boundaries

## How It Works

The Context Grounding Scorer evaluates responses through multiple mechanisms:

### 1. Semantic Similarity Analysis

The scorer computes embeddings for both the response and source documents using AWS Bedrock embedding models (default: Amazon Titan Embed Text v2). It then calculates cosine similarity between the response and each source document.

**Scoring Logic:**
- Uses the **maximum similarity** across all source documents as the grounding score
- Rationale: A response well-grounded in any single source is considered grounded
- Score range: 0.0 (no grounding) to 1.0 (perfect grounding)

### 2. Citation Detection

The scorer checks for explicit citations in the response:
- Numbered citations: `[1]`, `[2]`, `(1)`, `(2)`
- Source references: `(Source: ...)`, `(Ref: ...)`
- Attribution phrases: "According to", "As stated in", "Based on", "Citing"

Explicit citations increase confidence in grounding even if semantic similarity is moderate.

### 3. Grounding Thresholds

Three grounding levels are defined:
- **High Grounding** (≥0.8): Response strongly supported by sources
- **Moderate Grounding** (0.5-0.8): Response partially supported
- **Low Grounding** (<0.5): Response poorly supported or unrelated

## Configuration

### Initialization Parameters

```python
from src.trust_scoring.scorers.context_grounding_scorer import (
    ContextGroundingScorer
)

scorer = ContextGroundingScorer(
    similarity_analyzer=None,  # Optional custom analyzer
    high_grounding_threshold=0.8,  # High grounding threshold
    low_grounding_threshold=0.5,   # Low grounding threshold
    embedding_model_id="amazon.titan-embed-text-v2:0"  # Embedding model
)
```

### Parameters

- **similarity_analyzer** (Optional[SemanticSimilarityAnalyzer]): Custom similarity analyzer instance. If None, creates default analyzer.
- **high_grounding_threshold** (float): Minimum score for high grounding (default: 0.8)
- **low_grounding_threshold** (float): Maximum score for low grounding (default: 0.5)
- **embedding_model_id** (str): Bedrock embedding model ID (default: "amazon.titan-embed-text-v2:0")

### Supported Embedding Models

- **amazon.titan-embed-text-v2:0** (default): Latest Titan embedding model
- **amazon.titan-embed-text-v1**: Previous Titan version
- **cohere.embed-english-v3**: Cohere English embeddings
- **cohere.embed-multilingual-v3**: Cohere multilingual embeddings

## Usage

### Basic Usage

```python
import asyncio
from src.trust_scoring.scorers.context_grounding_scorer import (
    ContextGroundingScorer
)

async def evaluate_grounding():
    scorer = ContextGroundingScorer()
    
    response = "Paris is the capital of France and its largest city."
    sources = [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe."
    ]
    
    result = await scorer.calculate_context_grounding(
        response=response,
        source_documents=sources
    )
    
    print(f"Grounding Score: {result.score:.2f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Checks Passed: {result.checks_passed}")
    print(f"Max Similarity: {result.details['max_similarity']:.2f}")

asyncio.run(evaluate_grounding())
```

### Batch Scoring

```python
async def batch_evaluate():
    scorer = ContextGroundingScorer()
    
    items = [
        (
            "Python is a programming language.",
            ["Python is a high-level programming language."]
        ),
        (
            "The sky is blue.",
            ["The atmosphere scatters blue light."]
        ),
        (
            "Water boils at 100°C.",
            ["Water boils at 100 degrees Celsius at sea level."]
        )
    ]
    
    results = await scorer.calculate_batch_grounding(items)
    
    for i, result in enumerate(results):
        print(f"Item {i+1}: Score = {result.score:.2f}")

asyncio.run(batch_evaluate())
```

### Integration with Trust Scoring Engine

```python
from src.trust_scoring.trust_scoring_engine import TrustScoringEngine
from src.data_models.trust_score import TrustScoreConfig, TrustScoreWeights

async def full_trust_score():
    # Configure trust score weights
    config = TrustScoreConfig(
        weights=TrustScoreWeights(
            accuracy=0.25,
            consistency=0.20,
            safety=0.20,
            bias=0.15,
            context_grounding=0.20  # Context grounding weight
        )
    )
    
    engine = TrustScoringEngine(config=config)
    
    result = await engine.score_response(
        prompt="What is the capital of France?",
        response="Paris is the capital of France.",
        expected_response="Paris",
        source_documents=[
            "Paris is the capital and largest city of France."
        ]
    )
    
    print(f"Overall Trust Score: {result.overall_score:.2f}")
    print(f"Context Grounding: {
        result.dimension_scores['context_grounding'].score:.2f
    }")

asyncio.run(full_trust_score())
```

## Result Structure

The scorer returns a `DimensionScore` object with the following structure:

```python
DimensionScore(
    dimension=TrustDimension.CONTEXT_GROUNDING,
    score=0.85,  # Overall grounding score [0, 1]
    confidence=0.78,  # Confidence in the score [0, 1]
    details={
        "max_similarity": 0.85,  # Best match with any source
        "avg_similarity": 0.72,  # Average across all sources
        "min_similarity": 0.60,  # Worst match
        "num_sources": 3,  # Number of source documents
        "similarities": [0.85, 0.72, 0.60],  # All similarities
        "best_match_index": 0,  # Index of best matching source
        "best_match_preview": "Paris is the capital...",  # Preview
        "has_explicit_citations": True,  # Citation detected
        "response_length": 45,  # Response character count
        "embedding_model": "amazon.titan-embed-text-v2:0"
    },
    checks_passed=["high_grounding", "explicit_citations"],
    checks_failed=[]
)
```

## Interpretation Guide

### Score Ranges

| Score Range | Interpretation | Recommendation |
|-------------|----------------|----------------|
| 0.8 - 1.0 | High grounding | Response well-supported by sources |
| 0.5 - 0.8 | Moderate grounding | Response partially supported, review recommended |
| 0.0 - 0.5 | Low grounding | Response poorly supported, likely hallucination |

### Confidence Levels

- **High Confidence (>0.8)**: Consistent similarity across sources
- **Moderate Confidence (0.5-0.8)**: Some variance in similarities
- **Low Confidence (<0.5)**: High variance or missing sources

### Checks

**Passed Checks:**
- `high_grounding`: Score ≥ high_grounding_threshold
- `moderate_grounding`: Score between thresholds
- `explicit_citations`: Citations detected in response

**Failed Checks:**
- `low_grounding`: Score < low_grounding_threshold
- `no_explicit_citations`: No citations found
- `no_source_documents`: No sources provided
- `empty_response`: Response is empty
- `similarity_calculation_failed`: Error computing similarity

## Edge Cases

### No Source Documents

When no source documents are provided:
```python
result = await scorer.calculate_context_grounding(
    response="Some response",
    source_documents=None
)
# Returns: score=0.5, confidence=0.3, check_failed="no_source_documents"
```

### Empty Response

When the response is empty:
```python
result = await scorer.calculate_context_grounding(
    response="",
    source_documents=["Source"]
)
# Returns: score=0.0, confidence=1.0, check_failed="empty_response"
```

### Similarity Calculation Errors

When embedding generation fails:
```python
# Returns: score=0.0, confidence=0.0, 
# check_failed="similarity_calculation_failed"
```

## Best Practices

### 1. Provide Relevant Sources

Ensure source documents are:
- **Relevant**: Related to the expected response topic
- **Complete**: Contain sufficient information
- **Clean**: Free of formatting issues or noise

### 2. Adjust Thresholds by Use Case

Different applications require different thresholds:

**High-Stakes Applications** (medical, legal, financial):
```python
scorer = ContextGroundingScorer(
    high_grounding_threshold=0.9,  # Stricter
    low_grounding_threshold=0.7
)
```

**General Applications** (customer support, content generation):
```python
scorer = ContextGroundingScorer(
    high_grounding_threshold=0.8,  # Default
    low_grounding_threshold=0.5
)
```

**Creative Applications** (marketing, storytelling):
```python
scorer = ContextGroundingScorer(
    high_grounding_threshold=0.7,  # More lenient
    low_grounding_threshold=0.4
)
```

### 3. Combine with Other Dimensions

Context grounding is most effective when combined with other trust dimensions:

```python
config = TrustScoreConfig(
    weights=TrustScoreWeights(
        accuracy=0.25,
        consistency=0.20,
        safety=0.20,
        bias=0.15,
        context_grounding=0.20  # Balanced weight
    )
)
```

### 4. Monitor Confidence Levels

Low confidence scores may indicate:
- Inconsistent source quality
- Ambiguous response content
- Need for additional sources

### 5. Use Batch Scoring for Efficiency

When evaluating multiple responses:
```python
# More efficient than individual calls
results = await scorer.calculate_batch_grounding(items)
```

## Performance Considerations

### Embedding Generation

- **Latency**: ~100-300ms per embedding (depends on model and text length)
- **Cost**: ~$0.0001 per 1K tokens (Titan Embed v2)
- **Optimization**: Cache embeddings for frequently used sources

### Batch Processing

- Process multiple responses concurrently
- Typical throughput: 10-20 responses/second
- Bottleneck: Bedrock API rate limits

### Cost Optimization

```python
# For large-scale evaluation, consider:
# 1. Pre-compute source document embeddings
# 2. Use batch embedding APIs when available
# 3. Cache results for identical source sets
```

## Troubleshooting

### Low Scores Despite Good Grounding

**Possible Causes:**
- Response uses different phrasing than sources
- Sources are too general or too specific
- Embedding model not capturing semantic meaning

**Solutions:**
- Try different embedding models
- Adjust thresholds
- Improve source document quality

### High Variance in Similarities

**Possible Causes:**
- Sources cover different aspects of the topic
- Some sources are irrelevant
- Response addresses multiple topics

**Solutions:**
- Filter sources for relevance
- Split multi-topic responses
- Use topic-specific source sets

### API Errors

**Common Errors:**
- `ThrottlingException`: Rate limit exceeded
- `ValidationException`: Invalid input format
- `ResourceNotFoundException`: Model not available

**Solutions:**
- Implement retry logic with exponential backoff
- Validate inputs before API calls
- Check model availability in your region

## Examples

### Example 1: High Grounding with Citations

```python
response = "According to the study [1], Paris is the capital of France."
sources = ["Paris is the capital and largest city of France."]

result = await scorer.calculate_context_grounding(response, sources)
# Score: ~0.9, Checks: ["high_grounding", "explicit_citations"]
```

### Example 2: Low Grounding (Hallucination)

```python
response = "Paris is the capital of Germany."
sources = ["Berlin is the capital of Germany."]

result = await scorer.calculate_context_grounding(response, sources)
# Score: ~0.3, Checks: ["low_grounding"]
```

### Example 3: Moderate Grounding

```python
response = "France is a European country with Paris as its capital."
sources = ["France is located in Western Europe."]

result = await scorer.calculate_context_grounding(response, sources)
# Score: ~0.6, Checks: ["moderate_grounding"]
```

## Integration with Hallucination Detection

The Context Grounding Scorer complements the Hallucination Detector:

- **Context Grounding**: Measures overall semantic alignment
- **Hallucination Detection**: Identifies specific unsupported claims

Use both for comprehensive evaluation:

```python
# Context grounding for overall alignment
grounding_score = await context_scorer.calculate_context_grounding(
    response, sources
)

# Hallucination detection for specific claims
hallucination_result = await hallucination_detector.detect(
    response, sources
)

# Combined assessment
if grounding_score.score < 0.6 or hallucination_result.has_hallucinations:
    print("Response requires review")
```

## References

- **Requirements**: 5.1, 5.7
- **Design Document**: Section 5 (Trust Scoring Engine)
- **Related Components**:
  - `SemanticSimilarityAnalyzer`: Embedding generation and similarity
  - `HallucinationDetector`: Claim-level grounding analysis
  - `TrustScoringEngine`: Overall trust score calculation

## Version History

- **v1.0.0**: Initial implementation with semantic similarity and citation detection
