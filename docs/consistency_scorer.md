# Consistency Scorer

## Overview

The Consistency Scorer is a component of the Trust Scoring Engine that evaluates how consistent a model's responses are when given the same prompt multiple times. This helps identify models that produce reliable, deterministic outputs versus those that exhibit randomness or instability.

**Requirements**: 5.1, 5.4

## Key Features

- **Multiple Invocations**: Invokes the model N times (configurable, default 3) with identical prompts
- **Semantic Similarity**: Measures response variance using semantic similarity between all pairs
- **Pairwise Comparison**: Calculates average pairwise similarity as the consistency score
- **Score Range**: Returns scores in [0, 1] where 1.0 indicates perfect consistency
- **Configurable Thresholds**: Supports custom thresholds for high/moderate/low consistency
- **Detailed Metrics**: Provides variance, individual similarities, and response lengths

## How It Works

### Consistency Calculation Process

1. **Invoke Model Multiple Times**: The scorer sends the same prompt to the model N times
2. **Collect Responses**: All responses are collected concurrently for efficiency
3. **Calculate Pairwise Similarities**: Semantic similarity is computed between every pair of responses
4. **Average Similarities**: The mean of all pairwise similarities becomes the consistency score
5. **Classify Consistency**: Based on thresholds, responses are classified as high/moderate/low consistency

### Scoring Formula

For N responses, the consistency score is:

```
consistency_score = (Σ similarity(response_i, response_j)) / C(N, 2)
```

Where:
- `C(N, 2)` is the number of pairs: N × (N-1) / 2
- `similarity(a, b)` is the semantic similarity between responses a and b (using embeddings)

### Interpretation

- **Score ≥ 0.9**: High consistency - Model produces very similar responses
- **0.6 ≤ Score < 0.9**: Moderate consistency - Some variation but generally consistent
- **Score < 0.6**: Low consistency - Significant variation between responses

## Usage

### Basic Usage

```python
from src.trust_scoring.scorers.consistency_scorer import ConsistencyScorer
from src.clients.inference_client import InferenceClient
from src.registry.model_registry import ModelRegistry
from src.utils.rate_limiter import RateLimiter

# Initialize dependencies
registry = ModelRegistry()
rate_limiter = RateLimiter()
inference_client = InferenceClient(registry, rate_limiter)

# Create consistency scorer
scorer = ConsistencyScorer(
    inference_client=inference_client,
    num_samples=3  # Default: invoke model 3 times
)

# Calculate consistency
result = await scorer.calculate_consistency(
    prompt="What is the capital of France?",
    model_id="anthropic.claude-v2"
)

print(f"Consistency Score: {result.score}")
print(f"Confidence: {result.confidence}")
print(f"Checks Passed: {result.checks_passed}")
```

### Custom Configuration

```python
# Create scorer with custom settings
scorer = ConsistencyScorer(
    inference_client=inference_client,
    num_samples=5,  # Invoke 5 times for higher confidence
    high_consistency_threshold=0.95,  # Stricter threshold
    low_consistency_threshold=0.7
)

# Use custom inference parameters
result = await scorer.calculate_consistency(
    prompt="Explain quantum computing",
    model_id="anthropic.claude-v2",
    num_samples=5,  # Override default
    inference_params={
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9
    }
)
```

### Interpreting Results

```python
result = await scorer.calculate_consistency(
    prompt="What is 2+2?",
    model_id="anthropic.claude-v2"
)

# Overall consistency score
print(f"Score: {result.score}")  # 0.0 to 1.0

# Confidence in the score
print(f"Confidence: {result.confidence}")  # Higher with more samples

# Classification
if "high_consistency" in result.checks_passed:
    print("Model is highly consistent")
elif "moderate_consistency" in result.checks_passed:
    print("Model has moderate consistency")
elif "low_consistency" in result.checks_failed:
    print("Model has low consistency")

# Detailed metrics
details = result.details
print(f"Number of samples: {details['num_samples']}")
print(f"Pairwise similarities: {details['pairwise_similarities']}")
print(f"Variance: {details['variance']}")
print(f"Response lengths: {details['response_lengths']}")
```

## Configuration Options

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inference_client` | InferenceClient | Required | Client for invoking models |
| `similarity_analyzer` | SemanticSimilarityAnalyzer | Auto-created | Analyzer for computing semantic similarity |
| `num_samples` | int | 3 | Default number of model invocations |
| `high_consistency_threshold` | float | 0.9 | Threshold for high consistency classification |
| `low_consistency_threshold` | float | 0.6 | Threshold below which consistency is low |

### Method Parameters

#### `calculate_consistency()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | Required | The prompt to send to the model |
| `model_id` | str | Required | The model to evaluate |
| `num_samples` | int | None | Number of invocations (overrides default) |
| `inference_params` | dict | None | Custom inference parameters |

**Inference Parameters**:
- `max_tokens`: Maximum tokens to generate (default: 1024)
- `temperature`: Sampling temperature (default: 0.7)
- `top_p`: Nucleus sampling parameter (default: 0.9)
- `stop_sequences`: List of stop sequences

## Return Value

Returns a `DimensionScore` object with:

```python
DimensionScore(
    dimension=TrustDimension.CONSISTENCY,
    score=0.85,  # Consistency score [0, 1]
    confidence=0.7,  # Confidence in the score
    details={
        "num_samples": 3,
        "pairwise_similarities": [0.9, 0.8, 0.85],
        "mean_similarity": 0.85,
        "variance": 0.0017,
        "response_lengths": [45, 47, 43],
        "model_id": "anthropic.claude-v2"
    },
    checks_passed=["moderate_consistency"],
    checks_failed=[]
)
```

## Edge Cases

### Insufficient Samples

If `num_samples < 2`, returns:
```python
DimensionScore(
    score=1.0,
    confidence=0.0,
    checks_failed=["insufficient_samples"]
)
```

### Empty Prompt

If prompt is empty or whitespace-only, returns:
```python
DimensionScore(
    score=0.0,
    confidence=1.0,
    checks_failed=["empty_prompt"]
)
```

### All Empty Responses

If all model responses are empty, returns:
```python
DimensionScore(
    score=1.0,
    confidence=0.5,
    checks_passed=["all_empty"]
)
```

### Model Invocation Failure

If model invocation fails, returns:
```python
DimensionScore(
    score=0.0,
    confidence=0.0,
    checks_failed=["model_invocation_failed"],
    details={"error": "Failed to invoke model: <error message>"}
)
```

### Similarity Calculation Failure

If semantic similarity calculation fails for a pair, uses neutral score (0.5) for that pair and continues.

## Performance Considerations

### Concurrent Invocations

The scorer invokes the model N times concurrently using `asyncio.gather()` for efficiency. This means:
- All invocations happen in parallel
- Total time ≈ single invocation time (not N × invocation time)
- Rate limiting is still applied per provider

### Number of Samples

The number of pairwise comparisons grows quadratically:
- 2 samples: 1 comparison
- 3 samples: 3 comparisons
- 5 samples: 10 comparisons
- 10 samples: 45 comparisons

**Recommendation**: Use 3-5 samples for most cases. More samples increase confidence but also cost and latency.

### Cost Implications

Each consistency check costs N model invocations:
- 3 samples = 3× the cost of a single inference
- 5 samples = 5× the cost of a single inference

Consider this when evaluating large datasets.

## Best Practices

### Choosing Number of Samples

- **Quick Check**: 2-3 samples for rapid assessment
- **Standard**: 3-5 samples for balanced confidence and cost
- **High Confidence**: 5-10 samples for critical applications
- **Research**: 10+ samples for detailed analysis

### Temperature Settings

Consistency is affected by temperature:
- **Low temperature (0.0-0.3)**: Expect high consistency (deterministic)
- **Medium temperature (0.4-0.7)**: Expect moderate consistency
- **High temperature (0.8-1.0)**: Expect low consistency (creative/random)

### Use Cases

**High Consistency Required**:
- Factual Q&A systems
- Mathematical calculations
- Code generation
- Compliance-critical applications

**Moderate Consistency Acceptable**:
- Creative writing
- Brainstorming
- Exploratory analysis

**Low Consistency Expected**:
- Diverse idea generation
- Random sampling
- A/B testing different responses

## Integration with Trust Scoring

The consistency scorer is one of five dimensions in the Trust Scoring Engine:

```python
from src.trust_scoring.trust_scoring_engine import TrustScoringEngine

engine = TrustScoringEngine(inference_client)

# Consistency is calculated as part of overall trust score
trust_result = await engine.score_response(
    prompt="What is the capital of France?",
    response="Paris is the capital of France",
    model_id="anthropic.claude-v2"
)

# Access consistency dimension
consistency_score = trust_result.dimension_scores[TrustDimension.CONSISTENCY]
print(f"Consistency: {consistency_score.score}")
```

The consistency dimension typically has a weight of 0.20 (20%) in the overall trust score.

## Examples

### Example 1: Evaluating Factual Q&A

```python
# For factual questions, we expect high consistency
result = await scorer.calculate_consistency(
    prompt="What is 2 + 2?",
    model_id="anthropic.claude-v2",
    num_samples=3
)

# Should have high consistency (score ≥ 0.9)
assert result.score >= 0.9
assert "high_consistency" in result.checks_passed
```

### Example 2: Evaluating Creative Tasks

```python
# For creative tasks, consistency may be lower
result = await scorer.calculate_consistency(
    prompt="Write a creative story about a dragon",
    model_id="anthropic.claude-v2",
    num_samples=5,
    inference_params={"temperature": 0.9}  # High temperature
)

# May have lower consistency due to creative nature
print(f"Consistency: {result.score}")  # Might be 0.5-0.7
```

### Example 3: Comparing Models

```python
# Compare consistency across different models
models = [
    "anthropic.claude-v2",
    "anthropic.claude-instant-v1",
    "amazon.titan-text-express-v1"
]

prompt = "What is the capital of France?"

for model_id in models:
    result = await scorer.calculate_consistency(
        prompt=prompt,
        model_id=model_id,
        num_samples=5
    )
    print(f"{model_id}: {result.score:.3f}")
```

### Example 4: Batch Evaluation

```python
# Evaluate consistency across multiple prompts
prompts = [
    "What is 2+2?",
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?"
]

results = []
for prompt in prompts:
    result = await scorer.calculate_consistency(
        prompt=prompt,
        model_id="anthropic.claude-v2",
        num_samples=3
    )
    results.append({
        "prompt": prompt,
        "score": result.score,
        "variance": result.details["variance"]
    })

# Analyze results
avg_consistency = sum(r["score"] for r in results) / len(results)
print(f"Average consistency: {avg_consistency:.3f}")
```

## Troubleshooting

### Low Consistency Scores

If you're getting unexpectedly low consistency scores:

1. **Check temperature**: High temperature increases randomness
2. **Verify prompt**: Ambiguous prompts may yield varied responses
3. **Increase samples**: More samples give more reliable estimates
4. **Check model**: Some models are inherently less deterministic

### High Latency

If consistency checks are slow:

1. **Reduce samples**: Use 2-3 samples instead of 5+
2. **Check rate limits**: Ensure rate limiter isn't throttling
3. **Optimize inference params**: Reduce `max_tokens` if possible
4. **Use faster models**: Consider using smaller/faster models

### API Errors

If you encounter API errors:

1. **Check model availability**: Verify model is active in registry
2. **Verify credentials**: Ensure AWS credentials are valid
3. **Check rate limits**: You may be hitting provider rate limits
4. **Review error details**: Check `result.details["error"]` for specifics

## Related Components

- **AccuracyScorer**: Compares responses to expected answers
- **SafetyScorer**: Checks for harmful content
- **BiasScorer**: Detects demographic bias
- **GroundingScorer**: Measures context grounding
- **TrustScoringEngine**: Combines all dimensions into overall trust score

## References

- Requirements: 5.1, 5.4
- Design Document: Section 5 (Trust Scoring Engine)
- Implementation: `src/trust_scoring/scorers/consistency_scorer.py`
- Tests: `tests/unit/test_consistency_scorer.py`
