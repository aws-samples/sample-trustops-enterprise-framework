# Accuracy Scorer Documentation

## Overview

The `AccuracyScorer` is a component of the Trust Scoring Engine that evaluates the accuracy of model responses by comparing them against expected answers. It implements three complementary scoring methods to provide robust accuracy assessment.

**Requirements**: 5.1, 5.3

## Features

### Three Scoring Methods

1. **Exact Match**: Normalized string comparison
   - Case-insensitive
   - Punctuation removed
   - Whitespace normalized
   - Returns 1.0 for perfect match, 0.0 otherwise

2. **Fuzzy Match**: Levenshtein distance-based similarity
   - Tolerates typos and minor variations
   - Calculates edit distance between strings
   - Returns similarity ratio in [0, 1] range

3. **Semantic Similarity**: Embedding-based comparison
   - Uses Bedrock Titan Embed v2 model
   - Computes cosine similarity between embeddings
   - Recognizes paraphrased answers
   - Returns similarity score in [0, 1] range

### Combined Scoring

The final accuracy score is the **maximum** of all three methods. This ensures that:
- Exact matches score perfectly
- Minor typos don't severely penalize scores
- Paraphrased but correct answers score well
- The best method for each case is automatically selected

### Confidence Calculation

Confidence reflects the agreement between methods:
- **High confidence**: All methods agree (similar scores)
- **Low confidence**: Methods disagree (divergent scores)

Confidence is calculated based on score variance across methods.

## Usage

### Basic Usage

```python
from src.trust_scoring.scorers.accuracy_scorer import AccuracyScorer

# Create scorer instance
scorer = AccuracyScorer()

# Calculate accuracy
result = await scorer.calculate_accuracy(
    response="Paris is the capital of France",
    expected_response="Paris is the capital of France"
)

print(f"Score: {result.score}")
print(f"Confidence: {result.confidence}")
print(f"Checks passed: {result.checks_passed}")
```

### Custom Configuration

```python
from src.trust_scoring.scorers.accuracy_scorer import AccuracyScorer
from src.aws_clients.semantic_similarity_analyzer import SemanticSimilarityAnalyzer

# Create custom similarity analyzer
similarity_analyzer = SemanticSimilarityAnalyzer(
    embedding_model_id="amazon.titan-embed-text-v2:0"
)

# Create scorer with custom thresholds
scorer = AccuracyScorer(
    similarity_analyzer=similarity_analyzer,
    fuzzy_threshold=0.85,      # Higher threshold for fuzzy match
    semantic_threshold=0.75     # Higher threshold for semantic similarity
)

result = await scorer.calculate_accuracy(response, expected)
```

### Accessing Detailed Scores

```python
result = await scorer.calculate_accuracy(response, expected)

# Access individual method scores
print(f"Exact match: {result.details['exact_match_score']}")
print(f"Fuzzy match: {result.details['fuzzy_match_score']}")
print(f"Semantic similarity: {result.details['semantic_similarity_score']}")

# Check which methods passed
if "exact_match" in result.checks_passed:
    print("Exact match passed!")

if "semantic_similarity" in result.checks_passed:
    print("Semantic similarity passed!")
```

## API Reference

### AccuracyScorer

#### Constructor

```python
AccuracyScorer(
    similarity_analyzer: Optional[SemanticSimilarityAnalyzer] = None,
    fuzzy_threshold: float = 0.8,
    semantic_threshold: float = 0.7
)
```

**Parameters**:
- `similarity_analyzer`: Semantic similarity analyzer instance (default: creates new instance)
- `fuzzy_threshold`: Threshold for fuzzy match to pass (default: 0.8)
- `semantic_threshold`: Threshold for semantic similarity to pass (default: 0.7)

#### calculate_accuracy()

```python
async def calculate_accuracy(
    response: str,
    expected_response: Optional[str] = None
) -> DimensionScore
```

Calculate accuracy score for a response against expected answer.

**Parameters**:
- `response`: The model's response text
- `expected_response`: The expected/reference answer (optional)

**Returns**: `DimensionScore` with:
- `dimension`: TrustDimension.ACCURACY
- `score`: Final accuracy score in [0, 1]
- `confidence`: Confidence level in [0, 1]
- `details`: Dictionary with individual method scores
- `checks_passed`: List of checks that passed
- `checks_failed`: List of checks that failed

## DimensionScore Structure

```python
{
    "dimension": "accuracy",
    "score": 0.95,
    "confidence": 0.87,
    "details": {
        "exact_match_score": 0.0,
        "fuzzy_match_score": 0.85,
        "semantic_similarity_score": 0.95,
        "method_used": "maximum",
        "response_length": 35,
        "expected_length": 30
    },
    "checks_passed": ["semantic_similarity"],
    "checks_failed": ["exact_match", "fuzzy_match"]
}
```

## Examples

### Example 1: Exact Match

```python
response = "Paris is the capital of France"
expected = "Paris is the capital of France"

result = await scorer.calculate_accuracy(response, expected)
# Score: 1.0 (perfect match)
# All checks pass
```

### Example 2: Minor Typo

```python
response = "Paris is the capitol of France"  # typo: capitol
expected = "Paris is the capital of France"

result = await scorer.calculate_accuracy(response, expected)
# Score: ~0.97 (fuzzy match handles typo)
# Fuzzy match passes, exact match fails
```

### Example 3: Paraphrased Answer

```python
response = "The French capital is Paris"
expected = "Paris is the capital of France"

result = await scorer.calculate_accuracy(response, expected)
# Score: ~0.92 (semantic similarity recognizes paraphrase)
# Semantic similarity passes, exact/fuzzy fail
```

### Example 4: Wrong Answer

```python
response = "London is the capital of England"
expected = "Paris is the capital of France"

result = await scorer.calculate_accuracy(response, expected)
# Score: low (all methods detect incorrect answer)
# All checks fail
```

## Edge Cases

### Empty Response

```python
result = await scorer.calculate_accuracy("", "Paris is the capital")
# Score: 0.0
# Confidence: 1.0
# Checks failed: ["empty_response"]
```

### No Expected Response

```python
result = await scorer.calculate_accuracy("Paris is the capital", None)
# Score: 0.5 (neutral)
# Confidence: 0.3
# Checks failed: ["no_reference_answer"]
```

### API Failure

If the semantic similarity API fails, the scorer:
- Returns neutral score (0.5) for semantic similarity
- Falls back to exact and fuzzy match scores
- Continues to function (resilient to failures)

```python
# Even if semantic similarity fails:
result = await scorer.calculate_accuracy(response, expected)
# Score: max(exact_score, fuzzy_score, 0.5)
# Still returns valid result
```

## Performance Considerations

### Latency

- **Exact match**: < 1ms (string comparison)
- **Fuzzy match**: < 5ms (Levenshtein algorithm)
- **Semantic similarity**: 50-200ms (embedding API call)

Total latency: ~50-200ms (dominated by semantic similarity)

### Optimization Tips

1. **Batch Processing**: For multiple comparisons, consider batching semantic similarity calls
2. **Caching**: Cache embeddings for frequently used expected responses
3. **Threshold Tuning**: Adjust thresholds based on your use case requirements

## Integration with Trust Scoring Engine

The AccuracyScorer is designed to integrate with the Trust Scoring Engine:

```python
from src.trust_scoring.trust_scoring_engine import TrustScoringEngine
from src.trust_scoring.scorers.accuracy_scorer import AccuracyScorer

# Create trust scoring engine
trust_engine = TrustScoringEngine()

# The engine will use AccuracyScorer internally
# when calculating the accuracy dimension
trust_score = trust_engine.calculate_trust_score(
    response=response,
    prompt=prompt,
    source_documents=source_docs,
    expected_response=expected  # Used by accuracy scorer
)

# Access accuracy dimension score
accuracy_score = trust_score.dimension_scores[TrustDimension.ACCURACY]
```

## Testing

Comprehensive unit tests are available in `tests/unit/test_accuracy_scorer.py`:

```bash
# Run accuracy scorer tests
pytest tests/unit/test_accuracy_scorer.py -v

# Run with coverage
pytest tests/unit/test_accuracy_scorer.py --cov=src.trust_scoring.scorers.accuracy_scorer
```

## Demo

Run the demo script to see the accuracy scorer in action:

```bash
python demo/accuracy_scorer_demo.py
```

The demo demonstrates:
- Exact match scoring
- Fuzzy match scoring
- Semantic similarity scoring
- Combined scoring
- Edge case handling

## Best Practices

1. **Always provide expected responses**: The scorer works best with reference answers
2. **Use appropriate thresholds**: Adjust fuzzy and semantic thresholds based on your domain
3. **Monitor confidence**: Low confidence indicates method disagreement - may need review
4. **Handle edge cases**: Check for empty responses and missing expected answers
5. **Consider context**: Semantic similarity works best with sufficient context

## Troubleshooting

### Low Scores for Correct Answers

If correct answers score low:
- Check if expected response is properly formatted
- Verify semantic similarity analyzer is configured correctly
- Consider lowering thresholds for your use case

### High Scores for Wrong Answers

If wrong answers score high:
- Check if strings are structurally similar but semantically different
- Verify semantic similarity is working (not returning neutral 0.5)
- Consider raising thresholds

### API Errors

If semantic similarity fails:
- Check AWS credentials and permissions
- Verify Bedrock model access
- Check network connectivity
- The scorer will continue with exact/fuzzy match

## Related Documentation

- [Trust Scoring Engine](./trust_scoring_engine.md)
- [Semantic Similarity Analyzer](./semantic_similarity_analyzer.md)
- [Trust Score Data Models](./trust_score_models.md)

## Requirements Traceability

- **Requirement 5.1**: Define trust score dimensions (accuracy dimension)
- **Requirement 5.3**: Implement accuracy scorer with exact, fuzzy, and semantic matching
- **Requirement 5.17**: Write unit tests for dimension scorers
