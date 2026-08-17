# Confidence Intervals for Trust Scores

## Overview

The TrustOps Enterprise Framework provides confidence interval calculation for aggregate trust scores, allowing you to quantify the statistical uncertainty in your model evaluations. This feature helps you understand the reliability of your trust score measurements and make more informed decisions about model deployment.

## Key Features

- **Automatic Method Selection**: Automatically chooses between bootstrap and normal approximation based on sample size
- **Bootstrap Resampling**: For small samples (<30), uses bootstrap resampling with 10,000 iterations
- **Normal Approximation**: For large samples (≥30), uses efficient normal approximation
- **Configurable Confidence Levels**: Support for any confidence level (e.g., 90%, 95%, 99%)
- **Bounded Results**: Ensures confidence bounds stay within valid trust score range [0, 1]

## When to Use Confidence Intervals

Confidence intervals are essential when:

1. **Comparing Models**: Determine if performance differences are statistically significant
2. **Small Sample Sizes**: Understand the uncertainty when evaluating on limited data
3. **Deployment Decisions**: Assess the reliability of trust score improvements
4. **Reporting**: Provide stakeholders with statistical rigor in your evaluations

## Usage

### Basic Usage

```python
from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine

# Initialize the engine
engine = TrustScoringEngine()

# Your trust scores from an evaluation
trust_scores = [0.82, 0.78, 0.85, 0.79, 0.81, 0.83, 0.77, 0.84, 0.80, 0.82]

# Calculate 95% confidence interval (default)
result = engine.calculate_confidence_interval(
    scores=trust_scores,
    confidence_level=0.95,
    method="auto"  # Automatically selects best method
)

print(f"Mean Trust Score: {result['mean']:.3f}")
print(f"95% CI: [{result['lower_bound']:.3f}, {result['upper_bound']:.3f}]")
print(f"Method Used: {result['method']}")
print(f"Sample Size: {result['sample_size']}")
```

### Specifying Confidence Level

```python
# Calculate 99% confidence interval for higher confidence
result_99 = engine.calculate_confidence_interval(
    scores=trust_scores,
    confidence_level=0.99,
    method="auto"
)

# Calculate 90% confidence interval for narrower bounds
result_90 = engine.calculate_confidence_interval(
    scores=trust_scores,
    confidence_level=0.90,
    method="auto"
)
```

### Forcing a Specific Method

```python
# Force bootstrap method (useful for small samples or non-normal distributions)
result_bootstrap = engine.calculate_confidence_interval(
    scores=trust_scores,
    confidence_level=0.95,
    method="bootstrap"
)

# Force normal approximation (faster for large samples)
result_normal = engine.calculate_confidence_interval(
    scores=trust_scores,
    confidence_level=0.95,
    method="normal"
)
```

## Methods

### Bootstrap Resampling

**When Used**: Sample size < 30 (or manually specified)

**How It Works**:
1. Generates 10,000 bootstrap samples by resampling with replacement
2. Calculates the mean for each bootstrap sample
3. Uses percentiles of bootstrap means to determine confidence bounds

**Advantages**:
- No assumptions about score distribution
- Works well with small samples
- Robust to outliers

**Disadvantages**:
- Computationally more expensive
- Requires sufficient sample size (at least 2-3 scores)

### Normal Approximation

**When Used**: Sample size ≥ 30 (or manually specified)

**How It Works**:
1. Calculates standard error of the mean
2. Uses z-score for desired confidence level
3. Computes margin of error: z × standard_error
4. Bounds: mean ± margin_of_error

**Advantages**:
- Very fast computation
- Well-understood statistical properties
- Accurate for large samples

**Disadvantages**:
- Assumes approximately normal distribution
- Less accurate for very small samples

## Interpreting Results

### Result Dictionary

```python
{
    "mean": 0.812,              # Mean trust score
    "lower_bound": 0.785,       # Lower confidence bound
    "upper_bound": 0.839,       # Upper confidence bound
    "confidence_level": 0.95,   # Confidence level used
    "method": "normal",         # Method used
    "sample_size": 30           # Number of scores
}
```

### Confidence Level Interpretation

- **95% Confidence**: If you repeated the evaluation many times, 95% of the calculated intervals would contain the true mean
- **99% Confidence**: Higher confidence but wider interval
- **90% Confidence**: Lower confidence but narrower interval

### Interval Width

- **Narrow Interval** (< 0.05): High precision, reliable estimate
- **Moderate Interval** (0.05 - 0.15): Reasonable precision
- **Wide Interval** (> 0.15): Low precision, consider collecting more data

## Examples

### Example 1: Small Sample Evaluation

```python
# Evaluated 15 examples
small_sample_scores = [
    0.75, 0.78, 0.82, 0.76, 0.79,
    0.81, 0.77, 0.80, 0.78, 0.82,
    0.79, 0.81, 0.77, 0.80, 0.78
]

result = engine.calculate_confidence_interval(
    scores=small_sample_scores,
    confidence_level=0.95,
    method="auto"
)

# Output:
# Mean Trust Score: 0.789
# 95% CI: [0.772, 0.806]
# Method Used: bootstrap
# Sample Size: 15
```

### Example 2: Large Sample Evaluation

```python
# Evaluated 100 examples
import numpy as np
np.random.seed(42)
large_sample_scores = np.random.normal(loc=0.82, scale=0.08, size=100).tolist()
large_sample_scores = [max(0.0, min(1.0, s)) for s in large_sample_scores]

result = engine.calculate_confidence_interval(
    scores=large_sample_scores,
    confidence_level=0.95,
    method="auto"
)

# Output:
# Mean Trust Score: 0.823
# 95% CI: [0.807, 0.839]
# Method Used: normal
# Sample Size: 100
```

### Example 3: Comparing Two Models

```python
# Baseline model scores
baseline_scores = [0.72, 0.75, 0.78, 0.74, 0.76, 0.73, 0.77, 0.75, 0.74, 0.76] * 3

# Fine-tuned model scores
finetuned_scores = [0.82, 0.85, 0.88, 0.84, 0.86, 0.83, 0.87, 0.85, 0.84, 0.86] * 3

baseline_ci = engine.calculate_confidence_interval(baseline_scores, 0.95)
finetuned_ci = engine.calculate_confidence_interval(finetuned_scores, 0.95)

print(f"Baseline: {baseline_ci['mean']:.3f} [{baseline_ci['lower_bound']:.3f}, {baseline_ci['upper_bound']:.3f}]")
print(f"Fine-tuned: {finetuned_ci['mean']:.3f} [{finetuned_ci['lower_bound']:.3f}, {finetuned_ci['upper_bound']:.3f}]")

# Check if confidence intervals overlap
if baseline_ci['upper_bound'] < finetuned_ci['lower_bound']:
    print("✓ Fine-tuned model is statistically significantly better!")
else:
    print("⚠ Confidence intervals overlap - difference may not be significant")
```

## Best Practices

### 1. Sample Size Considerations

- **Minimum**: At least 10 scores for meaningful confidence intervals
- **Recommended**: 30+ scores for reliable normal approximation
- **Ideal**: 100+ scores for narrow, precise intervals

### 2. Choosing Confidence Level

- **95%**: Standard choice for most applications
- **99%**: Use when high confidence is critical (e.g., safety-critical systems)
- **90%**: Use for exploratory analysis or when wider intervals are acceptable

### 3. Method Selection

- **Auto**: Let the system choose (recommended)
- **Bootstrap**: Force for non-normal distributions or small samples
- **Normal**: Force for large samples when speed is critical

### 4. Interpreting Overlapping Intervals

When comparing two models:
- **No Overlap**: Strong evidence of difference
- **Slight Overlap**: Possible difference, consider additional testing
- **Substantial Overlap**: Insufficient evidence of difference

### 5. Reporting

Always report confidence intervals alongside mean scores:

```
Model A: 0.82 (95% CI: 0.78-0.86, n=50)
Model B: 0.75 (95% CI: 0.71-0.79, n=50)
```

## Integration with Evaluation Engine

The confidence interval calculator integrates seamlessly with the evaluation engine:

```python
from src.evaluation.evaluation_engine import EvaluationEngine
from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine

# Run evaluation
evaluation_report = await evaluation_engine.run_baseline_evaluation(config)

# Extract trust scores
trust_scores = [result.trust_score.overall_score 
                for result in evaluation_report.results]

# Calculate confidence interval
trust_engine = TrustScoringEngine()
ci_result = trust_engine.calculate_confidence_interval(
    scores=trust_scores,
    confidence_level=0.95
)

print(f"Evaluation Results:")
print(f"  Mean Trust Score: {ci_result['mean']:.3f}")
print(f"  95% CI: [{ci_result['lower_bound']:.3f}, {ci_result['upper_bound']:.3f}]")
print(f"  Sample Size: {ci_result['sample_size']}")
print(f"  Method: {ci_result['method']}")
```

## Technical Details

### Bootstrap Algorithm

1. Set random seed for reproducibility (seed=42)
2. Generate 10,000 bootstrap samples
3. For each bootstrap sample:
   - Randomly sample n scores with replacement
   - Calculate mean of bootstrap sample
4. Sort bootstrap means
5. Extract percentiles for confidence bounds

### Normal Approximation Formula

```
Standard Error (SE) = σ / √n
Margin of Error (ME) = z × SE
Lower Bound = mean - ME
Upper Bound = mean + ME

Where:
- σ = standard deviation of scores
- n = sample size
- z = z-score for confidence level (e.g., 1.96 for 95%)
```

### Boundary Handling

Since trust scores must be in [0, 1], confidence bounds are clipped:

```python
lower_bound = max(0.0, min(1.0, calculated_lower))
upper_bound = max(0.0, min(1.0, calculated_upper))
```

## Limitations

1. **Assumes Independence**: Scores should be from independent evaluations
2. **Sample Size**: Very small samples (<5) may produce unreliable intervals
3. **Distribution**: Normal approximation assumes approximately normal distribution
4. **Boundary Effects**: Scores near 0 or 1 may have asymmetric intervals

## References

- Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap
- Wasserman, L. (2004). All of Statistics: A Concise Course in Statistical Inference
- Requirements: 5.10 - Calculate confidence intervals for aggregate trust scores

## See Also

- [Trust Scoring Engine V2](trust_scoring_engine_v2.md)
- [Trust Score Calibration](trust_score_calibration.md)
- [Evaluation Engine](../evaluation/evaluation_engine.md)
