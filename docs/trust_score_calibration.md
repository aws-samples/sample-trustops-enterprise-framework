# Trust Score Calibration

## Overview

The Trust Score Calibration feature validates the accuracy of automated trust scores by comparing them against human-labeled ground truth data. This helps ensure that the trust scoring engine's assessments align with human judgment.

**Requirement**: 5.8 - Calibrate trust scores against human-labeled ground truth

## Purpose

Calibration serves several important purposes:

1. **Validation**: Verify that automated scores correlate with human expert assessments
2. **Tuning**: Identify when dimension weights need adjustment
3. **Confidence**: Provide statistical evidence of scoring accuracy
4. **Improvement**: Guide refinements to scoring algorithms

## How It Works

The calibration process:

1. Takes a set of responses with human-labeled trust scores
2. Calculates automated trust scores for the same responses
3. Compares automated vs. human scores using statistical measures
4. Generates a calibration report with correlation metrics

### Statistical Metrics

The calibration report includes:

- **Pearson Correlation**: Measures linear relationship between automated and human scores
- **Spearman Correlation**: Measures monotonic relationship (rank-based)
- **Mean Absolute Error (MAE)**: Average absolute difference between scores
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
- **P-values**: Statistical significance of correlations

## Usage

### Basic Calibration

```python
from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine

# Initialize the engine
engine = TrustScoringEngine()

# Prepare ground truth data
ground_truth_data = [
    {
        "prompt": "What is 2+2?",
        "response": "4",
        "expected_response": "4",
        "human_score": 0.95,  # Human expert's trust score
    },
    {
        "prompt": "What is the capital of France?",
        "response": "Paris is the capital",
        "expected_response": "Paris",
        "human_score": 0.90,
    },
    # ... more examples
]

# Run calibration
result = await engine.calibrate(ground_truth_data)

# View results
print(f"Pearson Correlation: {result['pearson_correlation']:.3f}")
print(f"Spearman Correlation: {result['spearman_correlation']:.3f}")
print(f"Mean Absolute Error: {result['mean_absolute_error']:.3f}")
print(f"RMSE: {result['rmse']:.3f}")
print(f"\nSummary: {result['summary']}")
```

### With Source Documents

For responses that should be grounded in source documents:

```python
ground_truth_data = [
    {
        "prompt": "What does the document say about climate change?",
        "response": "The document states that climate change is accelerating",
        "source_documents": [
            "Climate change is accelerating due to human activities.",
            "Global temperatures have risen 1.1°C since pre-industrial times."
        ],
        "human_score": 0.85,
    },
    # ... more examples
]

result = await engine.calibrate(ground_truth_data)
```

### With Model ID for Consistency Scoring

To include consistency scoring in calibration:

```python
from src.clients.inference_client import InferenceClient

# Initialize with inference client
inference_client = InferenceClient(...)
engine = TrustScoringEngine(inference_client=inference_client)

ground_truth_data = [
    {
        "prompt": "Explain photosynthesis",
        "response": "Photosynthesis is the process...",
        "model_id": "anthropic.claude-v2",
        "human_score": 0.80,
    },
    # ... more examples
]

result = await engine.calibrate(ground_truth_data)
```

## Ground Truth Data Format

Each ground truth item should include:

### Required Fields

- `prompt` (str): The original prompt/question
- `response` (str): The model's response
- `human_score` (float): Human expert's trust score in range [0, 1]

### Optional Fields

- `expected_response` (str): Expected/correct answer for accuracy scoring
- `source_documents` (list[str]): Context documents for grounding evaluation
- `model_id` (str): Model identifier for consistency scoring

## Interpreting Results

### Correlation Coefficients

**Pearson Correlation** (linear relationship):
- 0.8 - 1.0: Strong correlation - excellent alignment
- 0.5 - 0.8: Moderate correlation - good alignment
- 0.0 - 0.5: Weak correlation - needs improvement

**Spearman Correlation** (rank-based):
- Similar interpretation to Pearson
- More robust to outliers
- Better for non-linear relationships

### Error Metrics

**Mean Absolute Error (MAE)**:
- < 0.1: Excellent accuracy
- 0.1 - 0.2: Good accuracy
- 0.2 - 0.3: Acceptable accuracy
- > 0.3: Poor accuracy - review scoring criteria

**RMSE**:
- Always >= MAE
- Penalizes large errors more heavily
- Useful for identifying systematic biases

### P-values

- p < 0.05: Statistically significant correlation
- p >= 0.05: Correlation may be due to chance

## Calibration Report Structure

```python
{
    "pearson_correlation": 0.823,
    "pearson_pvalue": 0.001,
    "spearman_correlation": 0.815,
    "spearman_pvalue": 0.002,
    "mean_absolute_error": 0.087,
    "rmse": 0.112,
    "sample_size": 50,
    "score_pairs": [
        (0.85, 0.90),  # (predicted, actual)
        (0.72, 0.75),
        # ... more pairs
    ],
    "summary": "Calibration based on 50 samples shows strong correlation..."
}
```

## Best Practices

### Sample Size

- **Minimum**: 10-20 samples for basic validation
- **Recommended**: 50-100 samples for reliable statistics
- **Ideal**: 200+ samples for robust calibration

### Ground Truth Quality

1. **Expert Labeling**: Use domain experts for human scores
2. **Multiple Raters**: Average scores from multiple experts
3. **Clear Guidelines**: Establish consistent scoring criteria
4. **Diverse Examples**: Include varied quality levels and topics

### When to Calibrate

- **Initial Setup**: Validate scoring before production use
- **After Changes**: Re-calibrate when modifying weights or scorers
- **Periodic Review**: Regular calibration (e.g., quarterly)
- **Domain Shift**: When applying to new domains or use cases

### Improving Calibration

If calibration shows poor alignment:

1. **Adjust Weights**: Modify dimension weights in `TrustScoreConfig`
2. **Review Scorers**: Check individual dimension scorer logic
3. **Refine Criteria**: Update scoring criteria to match human judgment
4. **Add Context**: Provide more source documents or expected responses

## Example: Complete Calibration Workflow

```python
import asyncio
from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine
from src.data_models.trust_score import TrustScoreConfig, TrustScoreWeights

async def run_calibration():
    # Configure engine with custom weights
    config = TrustScoreConfig(
        weights=TrustScoreWeights(
            accuracy=0.30,
            consistency=0.15,
            safety=0.20,
            bias=0.15,
            context_grounding=0.20,
        ),
        review_threshold=0.6,
    )
    
    engine = TrustScoringEngine(config=config)
    
    # Load ground truth data
    ground_truth_data = [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of AI...",
            "expected_response": "Machine learning is a method of data analysis...",
            "human_score": 0.85,
        },
        # ... load more examples from file or database
    ]
    
    # Run calibration
    result = await engine.calibrate(ground_truth_data)
    
    # Analyze results
    print("=" * 60)
    print("CALIBRATION REPORT")
    print("=" * 60)
    print(f"Sample Size: {result['sample_size']}")
    print(f"\nCorrelation Metrics:")
    print(f"  Pearson:  {result['pearson_correlation']:.3f} (p={result['pearson_pvalue']:.4f})")
    print(f"  Spearman: {result['spearman_correlation']:.3f} (p={result['spearman_pvalue']:.4f})")
    print(f"\nError Metrics:")
    print(f"  MAE:  {result['mean_absolute_error']:.3f}")
    print(f"  RMSE: {result['rmse']:.3f}")
    print(f"\n{result['summary']}")
    print("=" * 60)
    
    # Check if calibration is acceptable
    if result['pearson_correlation'] >= 0.7 and result['mean_absolute_error'] <= 0.15:
        print("\n✓ Calibration PASSED - Scoring is well-aligned with human judgment")
    else:
        print("\n✗ Calibration NEEDS IMPROVEMENT - Consider adjusting weights")
        print("  Recommendations:")
        if result['pearson_correlation'] < 0.7:
            print("  - Review dimension weights")
            print("  - Check scorer implementations")
        if result['mean_absolute_error'] > 0.15:
            print("  - Increase sample size")
            print("  - Refine scoring criteria")
    
    return result

# Run the calibration
if __name__ == "__main__":
    result = asyncio.run(run_calibration())
```

## Integration with Evaluation Pipeline

Calibration can be integrated into the evaluation workflow:

```python
from src.evaluation.evaluation_engine import EvaluationEngine

# After running evaluations, calibrate against expert reviews
evaluation_results = await evaluation_engine.run_baseline_evaluation(config)

# Convert evaluation results to ground truth format
ground_truth_data = []
for result in evaluation_results.results:
    if result.expert_review_score:  # If expert reviewed
        ground_truth_data.append({
            "prompt": result.prompt,
            "response": result.actual_response,
            "expected_response": result.expected_response,
            "human_score": result.expert_review_score,
        })

# Calibrate
calibration_result = await engine.calibrate(ground_truth_data)
```

## API Reference

### `calibrate(ground_truth_data: list[dict]) -> dict`

Calibrate trust scores against human-labeled ground truth.

**Parameters**:
- `ground_truth_data`: List of ground truth items with human scores

**Returns**: Dictionary containing:
- `pearson_correlation`: Pearson correlation coefficient
- `pearson_pvalue`: P-value for Pearson correlation
- `spearman_correlation`: Spearman rank correlation
- `spearman_pvalue`: P-value for Spearman correlation
- `mean_absolute_error`: Average absolute difference
- `rmse`: Root mean squared error
- `sample_size`: Number of samples
- `score_pairs`: List of (predicted, actual) tuples
- `summary`: Human-readable summary

**Raises**:
- `ValueError`: If ground truth data is empty

## See Also

- [Trust Scoring Engine Documentation](trust_scoring_engine_v2.md)
- [Dimension Scorers](accuracy_scorer.md)
- [Evaluation Engine](../evaluation/evaluation_engine.md)
