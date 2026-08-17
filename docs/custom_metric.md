# Custom Metric Plugin Interface

## Overview

The Custom Metric Plugin Interface allows users to extend the TrustOps Trust Scoring Engine with custom scoring functions. This enables domain-specific trust metrics to be integrated seamlessly into the overall trust score calculation.

**Requirements**: 5.6, 5.11

## Features

- **Extensible Architecture**: Define custom scoring logic by extending the `CustomMetric` base class
- **Configuration Support**: Pass custom configuration parameters to your metrics
- **Registry Management**: Register, unregister, and manage multiple custom metrics
- **Seamless Integration**: Custom metrics integrate with the existing trust scoring workflow
- **Type Safety**: Full type hints and Pydantic validation

## Quick Start

### 1. Define a Custom Metric

Create a custom metric by extending the `CustomMetric` base class:

```python
from src.trust_scoring.custom_metric import CustomMetric
from src.data_models.trust_score import DimensionScore, TrustDimension

class DomainComplianceMetric(CustomMetric):
    """Custom metric for domain-specific compliance checking."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.compliance_keywords = config.get('keywords', [])
        self.threshold = config.get('threshold', 0.7)
    
    async def calculate(
        self,
        prompt: str,
        response: str,
        expected_response: str = None,
        source_documents: list[str] = None,
        model_id: str = None,
        **kwargs
    ) -> DimensionScore:
        """Calculate compliance score based on keyword presence."""
        # Count compliance keywords in response
        keyword_count = sum(
            1 for keyword in self.compliance_keywords
            if keyword.lower() in response.lower()
        )
        
        # Calculate score (0-1 range)
        score = min(keyword_count / len(self.compliance_keywords), 1.0) if self.compliance_keywords else 0.5
        
        # Determine checks passed/failed
        checks_passed = []
        checks_failed = []
        
        if score >= self.threshold:
            checks_passed.append("compliance_threshold_met")
        else:
            checks_failed.append("compliance_threshold_not_met")
        
        return DimensionScore(
            dimension=TrustDimension.ACCURACY,  # Or use a custom dimension
            score=score,
            confidence=0.9,
            details={
                "method": "keyword_compliance",
                "keyword_count": keyword_count,
                "total_keywords": len(self.compliance_keywords),
                "threshold": self.threshold
            },
            checks_passed=checks_passed,
            checks_failed=checks_failed
        )
```

### 2. Register the Custom Metric

Register your metric with the global registry:

```python
from src.trust_scoring.custom_metric import register_custom_metric

# Register the metric
register_custom_metric('domain_compliance', DomainComplianceMetric)
```

### 3. Use the Custom Metric

Instantiate and use your custom metric:

```python
from src.trust_scoring.custom_metric import get_global_registry

# Get the registry
registry = get_global_registry()

# Instantiate with configuration
config = {
    'keywords': ['HIPAA', 'compliant', 'secure', 'encrypted'],
    'threshold': 0.75
}
metric = registry.instantiate('domain_compliance', config=config)

# Calculate score
result = await metric.calculate(
    prompt="How do you handle patient data?",
    response="We handle patient data in a HIPAA compliant manner using encrypted storage."
)

print(f"Compliance Score: {result.score}")
print(f"Checks Passed: {result.checks_passed}")
```

## API Reference

### CustomMetric Base Class

Abstract base class for implementing custom metrics.

#### Methods

##### `__init__(config: Optional[dict] = None)`

Initialize the custom metric with optional configuration.

**Parameters:**
- `config` (dict, optional): Configuration dictionary for the metric

##### `calculate(prompt, response, **kwargs) -> DimensionScore` (abstract)

Calculate the custom metric score. Must be implemented by subclasses.

**Parameters:**
- `prompt` (str): The original prompt/query
- `response` (str): The model's response text
- `expected_response` (str, optional): Expected answer for comparison
- `source_documents` (list[str], optional): Context documents for grounding
- `model_id` (str, optional): Model ID for model-specific scoring
- `**kwargs`: Additional custom parameters

**Returns:**
- `DimensionScore`: Score result with value in [0, 1] range

##### `get_name() -> str`

Get the name of the custom metric. Defaults to class name.

**Returns:**
- `str`: Metric name

##### `get_description() -> str`

Get a description of what the metric measures.

**Returns:**
- `str`: Metric description

##### `validate_config() -> bool`

Validate the metric configuration. Override to implement custom validation.

**Returns:**
- `bool`: True if configuration is valid

### CustomMetricRegistry

Registry for managing custom metrics.

#### Methods

##### `register(name: str, metric_class: type[CustomMetric])`

Register a custom metric class.

**Parameters:**
- `name` (str): Unique identifier for the metric
- `metric_class` (type[CustomMetric]): Class extending CustomMetric

**Raises:**
- `ValueError`: If name already registered or class invalid

##### `unregister(name: str)`

Unregister a custom metric.

**Parameters:**
- `name` (str): Name of the metric to unregister

**Raises:**
- `KeyError`: If metric not registered

##### `get(name: str) -> type[CustomMetric]`

Get a registered metric class.

**Parameters:**
- `name` (str): Name of the metric

**Returns:**
- `type[CustomMetric]`: The metric class

**Raises:**
- `KeyError`: If metric not registered

##### `instantiate(name: str, config: Optional[dict] = None) -> CustomMetric`

Instantiate a registered metric.

**Parameters:**
- `name` (str): Name of the metric
- `config` (dict, optional): Configuration for the metric

**Returns:**
- `CustomMetric`: Instance of the metric

##### `list_metrics() -> list[str]`

List all registered metric names.

**Returns:**
- `list[str]`: List of registered metric names

##### `is_registered(name: str) -> bool`

Check if a metric is registered.

**Parameters:**
- `name` (str): Name of the metric

**Returns:**
- `bool`: True if registered

##### `clear()`

Clear all registered metrics.

### Global Registry Functions

##### `register_custom_metric(name: str, metric_class: type[CustomMetric])`

Convenience function to register a metric in the global registry.

##### `get_global_registry() -> CustomMetricRegistry`

Get the global custom metric registry instance.

## Advanced Examples

### Example 1: Sentiment Analysis Metric

```python
from src.trust_scoring.custom_metric import CustomMetric
from src.data_models.trust_score import DimensionScore, TrustDimension

class SentimentMetric(CustomMetric):
    """Custom metric for sentiment analysis."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.target_sentiment = config.get('target_sentiment', 'neutral')
        self.positive_words = ['good', 'great', 'excellent', 'positive']
        self.negative_words = ['bad', 'poor', 'negative', 'terrible']
    
    async def calculate(
        self,
        prompt: str,
        response: str,
        **kwargs
    ) -> DimensionScore:
        """Calculate sentiment score."""
        response_lower = response.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in response_lower)
        negative_count = sum(1 for word in self.negative_words if word in response_lower)
        
        # Calculate sentiment score
        if positive_count + negative_count == 0:
            sentiment_score = 0.5  # Neutral
        else:
            sentiment_score = positive_count / (positive_count + negative_count)
        
        # Adjust based on target sentiment
        if self.target_sentiment == 'positive':
            score = sentiment_score
        elif self.target_sentiment == 'negative':
            score = 1.0 - sentiment_score
        else:  # neutral
            score = 1.0 - abs(sentiment_score - 0.5) * 2
        
        return DimensionScore(
            dimension=TrustDimension.BIAS,
            score=score,
            confidence=0.8,
            details={
                "method": "sentiment_analysis",
                "positive_count": positive_count,
                "negative_count": negative_count,
                "target_sentiment": self.target_sentiment
            },
            checks_passed=["sentiment_appropriate"] if score > 0.6 else [],
            checks_failed=[] if score > 0.6 else ["sentiment_inappropriate"]
        )
```

### Example 2: Length Constraint Metric

```python
class LengthConstraintMetric(CustomMetric):
    """Custom metric for response length constraints."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.min_length = config.get('min_length', 50)
        self.max_length = config.get('max_length', 500)
        self.optimal_length = config.get('optimal_length', 200)
    
    async def calculate(
        self,
        prompt: str,
        response: str,
        **kwargs
    ) -> DimensionScore:
        """Calculate length constraint score."""
        length = len(response)
        
        # Check constraints
        checks_passed = []
        checks_failed = []
        
        if length < self.min_length:
            checks_failed.append("too_short")
            score = length / self.min_length
        elif length > self.max_length:
            checks_failed.append("too_long")
            score = self.max_length / length
        else:
            checks_passed.append("length_within_bounds")
            # Score based on proximity to optimal length
            deviation = abs(length - self.optimal_length)
            max_deviation = max(
                self.optimal_length - self.min_length,
                self.max_length - self.optimal_length
            )
            score = 1.0 - (deviation / max_deviation)
        
        return DimensionScore(
            dimension=TrustDimension.ACCURACY,
            score=max(0.0, min(1.0, score)),
            confidence=1.0,
            details={
                "method": "length_constraint",
                "length": length,
                "min_length": self.min_length,
                "max_length": self.max_length,
                "optimal_length": self.optimal_length
            },
            checks_passed=checks_passed,
            checks_failed=checks_failed
        )
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        return (
            self.min_length > 0 and
            self.max_length > self.min_length and
            self.min_length <= self.optimal_length <= self.max_length
        )
```

### Example 3: Using Multiple Custom Metrics

```python
from src.trust_scoring.custom_metric import CustomMetricRegistry

# Create a registry
registry = CustomMetricRegistry()

# Register multiple metrics
registry.register('sentiment', SentimentMetric)
registry.register('length', LengthConstraintMetric)
registry.register('compliance', DomainComplianceMetric)

# Instantiate all metrics
sentiment_metric = registry.instantiate('sentiment', {
    'target_sentiment': 'positive'
})

length_metric = registry.instantiate('length', {
    'min_length': 100,
    'max_length': 300,
    'optimal_length': 200
})

compliance_metric = registry.instantiate('compliance', {
    'keywords': ['secure', 'encrypted', 'compliant'],
    'threshold': 0.7
})

# Use all metrics on the same response
response = "Our secure and encrypted system ensures compliant data handling."

sentiment_result = await sentiment_metric.calculate(
    prompt="How secure is your system?",
    response=response
)

length_result = await length_metric.calculate(
    prompt="How secure is your system?",
    response=response
)

compliance_result = await compliance_metric.calculate(
    prompt="How secure is your system?",
    response=response
)

# Combine scores (simple average)
overall_custom_score = (
    sentiment_result.score +
    length_result.score +
    compliance_result.score
) / 3

print(f"Overall Custom Score: {overall_custom_score:.2f}")
```

## Best Practices

### 1. Score Range

Always ensure your custom metric returns scores in the [0, 1] range:

```python
score = max(0.0, min(1.0, calculated_score))
```

### 2. Confidence Levels

Set appropriate confidence levels based on the reliability of your metric:

- High confidence (0.9-1.0): Deterministic checks, exact matches
- Medium confidence (0.7-0.9): Statistical analysis, pattern matching
- Low confidence (0.5-0.7): Heuristic-based, uncertain measurements

### 3. Detailed Information

Provide detailed information in the `details` dictionary for debugging and analysis:

```python
details={
    "method": "your_method_name",
    "intermediate_values": {...},
    "thresholds_used": {...},
    "data_points_analyzed": count
}
```

### 4. Checks Passed/Failed

Use descriptive check names that clearly indicate what was tested:

```python
checks_passed=["keyword_present", "length_appropriate", "format_valid"]
checks_failed=["sentiment_negative", "contains_pii"]
```

### 5. Configuration Validation

Implement `validate_config()` to catch configuration errors early:

```python
def validate_config(self) -> bool:
    """Validate configuration."""
    if self.threshold < 0.0 or self.threshold > 1.0:
        return False
    if self.min_value >= self.max_value:
        return False
    return True
```

### 6. Error Handling

Handle errors gracefully and return neutral scores when necessary:

```python
async def calculate(self, prompt: str, response: str, **kwargs) -> DimensionScore:
    try:
        score = self._complex_calculation(response)
    except Exception as e:
        # Log error and return neutral score
        return DimensionScore(
            dimension=TrustDimension.ACCURACY,
            score=0.5,
            confidence=0.0,
            details={"error": str(e)},
            checks_passed=[],
            checks_failed=["calculation_error"]
        )
```

## Integration with TrustScoringEngine

Custom metrics can be integrated into the TrustScoringEngine workflow. While the current implementation focuses on the five standard dimensions, custom metrics can be:

1. **Calculated alongside standard dimensions** for additional insights
2. **Used for domain-specific filtering** before trust scoring
3. **Combined with standard scores** using custom weighting schemes
4. **Stored in evaluation results** for later analysis

Future versions will support direct integration of custom metrics into the weighted trust score calculation.

## Testing Custom Metrics

Always test your custom metrics thoroughly:

```python
import pytest
from your_module import YourCustomMetric

@pytest.mark.asyncio
async def test_custom_metric_score_range():
    """Test that scores are in valid range."""
    metric = YourCustomMetric()
    
    result = await metric.calculate(
        prompt="test",
        response="test response"
    )
    
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0

@pytest.mark.asyncio
async def test_custom_metric_with_config():
    """Test metric with configuration."""
    config = {"threshold": 0.8}
    metric = YourCustomMetric(config=config)
    
    assert metric.validate_config() is True
```

## Troubleshooting

### Metric Not Found

```python
KeyError: "Custom metric 'my_metric' is not registered"
```

**Solution**: Ensure you've registered the metric before trying to use it:

```python
register_custom_metric('my_metric', MyMetricClass)
```

### Invalid Metric Class

```python
ValueError: "Metric class must extend CustomMetric"
```

**Solution**: Ensure your metric class extends `CustomMetric`:

```python
class MyMetric(CustomMetric):  # Must extend CustomMetric
    async def calculate(self, ...):
        ...
```

### Abstract Method Not Implemented

```python
TypeError: "Can't instantiate abstract class MyMetric without an implementation for abstract method 'calculate'"
```

**Solution**: Implement the required `calculate()` method:

```python
class MyMetric(CustomMetric):
    async def calculate(self, prompt: str, response: str, **kwargs) -> DimensionScore:
        # Your implementation here
        pass
```

## See Also

- [Trust Scoring Engine Documentation](trust_scoring_engine_v2.md)
- [Data Models Documentation](../src/data_models/trust_score.py)
- [Accuracy Scorer Example](../src/trust_scoring/scorers/accuracy_scorer.py)
