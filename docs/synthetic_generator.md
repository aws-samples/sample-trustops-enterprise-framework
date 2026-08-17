# Synthetic Data Generator

## Overview

The Synthetic Data Generator provides functionality to augment datasets by generating paraphrased and varied examples using Large Language Models (LLMs). This is useful for increasing dataset size and diversity, which can improve model training quality and generalization.

## Features

- **Multiple Augmentation Strategies**: Paraphrase, diverse, and creative variations
- **Configurable Augmentation Factor**: Control how much to expand your dataset
- **Task-Aware Generation**: Context-aware prompts based on task type
- **Graceful Error Handling**: Continues generation even if some examples fail
- **Metadata Tracking**: Marks synthetic examples for traceability

## Installation

The synthetic data generator is part of the TrustOps Enterprise Framework and requires:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import asyncio
from src.datasets.synthetic_generator import SyntheticDataGenerator
from src.clients.inference_client import InferenceClient
from src.registry.model_registry import ModelRegistry
from src.utils.rate_limiter import RateLimiter

# Initialize dependencies
registry = ModelRegistry()
rate_limiter = RateLimiter()
inference_client = InferenceClient(registry, rate_limiter)

# Create generator
generator = SyntheticDataGenerator(
    inference_client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
)

# Original dataset
examples = [
    {
        "prompt": "What is the capital of France?",
        "completion": "The capital of France is Paris.",
        "category": "geography"
    },
    {
        "prompt": "Explain photosynthesis.",
        "completion": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "category": "science"
    }
]

# Generate synthetic examples (50% more examples)
synthetic = await generator.generate_synthetic(
    examples,
    augmentation_factor=1.5,
    strategy="paraphrase"
)

print(f"Generated {len(synthetic)} synthetic examples")
```

### Augmentation Strategies

#### 1. Paraphrase Strategy

Simple paraphrasing that maintains exact meaning while varying expression:

```python
synthetic = await generator.generate_synthetic(
    examples,
    augmentation_factor=2.0,  # Double the dataset
    strategy="paraphrase"
)
```

**Example:**
- Original: "What is the capital of France?"
- Paraphrased: "What is France's capital city?"

#### 2. Diverse Strategy

Creates variations with different phrasings, perspectives, or formality levels:

```python
synthetic = await generator.generate_synthetic(
    examples,
    augmentation_factor=1.5,
    strategy="diverse"
)
```

**Example:**
- Original: "What is the capital of France?"
- Diverse: "Could you tell me which city serves as France's capital?"

#### 3. Creative Strategy

Generates more substantial variations using different scenarios or contexts:

```python
synthetic = await generator.generate_synthetic(
    examples,
    augmentation_factor=1.5,
    strategy="creative"
)
```

**Example:**
- Original: "What is the capital of France?"
- Creative: "If you were planning a trip to France, which city would be the seat of government?"

### Task-Aware Generation

Provide task type for context-aware generation:

```python
from src.data_models.dataset import DatasetTaskType

synthetic = await generator.generate_synthetic(
    examples,
    augmentation_factor=1.5,
    strategy="paraphrase",
    task_type=DatasetTaskType.QA
)
```

Supported task types:
- `DatasetTaskType.QA` - Question answering
- `DatasetTaskType.SUMMARIZATION` - Text summarization
- `DatasetTaskType.CLASSIFICATION` - Text classification
- `DatasetTaskType.TEXT_GENERATION` - Text generation
- `DatasetTaskType.CHAT` - Conversational chat

### Convenience Function

Use the convenience function for quick generation:

```python
from src.datasets.synthetic_generator import generate_synthetic_data

synthetic = await generate_synthetic_data(
    examples=examples,
    inference_client=inference_client,
    augmentation_factor=2.0,
    strategy="diverse",
    task_type=DatasetTaskType.QA,
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
)
```

## Configuration

### Augmentation Factor

The augmentation factor determines how much to expand your dataset:

- `1.0` - No augmentation (returns empty list)
- `1.5` - 50% more examples (e.g., 100 → 150)
- `2.0` - Double the dataset (e.g., 100 → 200)
- `3.0` - Triple the dataset (e.g., 100 → 300)

**Formula:** `num_synthetic = len(examples) × (augmentation_factor - 1.0)`

### Model Selection

Choose an appropriate LLM for generation:

```python
# Fast and cost-effective (default)
generator = SyntheticDataGenerator(
    inference_client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
)

# Higher quality
generator = SyntheticDataGenerator(
    inference_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

# Maximum quality
generator = SyntheticDataGenerator(
    inference_client,
    model_id="anthropic.claude-3-opus-20240229-v1:0"
)
```

## Output Format

Synthetic examples preserve the original structure and add metadata:

```python
{
    "prompt": "What is France's capital city?",  # Paraphrased
    "completion": "Paris is the capital of France.",  # Optionally paraphrased
    "category": "geography",  # Preserved from original
    "_synthetic": True,  # Marks as synthetic
    "_augmentation_strategy": "paraphrase"  # Strategy used
}
```

## Field Support

The generator automatically detects and handles various field names:

**Prompt fields:**
- `prompt`
- `question`
- `text`
- `input`

**Completion fields:**
- `completion`
- `answer`
- `response`
- `output`
- `summary`

**Other fields:**
- All other fields are preserved unchanged
- Category/label fields are maintained for stratification

## Error Handling

The generator handles errors gracefully:

```python
# If some examples fail, generation continues
synthetic = await generator.generate_synthetic(
    examples,
    augmentation_factor=2.0,
    strategy="paraphrase"
)

# Successful examples are returned
# Failed examples are logged but don't stop generation
print(f"Successfully generated {len(synthetic)} examples")
```

Common errors:
- **LLM timeout**: Individual examples may timeout (30s default)
- **Rate limiting**: Handled by InferenceClient
- **Invalid responses**: Skipped and logged

## Best Practices

### 1. Start Small

Test with a small augmentation factor first:

```python
# Test with 10% augmentation
synthetic = await generator.generate_synthetic(
    examples[:10],  # Small sample
    augmentation_factor=1.1,
    strategy="paraphrase"
)
```

### 2. Choose Appropriate Strategy

- **Paraphrase**: Best for maintaining exact meaning (fine-tuning datasets)
- **Diverse**: Good for increasing variety (evaluation datasets)
- **Creative**: Best for exploring edge cases (robustness testing)

### 3. Review Quality

Always review a sample of synthetic examples:

```python
synthetic = await generator.generate_synthetic(examples, augmentation_factor=1.5)

# Review first 5 synthetic examples
for i, example in enumerate(synthetic[:5]):
    print(f"\n--- Synthetic Example {i+1} ---")
    print(f"Prompt: {example['prompt']}")
    print(f"Strategy: {example['_augmentation_strategy']}")
```

### 4. Combine with Original Data

Merge synthetic examples with original data:

```python
# Generate synthetic examples
synthetic = await generator.generate_synthetic(
    examples,
    augmentation_factor=1.5,
    strategy="paraphrase"
)

# Combine with original
augmented_dataset = examples + synthetic

print(f"Original: {len(examples)} examples")
print(f"Synthetic: {len(synthetic)} examples")
print(f"Total: {len(augmented_dataset)} examples")
```

### 5. Monitor Costs

Synthetic generation uses LLM inference:

```python
# Estimate cost before generation
num_examples = len(examples)
augmentation_factor = 2.0
num_synthetic = int(num_examples * (augmentation_factor - 1.0))

# Rough estimate: ~100 tokens per example
estimated_tokens = num_synthetic * 100
estimated_cost = estimated_tokens * 0.00025 / 1000  # Claude Haiku pricing

print(f"Estimated cost: ${estimated_cost:.2f}")
```

## Performance

### Generation Speed

Typical generation times (Claude 3 Haiku):

- **10 examples**: ~5-10 seconds
- **100 examples**: ~30-60 seconds
- **1000 examples**: ~5-10 minutes

### Concurrency

The generator processes examples sequentially to avoid rate limits. For faster generation:

1. Use a faster model (Claude Haiku)
2. Increase rate limits in your account
3. Process in batches

## Limitations

1. **Quality Variance**: Synthetic examples may vary in quality
2. **Semantic Drift**: Creative strategy may drift from original meaning
3. **Cost**: LLM inference costs scale with dataset size
4. **Rate Limits**: Subject to provider rate limits
5. **Language Support**: Best results with English; other languages may vary

## Troubleshooting

### Issue: No synthetic examples generated

**Solution:**
```python
# Check augmentation factor
if augmentation_factor <= 1.0:
    print("Augmentation factor must be > 1.0")

# Check examples have required fields
for ex in examples:
    if "prompt" not in ex and "question" not in ex:
        print(f"Example missing prompt field: {ex}")
```

### Issue: Low quality synthetic examples

**Solution:**
```python
# Try a better model
generator = SyntheticDataGenerator(
    inference_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"  # Higher quality
)

# Or use paraphrase strategy for more conservative generation
synthetic = await generator.generate_synthetic(
    examples,
    strategy="paraphrase"  # Most conservative
)
```

### Issue: Generation too slow

**Solution:**
```python
# Use faster model
generator = SyntheticDataGenerator(
    inference_client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0"  # Fastest
)

# Or reduce augmentation factor
synthetic = await generator.generate_synthetic(
    examples,
    augmentation_factor=1.2  # Smaller increase
)
```

## API Reference

### SyntheticDataGenerator

```python
class SyntheticDataGenerator:
    def __init__(
        self,
        inference_client: InferenceClient,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    )
```

**Parameters:**
- `inference_client`: InferenceClient for invoking LLMs
- `model_id`: Model to use for generation (default: Claude 3 Haiku)

### generate_synthetic

```python
async def generate_synthetic(
    self,
    examples: list[dict[str, Any]],
    augmentation_factor: float = 1.5,
    strategy: str = "paraphrase",
    task_type: Optional[DatasetTaskType] = None
) -> list[dict[str, Any]]
```

**Parameters:**
- `examples`: Original dataset examples
- `augmentation_factor`: Multiplier for dataset size (default: 1.5)
- `strategy`: Augmentation strategy ("paraphrase", "diverse", "creative")
- `task_type`: Optional task type for context-aware generation

**Returns:**
- List of synthetic examples with `_synthetic` and `_augmentation_strategy` metadata

**Raises:**
- `ValueError`: If examples is empty or augmentation_factor < 1.0

### generate_synthetic_data (convenience function)

```python
async def generate_synthetic_data(
    examples: list[dict[str, Any]],
    inference_client: InferenceClient,
    augmentation_factor: float = 1.5,
    strategy: str = "paraphrase",
    task_type: Optional[DatasetTaskType] = None,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
) -> list[dict[str, Any]]
```

Convenience function with same parameters as `generate_synthetic`.

## Examples

### Example 1: Augment QA Dataset

```python
qa_examples = [
    {"question": "What is Python?", "answer": "Python is a programming language."},
    {"question": "What is ML?", "answer": "ML is machine learning."}
]

synthetic = await generator.generate_synthetic(
    qa_examples,
    augmentation_factor=2.0,
    strategy="paraphrase",
    task_type=DatasetTaskType.QA
)

# Result: 2 synthetic examples
```

### Example 2: Create Diverse Variations

```python
examples = [
    {"prompt": "Summarize this article.", "completion": "Article summary..."}
]

synthetic = await generator.generate_synthetic(
    examples,
    augmentation_factor=3.0,
    strategy="diverse",
    task_type=DatasetTaskType.SUMMARIZATION
)

# Result: 2 diverse variations
```

### Example 3: Batch Processing

```python
# Process large dataset in batches
batch_size = 100
all_synthetic = []

for i in range(0, len(examples), batch_size):
    batch = examples[i:i+batch_size]
    synthetic = await generator.generate_synthetic(
        batch,
        augmentation_factor=1.5,
        strategy="paraphrase"
    )
    all_synthetic.extend(synthetic)
    print(f"Processed {i+len(batch)}/{len(examples)} examples")

print(f"Total synthetic: {len(all_synthetic)}")
```

## Related Documentation

- [Dataset Manager](dataset_manager.md) - Dataset management and preparation
- [Quality Analyzer](quality_analyzer.md) - Dataset quality analysis
- [Format Converter](format_converter.md) - Dataset format conversion
- [Splitter](splitter.md) - Train/validation/test splitting

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test cases in `tests/unit/test_synthetic_generator.py`
3. Consult the design document for implementation details
