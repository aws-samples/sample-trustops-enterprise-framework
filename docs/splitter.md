# Dataset Splitter

The Dataset Splitter provides functionality to split datasets into train, validation, and test sets with configurable ratios and optional stratified sampling.

## Overview

The `DatasetSplitter` class enables you to:
- Split datasets into train/validation/test sets with custom ratios
- Maintain reproducibility with random seeds
- Preserve category balance using stratified sampling
- Calculate split statistics and category distributions

## Requirements

**Requirement 2.13**: The Dataset_Manager SHALL split datasets into train/validation/test sets with configurable ratios.

## Usage

### Basic Split

Split a dataset with default 80/10/10 ratios:

```python
from src.datasets.splitter import DatasetSplitter

# Create sample dataset
examples = [
    {"id": i, "prompt": f"Question {i}", "completion": f"Answer {i}"}
    for i in range(100)
]

# Initialize splitter with seed for reproducibility
splitter = DatasetSplitter(random_seed=42)

# Split dataset
train, validation, test = splitter.split(examples)

print(f"Train: {len(train)} examples")
print(f"Validation: {len(validation)} examples")
print(f"Test: {len(test)} examples")
```

### Custom Ratios

Specify custom split ratios:

```python
# 70% train, 15% validation, 15% test
train, validation, test = splitter.split(
    examples,
    train_ratio=0.7,
    validation_ratio=0.15,
    test_ratio=0.15
)
```

### Stratified Sampling

Maintain category balance across splits using stratified sampling:

```python
# Dataset with categories
examples = [
    {"id": i, "text": f"Example {i}", "category": "A" if i < 60 else "B"}
    for i in range(100)
]

# Split with stratification by category
train, validation, test = splitter.split(
    examples,
    stratify_by="category"
)

# Each split will maintain the 60/40 A/B ratio
```

### Convenience Function

Use the convenience function for quick splits:

```python
from src.datasets.splitter import split_dataset

train, validation, test = split_dataset(
    examples,
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
    stratify_by="category",
    random_seed=42
)
```

## Split Statistics

Get detailed statistics about your splits:

```python
# Get split statistics
stats = splitter.get_split_statistics(
    train, validation, test,
    stratify_by="category"
)

print(f"Total examples: {stats['total_examples']}")
print(f"Train size: {stats['train_size']} ({stats['train_ratio']:.1%})")
print(f"Validation size: {stats['validation_size']} ({stats['validation_ratio']:.1%})")
print(f"Test size: {stats['test_size']} ({stats['test_ratio']:.1%})")

# View category distributions
if "category_distributions" in stats:
    print("\nCategory distributions:")
    for split_name, distribution in stats["category_distributions"].items():
        print(f"  {split_name}: {distribution}")
```

## API Reference

### DatasetSplitter

#### Constructor

```python
DatasetSplitter(random_seed: Optional[int] = 42)
```

**Parameters:**
- `random_seed`: Random seed for reproducibility. Set to `None` for non-deterministic splits.

#### split()

```python
split(
    examples: list[dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_by: Optional[str] = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
```

Split dataset into train, validation, and test sets.

**Parameters:**
- `examples`: List of dataset examples to split
- `train_ratio`: Ratio of examples for training set (default: 0.8)
- `validation_ratio`: Ratio of examples for validation set (default: 0.1)
- `test_ratio`: Ratio of examples for test set (default: 0.1)
- `stratify_by`: Optional field name to stratify by (e.g., 'category', 'label')

**Returns:**
- Tuple of (train_examples, validation_examples, test_examples)

**Raises:**
- `ValueError`: If ratios don't sum to 1.0 or if examples list is empty

#### get_split_statistics()

```python
get_split_statistics(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    stratify_by: Optional[str] = None
) -> dict[str, Any]
```

Calculate statistics about the split.

**Parameters:**
- `train`: Training set examples
- `validation`: Validation set examples
- `test`: Test set examples
- `stratify_by`: Optional field name that was used for stratification

**Returns:**
- Dictionary with split statistics including:
  - `total_examples`: Total number of examples
  - `train_size`, `validation_size`, `test_size`: Size of each split
  - `train_ratio`, `validation_ratio`, `test_ratio`: Actual ratios achieved
  - `category_distributions`: Category counts per split (if stratify_by provided)

## Stratification

### How It Works

Stratified sampling ensures that category proportions are maintained across all splits:

1. Examples are grouped by the stratification field
2. Each category is split independently using the specified ratios
3. Results are combined and shuffled

### Supported Field Names

The splitter automatically recognizes these field names for stratification:
- `category`
- `label`
- `class`
- `type`

You can specify any field name, and the splitter will fall back to common alternatives if the exact field is not found.

### Example

```python
# Dataset with imbalanced categories
examples = [
    *[{"id": i, "category": "A"} for i in range(70)],  # 70% category A
    *[{"id": i+70, "category": "B"} for i in range(30)]  # 30% category B
]

# Without stratification - categories may be unevenly distributed
train, val, test = splitter.split(examples)

# With stratification - maintains 70/30 ratio in all splits
train, val, test = splitter.split(examples, stratify_by="category")
```

## Best Practices

### 1. Use Reproducible Seeds

Always use a fixed seed for reproducibility in experiments:

```python
splitter = DatasetSplitter(random_seed=42)
```

### 2. Validate Ratios

Ensure ratios sum to 1.0:

```python
# Good
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15
assert abs(train_ratio + validation_ratio + test_ratio - 1.0) < 0.001

# Bad - will raise ValueError
train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.2  # Sum = 1.1
```

### 3. Use Stratification for Imbalanced Data

For datasets with imbalanced categories, always use stratification:

```python
# Check if dataset is imbalanced
from collections import Counter
categories = [ex["category"] for ex in examples]
distribution = Counter(categories)
print(f"Category distribution: {distribution}")

# Use stratification if imbalanced
if max(distribution.values()) / min(distribution.values()) > 2:
    train, val, test = splitter.split(examples, stratify_by="category")
```

### 4. Verify Split Quality

Always check split statistics after splitting:

```python
train, val, test = splitter.split(examples, stratify_by="category")
stats = splitter.get_split_statistics(train, val, test, stratify_by="category")

# Verify ratios are as expected
assert 0.79 <= stats["train_ratio"] <= 0.81
assert 0.09 <= stats["validation_ratio"] <= 0.11
assert 0.09 <= stats["test_ratio"] <= 0.11

# Verify category balance
for split_name, dist in stats["category_distributions"].items():
    print(f"{split_name}: {dist}")
```

### 5. Handle Small Datasets

For small datasets, be aware of rounding effects:

```python
# With 10 examples and 80/10/10 ratios:
# train=8, validation=1, test=1
examples = [{"id": i} for i in range(10)]
train, val, test = splitter.split(examples)
print(f"Sizes: train={len(train)}, val={len(val)}, test={len(test)}")
```

## Common Patterns

### Train-Only Split

Create only a training set:

```python
train, _, _ = splitter.split(
    examples,
    train_ratio=1.0,
    validation_ratio=0.0,
    test_ratio=0.0
)
```

### Train-Test Split (No Validation)

Create train and test sets without validation:

```python
train, _, test = splitter.split(
    examples,
    train_ratio=0.8,
    validation_ratio=0.0,
    test_ratio=0.2
)
```

### Equal Three-Way Split

Split evenly into three sets:

```python
train, val, test = splitter.split(
    examples,
    train_ratio=1/3,
    validation_ratio=1/3,
    test_ratio=1/3
)
```

## Error Handling

The splitter validates inputs and provides clear error messages:

```python
# Empty dataset
try:
    splitter.split([])
except ValueError as e:
    print(f"Error: {e}")  # "Cannot split empty dataset"

# Invalid ratios
try:
    splitter.split(examples, train_ratio=0.5, validation_ratio=0.3, test_ratio=0.3)
except ValueError as e:
    print(f"Error: {e}")  # "Ratios must sum to 1.0, got 1.1000..."

# Missing stratification field
try:
    splitter.split(examples, stratify_by="nonexistent_field")
except ValueError as e:
    print(f"Error: {e}")  # "Stratification field 'nonexistent_field' not found..."
```

## Integration with Dataset Manager

The splitter integrates with the Dataset Manager for complete dataset workflows:

```python
from src.datasets.dataset_manager import DatasetManager
from src.datasets.splitter import DatasetSplitter

# Load dataset
manager = DatasetManager()
dataset = manager.load_dataset("my_dataset_id")

# Split dataset
splitter = DatasetSplitter(random_seed=42)
train, val, test = splitter.split(
    dataset.examples,
    stratify_by="category"
)

# Save splits as separate datasets
train_dataset = manager.save_dataset(train, "my_dataset_train")
val_dataset = manager.save_dataset(val, "my_dataset_val")
test_dataset = manager.save_dataset(test, "my_dataset_test")
```

## Performance Considerations

- **Memory**: The splitter creates copies of the dataset, so memory usage is approximately 3x the original dataset size
- **Speed**: Splitting is O(n) for random splits and O(n log n) for stratified splits due to shuffling
- **Large Datasets**: For datasets with millions of examples, consider using streaming or chunked processing

## See Also

- [Dataset Manager](dataset_manager.md) - Complete dataset management
- [Quality Analyzer](quality_analyzer.md) - Dataset quality analysis
- [Format Converter](format_converter.md) - Dataset format conversion
