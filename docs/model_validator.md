# Model-Specific Format Validator

## Overview

The Model-Specific Format Validator ensures that datasets meet the specific format requirements for fine-tuning different model families on AWS Bedrock. Each model family (Claude, Titan, Llama) has unique requirements for field names, data structure, and constraints.

## Supported Model Families

### Claude (Anthropic)
- **Required Fields**: `prompt`, `completion`
- **Optional Fields**: `system` (for system prompts)
- **Format**: JSONL
- **Minimum Examples**: 32
- **Maximum Tokens**: 32,000 per example

### Titan (Amazon)
- **Required Fields**: `inputText`, `outputText`
- **Optional Fields**: None
- **Format**: JSONL
- **Minimum Examples**: 32
- **Maximum Tokens**: 8,192 per example

### Llama (Meta)
- **Required Fields**: `prompt`, `completion`
- **Optional Fields**: None
- **Format**: JSONL
- **Minimum Examples**: 32
- **Maximum Tokens**: 4,096 per example

## Usage

### Basic Validation

```python
from src.datasets.model_validator import ModelValidator

# Validate a dataset for a specific model
result = ModelValidator.validate_for_model(
    file_path="path/to/dataset.jsonl",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

if result.valid:
    print("Dataset is valid for fine-tuning!")
else:
    print("Validation errors:")
    for error in result.errors:
        print(f"  - {error}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
```

### Get Format Requirements

```python
# Get format requirements for a model
requirements = ModelValidator.get_format_requirements(
    "amazon.titan-text-express-v1"
)

print(f"Model Family: {requirements['model_family']}")
print(f"Format: {requirements['format']}")
print(f"Required Fields: {requirements['required_fields']}")
print(f"Min Examples: {requirements['min_examples']}")
print(f"Max Tokens: {requirements['max_tokens']}")
```

### Detect Model Family

```python
from src.datasets.model_validator import ModelFamily

# Detect which family a model belongs to
family = ModelValidator.detect_model_family(
    "meta.llama2-13b-chat-v1"
)

print(f"Model Family: {family.value}")  # Output: llama
```

## Example Dataset Formats

### Claude Format

```jsonl
{"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}
{"prompt": "Explain photosynthesis", "completion": "Photosynthesis is the process by which plants convert light energy into chemical energy.", "system": "You are a science teacher."}
```

### Titan Format

```jsonl
{"inputText": "What is the capital of France?", "outputText": "The capital of France is Paris."}
{"inputText": "Explain photosynthesis", "outputText": "Photosynthesis is the process by which plants convert light energy into chemical energy."}
```

### Llama Format

```jsonl
{"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}
{"prompt": "Explain photosynthesis", "completion": "Photosynthesis is the process by which plants convert light energy into chemical energy."}
```

## Validation Checks

The validator performs the following checks:

1. **Format Check**: Ensures the dataset is in JSONL format (required for all Bedrock fine-tuning)
2. **Required Fields**: Verifies all required fields are present for the model family
3. **Field Types**: Ensures fields are strings (not numbers, arrays, etc.)
4. **Empty Values**: Checks that required fields are not empty
5. **Minimum Examples**: Validates the dataset has enough examples (minimum 32)
6. **Token Limits**: Warns if examples exceed recommended token limits
7. **JSON Validity**: Ensures each line is valid JSON

## Error Handling

The validator returns a `ValidationResult` object with:
- `valid`: Boolean indicating if validation passed
- `errors`: List of error messages (validation failures)
- `warnings`: List of warning messages (recommendations)

```python
result = ModelValidator.validate_for_model(file_path, model_id)

# Check if valid
if result:  # ValidationResult can be used in boolean context
    print("Valid!")

# Access details
print(f"Valid: {result.valid}")
print(f"Errors: {len(result.errors)}")
print(f"Warnings: {len(result.warnings)}")
```

## Integration with Dataset Manager

The model validator integrates with the Dataset Manager for end-to-end validation:

```python
from src.datasets.dataset_manager import DatasetManager
from src.datasets.model_validator import ModelValidator

# Upload and validate dataset
dataset_manager = DatasetManager()
dataset = dataset_manager.upload_dataset(
    file_path="training_data.jsonl",
    name="Customer Support Dataset"
)

# Validate for specific model before fine-tuning
result = ModelValidator.validate_for_model(
    file_path="training_data.jsonl",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

if not result.valid:
    print("Dataset needs corrections before fine-tuning")
    for error in result.errors:
        print(f"  - {error}")
```

## Common Issues and Solutions

### Issue: "Missing required fields"
**Solution**: Ensure your dataset has the correct field names for the model family. Claude and Llama use `prompt`/`completion`, while Titan uses `inputText`/`outputText`.

### Issue: "Requires at least 32 examples"
**Solution**: Add more examples to your dataset. All Bedrock models require a minimum of 32 training examples.

### Issue: "AWS Bedrock fine-tuning requires JSONL format"
**Solution**: Convert your dataset to JSONL format using the format converter:
```python
from src.datasets.format_converter import FormatConverter

FormatConverter.convert(
    input_path="dataset.csv",
    output_path="dataset.jsonl",
    target_format=DatasetFormat.JSONL
)
```

### Issue: "Estimated tokens exceeds recommended maximum"
**Solution**: This is a warning, not an error. Consider splitting long examples into shorter ones or truncating text to stay within token limits.

## Best Practices

1. **Validate Early**: Run validation before starting expensive fine-tuning jobs
2. **Check Requirements**: Use `get_format_requirements()` to understand model-specific needs
3. **Handle Warnings**: Address token limit warnings to ensure optimal fine-tuning results
4. **Use Correct Format**: Always use JSONL format for Bedrock fine-tuning
5. **Test with Sample**: Validate a small sample dataset before processing large files

## API Reference

### ModelValidator

#### `validate_for_model(file_path, model_id, dataset_format=None)`
Validate a dataset for a specific model.

**Parameters:**
- `file_path` (str | Path): Path to the dataset file
- `model_id` (str): AWS Bedrock model identifier
- `dataset_format` (DatasetFormat, optional): Pre-detected format

**Returns:** `ValidationResult`

#### `detect_model_family(model_id)`
Detect model family from model ID.

**Parameters:**
- `model_id` (str): AWS Bedrock model identifier

**Returns:** `ModelFamily` enum

#### `get_format_requirements(model_id)`
Get format requirements for a model.

**Parameters:**
- `model_id` (str): AWS Bedrock model identifier

**Returns:** Dictionary with format requirements

### ValidationResult

#### Properties
- `valid` (bool): Whether validation passed
- `errors` (list[str]): List of error messages
- `warnings` (list[str]): List of warning messages

#### Methods
- `__bool__()`: Returns `valid` status for boolean context
- `__repr__()`: String representation of the result
