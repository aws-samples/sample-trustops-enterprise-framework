# Dataset Quality Analyzer

## Overview

The Dataset Quality Analyzer is a comprehensive tool for evaluating dataset quality across multiple dimensions. It helps data scientists and ML engineers understand and improve their training and evaluation datasets before using them for model fine-tuning or evaluation.

**Requirement:** 2.5 - Implement dataset quality analyzer

## Features

The quality analyzer calculates four key quality metrics:

1. **Completeness Score** (0-1): Measures the ratio of present required fields
2. **Diversity Score** (0-1): Measures the uniqueness of prompts
3. **Balance Score** (0-1): Measures the evenness of category distribution
4. **Token Statistics**: Provides detailed token counts and distributions

Additionally, it:
- Detects quality issues with severity levels (error, warning, info)
- Generates actionable recommendations for improvement
- Supports multiple dataset formats (JSONL, CSV, Parquet)
- Works with all task types (QA, summarization, classification, chat, etc.)

## Usage

### Basic Usage

```python
from src.datasets.quality_analyzer import QualityAnalyzer
from src.data_models.dataset import DatasetTaskType, DatasetFormat

# Create analyzer instance
analyzer = QualityAnalyzer()

# Analyze examples
examples = [
    {"prompt": "What is AI?", "completion": "Artificial Intelligence", "category": "tech"},
    {"prompt": "What is ML?", "completion": "Machine Learning", "category": "tech"},
    # ... more examples
]

report = analyzer.analyze(examples, DatasetTaskType.QA, DatasetFormat.JSONL)

# Access quality metrics
print(f"Completeness: {report.completeness_score:.2f}")
print(f"Diversity: {report.diversity_score:.2f}")
print(f"Balance: {report.balance_score:.2f}")

# View recommendations
for rec in report.recommendations:
    print(f"- {rec}")
```

### Analyzing a File

```python
from src.datasets.quality_analyzer import analyze_dataset_file
from src.data_models.dataset import DatasetTaskType, DatasetFormat

# Analyze a dataset file directly
report = analyze_dataset_file(
    "path/to/dataset.jsonl",
    DatasetTaskType.QA,
    DatasetFormat.JSONL
)

# Check for issues
for issue in report.issues:
    print(f"[{issue.severity}] {issue.message}")
```

## Quality Metrics

### Completeness Score

**Formula:** `1 - (missing_fields / total_expected_fields)`

Measures how many required fields are present in the dataset. A score of 1.0 means all examples have all required fields.

**Required fields by task type:**
- QA, Summarization, Classification, Text Generation: `prompt`, `completion`
- Chat: `messages`
- Custom: `prompt`

**Interpretation:**
- 1.0 - 0.9: Excellent completeness
- 0.9 - 0.7: Good, minor issues
- < 0.7: Poor, significant missing data

### Diversity Score

**Formula:** `unique_prompts / total_prompts`

Measures how unique the prompts are in the dataset. Higher diversity typically leads to better model generalization.

**Interpretation:**
- 1.0 - 0.8: Excellent diversity
- 0.8 - 0.5: Good diversity
- < 0.5: Low diversity, many duplicates

**Note:** Text is normalized (lowercase, whitespace) for comparison.

### Balance Score

**Formula:** `Shannon_entropy(categories) / log(num_categories)`

Measures how evenly distributed the categories are. Uses Shannon entropy normalized to [0, 1].

**Interpretation:**
- 1.0: Perfectly balanced
- 0.7 - 1.0: Well balanced
- < 0.7: Imbalanced distribution

**Note:** If no categories are present, returns 1.0 (no imbalance).

### Token Statistics

Provides detailed token count information:

- **total_tokens**: Sum of all tokens in the dataset
- **min_tokens**: Minimum tokens in any example
- **max_tokens**: Maximum tokens in any example
- **avg_tokens**: Average tokens per example
- **p95_tokens**: 95th percentile of token counts
- **prompt_tokens**: Total tokens in all prompts
- **completion_tokens**: Total tokens in all completions

**Token Counting:** Uses a simple approximation: `tokens ≈ words × 1.3`

## Quality Issues

The analyzer detects various quality issues with severity levels:

### Error Level
- Completeness < 0.7: Critical missing data

### Warning Level
- Completeness 0.7-0.9: Some missing data
- Diversity < 0.3: Very low diversity
- Balance < 0.5: Highly imbalanced
- Dataset size < 100: Small dataset

### Info Level
- Diversity 0.3-0.5: Moderate diversity issues
- Balance 0.5-0.7: Moderate imbalance

## Recommendations

The analyzer generates actionable recommendations with severity levels based on detected issues:

### Severity Levels

Recommendations are categorized by severity to help prioritize improvements:

- **[CRITICAL]**: Severe issues that will significantly impact model training or evaluation
- **[WARNING]**: Important issues that should be addressed for optimal results
- **[INFO]**: Minor improvements that may enhance quality
- **[SUCCESS]**: Positive feedback when dataset quality is excellent

### Completeness Recommendations

**Critical (< 0.7):**
- "Completeness is very low (X%). Immediately review and fill in missing required fields. Missing data can severely impact model training quality."

**Warning (0.7-0.9):**
- "Some examples are missing required fields (X% complete). Review dataset and fill in missing prompt or completion fields to improve training quality."

### Diversity Recommendations

**Critical (< 0.3):**
- "Diversity is very low (X%). Most prompts are duplicates. Add unique examples or use data augmentation techniques (paraphrasing, synonym replacement) to increase variety."

**Warning (0.3-0.5):**
- "Diversity is below optimal (X%). Consider adding more unique prompts or using synthetic data generation to improve model generalization."

**Info (0.5-0.7):**
- "Diversity could be improved (X%). Adding more varied examples may help model performance on edge cases."

### Balance Recommendations

**Critical (< 0.5):**
- "Categories are highly imbalanced (X%). Add examples to underrepresented categories or use stratified sampling to prevent model bias toward majority classes."

**Warning (0.5-0.7):**
- "Category distribution is uneven (X%). Consider balancing by adding examples to smaller categories or using weighted sampling during training."

**Info (0.7-0.85):**
- "Category balance could be improved (X%). More even distribution may help model performance across all categories."

### Token Length Recommendations

**Critical (max > 8000):**
- "Some examples exceed 8000 tokens (max: X). Most models have context limits of 4K-8K tokens. Split or truncate long examples to prevent training failures."

**Warning (max > 4000):**
- "Some examples are very long (max: X tokens). Verify they fit within your model's context window (typically 4K-8K tokens). Consider splitting or truncating if needed."

**Warning (avg < 10):**
- "Average token count is very low (X tokens). Add more detailed prompts and completions to provide sufficient context for model learning."

**Warning (min < 3):**
- "Some examples are very short (min: X tokens). Very brief examples may not provide enough context for effective model learning. Review and expand if needed."

**Warning (large variance):**
- "Large variance in example lengths (range: X tokens). Consider normalizing example lengths for more consistent training batches."

### Token Imbalance Recommendations

**Warning (prompt:completion ratio >= 10:1):**
- "Prompts are much longer than completions (ratio: X:1). Consider more concise prompts or more detailed completions for better training balance."

**Warning (prompt:completion ratio <= 0.1:1):**
- "Completions are much longer than prompts (ratio: 1:X). Ensure prompts provide sufficient context and instructions for the task."

**Info (moderate imbalance):**
- "Prompts/completions have notable length difference (ratio: X:1). Verify this matches your use case requirements."

### P95 Outlier Detection

**Info:**
- "95th percentile token count (X) is much higher than average (Y). You may have outlier examples that could affect training. Review longest examples."

### Success Message

When all metrics are within optimal ranges:
- "[SUCCESS] Dataset quality is excellent! All metrics are within optimal ranges. Your dataset is ready for model training or evaluation."

## Example Output

```
Quality Report: My Dataset
======================================================================

Quality Scores:
  Completeness: 0.95 / 1.00
  Diversity:    0.75 / 1.00
  Balance:      0.85 / 1.00

Token Statistics:
  Total tokens:      10,000
  Min tokens:        10
  Max tokens:        150
  Avg tokens:        50.0
  P95 tokens:        120
  Prompt tokens:     4,500
  Completion tokens: 5,500

Issues Detected (1):
  ⚠️ [WARNING] Dataset has low completeness score (0.95). 
     Some examples are missing required fields.

Recommendations (1):
  1. [WARNING] Some examples are missing required fields (95% complete). 
     Review dataset and fill in missing prompt or completion fields to 
     improve training quality.
```

### Example with Multiple Issues

```
Quality Report: Low Quality Dataset
======================================================================

Quality Scores:
  Completeness: 0.60 / 1.00
  Diversity:    0.25 / 1.00
  Balance:      0.40 / 1.00

Token Statistics:
  Total tokens:      500
  Min tokens:        2
  Max tokens:        5000
  Avg tokens:        8.0
  P95 tokens:        4500
  Prompt tokens:     400
  Completion tokens: 100

Issues Detected (4):
  ❌ [ERROR] Dataset has low completeness score (0.60).
  ⚠️ [WARNING] Dataset has low diversity score (0.25).
  ⚠️ [WARNING] Dataset has low balance score (0.40).
  ⚠️ [WARNING] Dataset is small (30 examples).

Recommendations (6):
  1. [CRITICAL] Completeness is very low (60%). Immediately review and 
     fill in missing required fields. Missing data can severely impact 
     model training quality.
  2. [CRITICAL] Diversity is very low (25%). Most prompts are duplicates. 
     Add unique examples or use data augmentation techniques.
  3. [CRITICAL] Categories are highly imbalanced (40%). Add examples to 
     underrepresented categories or use stratified sampling.
  4. [WARNING] Average token count is very low (8.0 tokens). Add more 
     detailed prompts and completions.
  5. [WARNING] Some examples are very long (max: 5000 tokens). Verify 
     they fit within your model's context window.
  6. [WARNING] Prompts are much longer than completions (ratio: 4.0:1). 
     Consider more concise prompts or more detailed completions.
```

## Integration with Dataset Manager

The quality analyzer is designed to integrate with the Dataset Manager:

```python
from src.datasets.dataset_manager import DatasetManager

# Upload and analyze in one step
manager = DatasetManager()
metadata = await manager.upload_dataset(
    "path/to/dataset.jsonl",
    name="My Dataset",
    task_type=DatasetTaskType.QA
)

# Quality report is automatically included
quality_report = metadata.quality_report
print(f"Completeness: {quality_report.completeness_score:.2f}")
```

## Supported Formats

The quality analyzer supports:

- **JSONL**: JSON Lines format (one JSON object per line)
- **CSV**: Comma-separated values with headers
- **Parquet**: Apache Parquet columnar format
- **HuggingFace**: HuggingFace dataset format (via file analysis)

## Supported Task Types

- **QA**: Question-answering datasets
- **Summarization**: Text summarization datasets
- **Classification**: Text classification datasets
- **Text Generation**: General text generation datasets
- **Chat**: Conversational/chat datasets
- **Custom**: Custom task types

## Field Extraction

The analyzer intelligently extracts prompts and completions from various field names:

**Prompt fields:** `prompt`, `question`, `text`, `input`, `messages[0].content`

**Completion fields:** `completion`, `answer`, `response`, `output`, `summary`, `messages[role=assistant].content`

## Best Practices

1. **Aim for high scores**: Target completeness > 0.9, diversity > 0.7, balance > 0.7
2. **Review recommendations**: Follow the actionable suggestions provided
3. **Iterate**: Re-analyze after making improvements
4. **Consider context**: Some use cases may require different thresholds
5. **Balance metrics**: Don't optimize one metric at the expense of others

## Performance Considerations

- **Memory**: Loads entire dataset into memory for analysis
- **Speed**: Processes ~1000 examples per second on typical hardware
- **Large datasets**: For datasets > 100K examples, consider sampling

## Limitations

1. **Token counting**: Uses approximation, not actual model tokenizer
2. **Semantic similarity**: Diversity based on text matching, not semantic meaning
3. **Language-agnostic**: Works with any language but optimized for English
4. **No content validation**: Doesn't check if completions are correct

## Demo

Run the demo script to see the quality analyzer in action:

```bash
python demo/quality_analyzer_demo.py
```

This demonstrates analysis of high, medium, and low quality datasets with detailed output.

## Testing

Comprehensive unit tests are available:

```bash
pytest tests/unit/test_quality_analyzer.py -v
```

Tests cover:
- All quality metric calculations
- Edge cases (empty datasets, single examples, etc.)
- File format support
- Unicode and special character handling
- Recommendation generation

## Related Components

- **Format Detector**: Auto-detects dataset format and task type
- **Dataset Parsers**: Parse various dataset formats
- **Model Validator**: Validates datasets against model requirements
- **Dataset Manager**: Orchestrates dataset operations including quality analysis
