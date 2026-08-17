# DatasetManager

The `DatasetManager` is a high-level orchestration class that integrates all dataset components in the TrustOps Enterprise Framework. It provides a unified API for managing the complete dataset lifecycle from upload to versioning.

## Overview

The DatasetManager orchestrates:
- **Upload**: Automatic format detection and validation
- **Quality Analysis**: Completeness, diversity, and balance metrics
- **PII Detection**: Identify and mask sensitive information
- **Transformations**: Splitting, format conversion, synthetic generation
- **Versioning**: Track changes and lineage
- **Model Validation**: Ensure compatibility with specific models

## Requirements

Implements requirements 2.1-2.19 from the TrustOps Enterprise Framework specification.

## Architecture

```
DatasetManager
├── FormatDetector      # Auto-detect format and task type
├── QualityAnalyzer     # Calculate quality metrics
├── PIIDetector         # Detect and mask PII
├── DatasetSplitter     # Split into train/val/test
├── SyntheticGenerator  # Generate augmented data
├── VersionManager      # Track versions and lineage
├── FormatConverter     # Convert between formats
└── ModelValidator      # Validate model compatibility
```

## Installation

```python
from src.datasets.dataset_manager import DatasetManager
from src.clients.inference_client import InferenceClient

# Initialize with AWS configuration
manager = DatasetManager(
    region="us-east-1",
    datasets_bucket="my-datasets-bucket",
    metadata_table="my-datasets-table",
    inference_client=inference_client,  # Optional, for synthetic generation
    max_dataset_size_gb=10.0,
)
```

## Core Operations

### 1. Upload Dataset

Upload a dataset with automatic format and task type detection:

```python
metadata = await manager.upload_dataset(
    file_path="data/training_data.jsonl",
    name="Customer Support QA",
    description="Question-answering dataset for customer support",
)

print(f"Dataset ID: {metadata.id}")
print(f"Format: {metadata.format.value}")
print(f"Task Type: {metadata.task_type.value}")
print(f"Row Count: {metadata.row_count}")
```

**Features:**
- Automatic format detection (JSONL, CSV, Parquet, HuggingFace)
- Automatic task type detection (QA, summarization, classification, etc.)
- File size validation (configurable limit, default 10GB)
- Quality analysis on upload
- Automatic versioning (creates v1)
- S3 storage with DynamoDB metadata

**Size Validation:**

The DatasetManager enforces a configurable maximum file size to prevent resource exhaustion and ensure efficient processing. By default, the limit is set to 10GB.

```python
# Use default 10GB limit
manager = DatasetManager(region="us-east-1")

# Configure custom limit (e.g., 5GB)
manager = DatasetManager(
    region="us-east-1",
    max_dataset_size_gb=5.0
)
```

When a file exceeds the limit, a clear error message is returned:

```python
try:
    await manager.upload_dataset(
        file_path="large_dataset.jsonl",
        name="Large Dataset"
    )
except ValueError as e:
    # Error message includes:
    # - Actual file size in GB
    # - Maximum allowed size in GB
    # - Remediation suggestions (split files, contact support)
    print(e)
    # Output: "Dataset size (15.23 GB) exceeds maximum allowed size (10.00 GB). 
    #          Please split the dataset into smaller files or contact support 
    #          to increase the limit."
```

**Key Benefits:**
- **Fail Fast**: Size validation occurs before expensive parsing operations
- **Clear Feedback**: Error messages include actual vs. maximum size
- **Actionable Guidance**: Suggests remediation steps (split files, contact support)
- **Configurable**: Adjust limits based on infrastructure capacity

### 2. Validate Dataset

Validate dataset quality and optionally check model compatibility:

```python
# Basic quality validation
quality_report = await manager.validate_dataset(dataset_id)

print(f"Completeness: {quality_report.completeness_score:.2f}")
print(f"Diversity: {quality_report.diversity_score:.2f}")
print(f"Balance: {quality_report.balance_score:.2f}")

# Show issues
for issue in quality_report.issues:
    print(f"[{issue.severity}] {issue.message}")

# Show recommendations
for rec in quality_report.recommendations:
    print(f"• {rec}")

# Validate against specific model
quality_report = await manager.validate_dataset(
    dataset_id,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)
```

**Quality Metrics:**
- **Completeness**: Ratio of complete examples (0-1)
- **Diversity**: Ratio of unique prompts (0-1)
- **Balance**: Category distribution evenness (0-1)
- **Token Statistics**: Min, max, avg, p95 token counts

### 3. Detect and Mask PII

Identify and mask personally identifiable information:

```python
# Detect PII
pii_result = await manager.detect_pii(dataset_id)

print(f"Rows with PII: {pii_result.rows_with_pii}")
print(f"PII Percentage: {pii_result.pii_percentage:.1f}%")

for pii_type, count in pii_result.pii_type_counts.items():
    print(f"  {pii_type.value}: {count} instances")

# Mask all PII types
masked_metadata = await manager.mask_pii(
    dataset_id,
    strategy="mask"  # Options: "mask", "redact", "remove"
)

# Mask specific PII types only
from src.datasets.pii_detector import PIIType

masked_metadata = await manager.mask_pii(
    dataset_id,
    strategy="mask",
    selective_types=[PIIType.EMAIL, PIIType.PHONE]
)
```

**PII Types Detected:**
- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- Credit card numbers
- IP addresses

**Masking Strategies:**
- `mask`: Replace with `[PII_TYPE]` placeholder
- `redact`: Remove PII text entirely
- `remove`: Delete rows containing PII

### 4. Split Dataset

Split dataset into train/validation/test sets:

```python
# Basic split
train, val, test = await manager.split_dataset(
    dataset_id,
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
)

print(f"Train: {train.row_count} rows")
print(f"Validation: {val.row_count} rows")
print(f"Test: {test.row_count} rows")

# Stratified split (maintains category balance)
train, val, test = await manager.split_dataset(
    dataset_id,
    train_ratio=0.7,
    validation_ratio=0.15,
    test_ratio=0.15,
    stratify_by="category",
)
```

**Features:**
- Configurable split ratios
- Stratified sampling by category
- Creates three new datasets
- Tracks lineage to parent dataset
- Maintains format and task type

### 5. Convert Format

Convert datasets between formats:

```python
# Convert to CSV
csv_metadata = await manager.convert_format(
    dataset_id,
    target_format=DatasetFormat.CSV
)

# Convert to Parquet
parquet_metadata = await manager.convert_format(
    dataset_id,
    target_format=DatasetFormat.PARQUET
)
```

**Supported Formats:**
- JSONL (JSON Lines)
- CSV (Comma-Separated Values)
- Parquet (Apache Parquet)
- HuggingFace (HuggingFace Datasets format)

### 6. Generate Synthetic Data

Augment datasets with synthetic examples:

```python
# Requires InferenceClient to be configured
augmented_metadata = await manager.generate_synthetic(
    dataset_id,
    augmentation_factor=1.5,  # 50% more examples
    strategy="paraphrase"  # Options: "paraphrase", "diverse", "creative"
)

print(f"Original: {metadata.row_count} rows")
print(f"Augmented: {augmented_metadata.row_count} rows")
```

**Strategies:**
- `paraphrase`: Simple paraphrasing maintaining exact meaning
- `diverse`: Variations with different phrasings and perspectives
- `creative`: Creative variations with different scenarios

### 7. Query Datasets

List and retrieve datasets:

```python
# Get specific dataset
metadata = await manager.get_dataset(dataset_id)

# List all datasets
all_datasets = await manager.list_datasets()

# Filter by task type
qa_datasets = await manager.list_datasets(
    task_type=DatasetTaskType.QA
)

# Filter by format
jsonl_datasets = await manager.list_datasets(
    format=DatasetFormat.JSONL
)

# Get version history
versions = await manager.get_version_history(dataset_id)
for version in versions:
    print(f"Version {version.version}: {version.row_count} rows")
```

## Data Models

### DatasetMetadata

```python
class DatasetMetadata(BaseModel):
    id: str                              # Unique identifier
    name: str                            # Human-readable name
    description: Optional[str]           # Description
    format: DatasetFormat                # JSONL, CSV, Parquet, etc.
    task_type: DatasetTaskType           # QA, summarization, etc.
    version: str                         # Version identifier (e.g., "v1")
    s3_uri: str                          # S3 storage location
    row_count: int                       # Number of examples
    token_stats: TokenStatistics         # Token statistics
    created_at: datetime                 # Creation timestamp
    checksum: str                        # SHA-256 checksum
    lineage: Optional[DatasetLineage]    # Lineage information
    quality_report: Optional[DatasetQualityReport]  # Quality metrics
```

### DatasetQualityReport

```python
class DatasetQualityReport(BaseModel):
    completeness_score: float            # 0-1, ratio of complete examples
    diversity_score: float               # 0-1, ratio of unique prompts
    balance_score: float                 # 0-1, category distribution evenness
    token_stats: TokenStatistics         # Token statistics
    issues: list[QualityIssue]           # Detected issues
    recommendations: list[str]           # Actionable recommendations
```

## Error Handling

The DatasetManager raises specific exceptions for different error conditions:

```python
try:
    metadata = await manager.upload_dataset(
        file_path="data.jsonl",
        name="My Dataset"
    )
except ValueError as e:
    # File not found, invalid format, size exceeded, etc.
    print(f"Validation error: {e}")
except RuntimeError as e:
    # AWS service errors, network issues, etc.
    print(f"Runtime error: {e}")
```

**Common Errors:**
- `ValueError`: Invalid input (file not found, format unsupported, size exceeded)
- `RuntimeError`: AWS service failures (S3, DynamoDB errors)

## Best Practices

### 1. Dataset Size Management

```python
# Set appropriate size limits
manager = DatasetManager(
    max_dataset_size_gb=5.0  # Adjust based on your needs
)

# For large datasets, split before upload
# Or use multipart upload (handled automatically for >100MB)
```

### 2. Quality Validation

```python
# Always validate after upload
quality_report = await manager.validate_dataset(dataset_id)

# Check for critical issues
critical_issues = [
    issue for issue in quality_report.issues
    if issue.severity == "error"
]

if critical_issues:
    print("Critical issues found! Fix before using for training.")
```

### 3. PII Handling

```python
# Always check for PII before using datasets
pii_result = await manager.detect_pii(dataset_id)

if pii_result.rows_with_pii > 0:
    # Mask PII before training
    masked_metadata = await manager.mask_pii(
        dataset_id,
        strategy="mask"
    )
    dataset_id = masked_metadata.id  # Use masked version
```

### 4. Version Management

```python
# Track versions for reproducibility
versions = await manager.get_version_history(dataset_id)

# Use specific version for training
training_version = versions[0]  # Latest version

# Record usage in version manager
manager.version_manager.record_usage(
    dataset_id=dataset_id,
    version_id=training_version.version,
    usage_type="fine_tuning",
    job_id="job-123",
    job_details={"model_id": "claude-3-sonnet"}
)
```

## Integration Examples

### With Fine-Tuning Pipeline

```python
from src.fine_tuning.fine_tuning_pipeline import FineTuningPipeline

# Upload and validate dataset
metadata = await manager.upload_dataset(
    file_path="training_data.jsonl",
    name="Fine-tuning Dataset"
)

# Validate quality
quality_report = await manager.validate_dataset(
    metadata.id,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

# Check if quality is sufficient
if quality_report.completeness_score < 0.9:
    print("Warning: Dataset quality may be insufficient")

# Use with fine-tuning pipeline
pipeline = FineTuningPipeline(...)
job = await pipeline.start_fine_tuning(
    config=FineTuningConfig(
        base_model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        training_data_id=metadata.id,
        ...
    )
)
```

### With Evaluation Engine

```python
from src.evaluation.evaluation_engine import EvaluationEngine

# Upload evaluation dataset
eval_metadata = await manager.upload_dataset(
    file_path="eval_data.jsonl",
    name="Evaluation Dataset"
)

# Split for baseline and comparative evaluation
train, val, test = await manager.split_dataset(
    eval_metadata.id,
    train_ratio=0.0,  # No training split needed
    validation_ratio=0.5,
    test_ratio=0.5,
)

# Use with evaluation engine
engine = EvaluationEngine(...)
report = await engine.run_baseline_evaluation(
    config=EvaluationConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        dataset_id=test.id,
        ...
    )
)
```

## Performance Considerations

### Upload Performance

- Files <100MB: Direct upload
- Files >100MB: Automatic multipart upload with progress tracking
- Files >10GB: Rejected by default (configurable)

### Caching

The DatasetManager uses caching for:
- Format detection results
- Quality analysis results
- Version metadata

### Concurrency

All async methods support concurrent execution:

```python
# Upload multiple datasets concurrently
datasets = await asyncio.gather(
    manager.upload_dataset("data1.jsonl", "Dataset 1"),
    manager.upload_dataset("data2.jsonl", "Dataset 2"),
    manager.upload_dataset("data3.jsonl", "Dataset 3"),
)
```

## Troubleshooting

### Issue: Upload fails with "File too large"

**Solution**: Increase the size limit or split the dataset:

```python
manager = DatasetManager(
    max_dataset_size_gb=20.0  # Increase limit
)
```

### Issue: Format detection fails

**Solution**: Explicitly specify format and task type:

```python
metadata = await manager.upload_dataset(
    file_path="data.txt",
    name="My Dataset",
    task_type=DatasetTaskType.TEXT_GENERATION  # Explicit task type
)
```

### Issue: PII detection misses some PII

**Solution**: The PII detector uses regex patterns. For advanced PII detection, integrate with specialized PII detection services.

### Issue: Synthetic generation fails

**Solution**: Ensure InferenceClient is configured:

```python
from src.clients.inference_client import InferenceClient

inference_client = InferenceClient(...)
manager = DatasetManager(
    inference_client=inference_client
)
```

## See Also

- [Format Detector](format_detector.md) - Automatic format detection
- [Quality Analyzer](quality_analyzer.md) - Dataset quality metrics
- [PII Detector](pii_detector.md) - PII detection and masking
- [Dataset Splitter](splitter.md) - Train/val/test splitting
- [Synthetic Generator](synthetic_generator.md) - Data augmentation
- [Version Manager](version_manager.md) - Version tracking
- [Model Validator](model_validator.md) - Model compatibility validation

## API Reference

See the [DatasetManager API documentation](../src/datasets/dataset_manager.py) for complete method signatures and parameters.
