# Dataset Version Manager

## Overview

The Dataset Version Manager provides comprehensive versioning capabilities for datasets in the TrustOps Enterprise Framework. It enables tracking of dataset changes over time, maintaining version history, ensuring data integrity through SHA-256 checksums, and tracking detailed lineage information including transformations and usage history.

**Requirements:** 2.9, 2.10, 2.14, 2.17, 2.18

## Features

- **Version Storage**: Store dataset versions in S3 with organized paths
- **Metadata Tracking**: Store version metadata in DynamoDB for fast querying
- **Integrity Verification**: Calculate and verify SHA-256 checksums
- **Version History**: List and retrieve all versions of a dataset
- **Lineage Tracking**: Track parent-child relationships between versions
- **Transformation Tracking**: Record all transformations applied to datasets
- **Usage History**: Track which evaluations and fine-tuning jobs used which datasets
- **Version Comparison**: Compare two versions to identify changes
- **Efficient Retrieval**: Get latest version or specific historical versions

## Architecture

### Storage Structure

**S3 Path Format:**
```
s3://{bucket}/datasets/{dataset_id}/versions/{version_id}/data
```

**DynamoDB Schema:**
- **Partition Key**: `dataset_id` (String)
- **Sort Key**: `version_id` (String)
- **Attributes**:
  - `s3_uri`: S3 location of the version
  - `checksum`: SHA-256 checksum for integrity
  - `size_bytes`: Size of the dataset in bytes
  - `created_at`: ISO 8601 timestamp
  - `description`: Optional version description
  - `parent_version_id`: Parent version for lineage tracking
  - `format`: Dataset format (jsonl, csv, parquet)
  - `task_type`: Task type (qa, summarization, etc.)
  - `row_count`: Number of rows in the dataset
  - `name`: Dataset name

### Version ID Format

Version IDs are automatically generated using timestamps:
```
v{YYYYMMDD}_{HHMMSS}_{microseconds}
```

Example: `v20240115_143022_123456`

## Usage

### Initialization

```python
from src.datasets.version_manager import DatasetVersionManager

# Initialize with default configuration
version_manager = DatasetVersionManager()

# Or with custom configuration
version_manager = DatasetVersionManager(
    region='us-west-2',
    datasets_bucket='my-datasets-bucket',
    versions_table='my-versions-table'
)
```

### Creating a Version

```python
from src.data_models.dataset import DatasetMetadata, DatasetFormat, DatasetTaskType

# Prepare dataset content
content = b'{"prompt": "What is AI?", "completion": "AI is..."}\n'

# Create metadata
metadata = DatasetMetadata(
    id="qa-dataset-001",
    name="QA Dataset",
    format=DatasetFormat.JSONL,
    task_type=DatasetTaskType.QA,
    version="1.0",
    s3_uri="s3://bucket/dataset",
    row_count=100,
    token_stats=token_stats,
    created_at=datetime.utcnow(),
    checksum="abc123"
)

# Create version
result = version_manager.create_version(
    dataset_id="qa-dataset-001",
    content=content,
    metadata=metadata,
    description="Initial version"
)

print(f"Created version: {result['version_id']}")
print(f"S3 URI: {result['s3_uri']}")
print(f"Checksum: {result['checksum']}")
```

### Creating a Version with Lineage

```python
# Create child version with parent reference
child_result = version_manager.create_version(
    dataset_id="qa-dataset-001",
    content=updated_content,
    metadata=updated_metadata,
    description="Added 50 new examples",
    parent_version_id=parent_version_id
)
```

### Retrieving a Version

```python
# Get specific version
version_data = version_manager.get_version(
    dataset_id="qa-dataset-001",
    version_id="v20240115_143022_123456",
    verify_checksum=True  # Verify integrity (default)
)

content = version_data['content']
metadata = version_data['metadata']
checksum = version_data['checksum']
```

### Listing Versions

```python
# List all versions (newest first)
versions = version_manager.list_versions("qa-dataset-001")

for version in versions:
    print(f"Version: {version['version_id']}")
    print(f"Created: {version['created_at']}")
    print(f"Size: {version['size_bytes']} bytes")
    print(f"Description: {version['description']}")
    print()

# List with limit
recent_versions = version_manager.list_versions(
    "qa-dataset-001",
    limit=5
)
```

### Getting Latest Version

```python
# Get the most recent version
latest = version_manager.get_latest_version("qa-dataset-001")

if latest:
    print(f"Latest version: {latest['version_id']}")
    print(f"Created: {latest['created_at']}")
else:
    print("No versions found")
```

### Comparing Versions

```python
# Compare two versions
comparison = version_manager.compare_versions(
    dataset_id="qa-dataset-001",
    version_id_1="v20240115_143022_123456",
    version_id_2="v20240116_091530_789012"
)

print(f"Size delta: {comparison['size_delta']} bytes")
print(f"Row count delta: {comparison['row_count_delta']}")
print(f"Checksums match: {comparison['checksum_match']}")

# Access full metadata
v1_metadata = comparison['version_1']
v2_metadata = comparison['version_2']
```

### Tracking Version Lineage

```python
# Get complete lineage chain
lineage = version_manager.get_version_lineage(
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012"
)

print("Version lineage (oldest to newest):")
for i, version in enumerate(lineage):
    print(f"{i+1}. {version['version_id']} - {version['description']}")
```

### Deleting a Version

```python
# Delete a specific version (permanent)
version_manager.delete_version(
    dataset_id="qa-dataset-001",
    version_id="v20240115_143022_123456"
)
```

## Checksum Verification

The version manager uses SHA-256 checksums to ensure data integrity:

```python
# Calculate checksum for content
content = b"dataset content"
checksum = version_manager.calculate_checksum(content)

# Checksums are automatically verified during retrieval
version_data = version_manager.get_version(
    dataset_id="qa-dataset-001",
    version_id="v20240115_143022_123456",
    verify_checksum=True  # Raises ValueError if mismatch
)

# Skip verification if needed (not recommended)
version_data = version_manager.get_version(
    dataset_id="qa-dataset-001",
    version_id="v20240115_143022_123456",
    verify_checksum=False
)
```

## Best Practices

### 1. Version Descriptions

Always provide meaningful descriptions when creating versions:

```python
result = version_manager.create_version(
    dataset_id="qa-dataset-001",
    content=content,
    metadata=metadata,
    description="Added 100 medical domain examples, fixed formatting issues"
)
```

### 2. Lineage Tracking

Maintain parent-child relationships for transformations:

```python
# Original dataset
v1 = version_manager.create_version(
    dataset_id="qa-dataset-001",
    content=original_content,
    metadata=metadata,
    description="Original dataset"
)

# After PII masking
v2 = version_manager.create_version(
    dataset_id="qa-dataset-001",
    content=masked_content,
    metadata=metadata,
    description="PII masked",
    parent_version_id=v1['version_id']
)

# After quality improvements
v3 = version_manager.create_version(
    dataset_id="qa-dataset-001",
    content=improved_content,
    metadata=metadata,
    description="Quality improvements applied",
    parent_version_id=v2['version_id']
)
```

### 3. Version Comparison

Compare versions before and after transformations:

```python
# Before transformation
before_version = version_manager.create_version(...)

# Apply transformation
transformed_content = apply_transformation(content)

# After transformation
after_version = version_manager.create_version(
    ...,
    parent_version_id=before_version['version_id']
)

# Compare to verify changes
comparison = version_manager.compare_versions(
    dataset_id,
    before_version['version_id'],
    after_version['version_id']
)

print(f"Size change: {comparison['size_delta']} bytes")
print(f"Row count change: {comparison['row_count_delta']}")
```

### 4. Checksum Verification

Always verify checksums when data integrity is critical:

```python
try:
    version_data = version_manager.get_version(
        dataset_id="qa-dataset-001",
        version_id=version_id,
        verify_checksum=True
    )
    # Checksum verified - data is intact
except ValueError as e:
    # Checksum mismatch - data may be corrupted
    print(f"Integrity check failed: {e}")
```

### 5. Version Cleanup

Periodically clean up old versions to manage storage costs:

```python
# List all versions
versions = version_manager.list_versions("qa-dataset-001")

# Keep only the last 10 versions
if len(versions) > 10:
    for version in versions[10:]:
        version_manager.delete_version(
            dataset_id="qa-dataset-001",
            version_id=version['version_id']
        )
```

## Error Handling

```python
from botocore.exceptions import ClientError

try:
    result = version_manager.create_version(
        dataset_id="qa-dataset-001",
        content=content,
        metadata=metadata
    )
except RuntimeError as e:
    print(f"Failed to create version: {e}")

try:
    version_data = version_manager.get_version(
        dataset_id="qa-dataset-001",
        version_id="v20240115_143022_123456"
    )
except ValueError as e:
    print(f"Version not found or checksum mismatch: {e}")
except RuntimeError as e:
    print(f"Failed to retrieve version: {e}")
```

## Integration with Dataset Manager

The version manager integrates with the Dataset Manager for automatic versioning:

```python
from src.datasets.dataset_manager import DatasetManager
from src.datasets.version_manager import DatasetVersionManager

dataset_manager = DatasetManager()
version_manager = DatasetVersionManager()

# Upload dataset
metadata = dataset_manager.upload_dataset(
    file_path="data/qa_dataset.jsonl",
    name="QA Dataset",
    task_type=DatasetTaskType.QA
)

# Create initial version
with open("data/qa_dataset.jsonl", "rb") as f:
    content = f.read()

version_result = version_manager.create_version(
    dataset_id=metadata.id,
    content=content,
    metadata=metadata,
    description="Initial upload"
)

# Apply transformation
masked_metadata = dataset_manager.mask_pii(
    dataset_id=metadata.id,
    strategy="mask"
)

# Create new version after transformation
with open(masked_metadata.s3_uri.replace("s3://", "/mnt/s3/"), "rb") as f:
    masked_content = f.read()

version_manager.create_version(
    dataset_id=metadata.id,
    content=masked_content,
    metadata=masked_metadata,
    description="PII masked",
    parent_version_id=version_result['version_id']
)
```

## Performance Considerations

### S3 Storage

- Versions are stored with organized paths for efficient retrieval
- Use S3 lifecycle policies to archive old versions to Glacier
- Enable S3 versioning on the bucket for additional protection

### DynamoDB Queries

- Queries use partition key (dataset_id) for efficient lookups
- Sort key (version_id) enables range queries and sorting
- Consider adding GSI for queries by creation date if needed

### Checksum Calculation

- SHA-256 calculation is CPU-intensive for large datasets
- Consider calculating checksums asynchronously for very large files
- Checksums are calculated once during version creation

## Monitoring and Observability

### CloudWatch Metrics

Monitor version manager operations:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Track version creation
cloudwatch.put_metric_data(
    Namespace='TrustOps/DatasetVersioning',
    MetricData=[
        {
            'MetricName': 'VersionsCreated',
            'Value': 1,
            'Unit': 'Count',
            'Dimensions': [
                {'Name': 'DatasetId', 'Value': dataset_id}
            ]
        }
    ]
)
```

### Logging

Enable detailed logging for troubleshooting:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version manager operations are logged automatically
version_manager = DatasetVersionManager()
```

## Security Considerations

### Access Control

- Use IAM policies to control access to S3 buckets and DynamoDB tables
- Implement least-privilege access for version operations
- Enable S3 bucket encryption at rest

### Data Integrity

- Always verify checksums when retrieving versions
- Use S3 versioning as an additional safety layer
- Implement backup and disaster recovery procedures

### Audit Trail

- All version operations are tracked in DynamoDB
- Use CloudTrail to audit API calls
- Maintain version lineage for compliance requirements

## Troubleshooting

### Common Issues

**Issue: Checksum mismatch during retrieval**
```
Solution: Data may be corrupted. Retrieve from S3 versioning or backup.
```

**Issue: Version not found**
```
Solution: Verify dataset_id and version_id are correct. Check DynamoDB table.
```

**Issue: S3 upload fails**
```
Solution: Check IAM permissions, bucket existence, and network connectivity.
```

**Issue: DynamoDB query timeout**
```
Solution: Reduce query scope with limit parameter. Check table capacity.
```

## API Reference

### DatasetVersionManager

#### `__init__(region, datasets_bucket, versions_table)`
Initialize the version manager.

#### `calculate_checksum(content: bytes) -> str`
Calculate SHA-256 checksum for content.

#### `create_version(dataset_id, content, metadata, description, parent_version_id) -> Dict`
Create a new dataset version.

#### `get_version(dataset_id, version_id, verify_checksum) -> Dict`
Retrieve a specific version.

#### `list_versions(dataset_id, limit) -> List[Dict]`
List all versions of a dataset.

#### `get_latest_version(dataset_id) -> Optional[Dict]`
Get the most recent version.

#### `compare_versions(dataset_id, version_id_1, version_id_2) -> Dict`
Compare two versions.

#### `get_version_lineage(dataset_id, version_id) -> List[Dict]`
Get the lineage chain for a version.

#### `delete_version(dataset_id, version_id) -> None`
Delete a specific version.

## Related Documentation

- [Dataset Manager](./dataset_manager.md)
- [Format Converter](./format_converter.md)
- [PII Detector](./pii_detector.md)
- [Quality Analyzer](./quality_analyzer.md)

## Requirements Mapping

- **Requirement 2.9**: Format converter implementation (used with versioning)
- **Requirement 2.14**: PII masking/redaction (creates new versions)
- **Requirement 2.17**: Dataset versioning with S3 storage, DynamoDB metadata, and SHA-256 checksums


## Lineage Tracking

The version manager provides comprehensive lineage tracking to maintain a complete audit trail of dataset transformations and usage.

### Recording Transformations

Track transformations applied to datasets (splits, PII masking, synthetic generation, etc.):

```python
# After splitting a dataset
version_manager.record_transformation(
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012",
    transformation_type="split",
    transformation_details={
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
        "stratify_by": "category"
    }
)

# After PII masking
version_manager.record_transformation(
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012",
    transformation_type="pii_masking",
    transformation_details={
        "strategy": "mask",
        "fields_masked": ["email", "phone", "ssn"],
        "rows_affected": 45
    }
)

# After synthetic data generation
version_manager.record_transformation(
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012",
    transformation_type="synthetic_generation",
    transformation_details={
        "augmentation_factor": 1.5,
        "strategy": "paraphrase",
        "model_used": "claude-v2"
    }
)
```

### Recording Usage History

Track when datasets are used in evaluations or fine-tuning jobs:

```python
# Record evaluation usage
version_manager.record_usage(
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012",
    usage_type="evaluation",
    job_id="eval-baseline-20240116-001",
    job_details={
        "model_id": "claude-v2",
        "evaluation_type": "baseline",
        "trust_score": 0.87
    }
)

# Record fine-tuning usage
version_manager.record_usage(
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012",
    usage_type="fine_tuning",
    job_id="ft-job-20240116-002",
    job_details={
        "base_model_id": "claude-v2",
        "epochs": 3,
        "learning_rate": 1e-5,
        "finetuned_model_id": "claude-v2-custom-001"
    }
)
```

### Querying Transformations

Retrieve all transformations applied to a dataset version:

```python
# Get all transformations
transformations = version_manager.get_transformations(
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012"
)

print("Transformations applied:")
for transform in transformations:
    print(f"- Type: {transform['transformation_type']}")
    print(f"  Details: {transform['transformation_details']}")
    print(f"  Timestamp: {transform['timestamp']}")
    print()
```

### Querying Usage History

Retrieve all usage records for a dataset version:

```python
# Get usage history
usage_history = version_manager.get_usage_history(
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012"
)

print("Usage history:")
for usage in usage_history:
    print(f"- Type: {usage['usage_type']}")
    print(f"  Job ID: {usage['job_id']}")
    print(f"  Details: {usage['job_details']}")
    print(f"  Timestamp: {usage['timestamp']}")
    print()
```

### Complete Lineage Information

Get all lineage information in one call:

```python
# Get complete lineage
lineage = version_manager.get_complete_lineage(
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012"
)

# Version chain (parent versions)
print("Version chain:")
for version in lineage['version_chain']:
    print(f"- {version['version_id']}: {version['description']}")

# Transformations applied
print("\nTransformations:")
for transform in lineage['transformations']:
    print(f"- {transform['transformation_type']}")

# Usage history
print("\nUsage history:")
for usage in lineage['usage_history']:
    print(f"- {usage['usage_type']}: {usage['job_id']}")
```

## Lineage Tracking Best Practices

### 1. Record Transformations Immediately

Always record transformations right after applying them:

```python
# Apply transformation
masked_content = apply_pii_masking(content)

# Create new version
new_version = version_manager.create_version(
    dataset_id="qa-dataset-001",
    content=masked_content,
    metadata=metadata,
    description="PII masked",
    parent_version_id=original_version_id
)

# Record transformation immediately
version_manager.record_transformation(
    dataset_id="qa-dataset-001",
    version_id=new_version['version_id'],
    transformation_type="pii_masking",
    transformation_details={
        "strategy": "mask",
        "fields": ["email", "phone"]
    }
)
```

### 2. Include Detailed Transformation Information

Provide comprehensive details for reproducibility:

```python
version_manager.record_transformation(
    dataset_id="qa-dataset-001",
    version_id=version_id,
    transformation_type="split",
    transformation_details={
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
        "stratify_by": "category",
        "random_seed": 42,
        "train_count": 800,
        "validation_count": 100,
        "test_count": 100
    }
)
```

### 3. Track All Dataset Usage

Record every time a dataset is used:

```python
# Before running evaluation
version_manager.record_usage(
    dataset_id="qa-dataset-001",
    version_id=version_id,
    usage_type="evaluation",
    job_id=evaluation_id,
    job_details={
        "model_id": model_id,
        "evaluation_type": "baseline",
        "started_at": datetime.utcnow().isoformat()
    }
)

# Run evaluation
results = run_evaluation(...)

# Update with results (optional - create new usage record)
version_manager.record_usage(
    dataset_id="qa-dataset-001",
    version_id=version_id,
    usage_type="evaluation",
    job_id=f"{evaluation_id}-completed",
    job_details={
        "model_id": model_id,
        "evaluation_type": "baseline",
        "trust_score": results['trust_score'],
        "completed_at": datetime.utcnow().isoformat()
    }
)
```

### 4. Use Lineage for Audit Trails

Generate audit reports using lineage information:

```python
def generate_audit_report(dataset_id: str, version_id: str):
    """Generate comprehensive audit report for a dataset version."""
    lineage = version_manager.get_complete_lineage(dataset_id, version_id)
    
    report = {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "ancestry": [v['version_id'] for v in lineage['version_chain']],
        "transformation_count": len(lineage['transformations']),
        "transformations": lineage['transformations'],
        "usage_count": len(lineage['usage_history']),
        "evaluations": [
            u for u in lineage['usage_history']
            if u['usage_type'] == 'evaluation'
        ],
        "fine_tuning_jobs": [
            u for u in lineage['usage_history']
            if u['usage_type'] == 'fine_tuning'
        ]
    }
    
    return report
```

## DynamoDB Schema

### Versions Table

**Table Name:** `trustops-dataset-versions`

**Keys:**
- Partition Key: `dataset_id` (String)
- Sort Key: `version_id` (String)

**Attributes:**
- `s3_uri`: S3 location
- `checksum`: SHA-256 checksum
- `size_bytes`: Size in bytes
- `created_at`: ISO 8601 timestamp
- `description`: Version description
- `parent_version_id`: Parent version for lineage
- `format`: Dataset format
- `task_type`: Task type
- `row_count`: Number of rows
- `name`: Dataset name

### Lineage Table

**Table Name:** `trustops-dataset-lineage`

**Keys:**
- Partition Key: `lineage_id` (String)

**Attributes:**
- `dataset_id`: Dataset identifier
- `version_id`: Version identifier
- `record_type`: "transformation" or "usage"
- `transformation_type`: Type of transformation (for transformation records)
- `transformation_details`: JSON string with transformation details
- `usage_type`: "evaluation" or "fine_tuning" (for usage records)
- `job_id`: Job identifier (for usage records)
- `job_details`: JSON string with job details
- `timestamp`: ISO 8601 timestamp

## Error Handling

The version manager provides clear error messages for common issues:

```python
try:
    version_data = version_manager.get_version(
        dataset_id="nonexistent",
        version_id="v20240115_143022_123456"
    )
except ValueError as e:
    print(f"Version not found: {e}")

try:
    version_manager.record_transformation(
        dataset_id="qa-dataset-001",
        version_id="v20240115_143022_123456",
        transformation_type="split",
        transformation_details={}
    )
except RuntimeError as e:
    print(f"Failed to record transformation: {e}")
```

## Integration Examples

### Integration with Dataset Splitter

```python
from src.datasets.splitter import DatasetSplitter
from src.datasets.version_manager import DatasetVersionManager

# Initialize
splitter = DatasetSplitter()
version_manager = DatasetVersionManager()

# Split dataset
train, val, test = splitter.split_dataset(
    dataset_path="data/qa_dataset.jsonl",
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1
)

# Create versions for each split
train_version = version_manager.create_version(
    dataset_id="qa-dataset-001-train",
    content=train,
    metadata=train_metadata,
    description="Training split",
    parent_version_id=original_version_id
)

# Record transformation
version_manager.record_transformation(
    dataset_id="qa-dataset-001-train",
    version_id=train_version['version_id'],
    transformation_type="split",
    transformation_details={
        "split_type": "train",
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1
    }
)
```

### Integration with Evaluation Engine

```python
from src.evaluation.evaluation_engine import EvaluationEngine
from src.datasets.version_manager import DatasetVersionManager

# Initialize
evaluation_engine = EvaluationEngine()
version_manager = DatasetVersionManager()

# Run evaluation
evaluation_id = "eval-baseline-001"
results = evaluation_engine.run_baseline_evaluation(
    model_id="claude-v2",
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012"
)

# Record usage
version_manager.record_usage(
    dataset_id="qa-dataset-001",
    version_id="v20240116_091530_789012",
    usage_type="evaluation",
    job_id=evaluation_id,
    job_details={
        "model_id": "claude-v2",
        "evaluation_type": "baseline",
        "trust_score": results.aggregate_metrics.mean_trust_score,
        "total_examples": results.total_examples
    }
)
```

## Performance Considerations

### Scan Operations

The lineage tracking methods use DynamoDB scan operations, which can be slower for large datasets. For production use with high volumes:

1. Consider implementing pagination for large result sets
2. Use appropriate filtering to limit scanned items
3. Monitor DynamoDB read capacity usage
4. Consider adding GSI (Global Secondary Index) for frequently queried patterns

### Caching

For frequently accessed lineage information, consider caching:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_lineage(dataset_id: str, version_id: str):
    """Get lineage with caching."""
    return version_manager.get_complete_lineage(dataset_id, version_id)
```

## See Also

- [Dataset Manager](dataset_manager.md) - Dataset preparation and management
- [Dataset Splitter](splitter.md) - Dataset splitting functionality
- [PII Detector](pii_detector.md) - PII detection and masking
- [Synthetic Generator](synthetic_generator.md) - Synthetic data generation
