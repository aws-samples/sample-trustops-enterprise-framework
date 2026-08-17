# PII Detector

## Overview

The PII Detector is a component of the TrustOps Enterprise Framework that identifies and masks Personally Identifiable Information (PII) in datasets using regex pattern matching. It helps ensure compliance with data privacy regulations by detecting common PII types and providing multiple remediation strategies before datasets are used for model training or evaluation.

## Features

- **Multiple PII Types**: Detects email addresses, phone numbers, SSNs, credit cards, and IP addresses
- **Luhn Validation**: Credit card numbers are validated using the Luhn algorithm to reduce false positives
- **Flexible Masking**: Three masking strategies (mask, redact, remove) with selective type filtering
- **Detailed Results**: Provides row-level and field-level PII detection information
- **Summary Statistics**: Reports total PII counts, affected rows, and percentage of rows with PII
- **Format Support**: Works with any dataset format (JSONL, CSV, Parquet) after parsing

## Supported PII Types

### 1. Email Addresses
- **Pattern**: Standard email format (user@domain.tld)
- **Example**: `john.doe@example.com`

### 2. Phone Numbers
- **Formats Supported**:
  - Standard: `555-123-4567`
  - No separators: `5551234567`
  - International: `+1-555-123-4567`
- **Note**: Parentheses format `(555) 123-4567` may not be detected by current regex

### 3. Social Security Numbers (SSN)
- **Format**: `XXX-XX-XXXX`
- **Example**: `123-45-6789`

### 4. Credit Card Numbers
- **Format**: 16 digits with optional separators (spaces or hyphens)
- **Validation**: Uses Luhn algorithm to validate card numbers
- **Examples**: 
  - `4111-1111-1111-1111`
  - `4111 1111 1111 1111`
  - `4111111111111111`

### 5. IP Addresses
- **IPv4**: Standard dotted decimal notation
  - Example: `192.168.1.1`
- **IPv6**: Full and compressed formats
  - Example: `2001:0db8:85a3:0000:0000:8a2e:0370:7334`
  - Example: `2001:db8::1`

## Usage

### Basic Detection

```python
from src.datasets.pii_detector import PIIDetector

# Create detector instance
detector = PIIDetector()

# Prepare dataset examples
examples = [
    {"prompt": "Contact john.doe@example.com for details"},
    {"prompt": "Call 555-123-4567"},
    {"prompt": "No PII here"},
]

# Detect PII
result = detector.detect(examples)

# Access results
print(f"Total rows: {result.total_rows}")
print(f"Rows with PII: {result.rows_with_pii}")
print(f"PII percentage: {result.pii_percentage}%")
print(f"Affected rows: {result.affected_rows}")
print(f"PII type counts: {result.pii_type_counts}")
```

### PII Masking

#### Strategy 1: MASK - Replace with Placeholders

```python
from src.datasets.pii_detector import PIIDetector, MaskingStrategy

detector = PIIDetector()

examples = [
    {"text": "Contact john@example.com or call 555-123-4567"},
    {"text": "No PII here"},
]

# Mask all PII with placeholders like [EMAIL], [PHONE]
masked_examples, report = detector.mask(
    examples,
    strategy=MaskingStrategy.MASK
)

print(masked_examples[0]["text"])
# Output: "Contact [EMAIL] or call [PHONE]"

print(f"Masked {report.pii_instances_masked} PII instances")
```

#### Strategy 2: REDACT - Remove PII Text

```python
# Remove PII text entirely (replace with empty string)
masked_examples, report = detector.mask(
    examples,
    strategy=MaskingStrategy.REDACT
)

print(masked_examples[0]["text"])
# Output: "Contact  or call "
```

#### Strategy 3: REMOVE - Delete Rows with PII

```python
# Remove entire rows containing PII
masked_examples, report = detector.mask(
    examples,
    strategy=MaskingStrategy.REMOVE
)

print(f"Original rows: {report.original_row_count}")
print(f"Remaining rows: {report.masked_row_count}")
print(f"Rows removed: {report.rows_removed}")
```

### Selective Masking by PII Type

```python
from src.datasets.pii_detector import PIIType

examples = [
    {
        "text": "Email: john@example.com, Phone: 555-123-4567, "
                "SSN: 123-45-6789"
    }
]

# Mask only SSN and credit cards, leave other PII types
masked_examples, report = detector.mask(
    examples,
    strategy=MaskingStrategy.MASK,
    selective_types=[PIIType.SSN, PIIType.CREDIT_CARD]
)

print(masked_examples[0]["text"])
# Output: "Email: john@example.com, Phone: 555-123-4567, SSN: [SSN]"

print(f"Selective types: {report.selective_types}")
print(f"Types masked: {report.pii_types_masked}")
```

### Complete Workflow: Detect, Review, Mask

```python
# Step 1: Detect PII
result = detector.detect(examples)

# Step 2: Review what was found
if result.rows_with_pii > 0:
    print(f"Found PII in {result.rows_with_pii} rows:")
    for pii_type, count in result.pii_type_counts.items():
        print(f"  - {pii_type}: {count} instances")
    
    # Step 3: Decide on masking strategy
    if PIIType.SSN in result.pii_type_counts:
        # Critical PII - remove rows
        masked_examples, report = detector.mask(
            examples,
            strategy=MaskingStrategy.REMOVE,
            selective_types=[PIIType.SSN, PIIType.CREDIT_CARD]
        )
    else:
        # Less critical - just mask
        masked_examples, report = detector.mask(
            examples,
            strategy=MaskingStrategy.MASK
        )
    
    print(f"\nMasking complete:")
    print(f"  Strategy: {report.strategy}")
    print(f"  PII instances masked: {report.pii_instances_masked}")
    print(f"  Rows removed: {report.rows_removed}")
```

### File-Based Detection

```python
from src.datasets.pii_detector import detect_pii_in_file

# Detect PII in a JSONL file
result = detect_pii_in_file("path/to/dataset.jsonl")

# Access results
for row_result in result.row_results:
    if row_result.has_pii:
        print(f"Row {row_result.row_index} contains PII:")
        for match in row_result.matches:
            print(f"  - {match.pii_type}: {match.value} in field '{match.field}'")
```

### Detailed Row Analysis

```python
# Analyze specific rows
for row_result in result.row_results:
    if row_result.has_pii:
        print(f"\nRow {row_result.row_index}:")
        print(f"  PII Types: {row_result.pii_types}")
        print(f"  Matches:")
        for match in row_result.matches:
            print(f"    - Type: {match.pii_type}")
            print(f"      Value: {match.value}")
            print(f"      Field: {match.field}")
            print(f"      Position: {match.start_idx}-{match.end_idx}")
```

## Data Models

### PIIType
Enumeration of supported PII types:
- `EMAIL`
- `PHONE`
- `SSN`
- `CREDIT_CARD`
- `IP_ADDRESS`

### MaskingStrategy
Enumeration of masking strategies:
- `MASK`: Replace PII with placeholder like `[EMAIL]`, `[PHONE]`
- `REDACT`: Remove PII text entirely (replace with empty string)
- `REMOVE`: Delete entire rows containing PII

### PIIMatch
Represents a single PII match:
- `pii_type`: Type of PII detected
- `value`: The matched PII value
- `field`: Field name where PII was found
- `start_idx`: Start index in the text
- `end_idx`: End index in the text

### PIIRowResult
PII detection result for a single row:
- `row_index`: Index of the row
- `has_pii`: Whether PII was found
- `pii_types`: List of PII types found
- `matches`: List of detailed matches

### PIIDetectionResult
Complete detection result for a dataset:
- `total_rows`: Total number of rows scanned
- `rows_with_pii`: Number of rows containing PII
- `pii_percentage`: Percentage of rows with PII
- `pii_type_counts`: Count of each PII type
- `affected_rows`: Indices of rows with PII
- `row_results`: Detailed results per row

### MaskingReport
Report of masking operations performed:
- `strategy`: Masking strategy used
- `original_row_count`: Number of rows before masking
- `masked_row_count`: Number of rows after masking
- `rows_removed`: Number of rows removed (for REMOVE strategy)
- `pii_instances_masked`: Total PII instances masked/redacted
- `pii_types_masked`: Count of each PII type masked
- `selective_types`: PII types that were selectively masked (None = all)

## Integration with Dataset Manager

The PII Detector integrates with the Dataset Manager for comprehensive dataset analysis:

```python
from src.datasets.dataset_manager import DatasetManager

# Dataset Manager will use PII Detector
manager = DatasetManager()

# Detect PII as part of dataset validation
pii_result = await manager.detect_pii(dataset_id)

# Mask PII if needed
masked_dataset = await manager.mask_pii(dataset_id, strategy="mask")
```

## Best Practices

### 1. Run PII Detection Early
Always run PII detection before using datasets for training or evaluation:
```python
# Check for PII before proceeding
result = detector.detect(examples)
if result.rows_with_pii > 0:
    print(f"Warning: {result.rows_with_pii} rows contain PII")
    # Decide whether to mask, remove, or proceed
```

### 2. Choose Appropriate Masking Strategy
Select the masking strategy based on your use case:

**Use MASK when:**
- You want to preserve dataset structure and row count
- The presence of PII is informative (e.g., "contact info" context)
- You need to maintain text length approximately

**Use REDACT when:**
- You want to completely remove PII but keep rows
- Text structure is less important
- You want minimal trace of PII

**Use REMOVE when:**
- Rows with PII cannot be used at all
- You have sufficient data after removal
- Compliance requires complete elimination

### 3. Use Selective Masking for Sensitive PII
Prioritize masking of high-risk PII types:
```python
# Remove rows with critical PII
critical_pii = [PIIType.SSN, PIIType.CREDIT_CARD]
masked_examples, report = detector.mask(
    examples,
    strategy=MaskingStrategy.REMOVE,
    selective_types=critical_pii
)

# Then mask remaining PII types
other_pii = [PIIType.EMAIL, PIIType.PHONE]
masked_examples, report = detector.mask(
    masked_examples,
    strategy=MaskingStrategy.MASK,
    selective_types=other_pii
)
```

### 4. Review Masking Reports
Always review masking reports to understand what was changed:
```python
masked_examples, report = detector.mask(examples, strategy=MaskingStrategy.MASK)

print(f"Masking Summary:")
print(f"  Original rows: {report.original_row_count}")
print(f"  Remaining rows: {report.masked_row_count}")
print(f"  PII instances masked: {report.pii_instances_masked}")
print(f"  By type: {report.pii_types_masked}")
```

### 5. Preserve Original Datasets
Keep a copy of the original dataset before masking:
```python
import copy

# Keep original
original_examples = copy.deepcopy(examples)

# Mask for use
masked_examples, report = detector.mask(
    examples,
    strategy=MaskingStrategy.MASK
)

# Use masked version for training
# Keep original for audit purposes
```

### 6. Document Masking Operations
Maintain audit trail of masking operations:
```python
import json
from datetime import datetime

# Perform masking
masked_examples, report = detector.mask(examples, strategy=MaskingStrategy.MASK)

# Log the operation
audit_log = {
    "timestamp": datetime.utcnow().isoformat(),
    "operation": "pii_masking",
    "strategy": report.strategy,
    "original_row_count": report.original_row_count,
    "masked_row_count": report.masked_row_count,
    "pii_instances_masked": report.pii_instances_masked,
    "pii_types_masked": report.pii_types_masked,
}

# Save audit log
with open("masking_audit.jsonl", "a") as f:
    f.write(json.dumps(audit_log) + "\n")
```

## Limitations

### Current Limitations

1. **Phone Number Formats**: Some formats like `(555) 123-4567` may not be detected
2. **IPv6 Patterns**: Very short IPv6 addresses like `::1` may not be detected
3. **International Formats**: Primarily supports US formats for phone numbers and SSNs
4. **Context Awareness**: Does not understand context (e.g., "call 911" might be flagged)

### False Positives

The detector may flag:
- Test data that looks like PII (e.g., `test@example.com`)
- Fictional data (e.g., `555-0100` to `555-0199` are reserved for fiction)
- IP addresses in technical documentation

### False Negatives

The detector may miss:
- PII in non-standard formats
- Obfuscated PII (e.g., `john dot doe at example dot com`)
- PII in non-English text
- Custom PII types not in the standard patterns

## Performance Considerations

- **Regex Performance**: Regex matching is fast for most datasets
- **Large Datasets**: For datasets with millions of rows, consider batch processing
- **Memory Usage**: All examples are processed in memory
- **Luhn Validation**: Credit card validation adds minimal overhead

## Security and Compliance

### Data Privacy
- PII detection is performed locally - no data is sent to external services
- Detection results should be treated as sensitive information
- Store detection results securely with appropriate access controls

### Compliance
- Use PII detection as part of GDPR, CCPA, and HIPAA compliance workflows
- Document PII detection results for audit trails
- Implement data retention policies for detection results

### Recommendations
1. Run PII detection on all datasets before use
2. Mask or remove PII before sharing datasets
3. Log PII detection events for compliance auditing
4. Regularly review and update PII patterns

## Troubleshooting

### No PII Detected When Expected
- Verify the PII format matches supported patterns
- Check if text fields are actually strings (not numbers or objects)
- Review regex patterns for your specific use case

### Too Many False Positives
- Review the specific PII types being flagged
- Consider implementing custom filtering logic
- Use the detailed match information to identify patterns

### Performance Issues
- Process datasets in smaller batches
- Consider parallel processing for very large datasets
- Profile regex patterns if performance is critical

## Future Enhancements

Potential improvements for future versions:
- Configurable regex patterns
- Custom PII type definitions
- Machine learning-based PII detection
- Support for more international formats
- Context-aware detection
- Confidence scores for matches
- Integration with external PII detection services

## Related Documentation

- [Dataset Manager](dataset_manager.md)
- [Data Quality Analyzer](quality_analyzer.md)
- [Dataset Parsers](dataset_parsers.md)

## Requirements

This component implements:
- **Requirement 2.7**: CSV parser and validator with column mapping configuration
- **Requirement 2.13**: Basic PII detector using regex patterns (email, phone, SSN, credit card, IP address)
- **Requirement 2.14**: PII masking/redaction with configurable replacement strategies
