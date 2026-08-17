# Dataset Templates

Domain-specific dataset templates for common enterprise use cases in the TrustOps Enterprise Framework.

**Requirement 2.19**: Domain-specific dataset templates for customer support, legal, medical, and financial domains.

## Overview

Dataset templates provide pre-built schemas with:
- **Field definitions** with data types and validation rules
- **PII sensitivity levels** for each field
- **Domain-specific validation rules** to ensure data quality
- **Quality thresholds** for dataset acceptance
- **Example data** demonstrating proper format

## Available Templates

### 1. Customer Support Template

For customer support ticket and conversation datasets.

**Task Type**: Question-Answering (QA)

**Required Fields**:
- `ticket_id`: Unique ticket identifier
- `customer_query`: Customer question or issue (10-5000 chars)
- `agent_response`: Agent's response (10-5000 chars)
- `category`: Issue category (billing, technical, account, product, shipping, returns, general)

**Optional Fields**:
- `priority`: Ticket priority (low, medium, high, urgent)
- `sentiment`: Customer sentiment (positive, neutral, negative)
- `resolution_status`: Resolution status (resolved, unresolved, escalated)
- `context`: Additional context (max 10000 chars)

**Validation Rules**:
- Query and response length balance check
- Professional tone validation in agent responses

**Quality Thresholds**:
- Minimum completeness: 95%
- Minimum diversity: 70%
- Minimum examples: 200

**Example**:
```python
from src.datasets.templates import CustomerSupportTemplate

template = CustomerSupportTemplate()

example = {
    "ticket_id": "CS-2024-001",
    "customer_query": "I was charged twice for my subscription this month.",
    "agent_response": "I've reviewed your account and processed a refund...",
    "category": "billing",
    "priority": "high",
    "sentiment": "negative",
    "resolution_status": "resolved"
}

is_valid, errors = template.validate_example(example)
```

### 2. Legal Template

For legal document analysis and contract review datasets.

> **⚠️ Important**: AI outputs from models trained with legal datasets do not
> constitute legal advice and cannot stand alone as a basis for decisions. All
> outputs require review by a licensed attorney. Enable Amazon Bedrock Guardrails
> for content filtering in production.

**Task Type**: Text Generation

**Required Fields**:
- `document_id`: Unique document identifier
- `document_type`: Type of legal document (contract, agreement, policy, terms_of_service, privacy_policy, compliance, litigation, regulatory)
- `document_text`: Legal document text (50-50000 chars)
- `prompt`: Analysis prompt or question (10-2000 chars)
- `expected_analysis`: Expected analysis or answer (20-10000 chars)

**Optional Fields**:
- `jurisdiction`: Legal jurisdiction
- `practice_area`: Legal practice area (corporate, intellectual_property, employment, real_estate, tax, litigation, regulatory, privacy)
- `risk_level`: Risk level (low, medium, high, critical)
- `citations`: Legal citations or references (list)

**Validation Rules**:
- Document length validation (minimum 100 chars)
- Analysis quality check (references document)
- Legal terminology validation

**Quality Thresholds**:
- Minimum completeness: 98%
- Minimum diversity: 75%
- Minimum examples: 100

**Example**:
```python
from src.datasets.templates import LegalTemplate

template = LegalTemplate()

example = {
    "document_id": "LEGAL-2024-001",
    "document_type": "contract",
    "document_text": "This Software License Agreement...",
    "prompt": "Identify key obligations on the Licensee.",
    "expected_analysis": "The Licensee has the following obligations...",
    "jurisdiction": "United States",
    "practice_area": "intellectual_property",
    "risk_level": "medium"
}

is_valid, errors = template.validate_example(example)
```

### 3. Medical Template

For medical Q&A and clinical notes datasets (HIPAA-compliant).

> **⚠️ Important**: AI outputs from models trained with medical datasets MUST NOT
> be used as sole decision-makers for clinical decisions. All outputs require review
> by licensed healthcare professionals. Enable Amazon Bedrock Guardrails for content
> filtering in production.

**Task Type**: Question-Answering (QA)

**Required Fields**:
- `case_id`: Unique case identifier (de-identified)
- `clinical_question`: Clinical question or scenario (20-5000 chars)
- `clinical_answer`: Clinical answer or recommendation (20-5000 chars)
- `specialty`: Medical specialty (cardiology, neurology, oncology, pediatrics, psychiatry, radiology, surgery, internal_medicine, emergency_medicine, general_practice)
- `hipaa_compliant`: HIPAA compliance verification flag (must be True)

**Optional Fields**:
- `patient_demographics`: De-identified demographics (age range, gender)
- `diagnosis_codes`: ICD-10 diagnosis codes (list)
- `urgency`: Clinical urgency (routine, urgent, emergent, critical)
- `evidence_level`: Evidence level (A, B, C, D, expert_opinion)
- `references`: Medical literature references (list)

**Validation Rules**:
- HIPAA compliance flag must be set to True
- No Protected Health Information (PHI) validation
- Clinical accuracy and terminology check

**Quality Thresholds**:
- Minimum completeness: 98%
- Minimum diversity: 80%
- Minimum examples: 150

**PII Sensitivity**: All clinical fields marked as HIGH or CRITICAL sensitivity.

**Example**:
```python
from src.datasets.templates import MedicalTemplate

template = MedicalTemplate()

example = {
    "case_id": "MED-2024-001",
    "clinical_question": "A 55-year-old male presents with chest pain...",
    "clinical_answer": "This presentation is consistent with acute STEMI...",
    "specialty": "cardiology",
    "patient_demographics": "55-65 years, male",
    "diagnosis_codes": ["I21.19"],
    "urgency": "critical",
    "evidence_level": "A",
    "hipaa_compliant": True
}

is_valid, errors = template.validate_example(example)
```

### 4. Financial Template

For financial analysis and risk assessment datasets.

> **⚠️ Important**: AI outputs for credit evaluation and risk assessment MUST NOT
> be used as sole decision-makers. They require human review and must comply with
> applicable financial regulations. Enable Amazon Bedrock Guardrails for financial
> decision-making contexts.

**Task Type**: Text Generation

**Required Fields**:
- `analysis_id`: Unique analysis identifier
- `analysis_type`: Type of analysis (risk_assessment, investment_analysis, credit_evaluation, fraud_detection, market_analysis, portfolio_review, compliance, transaction_analysis)
- `financial_data`: Financial data or scenario (50-10000 chars)
- `prompt`: Analysis prompt or question (10-2000 chars)
- `expected_analysis`: Expected analysis or recommendation (50-10000 chars)

**Optional Fields**:
- `risk_level`: Risk level (low, medium, high, critical)
- `asset_class`: Asset class (equities, fixed_income, commodities, real_estate, derivatives, cash, alternative)
- `regulatory_framework`: Applicable framework (SEC, FINRA, Basel_III, MiFID_II, Dodd_Frank, SOX, AML)
- `time_horizon`: Analysis time horizon (short_term, medium_term, long_term)
- `confidence_level`: Confidence level (low, medium, high)

**Validation Rules**:
- Financial terminology validation
- Quantitative analysis check (numbers, percentages)
- Risk disclosure validation

**Quality Thresholds**:
- Minimum completeness: 95%
- Minimum diversity: 75%
- Minimum examples: 150

**Example**:
```python
from src.datasets.templates import FinancialTemplate

template = FinancialTemplate()

example = {
    "analysis_id": "FIN-2024-001",
    "analysis_type": "risk_assessment",
    "financial_data": "Company XYZ has revenue $500M, EBITDA margin 22%...",
    "prompt": "Assess the financial risk profile of this company.",
    "expected_analysis": "Risk Assessment: MEDIUM-HIGH. Positive factors...",
    "risk_level": "medium",
    "asset_class": "equities",
    "time_horizon": "medium_term",
    "confidence_level": "high"
}

is_valid, errors = template.validate_example(example)
```

## Usage

### Basic Usage

```python
from src.datasets.templates import CustomerSupportTemplate

# Initialize template
template = CustomerSupportTemplate()

# Get template information
print(f"Template: {template.name}")
print(f"Task Type: {template.task_type}")
print(f"Required Fields: {template.get_required_fields()}")
print(f"PII Fields: {template.get_pii_fields()}")

# Validate a single example
example = {...}
is_valid, errors = template.validate_example(example)
if not is_valid:
    print(f"Validation errors: {errors}")

# Validate entire dataset
examples = [...]
is_valid, report = template.validate_dataset(examples)
print(f"Valid examples: {report['valid_examples']}/{report['total_examples']}")
print(f"Meets quality thresholds: {report['meets_quality_thresholds']}")

# Get JSON schema
schema = template.get_schema()
```

### Accessing Example Data

Each template includes example data demonstrating proper format:

```python
template = CustomerSupportTemplate()

# Get example data
examples = template.example_data
for example in examples:
    print(example)
```

### Custom Validation

Templates include domain-specific validation rules that are automatically applied:

```python
template = MedicalTemplate()

# This will fail HIPAA compliance check
example = {
    "case_id": "MED-001",
    "clinical_question": "What is the treatment?",
    "clinical_answer": "Treatment includes...",
    "specialty": "cardiology",
    "hipaa_compliant": False  # Will fail validation
}

is_valid, errors = template.validate_example(example)
# is_valid = False
# errors = ["hipaa_compliance: Example must be marked as HIPAA compliant"]
```

## PII Sensitivity Levels

Templates mark fields with appropriate PII sensitivity:

- **NONE**: No PII (e.g., ticket_id, category)
- **LOW**: Minimal PII risk (e.g., agent_response)
- **MEDIUM**: Moderate PII risk (e.g., customer_query, expected_analysis)
- **HIGH**: High PII risk (e.g., document_text, financial_data, clinical_question)
- **CRITICAL**: Critical PII (e.g., case_id in medical data)

Use `template.get_pii_fields()` to identify fields requiring PII protection.

## Quality Thresholds

Each template defines quality thresholds that datasets must meet:

```python
template = FinancialTemplate()
thresholds = template.quality_thresholds

print(f"Min completeness: {thresholds.min_completeness}")
print(f"Min diversity: {thresholds.min_diversity}")
print(f"Min balance: {thresholds.min_balance}")
print(f"Min examples: {thresholds.min_examples}")
```

## Integration with Dataset Manager

Templates can be used with the Dataset Manager for validation:

```python
from src.datasets.dataset_manager import DatasetManager
from src.datasets.templates import CustomerSupportTemplate

manager = DatasetManager()
template = CustomerSupportTemplate()

# Upload and validate dataset against template
dataset_metadata = await manager.upload_dataset(
    file_path="customer_support_data.jsonl",
    name="Customer Support Q&A",
    task_type=template.task_type
)

# Validate against template
quality_report = await manager.validate_dataset(dataset_metadata.id)

# Check if meets template thresholds
if quality_report.completeness_score >= template.quality_thresholds.min_completeness:
    print("Dataset meets completeness threshold")
```

## Best Practices

1. **Always validate examples** before adding to datasets
2. **Check PII fields** and apply appropriate masking
3. **Review quality thresholds** to ensure dataset meets standards
4. **Use example data** as reference for proper formatting
5. **Verify domain-specific rules** (e.g., HIPAA compliance for medical data)
6. **Monitor validation errors** to improve data quality

## Extending Templates

To create custom templates, extend the `DatasetTemplate` base class:

```python
from src.datasets.templates.base_template import (
    DatasetTemplate,
    FieldDefinition,
    PIISensitivity,
    QualityThresholds,
    ValidationRule
)
from src.data_models.dataset import DatasetTaskType

class CustomTemplate(DatasetTemplate):
    def _initialize_template(self) -> None:
        self._task_type = DatasetTaskType.CUSTOM
        
        self._fields = [
            FieldDefinition(
                name="field_name",
                data_type="str",
                required=True,
                description="Field description",
                pii_sensitivity=PIISensitivity.NONE
            ),
            # Add more fields...
        ]
        
        self._validation_rules = [
            ValidationRule(
                name="custom_rule",
                description="Custom validation rule",
                validator=self._custom_validator
            )
        ]
        
        self._quality_thresholds = QualityThresholds(
            min_completeness=0.95,
            min_diversity=0.7,
            min_examples=100
        )
        
        self._example_data = [
            {"field_name": "example value"},
        ]
    
    def _custom_validator(self, example: dict) -> tuple[bool, Optional[str]]:
        # Custom validation logic
        return True, None
```

## API Reference

### DatasetTemplate

Base class for all templates.

**Properties**:
- `name`: Template name
- `task_type`: Dataset task type
- `fields`: List of field definitions
- `validation_rules`: List of validation rules
- `quality_thresholds`: Quality threshold configuration
- `example_data`: Example data

**Methods**:
- `get_schema()`: Get JSON schema representation
- `validate_example(example)`: Validate single example
- `validate_dataset(examples)`: Validate entire dataset
- `get_pii_fields()`: Get list of PII-sensitive fields
- `get_required_fields()`: Get list of required fields

### FieldDefinition

Defines a dataset field with validation rules.

**Attributes**:
- `name`: Field name
- `data_type`: Expected data type (str, int, float, list, dict)
- `required`: Whether field is required
- `description`: Field description
- `pii_sensitivity`: PII sensitivity level
- `min_length`: Minimum string length
- `max_length`: Maximum string length
- `allowed_values`: List of allowed values
- `pattern`: Regex pattern for validation

### ValidationRule

Custom validation rule for datasets.

**Attributes**:
- `name`: Rule name
- `description`: Rule description
- `validator`: Validation function returning (is_valid, error_message)

### QualityThresholds

Quality thresholds for dataset validation.

**Attributes**:
- `min_completeness`: Minimum completeness score (0-1)
- `min_diversity`: Minimum diversity score (0-1)
- `min_balance`: Minimum balance score (0-1)
- `min_examples`: Minimum number of examples
- `max_examples`: Maximum number of examples (optional)
