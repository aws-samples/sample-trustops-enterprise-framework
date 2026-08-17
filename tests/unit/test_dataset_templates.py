"""
Unit tests for domain-specific dataset templates.

Requirement 2.19: Test domain-specific templates
"""

import pytest

from src.data_models.dataset import DatasetTaskType
from src.datasets.templates import (
    CustomerSupportTemplate,
    FinancialTemplate,
    LegalTemplate,
    MedicalTemplate,
)
from src.datasets.templates.base_template import PIISensitivity


class TestCustomerSupportTemplate:
    """Test customer support template."""

    def test_template_initialization(self):
        """Test template initializes correctly."""
        template = CustomerSupportTemplate()

        assert template.name == "CustomerSupport"
        assert template.task_type == DatasetTaskType.QA
        assert len(template.fields) == 8
        assert len(template.validation_rules) == 2
        assert len(template.example_data) == 2

    def test_required_fields(self):
        """Test required fields are correctly defined."""
        template = CustomerSupportTemplate()
        required = template.get_required_fields()

        assert "ticket_id" in required
        assert "customer_query" in required
        assert "agent_response" in required
        assert "category" in required
        assert "priority" not in required  # Optional field

    def test_pii_fields(self):
        """Test PII fields are correctly identified."""
        template = CustomerSupportTemplate()
        pii_fields = template.get_pii_fields()

        assert "customer_query" in pii_fields
        assert "agent_response" in pii_fields
        assert "context" in pii_fields
        assert "ticket_id" not in pii_fields

    def test_validate_valid_example(self):
        """Test validation of valid example."""
        template = CustomerSupportTemplate()
        example = {
            "ticket_id": "CS-001",
            "customer_query": "I need help with my account password reset.",
            "agent_response": "I can help you reset your password. Please follow these steps...",
            "category": "account",
            "priority": "medium",
        }

        is_valid, errors = template.validate_example(example)
        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_required_field(self):
        """Test validation fails for missing required field."""
        template = CustomerSupportTemplate()
        example = {
            "ticket_id": "CS-001",
            "customer_query": "I need help.",
            # Missing agent_response
            "category": "account",
        }

        is_valid, errors = template.validate_example(example)
        assert not is_valid
        assert any("agent_response" in error for error in errors)

    def test_validate_invalid_category(self):
        """Test validation fails for invalid category."""
        template = CustomerSupportTemplate()
        example = {
            "ticket_id": "CS-001",
            "customer_query": "I need help with my account.",
            "agent_response": "I can help you.",
            "category": "invalid_category",
        }

        is_valid, errors = template.validate_example(example)
        assert not is_valid
        assert any("category" in error for error in errors)

    def test_validate_query_too_short(self):
        """Test validation fails for too short query."""
        template = CustomerSupportTemplate()
        example = {
            "ticket_id": "CS-001",
            "customer_query": "Help",  # Too short
            "agent_response": "I can help you with that issue.",
            "category": "general",
        }

        is_valid, errors = template.validate_example(example)
        assert not is_valid
        assert any("too short" in error.lower() for error in errors)

    def test_validate_dataset(self):
        """Test dataset validation."""
        template = CustomerSupportTemplate()
        examples = template.example_data

        is_valid, report = template.validate_dataset(examples)

        assert report["total_examples"] == 2
        assert report["valid_examples"] == 2
        assert report["invalid_examples"] == 0

    def test_get_schema(self):
        """Test schema generation."""
        template = CustomerSupportTemplate()
        schema = template.get_schema()

        assert schema["name"] == "CustomerSupport"
        assert schema["task_type"] == "qa"
        assert len(schema["fields"]) == 8
        assert len(schema["validation_rules"]) == 2


class TestLegalTemplate:
    """Test legal template."""

    def test_template_initialization(self):
        """Test template initializes correctly."""
        template = LegalTemplate()

        assert template.name == "Legal"
        assert template.task_type == DatasetTaskType.TEXT_GENERATION
        assert len(template.fields) == 9
        assert len(template.validation_rules) == 3
        assert len(template.example_data) == 2

    def test_required_fields(self):
        """Test required fields are correctly defined."""
        template = LegalTemplate()
        required = template.get_required_fields()

        assert "document_id" in required
        assert "document_type" in required
        assert "document_text" in required
        assert "prompt" in required
        assert "expected_analysis" in required
        assert "jurisdiction" not in required  # Optional

    def test_pii_sensitivity_levels(self):
        """Test PII sensitivity levels are appropriate."""
        template = LegalTemplate()

        # Find document_text field
        doc_field = next(f for f in template.fields if f.name == "document_text")
        assert doc_field.pii_sensitivity == PIISensitivity.HIGH

    def test_validate_valid_example(self):
        """Test validation of valid legal example."""
        template = LegalTemplate()
        example = template.example_data[0]

        is_valid, errors = template.validate_example(example)
        assert is_valid
        assert len(errors) == 0

    def test_validate_document_too_short(self):
        """Test validation fails for too short document."""
        template = LegalTemplate()
        example = {
            "document_id": "LEGAL-001",
            "document_type": "contract",
            "document_text": "Short text",  # Too short
            "prompt": "Analyze this document",
            "expected_analysis": "This is a brief analysis of the document.",
        }

        is_valid, errors = template.validate_example(example)
        assert not is_valid

    def test_validate_invalid_document_type(self):
        """Test validation fails for invalid document type."""
        template = LegalTemplate()
        example = {
            "document_id": "LEGAL-001",
            "document_type": "invalid_type",
            "document_text": "This is a legal document with sufficient length for validation.",
            "prompt": "Analyze this document",
            "expected_analysis": "This is the analysis.",
        }

        is_valid, errors = template.validate_example(example)
        assert not is_valid
        assert any("document_type" in error for error in errors)

    def test_get_schema(self):
        """Test schema generation."""
        template = LegalTemplate()
        schema = template.get_schema()

        assert schema["name"] == "Legal"
        assert schema["task_type"] == "text_generation"
        assert "quality_thresholds" in schema


class TestMedicalTemplate:
    """Test medical template."""

    def test_template_initialization(self):
        """Test template initializes correctly."""
        template = MedicalTemplate()

        assert template.name == "Medical"
        assert template.task_type == DatasetTaskType.QA
        assert len(template.fields) == 10
        assert len(template.validation_rules) == 3
        assert len(template.example_data) == 2

    def test_required_fields(self):
        """Test required fields including HIPAA compliance."""
        template = MedicalTemplate()
        required = template.get_required_fields()

        assert "case_id" in required
        assert "clinical_question" in required
        assert "clinical_answer" in required
        assert "specialty" in required
        assert "hipaa_compliant" in required

    def test_pii_sensitivity_critical(self):
        """Test critical PII fields are marked."""
        template = MedicalTemplate()

        case_id_field = next(f for f in template.fields if f.name == "case_id")
        assert case_id_field.pii_sensitivity == PIISensitivity.CRITICAL

    def test_validate_valid_example(self):
        """Test validation of valid medical example."""
        template = MedicalTemplate()
        example = template.example_data[0]

        is_valid, errors = template.validate_example(example)
        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_hipaa_flag(self):
        """Test validation fails without HIPAA compliance flag."""
        template = MedicalTemplate()
        example = {
            "case_id": "MED-001",
            "clinical_question": "What is the treatment for condition X?",
            "clinical_answer": "The treatment includes medication Y and therapy Z.",
            "specialty": "cardiology",
            # Missing hipaa_compliant flag
        }

        is_valid, errors = template.validate_example(example)
        assert not is_valid
        assert any("hipaa" in error.lower() for error in errors)

    def test_validate_hipaa_not_compliant(self):
        """Test validation fails if HIPAA flag is False."""
        template = MedicalTemplate()
        example = {
            "case_id": "MED-001",
            "clinical_question": "What is the treatment?",
            "clinical_answer": "Treatment includes medication.",
            "specialty": "cardiology",
            "hipaa_compliant": False,
        }

        is_valid, errors = template.validate_example(example)
        assert not is_valid

    def test_validate_invalid_specialty(self):
        """Test validation fails for invalid specialty."""
        template = MedicalTemplate()
        example = {
            "case_id": "MED-001",
            "clinical_question": "What is the treatment?",
            "clinical_answer": "Treatment includes medication.",
            "specialty": "invalid_specialty",
            "hipaa_compliant": True,
        }

        is_valid, errors = template.validate_example(example)
        assert not is_valid
        assert any("specialty" in error for error in errors)

    def test_quality_thresholds(self):
        """Test medical template has strict quality thresholds."""
        template = MedicalTemplate()

        assert template.quality_thresholds.min_completeness == 0.98
        assert template.quality_thresholds.min_diversity == 0.8
        assert template.quality_thresholds.min_examples == 150


class TestFinancialTemplate:
    """Test financial template."""

    def test_template_initialization(self):
        """Test template initializes correctly."""
        template = FinancialTemplate()

        assert template.name == "Financial"
        assert template.task_type == DatasetTaskType.TEXT_GENERATION
        assert len(template.fields) == 10
        assert len(template.validation_rules) == 3
        assert len(template.example_data) == 2

    def test_required_fields(self):
        """Test required fields are correctly defined."""
        template = FinancialTemplate()
        required = template.get_required_fields()

        assert "analysis_id" in required
        assert "analysis_type" in required
        assert "financial_data" in required
        assert "prompt" in required
        assert "expected_analysis" in required
        assert "risk_level" not in required  # Optional

    def test_pii_sensitivity(self):
        """Test PII sensitivity for financial data."""
        template = FinancialTemplate()

        financial_data_field = next(
            f for f in template.fields if f.name == "financial_data"
        )
        assert financial_data_field.pii_sensitivity == PIISensitivity.HIGH

    def test_validate_valid_example(self):
        """Test validation of valid financial example."""
        template = FinancialTemplate()
        example = template.example_data[0]

        is_valid, errors = template.validate_example(example)
        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_analysis_type(self):
        """Test validation fails for invalid analysis type."""
        template = FinancialTemplate()
        example = {
            "analysis_id": "FIN-001",
            "analysis_type": "invalid_type",
            "financial_data": "Company has revenue of $100M and profit margin of 15%.",
            "prompt": "Analyze the financial health",
            "expected_analysis": "The company shows strong profitability.",
        }

        is_valid, errors = template.validate_example(example)
        assert not is_valid
        assert any("analysis_type" in error for error in errors)

    def test_validate_financial_data_too_short(self):
        """Test validation fails for insufficient financial data."""
        template = FinancialTemplate()
        example = {
            "analysis_id": "FIN-001",
            "analysis_type": "risk_assessment",
            "financial_data": "Short data",  # Too short
            "prompt": "Analyze the risk",
            "expected_analysis": "Risk is moderate based on the available information.",
        }

        is_valid, errors = template.validate_example(example)
        assert not is_valid

    def test_validate_dataset_quality_thresholds(self):
        """Test dataset validation with quality thresholds."""
        template = FinancialTemplate()

        # Create dataset with enough examples
        examples = template.example_data * 80  # 160 examples

        is_valid, report = template.validate_dataset(examples)

        assert report["total_examples"] == 160
        assert report["valid_examples"] == 160
        assert report["meets_quality_thresholds"]

    def test_get_schema(self):
        """Test schema generation."""
        template = FinancialTemplate()
        schema = template.get_schema()

        assert schema["name"] == "Financial"
        assert schema["task_type"] == "text_generation"
        assert len(schema["fields"]) == 10


class TestTemplateComparison:
    """Test comparison across templates."""

    def test_all_templates_have_examples(self):
        """Test all templates provide example data."""
        templates = [
            CustomerSupportTemplate(),
            LegalTemplate(),
            MedicalTemplate(),
            FinancialTemplate(),
        ]

        for template in templates:
            assert len(template.example_data) >= 2
            assert all(isinstance(ex, dict) for ex in template.example_data)

    def test_all_templates_have_validation_rules(self):
        """Test all templates have validation rules."""
        templates = [
            CustomerSupportTemplate(),
            LegalTemplate(),
            MedicalTemplate(),
            FinancialTemplate(),
        ]

        for template in templates:
            assert len(template.validation_rules) >= 2

    def test_all_templates_have_pii_fields(self):
        """Test all templates identify PII fields."""
        templates = [
            CustomerSupportTemplate(),
            LegalTemplate(),
            MedicalTemplate(),
            FinancialTemplate(),
        ]

        for template in templates:
            pii_fields = template.get_pii_fields()
            assert len(pii_fields) > 0

    def test_all_templates_generate_schema(self):
        """Test all templates can generate schema."""
        templates = [
            CustomerSupportTemplate(),
            LegalTemplate(),
            MedicalTemplate(),
            FinancialTemplate(),
        ]

        for template in templates:
            schema = template.get_schema()
            assert "name" in schema
            assert "task_type" in schema
            assert "fields" in schema
            assert "validation_rules" in schema
            assert "quality_thresholds" in schema

    def test_template_quality_thresholds_valid(self):
        """Test all templates have valid quality thresholds."""
        templates = [
            CustomerSupportTemplate(),
            LegalTemplate(),
            MedicalTemplate(),
            FinancialTemplate(),
        ]

        for template in templates:
            thresholds = template.quality_thresholds
            assert 0.0 <= thresholds.min_completeness <= 1.0
            assert 0.0 <= thresholds.min_diversity <= 1.0
            assert 0.0 <= thresholds.min_balance <= 1.0
            assert thresholds.min_examples > 0
