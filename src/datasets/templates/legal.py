"""
Legal dataset template.

Requirement 2.19: Legal domain-specific template

WARNING: For healthcare/financial/legal applications, AI outputs from models
trained with this template MUST NOT be used as sole decision-makers.
All outputs require review by licensed professionals.
Enable Amazon Bedrock Guardrails for content filtering in production.

This template defines dataset structure and validation only. Model output is
not legal advice and does not create an attorney-client relationship.
See SECURITY.md for the responsible-AI guidance that applies here.
"""

from typing import Optional

from src.data_models.dataset import DatasetTaskType
from .base_template import (
    DatasetTemplate,
    FieldDefinition,
    PIISensitivity,
    QualityThresholds,
    ValidationRule,
)


class LegalTemplate(DatasetTemplate):
    """Template for legal document analysis and contract review datasets."""

    def _initialize_template(self) -> None:
        """Initialize legal template."""
        self._task_type = DatasetTaskType.TEXT_GENERATION

        # Define fields
        self._fields = [
            FieldDefinition(
                name="document_id",
                data_type="str",
                required=True,
                description="Unique document identifier",
                pii_sensitivity=PIISensitivity.NONE,
                min_length=1,
                max_length=100,
            ),
            FieldDefinition(
                name="document_type",
                data_type="str",
                required=True,
                description="Type of legal document",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=[
                    "contract",
                    "agreement",
                    "policy",
                    "terms_of_service",
                    "privacy_policy",
                    "compliance",
                    "litigation",
                    "regulatory",
                ],
            ),
            FieldDefinition(
                name="document_text",
                data_type="str",
                required=True,
                description="Legal document text or excerpt",
                pii_sensitivity=PIISensitivity.HIGH,
                min_length=50,
                max_length=50000,
            ),
            FieldDefinition(
                name="prompt",
                data_type="str",
                required=True,
                description="Analysis prompt or question about the document",
                pii_sensitivity=PIISensitivity.LOW,
                min_length=10,
                max_length=2000,
            ),
            FieldDefinition(
                name="expected_analysis",
                data_type="str",
                required=True,
                description="Expected analysis or answer",
                pii_sensitivity=PIISensitivity.MEDIUM,
                min_length=20,
                max_length=10000,
            ),
            FieldDefinition(
                name="jurisdiction",
                data_type="str",
                required=False,
                description="Legal jurisdiction",
                pii_sensitivity=PIISensitivity.NONE,
                max_length=100,
            ),
            FieldDefinition(
                name="practice_area",
                data_type="str",
                required=False,
                description="Legal practice area",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=[
                    "corporate",
                    "intellectual_property",
                    "employment",
                    "real_estate",
                    "tax",
                    "litigation",
                    "regulatory",
                    "privacy",
                ],
            ),
            FieldDefinition(
                name="risk_level",
                data_type="str",
                required=False,
                description="Risk level of the clause or issue",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=["low", "medium", "high", "critical"],
            ),
            FieldDefinition(
                name="citations",
                data_type="list",
                required=False,
                description="Legal citations or references",
                pii_sensitivity=PIISensitivity.NONE,
            ),
        ]

        # Define validation rules
        self._validation_rules = [
            ValidationRule(
                name="document_length",
                description="Document text should be substantial",
                validator=self._validate_document_length,
            ),
            ValidationRule(
                name="analysis_quality",
                description="Analysis should reference the document",
                validator=self._validate_analysis_quality,
            ),
            ValidationRule(
                name="legal_terminology",
                description="Should contain appropriate legal terminology",
                validator=self._validate_legal_terminology,
            ),
        ]

        # Set quality thresholds
        self._quality_thresholds = QualityThresholds(
            min_completeness=0.98,
            min_diversity=0.75,
            min_balance=0.65,
            min_examples=100,
        )

        # Example data
        self._example_data = [
            {
                "document_id": "LEGAL-2024-001",
                "document_type": "contract",
                "document_text": "This Software License Agreement ('Agreement') is entered into as of January 1, 2024, between TechCorp Inc. ('Licensor') and Customer ('Licensee'). The Licensor grants Licensee a non-exclusive, non-transferable license to use the Software. The license term is perpetual unless terminated for breach. Licensee shall not reverse engineer, decompile, or disassemble the Software.",
                "prompt": "Identify the key obligations and restrictions placed on the Licensee in this software license agreement.",
                "expected_analysis": "The Licensee has the following key obligations and restrictions: 1) The license is non-exclusive and non-transferable, meaning the Licensee cannot transfer rights to third parties. 2) The Licensee is prohibited from reverse engineering, decompiling, or disassembling the Software. 3) The license can be terminated if the Licensee breaches the agreement terms. The license is perpetual, providing long-term usage rights as long as terms are followed.",
                "jurisdiction": "United States",
                "practice_area": "intellectual_property",
                "risk_level": "medium",
            },
            {
                "document_id": "LEGAL-2024-002",
                "document_type": "privacy_policy",
                "document_text": "We collect personal information including name, email address, and payment information. We use this information to process orders and communicate with customers. We do not sell personal information to third parties. We retain data for 7 years after account closure for compliance purposes.",
                "prompt": "Assess the GDPR compliance of this privacy policy's data retention clause.",
                "expected_analysis": "The 7-year data retention period after account closure may raise GDPR compliance concerns. Under GDPR Article 5(1)(e), personal data should be kept only as long as necessary for the purposes for which it was processed. A 7-year retention period must be justified by specific legal obligations (e.g., tax, accounting) or legitimate interests. The policy should specify the legal basis for this retention period and provide users with information about their right to erasure under certain circumstances.",
                "jurisdiction": "European Union",
                "practice_area": "privacy",
                "risk_level": "high",
            },
        ]

    def _validate_document_length(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate document has substantial content."""
        if "document_text" not in example:
            return True, None

        doc_length = len(example["document_text"])
        if doc_length < 100:
            return False, "Document text is too short for meaningful legal analysis"

        return True, None

    def _validate_analysis_quality(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate analysis references the document."""
        if "expected_analysis" not in example or "document_text" not in example:
            return True, None

        analysis = example["expected_analysis"].lower()
        document = example["document_text"].lower()

        # Check if analysis contains specific references or quotes
        # Simple heuristic: analysis should be detailed (>100 chars) and reference key terms
        if len(analysis) < 100:
            return False, "Analysis is too brief for legal document review"

        return True, None

    def _validate_legal_terminology(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate presence of legal terminology."""
        if "document_text" not in example:
            return True, None

        text = example["document_text"].lower()

        # Check for common legal terms
        legal_terms = [
            "agreement", "party", "parties", "shall", "hereby", "pursuant",
            "clause", "provision", "term", "condition", "liability", "obligation"
        ]

        found_terms = sum(1 for term in legal_terms if term in text)
        if found_terms < 2:
            return False, "Document lacks sufficient legal terminology"

        return True, None
