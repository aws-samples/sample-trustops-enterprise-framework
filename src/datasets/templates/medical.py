"""
Medical dataset template.

Requirement 2.19: Medical domain-specific template (HIPAA-compliant)

WARNING: For healthcare/financial/legal applications, AI outputs from models
trained with this template MUST NOT be used as sole decision-makers.
All outputs require review by licensed professionals.
Enable Amazon Bedrock Guardrails for content filtering in production.

This template defines dataset structure and validation only. It does not make
a model clinically safe, and it is not a substitute for regulatory review.
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


class MedicalTemplate(DatasetTemplate):
    """Template for medical Q&A and clinical notes datasets (HIPAA-compliant)."""

    def _initialize_template(self) -> None:
        """Initialize medical template."""
        self._task_type = DatasetTaskType.QA

        # Define fields
        self._fields = [
            FieldDefinition(
                name="case_id",
                data_type="str",
                required=True,
                description="Unique case identifier (de-identified)",
                pii_sensitivity=PIISensitivity.CRITICAL,
                min_length=1,
                max_length=100,
            ),
            FieldDefinition(
                name="clinical_question",
                data_type="str",
                required=True,
                description="Clinical question or scenario",
                pii_sensitivity=PIISensitivity.HIGH,
                min_length=20,
                max_length=5000,
            ),
            FieldDefinition(
                name="clinical_answer",
                data_type="str",
                required=True,
                description="Clinical answer or recommendation",
                pii_sensitivity=PIISensitivity.HIGH,
                min_length=20,
                max_length=5000,
            ),
            FieldDefinition(
                name="specialty",
                data_type="str",
                required=True,
                description="Medical specialty",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=[
                    "cardiology",
                    "neurology",
                    "oncology",
                    "pediatrics",
                    "psychiatry",
                    "radiology",
                    "surgery",
                    "internal_medicine",
                    "emergency_medicine",
                    "general_practice",
                ],
            ),
            FieldDefinition(
                name="patient_demographics",
                data_type="str",
                required=False,
                description="De-identified patient demographics (age range, gender)",
                pii_sensitivity=PIISensitivity.HIGH,
                max_length=200,
            ),
            FieldDefinition(
                name="diagnosis_codes",
                data_type="list",
                required=False,
                description="ICD-10 diagnosis codes",
                pii_sensitivity=PIISensitivity.MEDIUM,
            ),
            FieldDefinition(
                name="urgency",
                data_type="str",
                required=False,
                description="Clinical urgency level",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=["routine", "urgent", "emergent", "critical"],
            ),
            FieldDefinition(
                name="evidence_level",
                data_type="str",
                required=False,
                description="Evidence level for the recommendation",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=["A", "B", "C", "D", "expert_opinion"],
            ),
            FieldDefinition(
                name="references",
                data_type="list",
                required=False,
                description="Medical literature references",
                pii_sensitivity=PIISensitivity.NONE,
            ),
            FieldDefinition(
                name="hipaa_compliant",
                data_type="bool",
                required=True,
                description="Flag indicating HIPAA compliance verification",
                pii_sensitivity=PIISensitivity.NONE,
            ),
        ]

        # Define validation rules
        self._validation_rules = [
            ValidationRule(
                name="hipaa_compliance",
                description="Must be marked as HIPAA compliant",
                validator=self._validate_hipaa_compliance,
            ),
            ValidationRule(
                name="no_phi",
                description="Should not contain Protected Health Information",
                validator=self._validate_no_phi,
            ),
            ValidationRule(
                name="clinical_accuracy",
                description="Answer should be clinically appropriate",
                validator=self._validate_clinical_accuracy,
            ),
        ]

        # Set quality thresholds
        self._quality_thresholds = QualityThresholds(
            min_completeness=0.98,
            min_diversity=0.8,
            min_balance=0.7,
            min_examples=150,
        )

        # Example data
        self._example_data = [
            {
                "case_id": "MED-2024-001",
                "clinical_question": "A 55-year-old male patient presents with chest pain radiating to the left arm, diaphoresis, and shortness of breath for 30 minutes. ECG shows ST-segment elevation in leads II, III, and aVF. What is the most appropriate immediate management?",
                "clinical_answer": "This presentation is consistent with an acute inferior ST-elevation myocardial infarction (STEMI). Immediate management includes: 1) Activate cardiac catheterization lab for primary PCI (door-to-balloon time <90 minutes). 2) Administer aspirin 325mg, clopidogrel or ticagrelor loading dose. 3) Provide supplemental oxygen if SpO2 <90%. 4) Administer sublingual nitroglycerin if BP permits. 5) Initiate anticoagulation with heparin or bivalirudin. 6) Provide morphine for pain control. 7) Continuous cardiac monitoring. If PCI not available within 120 minutes, consider fibrinolytic therapy if no contraindications.",
                "specialty": "cardiology",
                "patient_demographics": "55-65 years, male",
                "diagnosis_codes": ["I21.19"],
                "urgency": "critical",
                "evidence_level": "A",
                "hipaa_compliant": True,
            },
            {
                "case_id": "MED-2024-002",
                "clinical_question": "What are the first-line treatment options for a newly diagnosed patient with type 2 diabetes mellitus with HbA1c of 7.8% and no contraindications?",
                "clinical_answer": "For newly diagnosed type 2 diabetes with HbA1c 7.8%, first-line treatment includes: 1) Lifestyle modifications: medical nutrition therapy, weight loss if overweight (target 5-10% reduction), regular physical activity (150 min/week). 2) Metformin: Start 500mg once or twice daily with meals, titrate up to 2000mg daily as tolerated. Metformin is preferred due to efficacy, safety profile, low cost, and potential cardiovascular benefits. 3) Set individualized HbA1c target (typically <7% for most adults). 4) Monitor HbA1c every 3 months until stable, then every 6 months. 5) Screen for complications: annual eye exam, foot exam, kidney function, lipid profile. If HbA1c remains >7% after 3 months on metformin, consider adding second agent based on patient factors.",
                "specialty": "internal_medicine",
                "patient_demographics": "Adult, newly diagnosed",
                "diagnosis_codes": ["E11.9"],
                "urgency": "routine",
                "evidence_level": "A",
                "hipaa_compliant": True,
            },
        ]

    def _validate_hipaa_compliance(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate HIPAA compliance flag is set."""
        if "hipaa_compliant" not in example:
            return False, "HIPAA compliance flag is required"

        if not example["hipaa_compliant"]:
            return False, "Example must be marked as HIPAA compliant"

        return True, None

    def _validate_no_phi(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate no Protected Health Information is present."""
        # Check for common PHI patterns
        phi_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b[A-Z]{2}\d{6}\b",  # Medical record number pattern
            r"\b\d{10}\b",  # Phone number
        ]

        import re

        for field in ["clinical_question", "clinical_answer", "patient_demographics"]:
            if field not in example:
                continue

            text = str(example[field])
            for pattern in phi_patterns:
                if re.search(pattern, text):
                    return False, f"Potential PHI detected in {field}"

        # Check for specific names (simple check)
        if "patient_demographics" in example:
            demographics = example["patient_demographics"].lower()
            # Should only contain age ranges and general demographics
            if any(word in demographics for word in ["mr.", "mrs.", "dr.", "patient name"]):
                return False, "Patient demographics should not contain names or titles"

        return True, None

    def _validate_clinical_accuracy(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate clinical answer is appropriate."""
        if "clinical_answer" not in example:
            return True, None

        answer = example["clinical_answer"]

        # Answer should be substantial
        if len(answer) < 50:
            return False, "Clinical answer is too brief"

        # Should contain clinical terminology
        clinical_terms = [
            "patient", "treatment", "diagnosis", "therapy", "management",
            "medication", "dose", "monitor", "assess", "recommend"
        ]

        answer_lower = answer.lower()
        found_terms = sum(1 for term in clinical_terms if term in answer_lower)

        if found_terms < 2:
            return False, "Clinical answer lacks appropriate medical terminology"

        return True, None
