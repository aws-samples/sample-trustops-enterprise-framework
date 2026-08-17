"""
Customer support dataset template.

Requirement 2.19: Customer support domain-specific template
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


class CustomerSupportTemplate(DatasetTemplate):
    """Template for customer support ticket and conversation datasets."""

    def _initialize_template(self) -> None:
        """Initialize customer support template."""
        self._task_type = DatasetTaskType.QA

        # Define fields
        self._fields = [
            FieldDefinition(
                name="ticket_id",
                data_type="str",
                required=True,
                description="Unique ticket identifier",
                pii_sensitivity=PIISensitivity.NONE,
                min_length=1,
                max_length=100,
            ),
            FieldDefinition(
                name="customer_query",
                data_type="str",
                required=True,
                description="Customer question or issue description",
                pii_sensitivity=PIISensitivity.MEDIUM,
                min_length=10,
                max_length=5000,
            ),
            FieldDefinition(
                name="agent_response",
                data_type="str",
                required=True,
                description="Agent's response to the customer",
                pii_sensitivity=PIISensitivity.LOW,
                min_length=10,
                max_length=5000,
            ),
            FieldDefinition(
                name="category",
                data_type="str",
                required=True,
                description="Issue category",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=[
                    "billing",
                    "technical",
                    "account",
                    "product",
                    "shipping",
                    "returns",
                    "general",
                ],
            ),
            FieldDefinition(
                name="priority",
                data_type="str",
                required=False,
                description="Ticket priority level",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=["low", "medium", "high", "urgent"],
            ),
            FieldDefinition(
                name="sentiment",
                data_type="str",
                required=False,
                description="Customer sentiment",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=["positive", "neutral", "negative"],
            ),
            FieldDefinition(
                name="resolution_status",
                data_type="str",
                required=False,
                description="Whether issue was resolved",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=["resolved", "unresolved", "escalated"],
            ),
            FieldDefinition(
                name="context",
                data_type="str",
                required=False,
                description="Additional context (product info, account history)",
                pii_sensitivity=PIISensitivity.MEDIUM,
                max_length=10000,
            ),
        ]

        # Define validation rules
        self._validation_rules = [
            ValidationRule(
                name="query_response_balance",
                description="Query and response should have similar lengths",
                validator=self._validate_query_response_balance,
            ),
            ValidationRule(
                name="professional_tone",
                description="Agent response should maintain professional tone",
                validator=self._validate_professional_tone,
            ),
        ]

        # Set quality thresholds
        self._quality_thresholds = QualityThresholds(
            min_completeness=0.95,
            min_diversity=0.7,
            min_balance=0.6,
            min_examples=200,
        )

        # Example data
        self._example_data = [
            {
                "ticket_id": "CS-2024-001",
                "customer_query": "I was charged twice for my subscription this month. Can you help me get a refund?",
                "agent_response": "I apologize for the inconvenience. I've reviewed your account and confirmed the duplicate charge. I've processed a refund for the extra charge, which should appear in your account within 3-5 business days. I've also added a note to prevent this from happening again.",
                "category": "billing",
                "priority": "high",
                "sentiment": "negative",
                "resolution_status": "resolved",
            },
            {
                "ticket_id": "CS-2024-002",
                "customer_query": "How do I reset my password? I can't log into my account.",
                "agent_response": "I can help you reset your password. Please click on the 'Forgot Password' link on the login page. You'll receive an email with instructions to create a new password. If you don't receive the email within 5 minutes, please check your spam folder or let me know.",
                "category": "account",
                "priority": "medium",
                "sentiment": "neutral",
                "resolution_status": "resolved",
            },
        ]

    def _validate_query_response_balance(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate that query and response have reasonable length balance."""
        if "customer_query" not in example or "agent_response" not in example:
            return True, None

        query_len = len(example["customer_query"])
        response_len = len(example["agent_response"])

        # Response should be at least 50% of query length and not more than 10x
        if response_len < query_len * 0.5:
            return False, "Agent response is too short compared to customer query"
        if response_len > query_len * 10:
            return False, "Agent response is excessively long compared to customer query"

        return True, None

    def _validate_professional_tone(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate professional tone in agent response."""
        if "agent_response" not in example:
            return True, None

        response = example["agent_response"].lower()

        # Check for unprofessional language
        unprofessional_words = ["stupid", "dumb", "idiot", "whatever", "dunno"]
        for word in unprofessional_words:
            if word in response:
                return False, f"Agent response contains unprofessional language: '{word}'"

        return True, None
