"""
Domain-specific dataset templates for common enterprise use cases.

This module provides pre-built dataset templates for customer support, legal,
medical, and financial domains with schema definitions, validation rules,
and quality thresholds.

Requirement 2.19: Domain-specific templates for common enterprise use cases
"""

from .base_template import DatasetTemplate, FieldDefinition, ValidationRule
from .customer_support import CustomerSupportTemplate
from .financial import FinancialTemplate
from .legal import LegalTemplate
from .medical import MedicalTemplate

__all__ = [
    "DatasetTemplate",
    "FieldDefinition",
    "ValidationRule",
    "CustomerSupportTemplate",
    "FinancialTemplate",
    "LegalTemplate",
    "MedicalTemplate",
]
