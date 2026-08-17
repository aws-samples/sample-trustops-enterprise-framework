"""Pre-built workflow templates for common orchestration patterns."""

from src.orchestration.templates.workflow_templates import (
    create_evaluation_only_template,
    create_comparison_only_template,
    create_full_pipeline_template,
    get_all_templates,
    get_template_by_name,
)

__all__ = [
    "create_evaluation_only_template",
    "create_comparison_only_template",
    "create_full_pipeline_template",
    "get_all_templates",
    "get_template_by_name",
]
