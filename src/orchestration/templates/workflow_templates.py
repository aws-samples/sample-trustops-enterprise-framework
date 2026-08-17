"""
Pre-built workflow templates for common orchestration patterns.

Provides templates for:
- Full pipeline: dataset → baseline → fine-tuning → post-tuning → comparison
- Evaluation only: dataset → baseline evaluation
- Comparison only: model comparison without fine-tuning

Requirements: 8.1, 8.17
"""

from __future__ import annotations

from typing import Optional

from src.data_models.workflow import (
    StepType,
    WorkflowDefinition,
    WorkflowStep,
)


def create_full_pipeline_template(
    dataset_config: Optional[dict] = None,
    baseline_config: Optional[dict] = None,
    fine_tuning_config: Optional[dict] = None,
    post_eval_config: Optional[dict] = None,
    comparison_config: Optional[dict] = None,
) -> WorkflowDefinition:
    """Create a full pipeline workflow template.

    Pipeline: dataset → baseline evaluation → fine-tuning →
    post-tuning evaluation → comparison.

    Args:
        dataset_config: Config for dataset preparation step.
        baseline_config: Config for baseline evaluation step.
        fine_tuning_config: Config for fine-tuning step.
        post_eval_config: Config for post-tuning evaluation step.
        comparison_config: Config for comparison step.

    Returns:
        WorkflowDefinition for the full pipeline.
    """
    steps = [
        WorkflowStep(
            step_id="dataset-prep",
            step_type=StepType.DATASET_PREPARATION,
            name="Dataset Preparation",
            config=dataset_config or {},
        ),
        WorkflowStep(
            step_id="baseline-eval",
            step_type=StepType.BASELINE_EVALUATION,
            name="Baseline Evaluation",
            config=baseline_config or {},
            depends_on=["dataset-prep"],
        ),
        WorkflowStep(
            step_id="fine-tuning",
            step_type=StepType.FINE_TUNING,
            name="Fine-Tuning",
            config=fine_tuning_config or {},
            depends_on=["dataset-prep"],
        ),
        WorkflowStep(
            step_id="post-eval",
            step_type=StepType.POST_TUNING_EVALUATION,
            name="Post-Tuning Evaluation",
            config=post_eval_config or {},
            depends_on=["fine-tuning"],
        ),
        WorkflowStep(
            step_id="comparison",
            step_type=StepType.COMPARISON,
            name="Model Comparison",
            config=comparison_config or {},
            depends_on=["baseline-eval", "post-eval"],
        ),
    ]
    return WorkflowDefinition(
        name="Full Pipeline",
        description=(
            "Complete workflow: dataset preparation → baseline evaluation "
            "→ fine-tuning → post-tuning evaluation → comparison"
        ),
        steps=steps,
    )


def create_evaluation_only_template(
    dataset_config: Optional[dict] = None,
    evaluation_config: Optional[dict] = None,
) -> WorkflowDefinition:
    """Create an evaluation-only workflow template.

    Pipeline: dataset → baseline evaluation.

    Args:
        dataset_config: Config for dataset preparation step.
        evaluation_config: Config for baseline evaluation step.

    Returns:
        WorkflowDefinition for evaluation only.
    """
    steps = [
        WorkflowStep(
            step_id="dataset-prep",
            step_type=StepType.DATASET_PREPARATION,
            name="Dataset Preparation",
            config=dataset_config or {},
        ),
        WorkflowStep(
            step_id="baseline-eval",
            step_type=StepType.BASELINE_EVALUATION,
            name="Baseline Evaluation",
            config=evaluation_config or {},
            depends_on=["dataset-prep"],
        ),
    ]
    return WorkflowDefinition(
        name="Evaluation Only",
        description="Evaluate a model on a dataset without fine-tuning",
        steps=steps,
    )


def create_comparison_only_template(
    comparison_config: Optional[dict] = None,
) -> WorkflowDefinition:
    """Create a comparison-only workflow template.

    Pipeline: model comparison without fine-tuning.

    Args:
        comparison_config: Config for comparison step.

    Returns:
        WorkflowDefinition for comparison only.
    """
    steps = [
        WorkflowStep(
            step_id="comparison",
            step_type=StepType.COMPARISON,
            name="Model Comparison",
            config=comparison_config or {},
        ),
    ]
    return WorkflowDefinition(
        name="Comparison Only",
        description="Compare two models without fine-tuning",
        steps=steps,
    )


_TEMPLATE_REGISTRY: dict[str, callable] = {
    "full_pipeline": create_full_pipeline_template,
    "evaluation_only": create_evaluation_only_template,
    "comparison_only": create_comparison_only_template,
}


def get_all_templates() -> list[WorkflowDefinition]:
    """Get all available workflow templates with default configs.

    Returns:
        List of WorkflowDefinition templates.
    """
    return [factory() for factory in _TEMPLATE_REGISTRY.values()]


def get_template_by_name(name: str) -> Optional[WorkflowDefinition]:
    """Get a workflow template by name.

    Args:
        name: Template name (full_pipeline, evaluation_only,
              comparison_only).

    Returns:
        WorkflowDefinition or None if not found.
    """
    factory = _TEMPLATE_REGISTRY.get(name)
    if factory is None:
        return None
    return factory()
