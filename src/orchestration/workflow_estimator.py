"""
Workflow cost and time estimator.

Estimates time and cost for a complete workflow before execution,
based on step types and configuration.

Requirements: 8.12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.data_models.workflow import StepType, WorkflowDefinition


@dataclass
class StepEstimate:
    """Estimated cost and time for a single step.

    Attributes:
        step_id: The step identifier.
        step_type: The type of step.
        estimated_duration_seconds: Estimated execution time.
        estimated_cost: Estimated cost in USD.
        confidence: Confidence level of the estimate.
    """

    step_id: str
    step_type: StepType
    estimated_duration_seconds: float = 0.0
    estimated_cost: float = 0.0
    confidence: str = "medium"


@dataclass
class WorkflowEstimate:
    """Estimated cost and time for a complete workflow.

    Attributes:
        estimated_duration_seconds: Total estimated execution time.
        estimated_cost: Total estimated cost in USD.
        cost_breakdown: Per-step cost breakdown.
        step_estimates: Individual step estimates.
        confidence: Overall confidence level.
    """

    estimated_duration_seconds: float = 0.0
    estimated_cost: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    step_estimates: list[StepEstimate] = field(default_factory=list)
    confidence: str = "medium"


# Default estimates per step type
_DEFAULT_DURATION: dict[StepType, float] = {
    StepType.DATASET_PREPARATION: 60.0,
    StepType.BASELINE_EVALUATION: 300.0,
    StepType.FINE_TUNING: 3600.0,
    StepType.POST_TUNING_EVALUATION: 300.0,
    StepType.COMPARISON: 120.0,
    StepType.APPROVAL_GATE: 0.0,
    StepType.CUSTOM: 60.0,
}

_DEFAULT_COST: dict[StepType, float] = {
    StepType.DATASET_PREPARATION: 0.01,
    StepType.BASELINE_EVALUATION: 0.50,
    StepType.FINE_TUNING: 10.00,
    StepType.POST_TUNING_EVALUATION: 0.50,
    StepType.COMPARISON: 0.10,
    StepType.APPROVAL_GATE: 0.00,
    StepType.CUSTOM: 0.05,
}


class WorkflowEstimator:
    """Estimates time and cost for workflow execution.

    Uses configurable per-step-type defaults and allows overrides
    via step configuration.

    Usage::

        estimator = WorkflowEstimator()
        estimate = estimator.estimate(workflow_definition)
    """

    def __init__(
        self,
        duration_overrides: Optional[dict[StepType, float]] = None,
        cost_overrides: Optional[dict[StepType, float]] = None,
    ) -> None:
        self._durations = dict(_DEFAULT_DURATION)
        if duration_overrides:
            self._durations.update(duration_overrides)
        self._costs = dict(_DEFAULT_COST)
        if cost_overrides:
            self._costs.update(cost_overrides)

    def estimate_step(self, step_id: str, step_type: StepType, config: dict | None = None) -> StepEstimate:
        """Estimate cost and time for a single step.

        Step config can override defaults with keys:
        - ``estimated_duration_seconds``
        - ``estimated_cost``

        Args:
            step_id: The step identifier.
            step_type: The type of step.
            config: Optional step configuration with overrides.

        Returns:
            StepEstimate for the step.
        """
        config = config or {}
        duration = config.get(
            "estimated_duration_seconds",
            self._durations.get(step_type, 60.0),
        )
        cost = config.get(
            "estimated_cost",
            self._costs.get(step_type, 0.05),
        )
        confidence = "high" if step_type in self._durations else "low"
        return StepEstimate(
            step_id=step_id,
            step_type=step_type,
            estimated_duration_seconds=float(duration),
            estimated_cost=float(cost),
            confidence=confidence,
        )

    def estimate(self, definition: WorkflowDefinition) -> WorkflowEstimate:
        """Estimate total cost and time for a workflow.

        Sums individual step estimates. Steps with dependencies
        are assumed sequential; independent steps could overlap
        but we use a conservative sequential estimate.

        Args:
            definition: The workflow definition to estimate.

        Returns:
            WorkflowEstimate with total and per-step breakdown.
        """
        step_estimates: list[StepEstimate] = []
        total_duration = 0.0
        total_cost = 0.0
        cost_breakdown: dict[str, float] = {}

        for step in definition.steps:
            est = self.estimate_step(
                step.step_id, step.step_type, step.config
            )
            step_estimates.append(est)
            total_duration += est.estimated_duration_seconds
            total_cost += est.estimated_cost
            cost_breakdown[step.step_id] = est.estimated_cost

        # Determine overall confidence
        confidences = [e.confidence for e in step_estimates]
        if all(c == "high" for c in confidences):
            overall = "high"
        elif any(c == "low" for c in confidences):
            overall = "low"
        else:
            overall = "medium"

        return WorkflowEstimate(
            estimated_duration_seconds=total_duration,
            estimated_cost=total_cost,
            cost_breakdown=cost_breakdown,
            step_estimates=step_estimates,
            confidence=overall,
        )
