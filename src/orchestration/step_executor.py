"""
Workflow step executor for individual step execution.

Executes individual workflow steps with logging, recording timestamps,
inputs, outputs, and duration for each step.

Requirements: 8.3
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


@dataclass
class StepExecutionResult:
    """Result of executing a single workflow step.

    Attributes:
        step_id: The step identifier.
        success: Whether the step completed successfully.
        output: Output data from the step execution.
        error: Error message if the step failed.
        started_at: Timestamp when execution started.
        completed_at: Timestamp when execution completed.
        duration_seconds: Duration of execution in seconds.
        inputs: The inputs provided to the step.
    """

    step_id: str
    success: bool
    output: dict = field(default_factory=dict)
    error: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 0.0
    inputs: dict = field(default_factory=dict)


# Type alias for step handler functions
StepHandler = Callable[[str, dict], dict]


class StepExecutor:
    """Executes individual workflow steps with logging.

    Records timestamps, inputs, outputs, and duration for each step.
    Supports registering custom handlers for different step types.

    Usage::

        executor = StepExecutor()
        executor.register_handler("custom", my_handler)
        result = executor.execute("step-1", "custom", {"key": "value"})
    """

    def __init__(self) -> None:
        self._handlers: dict[str, StepHandler] = {}
        self._execution_log: list[StepExecutionResult] = []

    def register_handler(self, step_type: str, handler: StepHandler) -> None:
        """Register a handler function for a step type.

        Args:
            step_type: The step type identifier.
            handler: A callable that takes (step_id, config) and returns output dict.
        """
        self._handlers[step_type] = handler

    def execute(
        self,
        step_id: str,
        step_type: str,
        config: dict | None = None,
        inputs: dict | None = None,
    ) -> StepExecutionResult:
        """Execute a single workflow step.

        Args:
            step_id: Unique identifier for the step.
            step_type: The type of step to execute.
            config: Configuration parameters for the step.
            inputs: Input data from previous steps.

        Returns:
            StepExecutionResult with execution details.
        """
        config = config or {}
        inputs = inputs or {}
        merged = {**config, **inputs}

        started_at = datetime.now(timezone.utc)

        handler = self._handlers.get(step_type)
        if handler is None:
            completed_at = datetime.now(timezone.utc)
            duration = (completed_at - started_at).total_seconds()
            result = StepExecutionResult(
                step_id=step_id,
                success=False,
                error=f"No handler registered for step type: {step_type}",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                inputs=merged,
            )
            self._execution_log.append(result)
            return result

        try:
            output = handler(step_id, merged)
            completed_at = datetime.now(timezone.utc)
            duration = (completed_at - started_at).total_seconds()
            result = StepExecutionResult(
                step_id=step_id,
                success=True,
                output=output or {},
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                inputs=merged,
            )
        except Exception as exc:
            completed_at = datetime.now(timezone.utc)
            duration = (completed_at - started_at).total_seconds()
            result = StepExecutionResult(
                step_id=step_id,
                success=False,
                error=str(exc),
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                inputs=merged,
            )

        self._execution_log.append(result)
        return result

    @property
    def execution_log(self) -> list[StepExecutionResult]:
        """Return a copy of the execution log."""
        return list(self._execution_log)

    def clear_log(self) -> None:
        """Clear the execution log."""
        self._execution_log = []
