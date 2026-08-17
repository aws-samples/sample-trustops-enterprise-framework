"""
Sequential and parallel step runners for workflow orchestration.

Executes workflow steps in dependency order, passing outputs from
completed steps as inputs to dependent steps. Supports both sequential
and parallel execution of independent steps.

Requirements: 8.1, 8.11
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from src.orchestration.step_executor import StepExecutionResult, StepExecutor


@dataclass
class StepNode:
    """A step with its dependency and execution metadata.

    Attributes:
        step_id: Unique step identifier.
        step_type: The type of step.
        config: Configuration for the step.
        depends_on: List of step IDs this step depends on.
    """

    step_id: str
    step_type: str
    config: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Result of running a set of steps.

    Attributes:
        results: Mapping of step_id to execution result.
        success: Whether all steps completed successfully.
        failed_steps: List of step IDs that failed.
    """

    results: dict[str, StepExecutionResult] = field(default_factory=dict)
    success: bool = True
    failed_steps: list[str] = field(default_factory=list)


def _topological_sort(steps: list[StepNode]) -> list[list[StepNode]]:
    """Sort steps into execution layers based on dependencies.

    Returns a list of layers where each layer contains steps that
    can execute in parallel (all dependencies are in earlier layers).

    Args:
        steps: List of step nodes with dependency information.

    Returns:
        List of layers, each containing independent steps.

    Raises:
        ValueError: If there are circular dependencies or missing deps.
    """
    step_map = {s.step_id: s for s in steps}
    in_degree: dict[str, int] = {s.step_id: 0 for s in steps}

    for step in steps:
        for dep in step.depends_on:
            if dep not in step_map:
                raise ValueError(
                    f"Step '{step.step_id}' depends on unknown step '{dep}'"
                )
            in_degree[step.step_id] += 1

    layers: list[list[StepNode]] = []
    remaining = set(step_map.keys())

    while remaining:
        # Find all steps with no unresolved dependencies
        ready = [
            sid for sid in remaining if in_degree[sid] == 0
        ]
        if not ready:
            raise ValueError(
                f"Circular dependency detected among steps: {remaining}"
            )

        layer = [step_map[sid] for sid in ready]
        layers.append(layer)

        for sid in ready:
            remaining.remove(sid)
            # Decrease in-degree for dependents
            for step in steps:
                if sid in step.depends_on and step.step_id in remaining:
                    in_degree[step.step_id] -= 1

    return layers


def _gather_inputs(
    step: StepNode,
    completed: dict[str, StepExecutionResult],
) -> dict:
    """Gather outputs from dependency steps as inputs.

    Args:
        step: The step that needs inputs.
        completed: Map of completed step results.

    Returns:
        Merged outputs from all dependency steps.
    """
    inputs: dict[str, Any] = {}
    for dep_id in step.depends_on:
        if dep_id in completed and completed[dep_id].success:
            inputs.update(completed[dep_id].output)
    return inputs


def run_sequential(
    steps: list[StepNode],
    executor: StepExecutor,
) -> RunResult:
    """Execute steps sequentially in dependency order.

    Steps are sorted topologically and executed one at a time.
    Outputs from completed steps are passed as inputs to dependent steps.
    Execution stops on the first failure.

    Args:
        steps: List of step nodes to execute.
        executor: The step executor to use.

    Returns:
        RunResult with all execution results.
    """
    layers = _topological_sort(steps)
    completed: dict[str, StepExecutionResult] = {}
    run_result = RunResult()

    for layer in layers:
        for step in layer:
            inputs = _gather_inputs(step, completed)
            result = executor.execute(
                step_id=step.step_id,
                step_type=step.step_type,
                config=step.config,
                inputs=inputs,
            )
            completed[step.step_id] = result
            run_result.results[step.step_id] = result

            if not result.success:
                run_result.success = False
                run_result.failed_steps.append(step.step_id)
                return run_result

    return run_result


def run_parallel(
    steps: list[StepNode],
    executor: StepExecutor,
    max_workers: int = 4,
) -> RunResult:
    """Execute independent steps in parallel.

    Steps are sorted into layers. Steps within the same layer
    (no dependencies between them) execute concurrently.
    Execution stops if any step in a layer fails.

    Args:
        steps: List of step nodes to execute.
        executor: The step executor to use.
        max_workers: Maximum number of concurrent workers.

    Returns:
        RunResult with all execution results.
    """
    layers = _topological_sort(steps)
    completed: dict[str, StepExecutionResult] = {}
    run_result = RunResult()

    for layer in layers:
        if len(layer) == 1:
            # Single step, no need for threading
            step = layer[0]
            inputs = _gather_inputs(step, completed)
            result = executor.execute(
                step_id=step.step_id,
                step_type=step.step_type,
                config=step.config,
                inputs=inputs,
            )
            completed[step.step_id] = result
            run_result.results[step.step_id] = result
            if not result.success:
                run_result.success = False
                run_result.failed_steps.append(step.step_id)
                return run_result
        else:
            # Multiple independent steps, run in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}
                for step in layer:
                    inputs = _gather_inputs(step, completed)
                    future = pool.submit(
                        executor.execute,
                        step_id=step.step_id,
                        step_type=step.step_type,
                        config=step.config,
                        inputs=inputs,
                    )
                    futures[future] = step.step_id

                for future in as_completed(futures):
                    step_id = futures[future]
                    result = future.result()
                    completed[step_id] = result
                    run_result.results[step_id] = result
                    if not result.success:
                        run_result.success = False
                        run_result.failed_steps.append(step_id)

            if not run_result.success:
                return run_result

    return run_result
