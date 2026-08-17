"""
Unit tests for sequential and parallel step runners.

Requirements: 8.1, 8.11
"""

import threading

import pytest

from src.orchestration.step_executor import StepExecutor
from src.orchestration.step_runner import (
    RunResult,
    StepNode,
    run_parallel,
    run_sequential,
)


def _echo_handler(step_id: str, config: dict) -> dict:
    return {"from": step_id, **config}


def _failing_handler(step_id: str, config: dict) -> dict:
    raise RuntimeError("boom")


def _thread_recorder(step_id: str, config: dict) -> dict:
    return {"thread": threading.current_thread().name}


def _make_executor(*step_types: str, handler=None) -> StepExecutor:
    executor = StepExecutor()
    h = handler or _echo_handler
    for st in step_types:
        executor.register_handler(st, h)
    return executor


# --- Sequential runner tests (Task 15.3) ---


class TestRunSequential:
    def test_single_step(self):
        steps = [StepNode("s1", "echo")]
        executor = _make_executor("echo")
        result = run_sequential(steps, executor)
        assert result.success is True
        assert "s1" in result.results

    def test_dependency_order(self):
        steps = [
            StepNode("s1", "echo"),
            StepNode("s2", "echo", depends_on=["s1"]),
            StepNode("s3", "echo", depends_on=["s2"]),
        ]
        executor = _make_executor("echo")
        result = run_sequential(steps, executor)
        assert result.success is True
        assert list(result.results.keys()) == ["s1", "s2", "s3"]

    def test_outputs_passed_as_inputs(self):
        steps = [
            StepNode("s1", "echo", config={"val": 42}),
            StepNode("s2", "echo", depends_on=["s1"]),
        ]
        executor = _make_executor("echo")
        result = run_sequential(steps, executor)
        # s2 should receive s1's output as input
        s2_inputs = result.results["s2"].inputs
        assert s2_inputs.get("val") == 42

    def test_stops_on_failure(self):
        steps = [
            StepNode("s1", "fail"),
            StepNode("s2", "echo", depends_on=["s1"]),
        ]
        executor = StepExecutor()
        executor.register_handler("fail", _failing_handler)
        executor.register_handler("echo", _echo_handler)
        result = run_sequential(steps, executor)
        assert result.success is False
        assert "s1" in result.failed_steps
        assert "s2" not in result.results

    def test_circular_dependency_raises(self):
        steps = [
            StepNode("s1", "echo", depends_on=["s2"]),
            StepNode("s2", "echo", depends_on=["s1"]),
        ]
        executor = _make_executor("echo")
        with pytest.raises(ValueError, match="Circular dependency"):
            run_sequential(steps, executor)

    def test_missing_dependency_raises(self):
        steps = [
            StepNode("s1", "echo", depends_on=["nonexistent"]),
        ]
        executor = _make_executor("echo")
        with pytest.raises(ValueError, match="unknown step"):
            run_sequential(steps, executor)

    def test_independent_steps_execute(self):
        steps = [
            StepNode("s1", "echo"),
            StepNode("s2", "echo"),
        ]
        executor = _make_executor("echo")
        result = run_sequential(steps, executor)
        assert result.success is True
        assert len(result.results) == 2


# --- Parallel runner tests (Task 15.4) ---


class TestRunParallel:
    def test_single_step(self):
        steps = [StepNode("s1", "echo")]
        executor = _make_executor("echo")
        result = run_parallel(steps, executor)
        assert result.success is True
        assert "s1" in result.results

    def test_independent_steps_run_in_parallel(self):
        steps = [
            StepNode("s1", "echo"),
            StepNode("s2", "echo"),
            StepNode("s3", "echo"),
        ]
        executor = _make_executor("echo")
        result = run_parallel(steps, executor, max_workers=3)
        assert result.success is True
        assert len(result.results) == 3

    def test_dependency_order_respected(self):
        steps = [
            StepNode("s1", "echo"),
            StepNode("s2", "echo", depends_on=["s1"]),
        ]
        executor = _make_executor("echo")
        result = run_parallel(steps, executor)
        assert result.success is True
        # s1 must complete before s2
        s1_completed = result.results["s1"].completed_at
        s2_started = result.results["s2"].started_at
        assert s1_completed <= s2_started

    def test_outputs_passed_to_dependents(self):
        steps = [
            StepNode("s1", "echo", config={"key": "value"}),
            StepNode("s2", "echo", depends_on=["s1"]),
        ]
        executor = _make_executor("echo")
        result = run_parallel(steps, executor)
        s2_inputs = result.results["s2"].inputs
        assert s2_inputs.get("key") == "value"

    def test_stops_on_failure(self):
        steps = [
            StepNode("s1", "fail"),
            StepNode("s2", "echo", depends_on=["s1"]),
        ]
        executor = StepExecutor()
        executor.register_handler("fail", _failing_handler)
        executor.register_handler("echo", _echo_handler)
        result = run_parallel(steps, executor)
        assert result.success is False
        assert "s1" in result.failed_steps

    def test_diamond_dependency(self):
        # s1 -> s2, s3 (parallel) -> s4
        steps = [
            StepNode("s1", "echo"),
            StepNode("s2", "echo", depends_on=["s1"]),
            StepNode("s3", "echo", depends_on=["s1"]),
            StepNode("s4", "echo", depends_on=["s2", "s3"]),
        ]
        executor = _make_executor("echo")
        result = run_parallel(steps, executor, max_workers=2)
        assert result.success is True
        assert len(result.results) == 4
