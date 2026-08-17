"""
Unit tests for the workflow step executor.

Requirements: 8.3
"""

from src.orchestration.step_executor import StepExecutionResult, StepExecutor


def _echo_handler(step_id: str, config: dict) -> dict:
    return {"echoed": config}


def _failing_handler(step_id: str, config: dict) -> dict:
    raise RuntimeError("step failed")


class TestStepExecutor:
    def test_execute_success(self):
        executor = StepExecutor()
        executor.register_handler("echo", _echo_handler)
        result = executor.execute("s1", "echo", config={"x": 1})
        assert result.success is True
        assert result.output == {"echoed": {"x": 1}}
        assert result.step_id == "s1"

    def test_execute_records_timestamps(self):
        executor = StepExecutor()
        executor.register_handler("echo", _echo_handler)
        result = executor.execute("s1", "echo")
        assert result.started_at <= result.completed_at
        assert result.duration_seconds >= 0.0

    def test_execute_records_inputs(self):
        executor = StepExecutor()
        executor.register_handler("echo", _echo_handler)
        result = executor.execute("s1", "echo", inputs={"prev": "data"})
        assert "prev" in result.inputs

    def test_execute_merges_config_and_inputs(self):
        executor = StepExecutor()
        executor.register_handler("echo", _echo_handler)
        result = executor.execute("s1", "echo", config={"a": 1}, inputs={"b": 2})
        assert result.inputs == {"a": 1, "b": 2}

    def test_execute_failure(self):
        executor = StepExecutor()
        executor.register_handler("fail", _failing_handler)
        result = executor.execute("s1", "fail")
        assert result.success is False
        assert "step failed" in result.error

    def test_execute_unknown_step_type(self):
        executor = StepExecutor()
        result = executor.execute("s1", "unknown")
        assert result.success is False
        assert "No handler registered" in result.error

    def test_execution_log(self):
        executor = StepExecutor()
        executor.register_handler("echo", _echo_handler)
        executor.execute("s1", "echo")
        executor.execute("s2", "echo")
        assert len(executor.execution_log) == 2

    def test_execution_log_returns_copy(self):
        executor = StepExecutor()
        executor.register_handler("echo", _echo_handler)
        executor.execute("s1", "echo")
        log = executor.execution_log
        log.clear()
        assert len(executor.execution_log) == 1

    def test_clear_log(self):
        executor = StepExecutor()
        executor.register_handler("echo", _echo_handler)
        executor.execute("s1", "echo")
        executor.clear_log()
        assert len(executor.execution_log) == 0

    def test_default_config_and_inputs(self):
        executor = StepExecutor()
        executor.register_handler("echo", _echo_handler)
        result = executor.execute("s1", "echo")
        assert result.success is True
        assert result.inputs == {}
