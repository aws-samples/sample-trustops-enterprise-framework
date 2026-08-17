"""Tests for workflow cost/time estimator module."""

import pytest

from src.data_models.workflow import (
    StepType,
    StepStatus,
    WorkflowDefinition,
    WorkflowStep,
)
from src.orchestration.workflow_estimator import (
    WorkflowEstimate,
    WorkflowEstimator,
)


def _make_step(step_id, step_type, config=None):
    return WorkflowStep(
        step_id=step_id,
        step_type=step_type,
        name=f"Step {step_id}",
        config=config or {},
    )


def _make_definition(steps):
    return WorkflowDefinition(name="test", steps=steps)


class TestEstimateStep:
    def test_default_evaluation_estimate(self):
        e = WorkflowEstimator()
        est = e.estimate_step("s1", StepType.BASELINE_EVALUATION)
        assert est.estimated_duration_seconds == 300.0
        assert est.estimated_cost == 0.50
        assert est.confidence == "high"

    def test_default_fine_tuning_estimate(self):
        e = WorkflowEstimator()
        est = e.estimate_step("s1", StepType.FINE_TUNING)
        assert est.estimated_duration_seconds == 3600.0
        assert est.estimated_cost == 10.00

    def test_config_overrides_defaults(self):
        e = WorkflowEstimator()
        est = e.estimate_step(
            "s1",
            StepType.BASELINE_EVALUATION,
            config={
                "estimated_duration_seconds": 600,
                "estimated_cost": 1.0,
            },
        )
        assert est.estimated_duration_seconds == 600.0
        assert est.estimated_cost == 1.0

    def test_constructor_overrides(self):
        e = WorkflowEstimator(
            duration_overrides={StepType.COMPARISON: 999.0},
            cost_overrides={StepType.COMPARISON: 5.0},
        )
        est = e.estimate_step("s1", StepType.COMPARISON)
        assert est.estimated_duration_seconds == 999.0
        assert est.estimated_cost == 5.0


class TestEstimateWorkflow:
    def test_single_step(self):
        e = WorkflowEstimator()
        defn = _make_definition([
            _make_step("s1", StepType.BASELINE_EVALUATION),
        ])
        est = e.estimate(defn)
        assert est.estimated_duration_seconds == 300.0
        assert est.estimated_cost == 0.50
        assert "s1" in est.cost_breakdown

    def test_multi_step_sums(self):
        e = WorkflowEstimator()
        defn = _make_definition([
            _make_step("s1", StepType.DATASET_PREPARATION),
            _make_step("s2", StepType.BASELINE_EVALUATION),
            _make_step("s3", StepType.FINE_TUNING),
        ])
        est = e.estimate(defn)
        expected_duration = 60.0 + 300.0 + 3600.0
        expected_cost = 0.01 + 0.50 + 10.00
        assert est.estimated_duration_seconds == pytest.approx(expected_duration)
        assert est.estimated_cost == pytest.approx(expected_cost)
        assert len(est.step_estimates) == 3

    def test_full_pipeline_estimate(self):
        e = WorkflowEstimator()
        defn = _make_definition([
            _make_step("s1", StepType.DATASET_PREPARATION),
            _make_step("s2", StepType.BASELINE_EVALUATION),
            _make_step("s3", StepType.FINE_TUNING),
            _make_step("s4", StepType.POST_TUNING_EVALUATION),
            _make_step("s5", StepType.COMPARISON),
        ])
        est = e.estimate(defn)
        assert est.estimated_duration_seconds > 0
        assert est.estimated_cost > 0
        assert len(est.cost_breakdown) == 5

    def test_confidence_all_high(self):
        e = WorkflowEstimator()
        defn = _make_definition([
            _make_step("s1", StepType.BASELINE_EVALUATION),
        ])
        est = e.estimate(defn)
        assert est.confidence == "high"

    def test_approval_gate_zero_cost(self):
        e = WorkflowEstimator()
        est = e.estimate_step("s1", StepType.APPROVAL_GATE)
        assert est.estimated_cost == 0.0
        assert est.estimated_duration_seconds == 0.0
