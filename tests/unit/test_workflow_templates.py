"""Tests for pre-built workflow templates."""

import pytest

from src.data_models.workflow import StepType
from src.orchestration.templates.workflow_templates import (
    create_comparison_only_template,
    create_evaluation_only_template,
    create_full_pipeline_template,
    get_all_templates,
    get_template_by_name,
)


class TestFullPipelineTemplate:
    def test_has_five_steps(self):
        t = create_full_pipeline_template()
        assert len(t.steps) == 5

    def test_step_types(self):
        t = create_full_pipeline_template()
        types = [s.step_type for s in t.steps]
        assert StepType.DATASET_PREPARATION in types
        assert StepType.BASELINE_EVALUATION in types
        assert StepType.FINE_TUNING in types
        assert StepType.POST_TUNING_EVALUATION in types
        assert StepType.COMPARISON in types

    def test_dependencies(self):
        t = create_full_pipeline_template()
        step_map = {s.step_id: s for s in t.steps}
        assert step_map["dataset-prep"].depends_on == []
        assert "dataset-prep" in step_map["baseline-eval"].depends_on
        assert "dataset-prep" in step_map["fine-tuning"].depends_on
        assert "fine-tuning" in step_map["post-eval"].depends_on
        assert "baseline-eval" in step_map["comparison"].depends_on
        assert "post-eval" in step_map["comparison"].depends_on

    def test_custom_config(self):
        t = create_full_pipeline_template(
            dataset_config={"dataset_id": "ds-1"},
            baseline_config={"model_id": "m-1"},
        )
        step_map = {s.step_id: s for s in t.steps}
        assert step_map["dataset-prep"].config == {"dataset_id": "ds-1"}
        assert step_map["baseline-eval"].config == {"model_id": "m-1"}

    def test_name_and_description(self):
        t = create_full_pipeline_template()
        assert t.name == "Full Pipeline"
        assert t.description is not None


class TestEvaluationOnlyTemplate:
    def test_has_two_steps(self):
        t = create_evaluation_only_template()
        assert len(t.steps) == 2

    def test_step_types(self):
        t = create_evaluation_only_template()
        types = [s.step_type for s in t.steps]
        assert StepType.DATASET_PREPARATION in types
        assert StepType.BASELINE_EVALUATION in types

    def test_dependency(self):
        t = create_evaluation_only_template()
        step_map = {s.step_id: s for s in t.steps}
        assert "dataset-prep" in step_map["baseline-eval"].depends_on


class TestComparisonOnlyTemplate:
    def test_has_one_step(self):
        t = create_comparison_only_template()
        assert len(t.steps) == 1

    def test_step_type(self):
        t = create_comparison_only_template()
        assert t.steps[0].step_type == StepType.COMPARISON

    def test_no_dependencies(self):
        t = create_comparison_only_template()
        assert t.steps[0].depends_on == []


class TestTemplateRegistry:
    def test_get_all_templates(self):
        templates = get_all_templates()
        assert len(templates) == 3
        names = {t.name for t in templates}
        assert "Full Pipeline" in names
        assert "Evaluation Only" in names
        assert "Comparison Only" in names

    def test_get_by_name(self):
        t = get_template_by_name("full_pipeline")
        assert t is not None
        assert t.name == "Full Pipeline"

    def test_get_by_name_not_found(self):
        assert get_template_by_name("nonexistent") is None
