"""Tests for dashboard demo data generators."""

import pytest

from dashboard.utils.demo_data import (
    generate_demo_models,
    generate_demo_datasets,
    generate_demo_evaluations,
    generate_demo_workflows,
    generate_demo_fine_tuning_jobs,
    generate_demo_trust_scores,
    generate_demo_comparison,
)


class TestGenerateDemoModels:
    def test_returns_list(self):
        models = generate_demo_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_model_has_required_fields(self):
        models = generate_demo_models()
        for m in models:
            assert "id" in m
            assert "provider" in m
            assert "name" in m
            assert "capabilities" in m
            assert "status" in m
            assert "fine_tuning_support" in m
            assert "max_tokens" in m

    def test_model_providers(self):
        models = generate_demo_models()
        providers = {m["provider"] for m in models}
        assert "bedrock" in providers

    def test_has_active_models(self):
        models = generate_demo_models()
        active = [m for m in models if m["status"] == "active"]
        assert len(active) > 0


class TestGenerateDemoDatasets:
    def test_returns_list(self):
        datasets = generate_demo_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_dataset_has_required_fields(self):
        datasets = generate_demo_datasets()
        for d in datasets:
            assert "id" in d
            assert "name" in d
            assert "format" in d
            assert "task_type" in d
            assert "row_count" in d

    def test_dataset_quality_scores(self):
        datasets = generate_demo_datasets()
        for d in datasets:
            quality = d.get("quality", {})
            if quality:
                for key in ("completeness", "diversity", "balance"):
                    assert 0 <= quality[key] <= 1


class TestGenerateDemoEvaluations:
    def test_returns_list(self):
        evals = generate_demo_evaluations()
        assert isinstance(evals, list)
        assert len(evals) > 0

    def test_evaluation_has_required_fields(self):
        evals = generate_demo_evaluations()
        for e in evals:
            assert "id" in e
            assert "model_id" in e
            assert "mean_trust_score" in e
            assert "total_cost" in e

    def test_trust_scores_in_range(self):
        evals = generate_demo_evaluations()
        for e in evals:
            assert 0 <= e["mean_trust_score"] <= 1


class TestGenerateDemoWorkflows:
    def test_returns_list(self):
        workflows = generate_demo_workflows()
        assert isinstance(workflows, list)
        assert len(workflows) > 0

    def test_workflow_has_steps(self):
        workflows = generate_demo_workflows()
        for wf in workflows:
            assert "steps" in wf
            assert len(wf["steps"]) > 0

    def test_workflow_statuses(self):
        workflows = generate_demo_workflows()
        statuses = {wf["status"] for wf in workflows}
        # Should have variety of statuses
        assert len(statuses) > 1


class TestGenerateDemoFineTuningJobs:
    def test_returns_list(self):
        jobs = generate_demo_fine_tuning_jobs()
        assert isinstance(jobs, list)
        assert len(jobs) > 0

    def test_job_has_training_metrics(self):
        jobs = generate_demo_fine_tuning_jobs()
        for job in jobs:
            assert "training_metrics" in job
            metrics = job["training_metrics"]
            for m in metrics:
                assert "epoch" in m
                assert "loss" in m


class TestGenerateDemoTrustScores:
    def test_returns_dict(self):
        scores = generate_demo_trust_scores()
        assert isinstance(scores, dict)

    def test_has_overall_score(self):
        scores = generate_demo_trust_scores()
        assert "overall" in scores
        assert 0 <= scores["overall"] <= 1

    def test_has_five_dimensions(self):
        scores = generate_demo_trust_scores()
        dims = scores["dimensions"]
        assert len(dims) == 5
        expected = {"accuracy", "consistency", "safety", "bias", "context_grounding"}
        assert set(dims.keys()) == expected

    def test_dimension_scores_in_range(self):
        scores = generate_demo_trust_scores()
        for dim_data in scores["dimensions"].values():
            assert 0 <= dim_data["score"] <= 1
            assert 0 <= dim_data["weight"] <= 1


class TestGenerateDemoComparison:
    def test_returns_dict(self):
        comp = generate_demo_comparison()
        assert isinstance(comp, dict)

    def test_has_two_models(self):
        comp = generate_demo_comparison()
        assert "model_1" in comp
        assert "model_2" in comp

    def test_has_improvement_metrics(self):
        comp = generate_demo_comparison()
        assert "improvement" in comp
        imp = comp["improvement"]
        assert "trust_score_delta" in imp
        assert "hallucination_reduction" in imp
        assert "p_value" in imp

    def test_has_recommendation(self):
        comp = generate_demo_comparison()
        assert "recommendation" in comp
        assert comp["recommendation"] in ("deploy", "iterate", "reject")

    def test_has_justification(self):
        comp = generate_demo_comparison()
        assert "justification" in comp
        assert len(comp["justification"]) > 0
