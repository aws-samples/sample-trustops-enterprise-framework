"""
Evaluation commands for TrustOps CLI.

Provides `trustops evaluate baseline` and `trustops evaluate compare`.

Requirements: 10.9, 10.10
"""

import asyncio
import json

import click
import yaml

from cli.auth_handler import AuthError
from cli.commands.auth import _handle_auth_error
from cli.utils.output import (
    print_header,
    print_success,
    print_error,
    print_info,
    print_table,
    print_progress,
)


def _get_evaluation_engine():
    """Create an EvaluationEngine instance."""
    from src.evaluation.evaluation_engine import EvaluationEngine
    return EvaluationEngine()


def _report_to_dict(report) -> dict:
    """Convert BaselineEvaluationReport to serializable dict."""
    metrics = report.aggregate_metrics
    data = {
        "evaluation_id": report.evaluation_id,
        "model_id": report.model_id,
        "dataset_id": report.dataset_id,
        "status": report.status,
        "total_examples": report.total_examples,
        "aggregate_metrics": {
            "total_examples": metrics.total_examples,
            "successful_examples": metrics.successful_examples,
            "failed_examples": metrics.failed_examples,
            "mean_trust_score": metrics.mean_trust_score,
            "median_trust_score": metrics.median_trust_score,
            "trust_score_std": metrics.trust_score_std,
            "mean_hallucination_rate": metrics.mean_hallucination_rate,
            "latency_p50_ms": metrics.latency_p50_ms,
            "latency_p95_ms": metrics.latency_p95_ms,
            "latency_p99_ms": metrics.latency_p99_ms,
            "total_input_tokens": metrics.total_input_tokens,
            "total_output_tokens": metrics.total_output_tokens,
            "total_cost": metrics.total_cost,
            "cost_per_query": metrics.cost_per_query,
        },
        "cost_summary": {
            "total_cost": report.cost_summary.total_cost,
            "inference_cost": report.cost_summary.inference_cost,
            "storage_cost": report.cost_summary.storage_cost,
            "cost_per_example": report.cost_summary.cost_per_example,
            "currency": report.cost_summary.currency,
        },
    }
    if report.created_at:
        data["created_at"] = report.created_at.isoformat()
    if report.completed_at:
        data["completed_at"] = report.completed_at.isoformat()
    if report.per_category_metrics:
        data["per_category_metrics"] = {}
        for cat, m in report.per_category_metrics.items():
            data["per_category_metrics"][cat] = {
                "mean_trust_score": m.mean_trust_score,
                "total_examples": m.total_examples,
            }
    return data


def _comparison_to_dict(report) -> dict:
    """Convert ComparisonReport to serializable dict."""
    data = {
        "comparison_id": report.comparison_id,
        "model_1_id": report.model_1_id,
        "model_2_id": report.model_2_id,
        "dataset_id": report.dataset_id,
        "recommendation": (
            report.recommendation.value
            if hasattr(report.recommendation, "value")
            else str(report.recommendation)
        ),
        "recommendation_justification": (
            report.recommendation_justification
        ),
        "improvement_metrics": {
            "trust_score_delta": (
                report.improvement_metrics.trust_score_delta
            ),
            "trust_score_delta_percent": (
                report.improvement_metrics.trust_score_delta_percent
            ),
            "hallucination_reduction": (
                report.improvement_metrics.hallucination_reduction
            ),
            "hallucination_reduction_percent": (
                report.improvement_metrics.hallucination_reduction_percent
            ),
            "latency_delta_ms": (
                report.improvement_metrics.latency_delta_ms
            ),
            "latency_delta_percent": (
                report.improvement_metrics.latency_delta_percent
            ),
            "cost_delta_per_query": (
                report.improvement_metrics.cost_delta_per_query
            ),
            "cost_delta_percent": (
                report.improvement_metrics.cost_delta_percent
            ),
            "statistical_significance": (
                report.improvement_metrics.statistical_significance
            ),
            "p_value": report.improvement_metrics.p_value,
        },
    }
    if report.created_at:
        data["created_at"] = report.created_at.isoformat()
    return data


@click.group()
def evaluate():
    """Run model evaluations."""
    pass


@evaluate.command("baseline")
@click.option(
    "--model", required=True, help="Model ID to evaluate"
)
@click.option(
    "--dataset", required=True, help="Dataset ID to use"
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help="Path to evaluation config JSON file",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def evaluate_baseline(ctx, model, dataset, config_file, output_format):
    """Run baseline evaluation on a model.

    Evaluates the model against the dataset and calculates
    trust scores, hallucination rates, and cost metrics.

    Requirements: 10.9
    """
    try:
        from src.data_models.evaluation import EvaluationConfig
        from src.data_models.model import InferenceRequest

        engine = _get_evaluation_engine()

        extra = {}
        if config_file:
            with open(config_file, "r") as f:
                extra = json.load(f)

        # Build inference_params with defaults if not provided
        inference_params = extra.pop("inference_params", None)
        if inference_params is None:
            inference_params = InferenceRequest(
                prompt="<dataset>"
            )
        elif isinstance(inference_params, dict):
            inference_params = InferenceRequest(
                prompt=inference_params.get(
                    "prompt", "<dataset>"
                ),
                **{
                    k: v for k, v in inference_params.items()
                    if k != "prompt"
                },
            )

        config = EvaluationConfig(
            model_id=model,
            dataset_id=dataset,
            inference_params=inference_params,
            **extra,
        )

        if output_format == "text":
            print_header("Baseline Evaluation")
            print_info(f"Model:   {model}")
            print_info(f"Dataset: {dataset}")
            print_progress("Running evaluation...")

        def progress_cb(completed, total):
            if output_format == "text":
                pct = (completed / total * 100) if total else 0
                print_progress(
                    f"Progress: {completed}/{total} "
                    f"({pct:.0f}%)"
                )

        report = asyncio.run(
            engine.run_baseline_evaluation(
                config=config,
                progress_callback=progress_cb,
            )
        )

        result = _report_to_dict(report)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(result, default_flow_style=False).strip()
            )
        else:
            m = report.aggregate_metrics
            print_success("Evaluation complete")
            print_info(f"\nEvaluation ID: {report.evaluation_id}")
            print_info(f"Status:        {report.status}")
            print_info(f"\nResults ({m.total_examples} examples):")
            print_info(f"  Mean Trust Score:       {m.mean_trust_score:.3f}")
            print_info(f"  Median Trust Score:     {m.median_trust_score:.3f}")
            print_info(f"  Hallucination Rate:     {m.mean_hallucination_rate:.3f}")
            print_info(f"  Latency P50:            {m.latency_p50_ms:.0f}ms")
            print_info(f"  Latency P95:            {m.latency_p95_ms:.0f}ms")
            print_info(f"  Total Cost:             ${m.total_cost:.4f}")
            print_info(f"  Cost Per Query:         ${m.cost_per_query:.6f}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        raise click.Abort()


@evaluate.command("compare")
@click.option(
    "--model-1", required=True, help="First model ID (baseline)"
)
@click.option(
    "--model-2", required=True, help="Second model ID (fine-tuned)"
)
@click.option(
    "--dataset", required=True, help="Dataset ID to use"
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help="Path to comparison config JSON file",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def evaluate_compare(
    ctx, model_1, model_2, dataset, config_file, output_format
):
    """Run comparative evaluation between two models.

    Compares both models on identical prompts and generates
    improvement metrics and a deployment recommendation.

    Requirements: 10.10
    """
    try:
        from src.data_models.evaluation import (
            ComparisonConfig,
            DeploymentThresholds,
        )
        from src.data_models.model import InferenceRequest

        engine = _get_evaluation_engine()

        extra = {}
        if config_file:
            with open(config_file, "r") as f:
                extra = json.load(f)

        # Build inference_params with defaults if not provided
        inference_params = extra.pop("inference_params", None)
        if inference_params is None:
            inference_params = InferenceRequest(
                prompt="<dataset>"
            )
        elif isinstance(inference_params, dict):
            inference_params = InferenceRequest(
                prompt=inference_params.get(
                    "prompt", "<dataset>"
                ),
                **{
                    k: v for k, v in inference_params.items()
                    if k != "prompt"
                },
            )

        # Build deploy_thresholds with defaults if not provided
        thresholds = extra.pop("deploy_thresholds", None)
        if thresholds is None:
            thresholds = DeploymentThresholds()
        elif isinstance(thresholds, dict):
            thresholds = DeploymentThresholds(**thresholds)

        config = ComparisonConfig(
            model_id_1=model_1,
            model_id_2=model_2,
            dataset_id=dataset,
            inference_params=inference_params,
            deploy_thresholds=thresholds,
            **extra,
        )

        if output_format == "text":
            print_header("Comparative Evaluation")
            print_info(f"Model 1 (baseline):   {model_1}")
            print_info(f"Model 2 (fine-tuned): {model_2}")
            print_info(f"Dataset:              {dataset}")
            print_progress("Running comparison...")

        report = asyncio.run(
            engine.run_comparative_evaluation(config=config)
        )

        result = _comparison_to_dict(report)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(result, default_flow_style=False).strip()
            )
        else:
            imp = report.improvement_metrics
            rec = (
                report.recommendation.value
                if hasattr(report.recommendation, "value")
                else str(report.recommendation)
            )
            print_success("Comparison complete")
            print_info(f"\nComparison ID: {report.comparison_id}")
            print_info(f"\nImprovement Metrics:")
            print_info(
                f"  Trust Score Delta:      "
                f"{imp.trust_score_delta:+.3f} "
                f"({imp.trust_score_delta_percent:+.1f}%)"
            )
            print_info(
                f"  Hallucination Reduction:"
                f" {imp.hallucination_reduction:+.3f} "
                f"({imp.hallucination_reduction_percent:+.1f}%)"
            )
            print_info(
                f"  Latency Delta:          "
                f"{imp.latency_delta_ms:+.0f}ms "
                f"({imp.latency_delta_percent:+.1f}%)"
            )
            print_info(
                f"  Cost Delta:             "
                f"${imp.cost_delta_per_query:+.6f} "
                f"({imp.cost_delta_percent:+.1f}%)"
            )
            print_info(
                f"  Statistical Significance:"
                f" {imp.statistical_significance:.3f}"
            )
            print_info(
                f"\nRecommendation: "
                f"{rec.upper()}"
            )
            print_info(
                f"  {report.recommendation_justification}"
            )

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Comparison failed: {e}")
        raise click.Abort()
