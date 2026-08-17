"""
Results management commands for TrustOps CLI.

Provides `trustops results get`, `trustops results export`,
and `trustops results compare`.

Requirements: 10.17, 10.18
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
    print_progress,
)


def _get_results_store():
    """Create a ResultsStore instance."""
    from src.storage.results_store import ResultsStore
    return ResultsStore()


def _result_to_dict(result) -> dict:
    """Convert evaluation result to serializable dict."""
    if isinstance(result, dict):
        return result
    data = {}
    for field in [
        "evaluation_id", "model_id", "dataset_id",
        "status", "total_examples",
    ]:
        if hasattr(result, field):
            data[field] = getattr(result, field)
    if hasattr(result, "aggregate_metrics"):
        m = result.aggregate_metrics
        data["aggregate_metrics"] = {
            "mean_trust_score": m.mean_trust_score,
            "median_trust_score": m.median_trust_score,
            "mean_hallucination_rate": m.mean_hallucination_rate,
            "total_cost": m.total_cost,
            "cost_per_query": m.cost_per_query,
            "latency_p50_ms": m.latency_p50_ms,
            "latency_p95_ms": m.latency_p95_ms,
        }
    if hasattr(result, "created_at") and result.created_at:
        data["created_at"] = result.created_at.isoformat()
    return data


@click.group()
def results():
    """Manage evaluation results."""
    pass


@results.command("get")
@click.argument("evaluation_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def results_get(ctx, evaluation_id, output_format):
    """Retrieve evaluation results by ID.

    Requirements: 10.17
    """
    try:
        store = _get_results_store()

        result = asyncio.run(
            store.get(evaluation_id)
        )

        if result is None:
            print_error(
                f"Results for '{evaluation_id}' not found"
            )
            raise click.Abort()

        data = _result_to_dict(result)

        if output_format == "json":
            click.echo(json.dumps(data, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(data, default_flow_style=False).strip()
            )
        else:
            print_header(f"Evaluation: {evaluation_id}")
            for key, value in data.items():
                if isinstance(value, dict):
                    print_info(f"  {key}:")
                    for k, v in value.items():
                        print_info(f"    {k}: {v}")
                else:
                    print_info(f"  {key}: {value}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except click.Abort:
        raise
    except Exception as e:
        print_error(f"Failed to get results: {e}")
        raise click.Abort()


@results.command("export")
@click.argument("evaluation_id")
@click.option(
    "--export-format",
    "export_fmt",
    type=click.Choice(["json", "csv", "pdf"]),
    default="json",
    help="Export file format",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Output file path",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def results_export(
    ctx, evaluation_id, export_fmt, output_path, output_format
):
    """Export evaluation results in JSON, CSV, or PDF format.

    Requirements: 10.18
    """
    try:
        store = _get_results_store()

        if output_format == "text":
            print_progress(
                f"Exporting '{evaluation_id}' as {export_fmt}..."
            )

        exported = asyncio.run(
            store.export(
                evaluation_id,
                format=export_fmt,
                output_path=output_path,
            )
        )

        result = {
            "evaluation_id": evaluation_id,
            "format": export_fmt,
            "output_path": (
                exported
                if isinstance(exported, str)
                else output_path or f"{evaluation_id}.{export_fmt}"
            ),
            "status": "exported",
        }

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(result, default_flow_style=False).strip()
            )
        else:
            print_success(
                f"Results exported to {result['output_path']}"
            )

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to export results: {e}")
        raise click.Abort()


@results.command("compare")
@click.argument("evaluation_id_1")
@click.argument("evaluation_id_2")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def results_compare(
    ctx, evaluation_id_1, evaluation_id_2, output_format
):
    """Compare two evaluation results side-by-side.

    Requirements: 10.17
    """
    try:
        store = _get_results_store()

        comparison = asyncio.run(
            store.compare(evaluation_id_1, evaluation_id_2)
        )

        if isinstance(comparison, dict):
            data = comparison
        else:
            data = {
                "evaluation_1": evaluation_id_1,
                "evaluation_2": evaluation_id_2,
                "comparison": (
                    comparison
                    if isinstance(comparison, dict)
                    else str(comparison)
                ),
            }

        if output_format == "json":
            click.echo(json.dumps(data, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(data, default_flow_style=False).strip()
            )
        else:
            print_header("Results Comparison")
            print_info(f"  Evaluation 1: {evaluation_id_1}")
            print_info(f"  Evaluation 2: {evaluation_id_2}")
            print_info("")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        print_info(f"  {key}:")
                        for k, v in value.items():
                            print_info(f"    {k}: {v}")
                    else:
                        print_info(f"  {key}: {value}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to compare results: {e}")
        raise click.Abort()
