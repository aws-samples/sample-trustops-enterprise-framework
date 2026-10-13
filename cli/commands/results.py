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
    """Create a ResultsStore instance with proper AWS clients."""
    from src.storage.results_store import ResultsStore
    from src.storage.path_convention import StoragePathConfig
    import boto3
    from config.aws_config import config

    session_kwargs = config.get_boto3_session_kwargs()
    s3 = boto3.client("s3", **session_kwargs)
    dynamodb = boto3.client("dynamodb", **session_kwargs)
    path_config = StoragePathConfig(
        bucket=config.results_bucket,
    )
    return ResultsStore(
        s3_client=s3,
        dynamodb_client=dynamodb,
        path_config=path_config,
    )


def _load_result_from_workflow(workflow_id: str):
    """Try to load evaluation results via the workflow's S3 results path.

    The baseline and foundation-compare commands store their results JSON
    in S3 and record the path in the workflow manifest (DynamoDB). This
    function looks up the workflow, reads the S3 object, and returns a
    dict that _result_to_dict can handle.
    """
    try:
        import boto3
        import json as _json
        from config.aws_config import config

        session_kwargs = config.get_boto3_session_kwargs()
        wm_ddb = boto3.client("dynamodb", **session_kwargs)

        # Look up the workflow manifest
        resp = wm_ddb.get_item(
            TableName=config.workflows_table,
            Key={"workflow_id": {"S": workflow_id}},
        )
        item = resp.get("Item")
        if not item:
            return None

        # Extract the results S3 URI from the manifest
        results_uri = item.get("results_s3_uri", {}).get("S", "")
        if not results_uri:
            config_str = item.get("configuration", {}).get("S", "")
            if config_str:
                cfg = _json.loads(config_str)
                results_uri = cfg.get("results_s3_uri", "")

        if not results_uri or not results_uri.startswith("s3://"):
            return None

        parts = results_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]

        s3 = boto3.client("s3", **session_kwargs)
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = _json.loads(obj["Body"].read().decode("utf-8"))
        return body
    except Exception:
        return None


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
        try:
            store = _get_results_store()
            result = store.get_result(evaluation_id)
        except Exception:
            result = None

        if result is None:
            # Fall back: the evaluation_id might be a workflow_id whose results
            # are stored in S3 but not indexed in the results DynamoDB table.
            result = _load_result_from_workflow(evaluation_id)

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

        exported = store.export_result(
                evaluation_id,
                format=export_fmt,
                output_path=output_path,
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

        comparison = store.compare_results(evaluation_id_1, evaluation_id_2)

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
