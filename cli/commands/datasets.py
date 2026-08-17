"""
Dataset management commands for TrustOps CLI.

Provides `trustops datasets upload`, `trustops datasets list`,
`trustops datasets analyze`, and `trustops datasets convert` commands.

Requirements: 10.5, 10.6, 10.7, 10.8
"""

import asyncio
import json
from typing import Optional

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
    print_warning,
    print_progress,
)


def _get_dataset_manager():
    """Create a DatasetManager instance."""
    from src.datasets.dataset_manager import DatasetManager
    return DatasetManager()


def _metadata_to_dict(meta) -> dict:
    """Convert DatasetMetadata to a serializable dict."""
    data = {
        "id": meta.id,
        "name": meta.name,
        "format": meta.format.value if hasattr(meta.format, "value") else str(meta.format),
        "task_type": meta.task_type.value if hasattr(meta.task_type, "value") else str(meta.task_type),
        "version": meta.version,
        "s3_uri": meta.s3_uri,
        "row_count": meta.row_count,
        "created_at": meta.created_at.isoformat() if meta.created_at else None,
        "checksum": meta.checksum,
    }
    if meta.description:
        data["description"] = meta.description
    if meta.token_stats:
        data["token_stats"] = {
            "total_tokens": meta.token_stats.total_tokens,
            "min_tokens": meta.token_stats.min_tokens,
            "max_tokens": meta.token_stats.max_tokens,
            "avg_tokens": meta.token_stats.avg_tokens,
            "p95_tokens": meta.token_stats.p95_tokens,
        }
    return data


def _quality_report_to_dict(report) -> dict:
    """Convert DatasetQualityReport to a serializable dict."""
    data = {
        "completeness_score": report.completeness_score,
        "diversity_score": report.diversity_score,
        "balance_score": report.balance_score,
        "token_stats": {
            "total_tokens": report.token_stats.total_tokens,
            "min_tokens": report.token_stats.min_tokens,
            "max_tokens": report.token_stats.max_tokens,
            "avg_tokens": report.token_stats.avg_tokens,
            "p95_tokens": report.token_stats.p95_tokens,
        },
        "issues": [
            {
                "severity": issue.severity,
                "category": issue.category,
                "message": issue.message,
            }
            for issue in report.issues
        ],
        "recommendations": report.recommendations,
    }
    return data


@click.group()
def datasets():
    """Manage datasets for evaluation and fine-tuning."""
    pass


@datasets.command("upload")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--name", required=True, help="Name for the dataset")
@click.option(
    "--task-type",
    type=click.Choice(["qa", "summarization", "classification", "text_generation", "chat", "custom"]),
    default=None,
    help="Task type (auto-detected if not specified)",
)
@click.option("--description", default=None, help="Dataset description")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def datasets_upload(ctx, file_path, name, task_type, description, output_format):
    """Upload a dataset with auto-detection of format and task type.

    Requirements: 10.5
    """
    try:
        from src.data_models.dataset import DatasetTaskType

        manager = _get_dataset_manager()

        task_type_enum = DatasetTaskType(task_type) if task_type else None

        if output_format == "text":
            print_progress(f"Uploading dataset '{name}' from {file_path}...")

        metadata = asyncio.run(
            manager.upload_dataset(
                file_path=file_path,
                name=name,
                task_type=task_type_enum,
                description=description,
            )
        )

        result = _metadata_to_dict(metadata)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(yaml.dump(result, default_flow_style=False).strip())
        else:
            print_success(f"Dataset '{name}' uploaded successfully")
            print_info(f"  ID:        {metadata.id}")
            print_info(f"  Format:    {result['format']}")
            print_info(f"  Task Type: {result['task_type']}")
            print_info(f"  Rows:      {metadata.row_count}")
            print_info(f"  S3 URI:    {metadata.s3_uri}")
            print_info(f"  Checksum:  {metadata.checksum}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to upload dataset: {e}")
        raise click.Abort()


@datasets.command("list")
@click.option(
    "--task-type",
    type=click.Choice(["qa", "summarization", "classification", "text_generation", "chat", "custom"]),
    default=None,
    help="Filter by task type",
)
@click.option(
    "--dataset-format",
    "dataset_fmt",
    type=click.Choice(["jsonl", "csv", "parquet", "huggingface"]),
    default=None,
    help="Filter by dataset format",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def datasets_list(ctx, task_type, dataset_fmt, output_format):
    """List datasets with optional filtering.

    Requirements: 10.6
    """
    try:
        from src.data_models.dataset import DatasetFormat, DatasetTaskType

        manager = _get_dataset_manager()

        task_type_enum = DatasetTaskType(task_type) if task_type else None
        format_enum = DatasetFormat(dataset_fmt) if dataset_fmt else None

        dataset_list = asyncio.run(
            manager.list_datasets(
                task_type=task_type_enum,
                format=format_enum,
            )
        )

        results = [_metadata_to_dict(d) for d in dataset_list]

        if output_format == "json":
            click.echo(json.dumps(results, indent=2))
        elif output_format == "yaml":
            click.echo(yaml.dump(results, default_flow_style=False).strip())
        else:
            if not results:
                print_warning("No datasets found matching the specified filters.")
                return

            print_header("Datasets")
            headers = ["ID", "Name", "Format", "Task Type", "Rows"]
            rows = [
                [
                    d["id"][:16] + "..." if len(d["id"]) > 19 else d["id"],
                    d["name"],
                    d["format"],
                    d["task_type"],
                    str(d["row_count"]),
                ]
                for d in results
            ]
            print_table(headers, rows)
            print_info(f"\nTotal: {len(results)} dataset(s)")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to list datasets: {e}")
        raise click.Abort()


@datasets.command("analyze")
@click.argument("dataset_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def datasets_analyze(ctx, dataset_id, output_format):
    """Run quality analysis on a dataset.

    Displays quality scores, token statistics, issues, and recommendations.

    Requirements: 10.7
    """
    try:
        manager = _get_dataset_manager()

        if output_format == "text":
            print_progress(f"Analyzing dataset '{dataset_id}'...")

        report = asyncio.run(
            manager.analyze_quality(dataset_id)
        )

        result = _quality_report_to_dict(report)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(yaml.dump(result, default_flow_style=False).strip())
        else:
            print_header("Dataset Quality Report")

            # Scores
            print_info("Quality Scores:")
            print_info(f"  Completeness: {report.completeness_score:.2f}")
            print_info(f"  Diversity:    {report.diversity_score:.2f}")
            print_info(f"  Balance:      {report.balance_score:.2f}")

            overall = (
                report.completeness_score
                + report.diversity_score
                + report.balance_score
            ) / 3
            print_info(f"  Overall:      {overall:.2f}")

            # Token stats
            print_info("\nToken Statistics:")
            ts = report.token_stats
            print_info(f"  Total:   {ts.total_tokens}")
            print_info(f"  Min:     {ts.min_tokens}")
            print_info(f"  Max:     {ts.max_tokens}")
            print_info(f"  Average: {ts.avg_tokens:.1f}")
            print_info(f"  P95:     {ts.p95_tokens}")

            # Issues
            if report.issues:
                print_info(f"\nIssues ({len(report.issues)}):")
                for issue in report.issues:
                    severity_color = {
                        "error": "red",
                        "warning": "yellow",
                        "info": "blue",
                    }.get(issue.severity, "white")
                    click.echo(
                        f"  [{click.style(issue.severity.upper(), fg=severity_color)}] "
                        f"{issue.category}: {issue.message}"
                    )

            # Recommendations
            if report.recommendations:
                print_info(f"\nRecommendations ({len(report.recommendations)}):")
                for i, rec in enumerate(report.recommendations, 1):
                    print_info(f"  {i}. {rec}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to analyze dataset: {e}")
        raise click.Abort()


@datasets.command("convert")
@click.argument("dataset_id")
@click.option(
    "--target-format",
    "target_format",
    required=True,
    type=click.Choice(["jsonl", "csv", "parquet"]),
    help="Target format for conversion",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def datasets_convert(ctx, dataset_id, target_format, output_format):
    """Convert a dataset to a different format.

    Requirements: 10.8
    """
    try:
        from src.data_models.dataset import DatasetFormat

        manager = _get_dataset_manager()
        target_fmt = DatasetFormat(target_format)

        if output_format == "text":
            print_progress(f"Converting dataset '{dataset_id}' to {target_format}...")

        metadata = asyncio.run(
            manager.convert_format(dataset_id, target_fmt)
        )

        result = _metadata_to_dict(metadata)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(yaml.dump(result, default_flow_style=False).strip())
        else:
            print_success(f"Dataset converted to {target_format}")
            print_info(f"  New ID:    {metadata.id}")
            print_info(f"  Format:    {result['format']}")
            print_info(f"  Rows:      {metadata.row_count}")
            print_info(f"  S3 URI:    {metadata.s3_uri}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to convert dataset: {e}")
        raise click.Abort()
