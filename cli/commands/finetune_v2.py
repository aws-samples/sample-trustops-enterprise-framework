"""
Fine-tuning commands for TrustOps CLI (subcommands).

Provides `trustops finetune start`, `trustops finetune status`,
and `trustops finetune stop`, wired to the synchronous
FineTuningPipeline API.

Requirements: 10.11, 10.12
"""

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
    print_warning,
)


def _get_pipeline():
    """Create a FineTuningPipeline instance."""
    from src.fine_tuning.fine_tuning_pipeline import FineTuningPipeline
    return FineTuningPipeline()


def _job_to_dict(job) -> dict:
    """Convert FineTuningJob to serializable dict."""
    data = {
        "job_id": job.job_id,
        "status": (
            job.status.value
            if hasattr(job.status, "value")
            else str(job.status)
        ),
        "base_model_id": job.base_model_id,
    }
    if hasattr(job, "finetuned_model_id") and job.finetuned_model_id:
        data["finetuned_model_id"] = job.finetuned_model_id
    if hasattr(job, "estimated_cost"):
        data["estimated_cost"] = job.estimated_cost
    if hasattr(job, "actual_cost") and job.actual_cost is not None:
        data["actual_cost"] = job.actual_cost
    if hasattr(job, "created_at") and job.created_at:
        data["created_at"] = job.created_at.isoformat()
    if hasattr(job, "completed_at") and job.completed_at:
        data["completed_at"] = job.completed_at.isoformat()
    if hasattr(job, "error_message") and job.error_message:
        data["error_message"] = job.error_message
    if hasattr(job, "training_metrics") and job.training_metrics:
        data["training_metrics"] = [
            {
                "epoch": m.epoch,
                "step": m.step,
                "training_loss": m.training_loss,
                "validation_loss": m.validation_loss,
                "learning_rate": m.learning_rate,
            }
            for m in job.training_metrics
        ]
    if hasattr(job, "hyperparameters") and job.hyperparameters:
        hp = job.hyperparameters
        data["hyperparameters"] = {
            "learning_rate": hp.learning_rate,
            "epochs": hp.epochs,
            "batch_size": hp.batch_size,
        }
    return data


def _cost_estimate_to_dict(estimate) -> dict:
    """Convert CostEstimate to serializable dict."""
    return {
        "estimated_training_cost": estimate.estimated_training_cost,
        "estimated_duration_hours": estimate.estimated_duration_hours,
        "cost_breakdown": estimate.cost_breakdown,
        "currency": estimate.currency,
        "confidence": estimate.confidence,
    }


@click.group("finetune")
def finetune_v2():
    """Manage fine-tuning jobs."""
    pass


@finetune_v2.command("start")
@click.option(
    "--model", required=True, help="Base model ID to fine-tune"
)
@click.option(
    "--dataset", required=True, help="Training dataset ID"
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help="Path to fine-tuning config JSON file",
)
@click.option(
    "--num-examples",
    type=int,
    default=None,
    help="Number of training examples (used for cost estimation)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def finetune_start(ctx, model, dataset, config_file, num_examples,
                   output_format):
    """Start a fine-tuning job.

    Validates the model and dataset, displays a cost estimate,
    and registers the fine-tuning job.

    Requirements: 10.11
    """
    try:
        from src.data_models.fine_tuning import (
            FineTuningConfig,
            HyperparameterConfig,
        )

        pipeline = _get_pipeline()

        extra = {}
        if config_file:
            with open(config_file, "r") as f:
                extra = json.load(f)

        # Build hyperparameters with defaults if not provided
        hp = extra.pop("hyperparameters", None)
        if hp is None:
            hp = HyperparameterConfig()
        elif isinstance(hp, dict):
            hp = HyperparameterConfig(**hp)

        est_examples = num_examples or extra.pop("num_examples", 100)

        config = FineTuningConfig(
            base_model_id=model,
            training_data_id=dataset,
            hyperparameters=hp,
            job_name=extra.pop(
                "job_name", f"ft-{model[:8]}"
            ),
            output_model_name=extra.pop(
                "output_model_name", f"{model}-finetuned"
            ),
            **{
                k: v for k, v in extra.items()
                if k not in (
                    "base_model_id", "training_data_id",
                )
            },
        )

        # Get cost estimate first (synchronous API)
        estimate = pipeline.estimate_cost(
            config, num_examples=est_examples
        )

        if output_format == "text":
            print_header("Fine-Tuning")
            print_info(f"Model:   {model}")
            print_info(f"Dataset: {dataset}")
            print_info(f"\nCost Estimate:")
            print_info(
                f"  Training Cost: "
                f"${estimate.estimated_training_cost:.2f}"
            )
            print_info(
                f"  Duration:      "
                f"~{estimate.estimated_duration_hours:.1f} hours"
            )
            print_info(
                f"  Confidence:    {estimate.confidence}"
            )
            print_progress("Registering fine-tuning job...")

        job = pipeline.start(config)

        result = _job_to_dict(job)
        result["cost_estimate"] = _cost_estimate_to_dict(estimate)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(
                    result, default_flow_style=False
                ).strip()
            )
        else:
            print_success(
                f"Fine-tuning job started: {job.job_id}"
            )
            status_val = (
                job.status.value
                if hasattr(job.status, "value")
                else str(job.status)
            )
            print_info(f"  Status: {status_val}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Fine-tuning failed: {e}")
        raise click.Abort()


@finetune_v2.command("status")
@click.argument("job_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def finetune_status(ctx, job_id, output_format):
    """Display fine-tuning job status, progress, and metrics.

    Requirements: 10.12
    """
    try:
        pipeline = _get_pipeline()

        job = pipeline.get_status(job_id)

        if job is None:
            print_error(f"Job '{job_id}' not found")
            raise click.Abort()

        result = _job_to_dict(job)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(
                    result, default_flow_style=False
                ).strip()
            )
        else:
            status_val = result.get("status", "unknown")
            print_header(f"Fine-Tuning Job: {job_id}")
            print_info(f"  Status:     {status_val}")
            print_info(
                f"  Base Model: {result['base_model_id']}"
            )
            if result.get("finetuned_model_id"):
                print_info(
                    f"  Fine-Tuned: "
                    f"{result['finetuned_model_id']}"
                )
            if result.get("estimated_cost") is not None:
                print_info(
                    f"  Est. Cost:  "
                    f"${result['estimated_cost']:.2f}"
                )
            if result.get("actual_cost") is not None:
                print_info(
                    f"  Actual Cost:"
                    f" ${result['actual_cost']:.2f}"
                )

            metrics = result.get("training_metrics", [])
            if metrics:
                print_info(f"\nTraining Metrics:")
                latest = metrics[-1]
                print_info(
                    f"  Epoch:           {latest['epoch']}"
                )
                print_info(
                    f"  Training Loss:   "
                    f"{latest['training_loss']:.4f}"
                )
                if latest.get("validation_loss") is not None:
                    print_info(
                        f"  Validation Loss: "
                        f"{latest['validation_loss']:.4f}"
                    )

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except click.Abort:
        raise
    except Exception as e:
        print_error(f"Failed to get job status: {e}")
        raise click.Abort()


@finetune_v2.command("stop")
@click.argument("job_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def finetune_stop(ctx, job_id, output_format):
    """Stop a running fine-tuning job.

    Requirements: 10.12
    """
    try:
        pipeline = _get_pipeline()

        job = pipeline.stop(job_id)

        if job is None:
            print_error(f"Job '{job_id}' not found")
            raise click.Abort()

        result = _job_to_dict(job)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(
                    result, default_flow_style=False
                ).strip()
            )
        else:
            print_success(f"Job '{job_id}' stopped")
            status_val = result.get("status", "unknown")
            print_info(f"  Status: {status_val}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except click.Abort:
        raise
    except Exception as e:
        print_error(f"Failed to stop job: {e}")
        raise click.Abort()
