"""
Workflow management commands for TrustOps CLI.

Provides `trustops workflows list`, `trustops workflows run`,
`trustops workflows status`, `trustops workflows resume`,
and `trustops workflows reproduce`.

Requirements: 10.13, 10.14, 10.15, 10.16
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
    print_warning,
)


def _get_orchestrator():
    """Create a WorkflowOrchestrator instance."""
    from src.orchestration.workflow_orchestrator import (
        WorkflowOrchestrator,
    )
    return WorkflowOrchestrator()


def _workflow_to_dict(wf) -> dict:
    """Convert workflow data to serializable dict."""
    data = {
        "workflow_id": wf.workflow_id,
        "status": (
            wf.status.value
            if hasattr(wf.status, "value")
            else str(wf.status)
        ),
    }
    if hasattr(wf, "workflow_type"):
        data["workflow_type"] = wf.workflow_type
    if hasattr(wf, "created_at") and wf.created_at:
        data["created_at"] = wf.created_at.isoformat()
    if hasattr(wf, "completed_at") and wf.completed_at:
        data["completed_at"] = wf.completed_at.isoformat()
    if hasattr(wf, "steps") and wf.steps:
        data["steps"] = [
            {
                "name": s.name if hasattr(s, "name") else str(s),
                "status": (
                    s.status.value
                    if hasattr(s, "status")
                    and hasattr(s.status, "value")
                    else str(getattr(s, "status", "unknown"))
                ),
            }
            for s in wf.steps
        ]
    if hasattr(wf, "total_cost"):
        data["total_cost"] = wf.total_cost
    if hasattr(wf, "estimated_cost"):
        data["estimated_cost"] = wf.estimated_cost
    return data


@click.group()
def workflows():
    """Manage TrustOps workflows."""
    pass


@workflows.command("list")
@click.option(
    "--status",
    "wf_status",
    type=click.Choice([
        "pending", "running", "completed",
        "failed", "paused", "cancelled",
    ]),
    default=None,
    help="Filter by workflow status",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def workflows_list(ctx, wf_status, output_format):
    """List workflows with optional filtering.

    Requirements: 10.13
    """
    try:
        orchestrator = _get_orchestrator()

        wf_list = asyncio.run(
            orchestrator.list_workflows(status=wf_status)
        )

        results = [_workflow_to_dict(w) for w in wf_list]

        if output_format == "json":
            click.echo(json.dumps(results, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(results, default_flow_style=False).strip()
            )
        else:
            if not results:
                print_warning("No workflows found.")
                return

            print_header("Workflows")
            headers = ["ID", "Status", "Type", "Created"]
            rows = [
                [
                    w.get("workflow_id", "")[:20],
                    w.get("status", ""),
                    w.get("workflow_type", "N/A"),
                    w.get("created_at", "N/A")[:19]
                    if w.get("created_at")
                    else "N/A",
                ]
                for w in results
            ]
            print_table(headers, rows)
            print_info(f"\nTotal: {len(results)} workflow(s)")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to list workflows: {e}")
        raise click.Abort()


@workflows.command("run")
@click.argument("template")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help="Path to workflow configuration JSON file",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def workflows_run(ctx, template, config_file, output_format):
    """Run a workflow from a template.

    Available templates: full_pipeline, evaluation_only,
    comparison_only.

    Requirements: 10.14
    """
    try:
        orchestrator = _get_orchestrator()

        wf_config = {}
        if config_file:
            with open(config_file, "r") as f:
                wf_config = json.load(f)

        if output_format == "text":
            print_header("Run Workflow")
            print_info(f"Template: {template}")
            print_progress("Starting workflow...")

        workflow = asyncio.run(
            orchestrator.run_workflow(
                template_name=template,
                config=wf_config,
            )
        )

        result = _workflow_to_dict(workflow)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(result, default_flow_style=False).strip()
            )
        else:
            print_success(
                f"Workflow started: {workflow.workflow_id}"
            )
            status_val = (
                workflow.status.value
                if hasattr(workflow.status, "value")
                else str(workflow.status)
            )
            print_info(f"  Status: {status_val}")
            if hasattr(workflow, "estimated_cost"):
                print_info(
                    f"  Estimated Cost: "
                    f"${workflow.estimated_cost:.2f}"
                )

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to run workflow: {e}")
        raise click.Abort()


@workflows.command("status")
@click.argument("workflow_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def workflows_status(ctx, workflow_id, output_format):
    """Display workflow status, step progress, and cost.

    Requirements: 10.15
    """
    try:
        orchestrator = _get_orchestrator()

        workflow = asyncio.run(
            orchestrator.get_workflow_status(workflow_id)
        )

        if workflow is None:
            print_error(f"Workflow '{workflow_id}' not found")
            raise click.Abort()

        result = _workflow_to_dict(workflow)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(result, default_flow_style=False).strip()
            )
        else:
            status_val = result.get("status", "unknown")
            print_header(f"Workflow: {workflow_id}")
            print_info(f"  Status: {status_val}")
            if result.get("created_at"):
                print_info(f"  Created: {result['created_at']}")
            if result.get("total_cost") is not None:
                print_info(
                    f"  Total Cost: ${result['total_cost']:.4f}"
                )

            steps = result.get("steps", [])
            if steps:
                print_info(f"\nSteps ({len(steps)}):")
                for s in steps:
                    print_info(
                        f"  {s['name']}: {s['status']}"
                    )

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except click.Abort:
        raise
    except Exception as e:
        print_error(f"Failed to get workflow status: {e}")
        raise click.Abort()


@workflows.command("resume")
@click.argument("workflow_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def workflows_resume(ctx, workflow_id, output_format):
    """Resume a paused or failed workflow from last checkpoint.

    Requirements: 10.16
    """
    try:
        orchestrator = _get_orchestrator()

        if output_format == "text":
            print_progress(
                f"Resuming workflow '{workflow_id}'..."
            )

        workflow = asyncio.run(
            orchestrator.resume_workflow(workflow_id)
        )

        result = _workflow_to_dict(workflow)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(result, default_flow_style=False).strip()
            )
        else:
            print_success(
                f"Workflow '{workflow_id}' resumed"
            )
            status_val = result.get("status", "unknown")
            print_info(f"  Status: {status_val}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to resume workflow: {e}")
        raise click.Abort()


@workflows.command("reproduce")
@click.argument("workflow_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def workflows_reproduce(ctx, workflow_id, output_format):
    """Reproduce a completed workflow with identical configuration.

    Requirements: 10.16
    """
    try:
        orchestrator = _get_orchestrator()

        if output_format == "text":
            print_progress(
                f"Reproducing workflow '{workflow_id}'..."
            )

        workflow = asyncio.run(
            orchestrator.reproduce_workflow(workflow_id)
        )

        result = _workflow_to_dict(workflow)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(
                yaml.dump(result, default_flow_style=False).strip()
            )
        else:
            print_success(
                f"Workflow reproduced: {workflow.workflow_id}"
            )
            print_info(
                f"  Original: {workflow_id}"
            )
            status_val = result.get("status", "unknown")
            print_info(f"  Status:   {status_val}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to reproduce workflow: {e}")
        raise click.Abort()
