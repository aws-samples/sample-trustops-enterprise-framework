"""Workflow status command."""
import click
from datetime import datetime
from cli.utils.output import (
    print_header, print_success, print_error, print_info, print_warning
)
from cli.config import update_env_from_config


@click.command()
@click.argument('workflow-id', required=False)
@click.option(
    '--list',
    'list_all',
    is_flag=True,
    help='List all workflows'
)
@click.option(
    '--type',
    'workflow_type',
    type=click.Choice(['baseline', 'comparative', 'fine-tuning']),
    help='Filter by workflow type'
)
@click.option(
    '--status-filter',
    type=click.Choice(['created', 'running', 'completed', 'failed']),
    help='Filter by workflow status'
)
@click.option(
    '--limit',
    type=int,
    default=10,
    help='Maximum number of workflows to display'
)
@click.pass_context
def status(ctx, workflow_id, list_all, workflow_type, status_filter, limit):
    """
    Check workflow status or list recent workflows.
    
    Examples:
    
        # Check specific workflow status
        trustops status baseline-20240115-abc123
        
        # List all workflows
        trustops status --list
        
        # List completed baseline evaluations
        trustops status --list --type baseline --status-filter completed
    """
    # Update environment from config
    config = ctx.obj.get('config', {})
    update_env_from_config(config)
    
    try:
        from src.orchestration.workflow_manager import WorkflowManager
        
        workflow_manager = WorkflowManager()
        
        if list_all:
            # List workflows with filters
            print_header("Workflow History")
            
            filters = {}
            if workflow_type:
                filters['workflow_type'] = workflow_type
            if status_filter:
                filters['status'] = status_filter
            
            workflows = workflow_manager.get_workflow_history(filters)
            
            if not workflows:
                print_info("No workflows found.")
                return
            
            # Display workflows
            workflows = workflows[:limit]
            
            for i, wf in enumerate(workflows, 1):
                status_symbol = _get_status_symbol(wf.get('status', 'unknown'))
                
                print_info(f"\n{i}. {wf.get('workflow_id', 'N/A')}")
                print_info(f"   Type: {wf.get('workflow_type', 'N/A')}")
                print_info(f"   Status: {status_symbol} {wf.get('status', 'N/A')}")
                print_info(f"   Created: {_format_timestamp(wf.get('created_at'))}")
                
                if wf.get('completed_at'):
                    print_info(
                        f"   Completed: {_format_timestamp(wf.get('completed_at'))}"
                    )
            
            if len(workflows) == limit:
                print_info(f"\nShowing {limit} most recent workflows.")
                print_info("Use --limit to show more.")
        
        elif workflow_id:
            # Show specific workflow details
            print_header(f"Workflow Status: {workflow_id}")
            
            manifest = workflow_manager.get_workflow(workflow_id)
            
            if not manifest:
                print_error(f"Workflow {workflow_id} not found.")
                raise click.Abort()
            
            status_symbol = _get_status_symbol(manifest.status)
            
            print_info(f"Workflow ID: {manifest.workflow_id}")
            print_info(f"Type: {manifest.workflow_type}")
            print_info(f"Status: {status_symbol} {manifest.status}")
            print_info(f"Created: {_format_timestamp(manifest.created_at.isoformat())}")
            print_info(f"Created By: {manifest.created_by}")
            
            if manifest.completed_at:
                print_info(
                    f"Completed: "
                    f"{_format_timestamp(manifest.completed_at.isoformat())}"
                )
                duration = (manifest.completed_at - manifest.created_at).total_seconds()
                print_info(f"Duration: {_format_duration(duration)}")
            
            # Configuration
            print_info("\nConfiguration:")
            for key, value in manifest.configuration.items():
                print_info(f"  {key}: {value}")
            
            # Model IDs
            if manifest.model_ids:
                print_info("\nModels:")
                for model_id in manifest.model_ids:
                    print_info(f"  - {model_id}")
            
            # Dataset
            if manifest.dataset_s3_uri:
                print_info(f"\nDataset: {manifest.dataset_s3_uri}")
            
            # Results
            if manifest.results_s3_uri:
                print_info(f"Results: {manifest.results_s3_uri}")
            
            # Recent events
            if manifest.events:
                print_info("\nRecent Events:")
                recent_events = manifest.events[-5:]  # Last 5 events
                for event in recent_events:
                    timestamp = event.get('timestamp', 'N/A')
                    event_type = event.get('event_type', 'N/A')
                    print_info(f"  [{_format_timestamp(timestamp)}] {event_type}")
                    
                    # Show details for status updates
                    if event_type == 'status_update':
                        details = event.get('details', {})
                        if 'stage' in details:
                            print_info(f"    Stage: {details['stage']}")
        
        else:
            print_error("Please provide a workflow ID or use --list")
            print_info("Usage: trustops status WORKFLOW_ID")
            print_info("       trustops status --list")
            raise click.Abort()
    
    except Exception as e:
        print_error(f"Failed to retrieve status: {e}")
        raise click.Abort()


def _get_status_symbol(status: str) -> str:
    """Get symbol for workflow status."""
    symbols = {
        'created': '○',
        'running': '◐',
        'completed': '●',
        'failed': '✗'
    }
    return symbols.get(status, '?')


def _format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp for display."""
    if not timestamp_str:
        return 'N/A'
    
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except (ValueError, AttributeError):
        return timestamp_str


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
