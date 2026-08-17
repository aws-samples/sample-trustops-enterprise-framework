"""
TrustOps CLI - Command-line interface for TrustOps AWS Demo.

This CLI provides commands for:
- Baseline model evaluation
- Fine-tuning workflows
- Comparative evaluation
- Workflow status tracking
- Workflow reproduction
"""
import click
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from cli.commands.baseline import baseline
from cli.commands.compare import compare
from cli.commands.foundation_compare import foundation_compare
from cli.commands.status import status
from cli.commands.reproduce import reproduce
from cli.commands.auth import auth
from cli.commands.models import models
from cli.commands.datasets import datasets
from cli.commands.evaluate import evaluate
from cli.commands.finetune_v2 import finetune_v2
from cli.commands.workflows import workflows
from cli.commands.results import results
from cli.commands.config_cmd import config_cmd
from cli.config import load_config, save_config, get_config_path


@click.group()
@click.version_option(version='0.1.0', prog_name='trustops')
@click.option(
    '--config',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
@click.pass_context
def cli(ctx, config):
    """
    TrustOps - Trust-first fine-tuning and evaluation for GenAI.
    
    A framework for quantifying trust in LLM outputs, comparing models,
    detecting hallucinations, and optimizing cost-performance tradeoffs.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        ctx.obj['config_path'] = config
        ctx.obj['config'] = load_config(config)
    else:
        ctx.obj['config_path'] = get_config_path()
        ctx.obj['config'] = load_config()


@cli.command()
@click.option(
    '--region',
    prompt='AWS Region',
    default='us-east-1',
    help='AWS region for TrustOps resources'
)
@click.option(
    '--datasets-bucket',
    prompt='Datasets S3 Bucket',
    help='S3 bucket for evaluation datasets'
)
@click.option(
    '--results-bucket',
    prompt='Results S3 Bucket',
    help='S3 bucket for evaluation results'
)
@click.option(
    '--artifacts-bucket',
    prompt='Artifacts S3 Bucket',
    help='S3 bucket for workflow artifacts'
)
@click.option(
    '--workflows-table',
    prompt='Workflows DynamoDB Table',
    default='trustops-workflows',
    help='DynamoDB table for workflow tracking'
)
@click.pass_context
def configure(ctx, region, datasets_bucket, results_bucket, 
              artifacts_bucket, workflows_table):
    """Configure TrustOps AWS settings."""
    config_data = {
        'aws': {
            'region': region,
            'datasets_bucket': datasets_bucket,
            'results_bucket': results_bucket,
            'artifacts_bucket': artifacts_bucket,
            'workflows_table': workflows_table
        },
        'trust_scoring': {
            'threshold': 0.7,
            'hallucination_similarity_threshold': 0.7
        },
        'models': {
            'default_embedding_model': 'amazon.titan-embed-text-v1',
            'default_foundation_model': 'anthropic.claude-v2'
        }
    }
    
    config_path = ctx.obj.get('config_path', get_config_path())
    save_config(config_data, config_path)
    
    click.echo(f"\n✓ Configuration saved to {config_path}")
    click.echo("\nYou can now run TrustOps commands.")


# Register command groups
cli.add_command(baseline)
cli.add_command(compare)
cli.add_command(foundation_compare)
cli.add_command(status)
cli.add_command(reproduce)
cli.add_command(auth)
cli.add_command(models)
cli.add_command(datasets)
cli.add_command(evaluate)
# `finetune` is provided by the subcommand group in finetune_v2
# (start / status / stop). The older single-command implementation was
# removed to eliminate a command-name collision on "finetune".
cli.add_command(finetune_v2)
cli.add_command(workflows)
cli.add_command(results)
cli.add_command(config_cmd)


def main():
    """Entry point for the CLI."""
    try:
        cli(obj={})
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
