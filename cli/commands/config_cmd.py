"""
Configuration commands for TrustOps CLI.

Provides `trustops config show` and `trustops config set`.

Requirements: 10.19
"""

import json

import click
import yaml

from cli.config import load_config, save_config, get_config_path
from cli.utils.output import (
    print_header,
    print_success,
    print_error,
    print_info,
)


@click.group("config")
def config_cmd():
    """Display and modify CLI configuration."""
    pass


@config_cmd.command("show")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def config_show(ctx, output_format):
    """Display current CLI configuration.

    Requirements: 10.19
    """
    config_path = ctx.obj.get("config_path", get_config_path())
    config = load_config(str(config_path))

    result = {
        "config_path": str(config_path),
        **config,
    }

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    elif output_format == "yaml":
        click.echo(
            yaml.dump(result, default_flow_style=False).strip()
        )
    else:
        print_header("TrustOps Configuration")
        print_info(f"Config file: {config_path}")
        print_info("")
        _print_nested(config)


@config_cmd.command("set")
@click.argument("key")
@click.argument("value")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def config_set(ctx, key, value, output_format):
    """Set a configuration value.

    KEY uses dot notation, e.g. aws.region or
    trust_scoring.threshold.

    Requirements: 10.19
    """
    config_path = ctx.obj.get("config_path", get_config_path())
    config = load_config(str(config_path))

    # Parse dot-notation key
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]

    # Try to parse value as number or boolean
    parsed_value = _parse_value(value)
    target[parts[-1]] = parsed_value

    save_config(config, str(config_path))

    result = {"key": key, "value": parsed_value, "status": "updated"}

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    elif output_format == "yaml":
        click.echo(
            yaml.dump(result, default_flow_style=False).strip()
        )
    else:
        print_success(f"Set {key} = {parsed_value}")


def _parse_value(value: str):
    """Parse a string value to appropriate type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _print_nested(data: dict, indent: int = 0):
    """Print nested dict with indentation."""
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            print_info(f"{prefix}{key}:")
            _print_nested(value, indent + 1)
        else:
            print_info(f"{prefix}{key}: {value}")
