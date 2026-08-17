"""
Model management commands for TrustOps CLI.

Provides `trustops models list`, `trustops models register`,
and `trustops models info` commands.

Requirements: 10.2, 10.3, 10.4
"""

import asyncio
import json
from typing import Optional

import click
import yaml

from cli.auth_handler import AuthHandler, AuthError
from cli.commands.auth import _handle_auth_error
from cli.utils.output import (
    print_header,
    print_success,
    print_error,
    print_info,
    print_table,
    print_warning,
)


def _get_registry():
    """Create a ModelRegistry instance."""
    from src.registry.model_registry import ModelRegistry
    return ModelRegistry()


def _model_to_dict(model) -> dict:
    """Convert a ModelMetadata to a serializable dict."""
    data = {
        "id": model.id,
        "provider": model.provider.value if hasattr(model.provider, "value") else str(model.provider),
        "name": model.name,
        "status": model.status.value if hasattr(model.status, "value") else str(model.status),
        "capabilities": [
            c.value if hasattr(c, "value") else str(c) for c in model.capabilities
        ],
        "fine_tuning_support": model.fine_tuning_support,
        "max_tokens": model.max_tokens,
        "region": model.region,
    }
    if model.pricing:
        data["pricing"] = {
            "input_price_per_1k_tokens": model.pricing.input_price_per_1k_tokens,
            "output_price_per_1k_tokens": model.pricing.output_price_per_1k_tokens,
            "currency": model.pricing.currency,
        }
        if model.pricing.fine_tuning_price_per_1k_tokens is not None:
            data["pricing"]["fine_tuning_price_per_1k_tokens"] = (
                model.pricing.fine_tuning_price_per_1k_tokens
            )
    return data


@click.group()
def models():
    """Manage models in the TrustOps registry."""
    pass


@models.command("list")
@click.option(
    "--provider",
    type=click.Choice(["bedrock", "sagemaker", "external_api"]),
    default=None,
    help="Filter by model provider",
)
@click.option(
    "--capability",
    type=click.Choice([
        "text_generation", "chat", "embedding",
        "fine_tunable", "multimodal", "completion",
    ]),
    default=None,
    help="Filter by model capability",
)
@click.option(
    "--fine-tunable",
    is_flag=True,
    default=False,
    help="Show only fine-tunable models",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def models_list(ctx, provider, capability, fine_tunable, output_format):
    """List available models with optional filtering.

    Requirements: 10.2
    """
    try:
        from src.data_models.model import ModelProvider, ModelCapability

        registry = _get_registry()

        provider_enum = ModelProvider(provider) if provider else None
        capability_enum = ModelCapability(capability) if capability else None

        model_list = asyncio.run(
            registry.list_models(
                provider=provider_enum,
                capability=capability_enum,
                fine_tunable_only=fine_tunable,
            )
        )

        results = [_model_to_dict(m) for m in model_list]

        if output_format == "json":
            click.echo(json.dumps(results, indent=2))
        elif output_format == "yaml":
            click.echo(yaml.dump(results, default_flow_style=False).strip())
        else:
            if not results:
                print_warning("No models found matching the specified filters.")
                return

            print_header("Available Models")
            headers = ["ID", "Provider", "Name", "Status", "Fine-Tunable"]
            rows = [
                [
                    m["id"],
                    m["provider"],
                    m["name"],
                    m["status"],
                    "Yes" if m["fine_tuning_support"] else "No",
                ]
                for m in results
            ]
            print_table(headers, rows)
            print_info(f"\nTotal: {len(results)} model(s)")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to list models: {e}")
        raise click.Abort()


@models.command("register")
@click.option("--provider", required=True, type=click.Choice(["bedrock", "sagemaker", "external_api"]), help="Model provider")
@click.option("--config", "config_file", type=click.Path(exists=True), default=None, help="Path to model configuration JSON file")
@click.option("--model-id", required=True, help="Unique model identifier")
@click.option("--name", required=True, help="Human-readable model name")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def models_register(ctx, provider, config_file, model_id, name, output_format):
    """Register an external model in the TrustOps registry.

    Requirements: 10.3
    """
    try:
        from src.data_models.model import (
            ModelMetadata,
            ModelProvider,
            ModelStatus,
        )

        extra_config = {}
        if config_file:
            with open(config_file, "r") as f:
                extra_config = json.load(f)

        metadata = ModelMetadata(
            id=model_id,
            provider=ModelProvider(provider),
            name=name,
            capabilities=extra_config.get("capabilities", []),
            status=ModelStatus.ACTIVE,
            fine_tuning_support=extra_config.get("fine_tuning_support", False),
            max_tokens=extra_config.get("max_tokens", 4096),
            region=extra_config.get("region", "us-east-1"),
            metadata=extra_config.get("metadata", {}),
        )

        registry = _get_registry()
        asyncio.run(
            registry.register_model(metadata)
        )

        result = _model_to_dict(metadata)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(yaml.dump(result, default_flow_style=False).strip())
        else:
            print_success(f"Model '{model_id}' registered successfully")
            print_info(f"  Provider: {provider}")
            print_info(f"  Name:     {name}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except Exception as e:
        print_error(f"Failed to register model: {e}")
        raise click.Abort()


@models.command("info")
@click.argument("model_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def models_info(ctx, model_id, output_format):
    """Display detailed information about a model.

    Requirements: 10.4
    """
    try:
        registry = _get_registry()
        model = asyncio.run(
            registry.get_model(model_id)
        )

        if model is None:
            print_error(f"Model '{model_id}' not found")
            raise click.Abort()

        result = _model_to_dict(model)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        elif output_format == "yaml":
            click.echo(yaml.dump(result, default_flow_style=False).strip())
        else:
            print_header(f"Model: {model.name}")
            print_info(f"  ID:              {model.id}")
            print_info(f"  Provider:        {result['provider']}")
            print_info(f"  Status:          {result['status']}")
            print_info(f"  Fine-Tunable:    {'Yes' if model.fine_tuning_support else 'No'}")
            print_info(f"  Max Tokens:      {model.max_tokens}")
            print_info(f"  Region:          {model.region}")
            print_info(f"  Capabilities:    {', '.join(result['capabilities'])}")

            if model.pricing:
                print_info("\n  Pricing:")
                print_info(f"    Input:         ${model.pricing.input_price_per_1k_tokens}/1k tokens")
                print_info(f"    Output:        ${model.pricing.output_price_per_1k_tokens}/1k tokens")
                if model.pricing.fine_tuning_price_per_1k_tokens is not None:
                    print_info(f"    Fine-Tuning:   ${model.pricing.fine_tuning_price_per_1k_tokens}/1k tokens")
                print_info(f"    Currency:      {model.pricing.currency}")

    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()
    except click.Abort:
        raise
    except Exception as e:
        print_error(f"Failed to get model info: {e}")
        raise click.Abort()
