"""
Authentication commands for TrustOps CLI.

Provides `trustops auth check` and `trustops auth configure` commands.

Requirements: 10.15, 10.22, 10.23, 10.24
"""

import click
import json
import yaml

from pathlib import Path

from cli.auth_handler import (
    AuthHandler,
    AuthError,
    ExpiredTokenError,
    MissingCredentialsError,
    PermissionDeniedError,
)
from cli.utils.output import (
    print_header,
    print_success,
    print_error,
    print_info,
    print_warning,
)


def _handle_auth_error(e: AuthError) -> None:
    """Handle authentication errors with clear, actionable messages.

    Requirements: 10.16, 10.24
    """
    if isinstance(e, ExpiredTokenError):
        print_error("Authentication failed: credentials have expired")
        print_info(str(e))
    elif isinstance(e, MissingCredentialsError):
        print_error("Authentication failed: no credentials found")
        print_info(str(e))
    elif isinstance(e, PermissionDeniedError):
        print_error("Authentication failed: permission denied")
        print_info(str(e))
    else:
        print_error(f"Authentication failed: {e}")


@click.group()
def auth():
    """Manage AWS authentication for TrustOps."""
    pass


@auth.command("check")
@click.option("--profile", default=None, help="AWS profile name to use")
@click.option("--region", default=None, help="AWS region")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def auth_check(ctx, profile, region, output_format):
    """Validate current AWS credentials and display identity.

    Checks that valid AWS credentials are configured and displays
    the authenticated identity including ARN, account ID, and user ID.

    Requirements: 10.22
    """
    handler = AuthHandler(profile=profile, region=region)

    try:
        identity = handler.check_credentials()
    except AuthError as e:
        _handle_auth_error(e)
        raise click.Abort()

    result = {
        "arn": identity.arn,
        "account_id": identity.account_id,
        "user_id": identity.user_id,
        "auth_method": identity.auth_method.value,
    }

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    elif output_format == "yaml":
        click.echo(yaml.dump(result, default_flow_style=False).strip())
    else:
        print_header("AWS Authentication Status")
        print_success("Credentials are valid")
        print_info(f"  ARN:         {identity.arn}")
        print_info(f"  Account ID:  {identity.account_id}")
        print_info(f"  User ID:     {identity.user_id}")
        print_info(f"  Auth Method: {identity.auth_method.value}")


@auth.command("configure")
@click.pass_context
def auth_configure(ctx):
    """Interactive setup wizard for AWS credentials.

    Guides you through configuring AWS credentials or SSO
    for use with TrustOps.

    Requirements: 10.23
    """
    print_header("TrustOps AWS Authentication Setup")

    method = click.prompt(
        "Select authentication method",
        type=click.Choice(["credentials", "sso", "env"]),
        default="credentials",
    )

    if method == "credentials":
        _configure_credentials()
    elif method == "sso":
        _configure_sso()
    elif method == "env":
        _configure_env_vars()


def _configure_credentials():
    """Guide user through credentials file setup."""
    print_info("\nConfiguring AWS credentials file (~/.aws/credentials)")

    access_key = click.prompt("AWS Access Key ID")
    secret_key = click.prompt("AWS Secret Access Key", hide_input=True)
    region = click.prompt("Default region", default="us-east-1")
    profile = click.prompt("Profile name", default="default")

    aws_dir = Path.home() / ".aws"
    aws_dir.mkdir(exist_ok=True)

    creds_path = aws_dir / "credentials"
    config_path = aws_dir / "config"

    # Write credentials
    creds_content = f"\n[{profile}]\naws_access_key_id = {access_key}\naws_secret_access_key = {secret_key}\n"
    with open(creds_path, "a") as f:
        f.write(creds_content)

    # Write config
    config_content = f"\n[profile {profile}]\nregion = {region}\n"
    with open(config_path, "a") as f:
        f.write(config_content)

    print_success(f"Credentials saved to {creds_path}")
    print_info(f"Profile '{profile}' configured with region '{region}'")

    # Verify
    handler = AuthHandler(profile=profile if profile != "default" else None, region=region)
    try:
        identity = handler.check_credentials()
        print_success(f"Verified: {identity.arn}")
    except AuthError as e:
        print_warning(f"Credentials saved but verification failed: {e}")


def _configure_sso():
    """Guide user through SSO setup."""
    print_info("\nConfiguring AWS SSO")

    sso_start_url = click.prompt("SSO Start URL")
    sso_region = click.prompt("SSO Region", default="us-east-1")
    sso_account_id = click.prompt("SSO Account ID")
    sso_role_name = click.prompt("SSO Role Name")
    profile = click.prompt("Profile name", default="trustops-sso")
    region = click.prompt("Default region", default="us-east-1")

    aws_dir = Path.home() / ".aws"
    aws_dir.mkdir(exist_ok=True)

    config_path = aws_dir / "config"
    config_content = (
        f"\n[profile {profile}]\n"
        f"sso_start_url = {sso_start_url}\n"
        f"sso_region = {sso_region}\n"
        f"sso_account_id = {sso_account_id}\n"
        f"sso_role_name = {sso_role_name}\n"
        f"region = {region}\n"
    )
    with open(config_path, "a") as f:
        f.write(config_content)

    print_success(f"SSO profile '{profile}' saved to {config_path}")
    print_info("Run the following to log in:")
    print_info(f"  aws sso login --profile {profile}")


def _configure_env_vars():
    """Guide user through environment variable setup."""
    print_info("\nConfiguring via environment variables")
    print_info("Set the following environment variables in your shell:\n")
    print_info("  export AWS_ACCESS_KEY_ID=<your-access-key>")
    print_info("  export AWS_SECRET_ACCESS_KEY=<your-secret-key>")
    print_info("  export AWS_DEFAULT_REGION=us-east-1")
    print_info("\nOptionally for temporary credentials:")
    print_info("  export AWS_SESSION_TOKEN=<your-session-token>")
    print_success("Add these to your shell profile for persistence.")
