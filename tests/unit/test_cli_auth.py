"""
Tests for CLI authentication handler and auth commands.

Tests tasks 19.1-19.4:
- AuthHandler credential detection and validation
- `trustops auth check` command
- `trustops auth configure` command
- Authentication error handling
"""

import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from cli.auth_handler import (
    AuthHandler,
    AuthIdentity,
    AuthMethod,
    AuthError,
    ExpiredTokenError,
    MissingCredentialsError,
    PermissionDeniedError,
)
from cli.commands.auth import auth
from cli.main import cli


# ─── AuthHandler unit tests (Task 19.1) ───


class TestAuthMethodDetection:
    """Test AuthHandler.detect_auth_method()."""

    def test_detects_env_credentials(self):
        handler = AuthHandler()
        with patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        }):
            assert handler.detect_auth_method() == AuthMethod.ENVIRONMENT_VARIABLES

    def test_detects_credentials_file(self, tmp_path):
        handler = AuthHandler()
        with patch.dict(os.environ, {}, clear=True):
            # Remove env vars that would take priority
            env = {k: v for k, v in os.environ.items()
                   if k not in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")}
            with patch.dict(os.environ, env, clear=True):
                with patch.object(handler, "_has_credentials_file", return_value=True):
                    assert handler.detect_auth_method() == AuthMethod.CREDENTIALS_FILE

    def test_detects_iam_role(self):
        handler = AuthHandler()
        with patch.dict(os.environ, {
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/creds",
        }, clear=True):
            with patch.object(handler, "_has_credentials_file", return_value=False):
                assert handler.detect_auth_method() == AuthMethod.IAM_ROLE

    def test_detects_sso_profile(self, tmp_path):
        config_file = tmp_path / ".aws" / "config"
        config_file.parent.mkdir(parents=True)
        config_file.write_text("[profile my-sso]\nsso_start_url = https://example.awsapps.com/start\n")

        handler = AuthHandler(profile="my-sso")
        with patch.dict(os.environ, {}, clear=True):
            with patch("cli.auth_handler.Path.home", return_value=tmp_path):
                assert handler.detect_auth_method() == AuthMethod.SSO_PROFILE

    def test_returns_unknown_when_no_method(self):
        handler = AuthHandler()
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(handler, "_has_credentials_file", return_value=False):
                with patch.object(handler, "_has_iam_role", return_value=False):
                    assert handler.detect_auth_method() == AuthMethod.UNKNOWN


class TestAuthHandlerCheckCredentials:
    """Test AuthHandler.check_credentials()."""

    def test_returns_identity_on_success(self):
        handler = AuthHandler()
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:user/testuser",
            "Account": "123456789012",
            "UserId": "AIDAEXAMPLE",
        }
        mock_session = MagicMock()
        mock_session.client.return_value = mock_sts

        with patch("cli.auth_handler.boto3") as mock_boto3:
            mock_boto3.Session.return_value = mock_session
            identity = handler.check_credentials()

        assert isinstance(identity, AuthIdentity)
        assert identity.arn == "arn:aws:iam::123456789012:user/testuser"
        assert identity.account_id == "123456789012"
        assert identity.user_id == "AIDAEXAMPLE"

    def test_raises_missing_credentials_error(self):
        handler = AuthHandler()
        from botocore.exceptions import NoCredentialsError

        mock_session = MagicMock()
        mock_session.client.return_value.get_caller_identity.side_effect = (
            NoCredentialsError()
        )

        with patch("cli.auth_handler.boto3") as mock_boto3:
            mock_boto3.Session.return_value = mock_session
            with pytest.raises(MissingCredentialsError):
                handler.check_credentials()

    def test_raises_expired_token_error(self):
        handler = AuthHandler()
        from botocore.exceptions import ClientError

        error_response = {
            "Error": {"Code": "ExpiredToken", "Message": "Token expired"}
        }
        mock_session = MagicMock()
        mock_session.client.return_value.get_caller_identity.side_effect = (
            ClientError(error_response, "GetCallerIdentity")
        )

        with patch("cli.auth_handler.boto3") as mock_boto3:
            mock_boto3.Session.return_value = mock_session
            with pytest.raises(ExpiredTokenError):
                handler.check_credentials()

    def test_raises_permission_denied_error(self):
        handler = AuthHandler()
        from botocore.exceptions import ClientError

        error_response = {
            "Error": {"Code": "AccessDenied", "Message": "Not authorized"}
        }
        mock_session = MagicMock()
        mock_session.client.return_value.get_caller_identity.side_effect = (
            ClientError(error_response, "GetCallerIdentity")
        )

        with patch("cli.auth_handler.boto3") as mock_boto3:
            mock_boto3.Session.return_value = mock_session
            with pytest.raises(PermissionDeniedError):
                handler.check_credentials()

    def test_uses_profile_when_provided(self):
        handler = AuthHandler(profile="my-profile", region="eu-west-1")
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:user/testuser",
            "Account": "123456789012",
            "UserId": "AIDAEXAMPLE",
        }
        mock_session = MagicMock()
        mock_session.client.return_value = mock_sts

        with patch("cli.auth_handler.boto3") as mock_boto3:
            mock_boto3.Session.return_value = mock_session
            handler.check_credentials()
            mock_boto3.Session.assert_called_once_with(
                profile_name="my-profile", region_name="eu-west-1"
            )


class TestAuthHandlerHelpers:
    """Test AuthHandler helper methods."""

    def test_has_env_credentials_true(self):
        handler = AuthHandler()
        with patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "AKID",
            "AWS_SECRET_ACCESS_KEY": "SECRET",
        }):
            assert handler._has_env_credentials() is True

    def test_has_env_credentials_false_missing_key(self):
        handler = AuthHandler()
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "AKID"}, clear=True):
            assert handler._has_env_credentials() is False

    def test_has_credentials_file(self, tmp_path):
        handler = AuthHandler()
        creds = tmp_path / ".aws" / "credentials"
        creds.parent.mkdir(parents=True)
        creds.write_text("[default]\naws_access_key_id = AKID\n")
        with patch("cli.auth_handler.Path.home", return_value=tmp_path):
            assert handler._has_credentials_file() is True

    def test_has_no_credentials_file(self, tmp_path):
        handler = AuthHandler()
        with patch("cli.auth_handler.Path.home", return_value=tmp_path):
            assert handler._has_credentials_file() is False

    def test_is_sso_profile_true(self, tmp_path):
        config = tmp_path / ".aws" / "config"
        config.parent.mkdir(parents=True)
        config.write_text(
            "[profile sso-test]\nsso_start_url = https://example.awsapps.com/start\n"
            "sso_region = us-east-1\n"
        )
        handler = AuthHandler(profile="sso-test")
        with patch("cli.auth_handler.Path.home", return_value=tmp_path):
            assert handler._is_sso_profile("sso-test") is True

    def test_is_sso_profile_false(self, tmp_path):
        config = tmp_path / ".aws" / "config"
        config.parent.mkdir(parents=True)
        config.write_text("[profile regular]\nregion = us-east-1\n")
        handler = AuthHandler(profile="regular")
        with patch("cli.auth_handler.Path.home", return_value=tmp_path):
            assert handler._is_sso_profile("regular") is False


# ─── CLI auth check command tests (Task 19.2) ───


class TestAuthCheckCommand:
    """Test `trustops auth check` CLI command."""

    def test_auth_check_success_text(self):
        runner = CliRunner()
        mock_identity = AuthIdentity(
            arn="arn:aws:iam::123456789012:user/testuser",
            account_id="123456789012",
            user_id="AIDAEXAMPLE",
            auth_method=AuthMethod.ENVIRONMENT_VARIABLES,
        )
        with patch(
            "cli.commands.auth.AuthHandler.check_credentials",
            return_value=mock_identity,
        ):
            result = runner.invoke(cli, ["auth", "check"])
            assert result.exit_code == 0
            assert "123456789012" in result.output
            assert "AIDAEXAMPLE" in result.output
            assert "Credentials are valid" in result.output

    def test_auth_check_success_json(self):
        runner = CliRunner()
        mock_identity = AuthIdentity(
            arn="arn:aws:iam::123456789012:user/testuser",
            account_id="123456789012",
            user_id="AIDAEXAMPLE",
            auth_method=AuthMethod.ENVIRONMENT_VARIABLES,
        )
        with patch(
            "cli.commands.auth.AuthHandler.check_credentials",
            return_value=mock_identity,
        ):
            result = runner.invoke(cli, ["auth", "check", "--format", "json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["account_id"] == "123456789012"
            assert data["auth_method"] == "environment_variables"

    def test_auth_check_success_yaml(self):
        runner = CliRunner()
        mock_identity = AuthIdentity(
            arn="arn:aws:iam::123456789012:user/testuser",
            account_id="123456789012",
            user_id="AIDAEXAMPLE",
            auth_method=AuthMethod.CREDENTIALS_FILE,
        )
        with patch(
            "cli.commands.auth.AuthHandler.check_credentials",
            return_value=mock_identity,
        ):
            result = runner.invoke(cli, ["auth", "check", "--format", "yaml"])
            assert result.exit_code == 0
            assert "account_id: '123456789012'" in result.output

    def test_auth_check_missing_credentials(self):
        runner = CliRunner()
        with patch(
            "cli.commands.auth.AuthHandler.check_credentials",
            side_effect=MissingCredentialsError("No credentials found"),
        ):
            result = runner.invoke(cli, ["auth", "check"])
            assert result.exit_code != 0
            assert "no credentials found" in result.output.lower()

    def test_auth_check_expired_token(self):
        runner = CliRunner()
        with patch(
            "cli.commands.auth.AuthHandler.check_credentials",
            side_effect=ExpiredTokenError("Token expired"),
        ):
            result = runner.invoke(cli, ["auth", "check"])
            assert result.exit_code != 0
            assert "expired" in result.output.lower()

    def test_auth_check_permission_denied(self):
        runner = CliRunner()
        with patch(
            "cli.commands.auth.AuthHandler.check_credentials",
            side_effect=PermissionDeniedError("Access denied"),
        ):
            result = runner.invoke(cli, ["auth", "check"])
            assert result.exit_code != 0
            assert "permission denied" in result.output.lower()

    def test_auth_check_with_profile(self):
        runner = CliRunner()
        mock_identity = AuthIdentity(
            arn="arn:aws:iam::123456789012:user/testuser",
            account_id="123456789012",
            user_id="AIDAEXAMPLE",
            auth_method=AuthMethod.SSO_PROFILE,
        )
        with patch(
            "cli.commands.auth.AuthHandler.check_credentials",
            return_value=mock_identity,
        ) as mock_check:
            with patch(
                "cli.commands.auth.AuthHandler.__init__",
                return_value=None,
            ) as mock_init:
                result = runner.invoke(
                    cli, ["auth", "check", "--profile", "my-profile"]
                )
                # Verify the handler was created (init called)
                assert mock_init.called


# ─── CLI auth configure command tests (Task 19.3) ───


class TestAuthConfigureCommand:
    """Test `trustops auth configure` CLI command."""

    def test_configure_credentials_method(self, tmp_path):
        runner = CliRunner()
        # Create the .aws directory structure
        aws_dir = tmp_path / ".aws"
        aws_dir.mkdir()
        (aws_dir / "credentials").write_text("")
        (aws_dir / "config").write_text("")

        with patch("cli.commands.auth.Path") as mock_path_cls:
            mock_path_cls.home.return_value = tmp_path
            with patch(
                "cli.commands.auth.AuthHandler.check_credentials",
                side_effect=AuthError("skip verification"),
            ):
                result = runner.invoke(
                    cli,
                    ["auth", "configure"],
                    input="credentials\nAKID\nSECRET\nus-east-1\ndefault\n",
                )
                assert result.exit_code == 0
                assert "Credentials saved" in result.output or "verification failed" in result.output

    def test_configure_env_method(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["auth", "configure"],
            input="env\n",
        )
        assert result.exit_code == 0
        assert "AWS_ACCESS_KEY_ID" in result.output

    def test_configure_sso_method(self, tmp_path):
        runner = CliRunner()
        aws_dir = tmp_path / ".aws"
        aws_dir.mkdir()
        (aws_dir / "config").write_text("")

        with patch("cli.commands.auth.Path") as mock_path_cls:
            mock_path_cls.home.return_value = tmp_path
            result = runner.invoke(
                cli,
                ["auth", "configure"],
                input="sso\nhttps://example.awsapps.com/start\nus-east-1\n123456789012\nAdminRole\ntrustops-sso\nus-east-1\n",
            )
            assert result.exit_code == 0
            assert "SSO profile" in result.output


# ─── Authentication error handler tests (Task 19.4) ───


class TestAuthErrorHandler:
    """Test authentication error handling with clear messages."""

    def test_expired_token_message(self, capsys):
        from cli.commands.auth import _handle_auth_error

        _handle_auth_error(ExpiredTokenError("Token expired. Refresh your session."))
        captured = capsys.readouterr()
        assert "expired" in captured.out.lower() or "expired" in captured.err.lower()

    def test_missing_credentials_message(self, capsys):
        from cli.commands.auth import _handle_auth_error

        _handle_auth_error(MissingCredentialsError("No credentials found"))
        captured = capsys.readouterr()
        assert "no credentials found" in (captured.out + captured.err).lower()

    def test_permission_denied_message(self, capsys):
        from cli.commands.auth import _handle_auth_error

        _handle_auth_error(PermissionDeniedError("Access denied"))
        captured = capsys.readouterr()
        assert "permission denied" in (captured.out + captured.err).lower()

    def test_generic_auth_error_message(self, capsys):
        from cli.commands.auth import _handle_auth_error

        _handle_auth_error(AuthError("Something went wrong"))
        captured = capsys.readouterr()
        assert "Something went wrong" in (captured.out + captured.err)
