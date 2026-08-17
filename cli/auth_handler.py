"""
Authentication handler for TrustOps CLI.

Supports multiple AWS authentication methods:
- AWS credentials file (~/.aws/credentials)
- Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- IAM roles (for EC2/Lambda)
- AWS SSO profiles

Requirements: 10.15
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
)

try:
    from botocore.exceptions import TokenRetrievalError
except ImportError:
    TokenRetrievalError = Exception


class AuthMethod(str, Enum):
    """Supported authentication methods."""
    CREDENTIALS_FILE = "credentials_file"
    ENVIRONMENT_VARIABLES = "environment_variables"
    IAM_ROLE = "iam_role"
    SSO_PROFILE = "sso_profile"
    UNKNOWN = "unknown"


@dataclass
class AuthIdentity:
    """Represents the authenticated AWS identity."""
    arn: str
    account_id: str
    user_id: str
    auth_method: AuthMethod


class AuthError(Exception):
    """Base exception for authentication errors."""
    pass


class ExpiredTokenError(AuthError):
    """Raised when AWS credentials have expired."""
    pass


class MissingCredentialsError(AuthError):
    """Raised when no AWS credentials are found."""
    pass


class PermissionDeniedError(AuthError):
    """Raised when credentials lack required permissions."""
    pass


class AuthHandler:
    """Handles AWS authentication for the TrustOps CLI.

    Supports credentials file, environment variables, IAM roles,
    and AWS SSO profiles.

    Requirements: 10.15, 10.21
    """

    def __init__(self, profile: Optional[str] = None, region: Optional[str] = None):
        self._profile = profile
        self._region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    def detect_auth_method(self) -> AuthMethod:
        """Detect which authentication method is available.

        Returns:
            The detected AuthMethod.
        """
        if self._has_env_credentials():
            return AuthMethod.ENVIRONMENT_VARIABLES
        if self._profile and self._is_sso_profile(self._profile):
            return AuthMethod.SSO_PROFILE
        if self._has_credentials_file():
            return AuthMethod.CREDENTIALS_FILE
        if self._has_iam_role():
            return AuthMethod.IAM_ROLE
        return AuthMethod.UNKNOWN

    def check_credentials(self) -> AuthIdentity:
        """Validate current credentials and return identity info.

        Returns:
            AuthIdentity with ARN, account ID, and user ID.

        Raises:
            MissingCredentialsError: If no credentials are found.
            ExpiredTokenError: If credentials have expired.
            PermissionDeniedError: If credentials lack permissions.
            AuthError: For other authentication failures.
        """
        session_kwargs = {}
        if self._profile:
            session_kwargs["profile_name"] = self._profile
        if self._region:
            session_kwargs["region_name"] = self._region

        try:
            session = boto3.Session(**session_kwargs)
            sts = session.client("sts")
            identity = sts.get_caller_identity()

            auth_method = self.detect_auth_method()
            return AuthIdentity(
                arn=identity["Arn"],
                account_id=identity["Account"],
                user_id=identity["UserId"],
                auth_method=auth_method,
            )
        except NoCredentialsError:
            raise MissingCredentialsError(
                "No AWS credentials found. Configure credentials using:\n"
                "  1. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
                "  2. AWS credentials file: ~/.aws/credentials\n"
                "  3. AWS SSO: aws sso login --profile <profile>\n"
                "  4. IAM role (on EC2/Lambda)"
            )
        except TokenRetrievalError:
            raise ExpiredTokenError(
                "AWS credentials have expired. Refresh using:\n"
                "  - For SSO: aws sso login --profile <profile>\n"
                "  - For temporary credentials: refresh your session token"
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("AccessDenied", "AccessDeniedException"):
                raise PermissionDeniedError(
                    f"Permission denied: {e.response['Error'].get('Message', str(e))}\n"
                    "Ensure your credentials have sts:GetCallerIdentity permission."
                )
            if error_code in ("ExpiredToken", "ExpiredTokenException"):
                raise ExpiredTokenError(
                    "AWS credentials have expired. Refresh your session."
                )
            raise AuthError(f"Authentication failed: {e}")
        except Exception as e:
            raise AuthError(f"Authentication failed: {e}")

    def get_session(self) -> "boto3.Session":
        """Create a boto3 session with the configured credentials.

        Returns:
            A configured boto3.Session.
        """
        session_kwargs = {}
        if self._profile:
            session_kwargs["profile_name"] = self._profile
        if self._region:
            session_kwargs["region_name"] = self._region
        return boto3.Session(**session_kwargs)

    def _has_env_credentials(self) -> bool:
        """Check if AWS credentials are set via environment variables."""
        return bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        )

    def _has_credentials_file(self) -> bool:
        """Check if AWS credentials file exists."""
        creds_path = Path.home() / ".aws" / "credentials"
        return creds_path.exists()

    def _has_iam_role(self) -> bool:
        """Check if running on an instance with an IAM role.

        Checks for the EC2 metadata endpoint indicator.
        """
        return bool(os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"))

    def _is_sso_profile(self, profile: str) -> bool:
        """Check if the given profile is configured for SSO."""
        config_path = Path.home() / ".aws" / "config"
        if not config_path.exists():
            return False
        try:
            content = config_path.read_text()
            in_profile = False
            for line in content.splitlines():
                stripped = line.strip()
                if stripped == f"[profile {profile}]" or stripped == f"[{profile}]":
                    in_profile = True
                elif stripped.startswith("["):
                    in_profile = False
                elif in_profile and "sso_" in stripped:
                    return True
        except OSError:
            pass
        return False
