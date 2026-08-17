"""AWS configuration management."""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BucketNotConfiguredError(ValueError):
    """Raised when a required S3 bucket name is not configured."""


def _env_bucket(env_var: str) -> Optional[str]:
    """
    Read a bucket name from the environment, treating blank values as unset.

    Args:
        env_var: Environment variable holding the bucket name

    Returns:
        The trimmed bucket name, or None if unset or whitespace-only
    """
    return os.getenv(env_var, "").strip() or None


def _required_bucket(env_var: str, purpose: str) -> str:
    """
    Resolve a bucket name from the environment, with no fallback default.

    Bucket names are globally unique across all of AWS. Shipping a short,
    predictable default invites bucket squatting: an attacker can create that
    name in their own account, and any deployment that falls back to the
    default would then read from or write to a bucket they control. Requiring
    the name to be set means a misconfigured deployment fails loudly instead of
    silently targeting someone else's bucket. Always qualify names with account
    and region: trustops-<purpose>-<ACCOUNT_ID>-<REGION>.

    Args:
        env_var: Environment variable holding the bucket name
        purpose: Short description used in the suggested naming convention

    Returns:
        The configured bucket name

    Raises:
        BucketNotConfiguredError: If the variable is unset or empty
    """
    value = os.getenv(env_var, "").strip()
    if not value:
        raise BucketNotConfiguredError(
            f"{env_var} env var required. "
            f"Use: trustops-{purpose}-<ACCOUNT_ID>-<REGION>"
        )
    return value


@dataclass
class AWSConfig:
    """AWS service configuration."""

    # AWS credentials and region
    region: str = os.getenv("AWS_REGION", "us-east-1")
    profile: Optional[str] = os.getenv("AWS_PROFILE")

    # S3 buckets. Deliberately no defaults - see _required_bucket. These are
    # resolved lazily so that importing this module (for `--help`, unit tests,
    # or any command that touches no bucket) does not require them to be set;
    # the error surfaces on first use instead. A whitespace-only value counts
    # as unset so a blank line in .env cannot become a bucket name.
    _datasets_bucket: Optional[str] = field(
        default_factory=lambda: _env_bucket("TRUSTOPS_DATASETS_BUCKET"),
        repr=False,
    )
    _results_bucket: Optional[str] = field(
        default_factory=lambda: _env_bucket("TRUSTOPS_RESULTS_BUCKET"),
        repr=False,
    )
    _artifacts_bucket: Optional[str] = field(
        default_factory=lambda: _env_bucket("TRUSTOPS_ARTIFACTS_BUCKET"),
        repr=False,
    )

    @property
    def datasets_bucket(self) -> str:
        """Datasets bucket name. Raises if TRUSTOPS_DATASETS_BUCKET is unset."""
        return self._datasets_bucket or _required_bucket(
            "TRUSTOPS_DATASETS_BUCKET", "datasets"
        )

    @datasets_bucket.setter
    def datasets_bucket(self, value: Optional[str]) -> None:
        self._datasets_bucket = (value or "").strip() or None

    @property
    def results_bucket(self) -> str:
        """Results bucket name. Raises if TRUSTOPS_RESULTS_BUCKET is unset."""
        return self._results_bucket or _required_bucket(
            "TRUSTOPS_RESULTS_BUCKET", "results"
        )

    @results_bucket.setter
    def results_bucket(self, value: Optional[str]) -> None:
        self._results_bucket = (value or "").strip() or None

    @property
    def artifacts_bucket(self) -> str:
        """Artifacts bucket name. Raises if TRUSTOPS_ARTIFACTS_BUCKET is unset."""
        return self._artifacts_bucket or _required_bucket(
            "TRUSTOPS_ARTIFACTS_BUCKET", "artifacts"
        )

    @artifacts_bucket.setter
    def artifacts_bucket(self, value: Optional[str]) -> None:
        self._artifacts_bucket = (value or "").strip() or None

    # Convenience property for s3_bucket (used in some places)
    @property
    def s3_bucket(self) -> str:
        """Get the primary S3 bucket (datasets bucket)."""
        return self.datasets_bucket

    # DynamoDB tables
    workflows_table: str = os.getenv("TRUSTOPS_WORKFLOWS_TABLE", "trustops-workflows")
    models_table: str = os.getenv("TRUSTOPS_MODELS_TABLE", "trustops-models")
    
    # CloudWatch
    log_group: str = os.getenv("TRUSTOPS_LOG_GROUP", "/aws/trustops")
    
    # Bedrock Knowledge Bases
    knowledge_base_id: Optional[str] = os.getenv("KNOWLEDGE_BASE_ID")
    knowledge_base_data_source_id: Optional[str] = os.getenv("KNOWLEDGE_BASE_DATA_SOURCE_ID")
    
    # Trust scoring
    trust_score_threshold: float = float(os.getenv("TRUST_SCORE_THRESHOLD", "0.7"))
    hallucination_similarity_threshold: float = float(
        os.getenv("HALLUCINATION_SIMILARITY_THRESHOLD", "0.7")
    )
    
    # Lambda configuration
    lambda_timeout: int = int(os.getenv("LAMBDA_TIMEOUT", "900"))
    lambda_memory: int = int(os.getenv("LAMBDA_MEMORY", "3008"))
    
    # Model configuration
    default_embedding_model: str = os.getenv(
        "DEFAULT_EMBEDDING_MODEL", "amazon.titan-embed-text-v1"
    )
    default_foundation_model: str = os.getenv(
        "DEFAULT_FOUNDATION_MODEL", "anthropic.claude-v2"
    )
    
    # Bedrock fine-tuning
    bedrock_execution_role_arn: str = os.getenv("BEDROCK_EXECUTION_ROLE_ARN", "")
    
    # Cost configuration (USD per 1000 tokens)
    claude_v2_input_cost: float = float(os.getenv("CLAUDE_V2_INPUT_COST", "0.008"))
    claude_v2_output_cost: float = float(os.getenv("CLAUDE_V2_OUTPUT_COST", "0.024"))
    
    def get_boto3_session_kwargs(self) -> dict:
        """Get kwargs for boto3 session creation."""
        kwargs = {"region_name": self.region}
        if self.profile:
            kwargs["profile_name"] = self.profile
        return kwargs
    
    @classmethod
    def from_env(cls) -> 'AWSConfig':
        """Create AWSConfig instance from environment variables."""
        return cls()


# Global configuration instance
config = AWSConfig()
