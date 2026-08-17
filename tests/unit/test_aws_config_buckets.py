"""
Tests for S3 bucket name resolution in AWSConfig.

Bucket names are globally unique across AWS, so the framework must not ship a
short predictable default: another account could create that bucket first and
receive reads and writes from any deployment that fell back to the default.
These tests pin the fail-closed behaviour.
"""
import os
from unittest.mock import patch

import pytest

from config.aws_config import AWSConfig, BucketNotConfiguredError

BUCKET_VARS = (
    "TRUSTOPS_DATASETS_BUCKET",
    "TRUSTOPS_RESULTS_BUCKET",
    "TRUSTOPS_ARTIFACTS_BUCKET",
)


@pytest.fixture
def no_bucket_env():
    """Clear every bucket variable for the duration of a test."""
    with patch.dict(os.environ, {var: "" for var in BUCKET_VARS}, clear=False):
        yield


class TestBucketRequired:
    """A missing bucket name must raise rather than silently default."""

    @pytest.mark.parametrize(
        "attr,env_var",
        [
            ("datasets_bucket", "TRUSTOPS_DATASETS_BUCKET"),
            ("results_bucket", "TRUSTOPS_RESULTS_BUCKET"),
            ("artifacts_bucket", "TRUSTOPS_ARTIFACTS_BUCKET"),
        ],
    )
    def test_unset_bucket_raises(self, no_bucket_env, attr, env_var):
        cfg = AWSConfig()

        with pytest.raises(BucketNotConfiguredError, match=env_var):
            getattr(cfg, attr)

    def test_error_names_the_variable_and_convention(self, no_bucket_env):
        cfg = AWSConfig()

        with pytest.raises(BucketNotConfiguredError) as exc:
            cfg.datasets_bucket

        message = str(exc.value)
        assert "TRUSTOPS_DATASETS_BUCKET" in message
        assert "<ACCOUNT_ID>" in message
        assert "<REGION>" in message

    def test_whitespace_only_is_treated_as_unset(self):
        with patch.dict(
            os.environ, {"TRUSTOPS_DATASETS_BUCKET": "   "}, clear=False
        ):
            cfg = AWSConfig()
            with pytest.raises(BucketNotConfiguredError):
                cfg.datasets_bucket

    def test_no_predictable_default_is_returned(self, no_bucket_env):
        """Regression guard: the old code returned a short unqualified name."""
        cfg = AWSConfig()

        try:
            value = cfg.datasets_bucket
        except BucketNotConfiguredError:
            return  # correct behaviour

        pytest.fail(f"expected a raise, got squattable default {value!r}")

    def test_s3_bucket_property_also_fails_closed(self, no_bucket_env):
        cfg = AWSConfig()

        with pytest.raises(BucketNotConfiguredError):
            cfg.s3_bucket

    def test_importing_config_does_not_require_buckets(self, no_bucket_env):
        """Construction must stay lazy so `--help` and tests still work."""
        cfg = AWSConfig()

        assert cfg.region  # unrelated fields resolve fine


class TestBucketConfigured:
    """A configured bucket name is returned unchanged."""

    def test_configured_values_are_used(self):
        env = {
            "TRUSTOPS_DATASETS_BUCKET": "my-datasets-111122223333-us-east-1",
            "TRUSTOPS_RESULTS_BUCKET": "my-results-111122223333-us-east-1",
            "TRUSTOPS_ARTIFACTS_BUCKET": "my-artifacts-111122223333-us-east-1",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = AWSConfig()

            assert cfg.datasets_bucket == env["TRUSTOPS_DATASETS_BUCKET"]
            assert cfg.results_bucket == env["TRUSTOPS_RESULTS_BUCKET"]
            assert cfg.artifacts_bucket == env["TRUSTOPS_ARTIFACTS_BUCKET"]
            assert cfg.s3_bucket == env["TRUSTOPS_DATASETS_BUCKET"]

    def test_setter_overrides_environment(self):
        with patch.dict(
            os.environ, {"TRUSTOPS_DATASETS_BUCKET": "from-env"}, clear=False
        ):
            cfg = AWSConfig()
            cfg.datasets_bucket = "explicit-override"

            assert cfg.datasets_bucket == "explicit-override"

    def test_setter_clearing_restores_fail_closed(self, no_bucket_env):
        cfg = AWSConfig()
        cfg.results_bucket = "temporary"
        assert cfg.results_bucket == "temporary"

        cfg.results_bucket = None

        with pytest.raises(BucketNotConfiguredError):
            cfg.results_bucket
