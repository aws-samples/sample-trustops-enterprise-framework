"""Pytest configuration and fixtures for TrustOps tests."""

import os

# Pin the environment BEFORE any test module imports config.aws_config.
#
# config.aws_config calls load_dotenv() and resolves its dataclass field
# defaults from os.getenv at import time. Without this block, a developer's
# local .env (which points at real deployed buckets/tables) would leak into
# the suite and tests would hit real AWS instead of moto/mock doubles.
# load_dotenv() does not overwrite variables that are already set, so
# assigning them here wins over .env.
_TEST_ENV = {
    # Fake credentials so a stray real call fails fast instead of
    # authenticating against a live account.
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SECURITY_TOKEN": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_REGION": "us-east-1",
    # Resource names the test fixtures actually create. Buckets follow the
    # trustops-test-<purpose>-<ACCOUNT_ID>-<REGION> convention with the AWS
    # documentation example account 123456789012, so no fixture models a
    # squattable short name even inside moto, and the names cannot be mistaken
    # for the deployable ones during a security review.
    "TRUSTOPS_REGION": "us-east-1",
    "TRUSTOPS_DATASETS_BUCKET": "trustops-test-datasets-123456789012-us-east-1",
    "TRUSTOPS_RESULTS_BUCKET": "trustops-test-results-123456789012-us-east-1",
    "TRUSTOPS_ARTIFACTS_BUCKET": "trustops-test-artifacts-123456789012-us-east-1",
    "TRUSTOPS_WORKFLOWS_TABLE": "trustops-workflows",
    "TRUSTOPS_MODELS_TABLE": "trustops-models",
    "TRUSTOPS_LOG_GROUP": "/aws/trustops",
    # Leave the Knowledge Base unset so clients default to mock mode.
    "KNOWLEDGE_BASE_ID": "",
    "KNOWLEDGE_BASE_DATA_SOURCE_ID": "",
}
for _key, _value in _TEST_ENV.items():
    os.environ[_key] = _value

# Do not let a local AWS profile redirect tests at a real account.
os.environ.pop("AWS_PROFILE", None)

import pytest
from unittest.mock import MagicMock
import boto3
from moto import mock_aws


@pytest.fixture(scope="session")
def aws_credentials():
    """Mock AWS credentials for testing."""
    for key, value in _TEST_ENV.items():
        os.environ[key] = value


@pytest.fixture
def mock_s3_client(aws_credentials):
    """Create a mocked S3 client."""
    with mock_aws():
        yield boto3.client("s3", region_name="us-east-1")


@pytest.fixture
def mock_dynamodb_client(aws_credentials):
    """Create a mocked DynamoDB client."""
    with mock_aws():
        yield boto3.client("dynamodb", region_name="us-east-1")


@pytest.fixture
def mock_logs_client(aws_credentials):
    """Create a mocked CloudWatch Logs client."""
    with mock_aws():
        yield boto3.client("logs", region_name="us-east-1")


@pytest.fixture
def sample_evaluation_dataset():
    """Sample evaluation dataset for testing."""
    return [
        {
            "prompt": "What is the capital of France?",
            "expected_response": "Paris",
            "source_documents": ["France is a country in Europe. Its capital is Paris."],
            "category": "geography",
            "metadata": {}
        },
        {
            "prompt": "Explain photosynthesis.",
            "expected_response": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "source_documents": [
                "Photosynthesis is a process used by plants to convert light into energy.",
                "Plants use chlorophyll to capture sunlight for photosynthesis."
            ],
            "category": "science",
            "metadata": {}
        }
    ]


@pytest.fixture
def sample_model_response():
    """Sample model response for testing."""
    return {
        "response_id": "test-response-001",
        "model_id": "anthropic.claude-v2",
        "prompt": "What is the capital of France?",
        "response_text": "The capital of France is Paris.",
        "input_tokens": 10,
        "output_tokens": 8,
        "latency_ms": 250.5,
        "timestamp": "2024-01-01T00:00:00Z",
        "metadata": {}
    }


@pytest.fixture
def sample_trust_score():
    """Sample trust score for testing."""
    return {
        "overall_score": 0.85,
        "components": {
            "context_grounding": 0.9,
            "output_structure": 0.8,
            "uncertainty_indicators": 0.85,
            "factual_consistency": 0.9,
            "response_completeness": 0.8
        },
        "confidence_level": "high",
        "flagged_for_review": False,
        "explanation": "High confidence response with good grounding."
    }
