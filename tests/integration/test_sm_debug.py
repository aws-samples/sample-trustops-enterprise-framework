"""Debug test for SageMaker adapter mocking."""
import io
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.base_adapter import InferenceRequest, InferenceResponse
from src.adapters.sagemaker_adapter import SageMakerAdapter


@pytest.mark.asyncio
async def test_sagemaker_debug():
    """Minimal SageMaker mock test."""
    with patch("src.adapters.sagemaker_adapter.boto3"):
        adapter = SageMakerAdapter(
            endpoint_name="test-endpoint", region="us-east-1"
        )

    response_bytes = b'{"generated_text": "AI is artificial intelligence."}'
    body_stream = io.BytesIO(response_bytes)
    mock_runtime = MagicMock()
    mock_runtime.invoke_endpoint.return_value = {
        "Body": body_stream,
        "ResponseMetadata": {"HTTPHeaders": {}},
    }
    adapter.sagemaker_runtime = mock_runtime

    # Verify the attribute was set
    assert adapter.sagemaker_runtime is mock_runtime

    request = InferenceRequest(prompt="What is AI?", max_tokens=256, temperature=0.5)
    response = await adapter.invoke(request)
    assert isinstance(response, InferenceResponse)
    assert response.text == "AI is artificial intelligence."
