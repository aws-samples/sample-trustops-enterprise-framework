"""Abstract base class for model provider adapters.

This module defines the unified interface that all model provider adapters
(Bedrock, SageMaker, External API) must implement to provide consistent
model invocation, discovery, and management capabilities.
"""

from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.data_models.model import ModelMetadata


class InferenceRequest(BaseModel):
    """Request parameters for model inference.

    Attributes:
        prompt: The input text prompt for the model
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0-1.0)
        top_p: Nucleus sampling parameter
        stop_sequences: Optional list of sequences that stop generation
    """
    prompt: str
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop_sequences: Optional[list[str]] = None


class InferenceResponse(BaseModel):
    """Response from model inference.

    Attributes:
        text: The generated text response
        input_tokens: Number of tokens in the input
        output_tokens: Number of tokens in the output
        latency_ms: Response latency in milliseconds
        model_id: Identifier of the model that generated the response
        finish_reason: Reason for completion (e.g., 'stop', 'length',
            'error')
    """
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    finish_reason: str


class BaseModelAdapter(ABC):
    """Abstract base class for model provider adapters.

    This class defines the unified interface that all model provider
    adapters must implement. It provides methods for model invocation,
    discovery, validation, and capability checking.

    Implementations must handle provider-specific authentication, rate
    limiting, error handling, and API differences while presenting a
    consistent interface.
    """

    @abstractmethod
    async def invoke(
        self, request: InferenceRequest
    ) -> InferenceResponse:
        """Invoke the model with the given request.

        Args:
            request: The inference request containing prompt and parameters

        Returns:
            InferenceResponse containing the generated text and metadata

        Raises:
            ValueError: If the request parameters are invalid
            RuntimeError: If the model invocation fails
        """
        pass

    @abstractmethod
    async def invoke_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[str]:
        """Stream model response tokens as they are generated.

        Args:
            request: The inference request containing prompt and parameters

        Yields:
            str: Individual tokens or chunks of the generated response

        Raises:
            ValueError: If the request parameters are invalid
            RuntimeError: If the model invocation fails
            NotImplementedError: If streaming is not supported by this
                adapter
        """
        pass

    @abstractmethod
    async def list_models(self) -> list["ModelMetadata"]:
        """List all available models from this provider.

        Returns:
            List of ModelMetadata objects describing available models

        Raises:
            RuntimeError: If model discovery fails
        """
        pass

    @abstractmethod
    async def get_model_info(self, model_id: str) -> "ModelMetadata":
        """Get metadata for a specific model.

        Args:
            model_id: The unique identifier of the model

        Returns:
            ModelMetadata object with model details

        Raises:
            ValueError: If the model_id is not found
            RuntimeError: If metadata retrieval fails
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate connectivity to the provider.

        This method should perform a lightweight check to verify that the
        adapter can successfully communicate with the provider's API.

        Returns:
            True if connection is valid, False otherwise
        """
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if this adapter supports streaming responses.

        Returns:
            True if streaming is supported, False otherwise
        """
        pass

    @abstractmethod
    def supports_fine_tuning(self, model_id: str) -> bool:
        """Check if the specified model supports fine-tuning.

        Args:
            model_id: The unique identifier of the model

        Returns:
            True if the model supports fine-tuning, False otherwise
        """
        pass
