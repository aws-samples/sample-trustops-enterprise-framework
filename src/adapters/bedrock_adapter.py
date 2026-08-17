"""Bedrock adapter for AWS Bedrock foundation models.

This module implements the BaseModelAdapter interface for AWS Bedrock,
providing model discovery, invocation, and streaming capabilities.

Requirements: 1.1, 1.4, 1.6, 1.7
"""

import asyncio
import json
from typing import AsyncIterator, Optional
from botocore.exceptions import ClientError

from src.adapters.base_adapter import (
    BaseModelAdapter,
    InferenceRequest,
    InferenceResponse,
)
from src.aws_clients.bedrock_client import BedrockClient
from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
    ModelPricing,
)


class BedrockAdapter(BaseModelAdapter):
    """Adapter for AWS Bedrock foundation models.

    This adapter wraps the existing BedrockClient and implements the
    BaseModelAdapter interface, adding model discovery via
    bedrock.list_foundation_models() and providing unified invocation.

    Requirements:
        - 1.1: Automatic discovery of AWS Bedrock models
        - 1.4: Maintain metadata including provider, capabilities, pricing
        - 1.6: Unified inference interface
        - 1.7: Handle authentication, rate limiting, and error handling
    """

    def __init__(self, model_id: str, region: Optional[str] = None):
        """Initialize the Bedrock adapter.

        Args:
            model_id: The Bedrock model identifier to use for invocations
            region: AWS region (defaults to config.region)
        """
        self.model_id = model_id
        self.client = BedrockClient(region=region)
        self.region = region or self.client.region

    async def invoke(self, request: InferenceRequest) -> InferenceResponse:
        """Invoke a Bedrock model with the given request.

        Args:
            request: The inference request containing prompt and parameters

        Returns:
            InferenceResponse containing the generated text and metadata

        Raises:
            ValueError: If the request parameters are invalid
            RuntimeError: If the model invocation fails
        """
        try:
            # Run the synchronous invoke_model in a thread pool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.invoke_model(
                    model_id=self.model_id,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    **({"stop": request.stop_sequences} if request.stop_sequences else {})
                )
            )

            return InferenceResponse(
                text=result['response_text'],
                input_tokens=result['input_tokens'],
                output_tokens=result['output_tokens'],
                latency_ms=result['latency_ms'],
                model_id=result['model_id'],
                finish_reason='stop'  # Bedrock doesn't always provide this
            )

        except ValueError as e:
            raise ValueError(f"Invalid request parameters: {str(e)}") from e
        except RuntimeError as e:
            raise RuntimeError(f"Model invocation failed: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during invocation: {str(e)}") from e

    async def invoke_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[str]:
        """Stream model response tokens as they are generated.

        Note: This is a placeholder implementation. Full streaming support
        requires using the Bedrock streaming API.

        Args:
            request: The inference request containing prompt and parameters

        Yields:
            str: Individual tokens or chunks of the generated response

        Raises:
            NotImplementedError: Streaming is not yet fully implemented
        """
        # For now, fall back to non-streaming and yield the full response
        response = await self.invoke(request)
        yield response.text

    async def list_models(self) -> list[ModelMetadata]:
        """List all available Bedrock foundation models.

        Uses bedrock.list_foundation_models() to discover available models
        in the configured region.

        Returns:
            List of ModelMetadata objects describing available models

        Raises:
            RuntimeError: If model discovery fails
        """
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.bedrock.list_foundation_models()
            )

            models = []
            for model_summary in response.get('modelSummaries', []):
                model_id = model_summary.get('modelId')
                if not model_id:
                    continue

                # Get detailed model configuration
                try:
                    config = await loop.run_in_executor(
                        None,
                        lambda mid=model_id: self.client.get_model_config(mid)
                    )

                    # Determine capabilities
                    capabilities = [ModelCapability.TEXT_GENERATION]
                    if 'CHAT' in model_summary.get('inferenceTypesSupported', []):
                        capabilities.append(ModelCapability.CHAT)
                    if model_summary.get('customizationsSupported'):
                        capabilities.append(ModelCapability.FINE_TUNABLE)

                    # Extract pricing
                    pricing_data = config.get('pricing', {})
                    pricing = ModelPricing(
                        input_price_per_1k_tokens=pricing_data.get('input', 0.0),
                        output_price_per_1k_tokens=pricing_data.get('output', 0.0),
                        currency="USD"
                    )

                    # Create metadata
                    metadata = ModelMetadata(
                        id=model_id,
                        provider=ModelProvider.BEDROCK,
                        name=config.get('model_name', model_id),
                        capabilities=capabilities,
                        status=ModelStatus.ACTIVE,
                        fine_tuning_support=bool(model_summary.get('customizationsSupported')),
                        input_modalities=config.get('input_modalities', ['text']),
                        output_modalities=config.get('output_modalities', ['text']),
                        max_tokens=4096,  # Default, varies by model
                        region=self.region,
                        pricing=pricing,
                        metadata={
                            'provider_name': config.get('provider_name'),
                            'model_arn': config.get('model_arn'),
                            'response_streaming_supported': config.get('response_streaming_supported', False),
                            'customizations_supported': config.get('customizations_supported', []),
                            'inference_types_supported': config.get('inference_types_supported', []),
                        }
                    )

                    models.append(metadata)

                except Exception as e:
                    # Log error but continue with other models
                    print(f"Warning: Failed to get config for model {model_id}: {str(e)}")
                    continue

            return models

        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', str(e))
            raise RuntimeError(f"Failed to list Bedrock models: {error_message}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during model discovery: {str(e)}") from e

    async def get_model_info(self, model_id: str) -> ModelMetadata:
        """Get metadata for a specific Bedrock model.

        Args:
            model_id: The unique identifier of the model

        Returns:
            ModelMetadata object with model details

        Raises:
            ValueError: If the model_id is not found
            RuntimeError: If metadata retrieval fails
        """
        try:
            loop = asyncio.get_running_loop()
            config = await loop.run_in_executor(
                None,
                lambda: self.client.get_model_config(model_id)
            )

            # Determine capabilities
            capabilities = [ModelCapability.TEXT_GENERATION]
            if 'CHAT' in config.get('inference_types_supported', []):
                capabilities.append(ModelCapability.CHAT)
            if config.get('customizations_supported'):
                capabilities.append(ModelCapability.FINE_TUNABLE)

            # Extract pricing
            pricing_data = config.get('pricing', {})
            pricing = ModelPricing(
                input_price_per_1k_tokens=pricing_data.get('input', 0.0),
                output_price_per_1k_tokens=pricing_data.get('output', 0.0),
                currency="USD"
            )

            # Create metadata
            metadata = ModelMetadata(
                id=model_id,
                provider=ModelProvider.BEDROCK,
                name=config.get('model_name', model_id),
                capabilities=capabilities,
                status=ModelStatus.ACTIVE,
                fine_tuning_support=bool(config.get('customizations_supported')),
                input_modalities=config.get('input_modalities', ['text']),
                output_modalities=config.get('output_modalities', ['text']),
                max_tokens=4096,  # Default, varies by model
                region=self.region,
                pricing=pricing,
                metadata={
                    'provider_name': config.get('provider_name'),
                    'model_arn': config.get('model_arn'),
                    'response_streaming_supported': config.get('response_streaming_supported', False),
                    'customizations_supported': config.get('customizations_supported', []),
                    'inference_types_supported': config.get('inference_types_supported', []),
                }
            )

            return metadata

        except ValueError as e:
            raise ValueError(f"Model not found: {model_id}") from e
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFoundException':
                raise ValueError(f"Model not found: {model_id}") from e
            error_message = e.response.get('Error', {}).get('Message', str(e))
            raise RuntimeError(f"Failed to get model info: {error_message}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error retrieving model info: {str(e)}") from e

    async def validate_connection(self) -> bool:
        """Validate connectivity to AWS Bedrock.

        Performs a lightweight check by attempting to list foundation models.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.bedrock.list_foundation_models(maxResults=1)
            )
            return True
        except Exception:
            return False

    def supports_streaming(self) -> bool:
        """Check if this adapter supports streaming responses.

        Returns:
            True if streaming is supported, False otherwise
        """
        # Bedrock supports streaming for most models, but implementation
        # is not complete yet
        return False

    def supports_fine_tuning(self, model_id: str) -> bool:
        """Check if the specified Bedrock model supports fine-tuning.

        Args:
            model_id: The unique identifier of the model

        Returns:
            True if the model supports fine-tuning, False otherwise
        """
        try:
            config = self.client.get_model_config(model_id)
            customizations = config.get('customizations_supported', [])
            return bool(customizations)
        except Exception:
            return False
