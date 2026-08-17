"""SageMaker adapter for AWS SageMaker deployed models.

This module implements the BaseModelAdapter interface for AWS SageMaker,
providing model invocation and endpoint management capabilities.

Requirements: 1.2, 1.6
"""

import asyncio
import json
import time
from typing import AsyncIterator, Optional
from botocore.exceptions import ClientError
import boto3

from src.adapters.base_adapter import (
    BaseModelAdapter,
    InferenceRequest,
    InferenceResponse,
)
from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
    ModelPricing,
)
from config.aws_config import config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SageMakerAdapter(BaseModelAdapter):
    """Adapter for AWS SageMaker deployed models.

    This adapter provides integration with SageMaker endpoints, implementing
    endpoint validation via describe_endpoint() and invocation via
    invoke_endpoint().

    Requirements:
        - 1.2: SageMaker endpoint validation and registration
        - 1.6: Unified inference interface for SageMaker models
    """

    def __init__(self, endpoint_name: str, region: Optional[str] = None):
        """Initialize the SageMaker adapter.

        Args:
            endpoint_name: The SageMaker endpoint name to use for invocations
            region: AWS region (defaults to config.region)
        """
        self.endpoint_name = endpoint_name
        self.region = region or config.region
        
        # Initialize boto3 clients
        session_kwargs = config.get_boto3_session_kwargs()
        if region:
            session_kwargs['region_name'] = region
        
        session = boto3.Session(**session_kwargs)
        self.sagemaker_runtime = session.client('sagemaker-runtime')
        self.sagemaker = session.client('sagemaker')

    async def invoke(self, request: InferenceRequest) -> InferenceResponse:
        """Invoke a SageMaker endpoint with the given request.

        Args:
            request: The inference request containing prompt and parameters

        Returns:
            InferenceResponse containing the generated text and metadata

        Raises:
            ValueError: If the request parameters are invalid
            RuntimeError: If the model invocation fails
        """
        try:
            start_time = time.time()
            
            # Format request body for SageMaker endpoint
            # Using a generic format that works with most text generation models
            body = {
                'inputs': request.prompt,
                'parameters': {
                    'max_new_tokens': request.max_tokens,
                    'temperature': request.temperature,
                    'top_p': request.top_p,
                }
            }
            
            if request.stop_sequences:
                body['parameters']['stop'] = request.stop_sequences
            
            # Run the synchronous invoke_endpoint in a thread pool
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType='application/json',
                    Accept='application/json',
                    Body=json.dumps(body)
                )
            )
            
            # Parse response
            response_body = json.loads(response['Body'].read().decode('utf-8'))
            
            # Extract response text (handle different response formats)
            if isinstance(response_body, dict):
                # Try common response formats
                response_text = (
                    response_body.get('generated_text') or
                    response_body.get('outputs') or
                    response_body.get('predictions') or
                    response_body.get('text') or
                    str(response_body)
                )
                
                # Handle list responses
                if isinstance(response_text, list) and response_text:
                    response_text = response_text[0]
                    if isinstance(response_text, dict):
                        response_text = response_text.get('generated_text', str(response_text))
            else:
                response_text = str(response_body)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimate token counts (simple approximation)
            input_tokens = self._estimate_tokens(request.prompt)
            output_tokens = self._estimate_tokens(response_text)
            
            return InferenceResponse(
                text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                model_id=self.endpoint_name,
                finish_reason='stop'
            )

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == 'ValidationError':
                raise ValueError(f"Invalid request parameters: {error_message}") from e
            elif error_code == 'ModelError':
                raise RuntimeError(f"Model invocation failed: {error_message}") from e
            elif error_code == 'ServiceUnavailable':
                raise RuntimeError(f"SageMaker endpoint unavailable: {error_message}") from e
            else:
                raise RuntimeError(f"SageMaker API error: {error_message}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse SageMaker response: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during invocation: {str(e)}") from e

    async def invoke_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[str]:
        """Stream model response tokens as they are generated.

        Note: This is a placeholder implementation. Full streaming support
        requires using the SageMaker streaming API if supported by the endpoint.

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
        """List all available SageMaker endpoints.

        Returns:
            List of ModelMetadata objects describing available endpoints

        Raises:
            RuntimeError: If endpoint discovery fails
        """
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sagemaker.list_endpoints(
                    StatusEquals='InService',
                    MaxResults=100
                )
            )

            models = []
            for endpoint_summary in response.get('Endpoints', []):
                endpoint_name = endpoint_summary.get('EndpointName')
                if not endpoint_name:
                    continue

                try:
                    # Get detailed endpoint information
                    endpoint_info = await self._get_endpoint_info(endpoint_name)
                    
                    # Create metadata
                    metadata = ModelMetadata(
                        id=endpoint_name,
                        provider=ModelProvider.SAGEMAKER,
                        name=endpoint_name,
                        capabilities=[ModelCapability.TEXT_GENERATION],
                        status=self._map_endpoint_status(endpoint_summary.get('EndpointStatus')),
                        fine_tuning_support=False,  # SageMaker endpoints are already deployed models
                        input_modalities=['text'],
                        output_modalities=['text'],
                        max_tokens=4096,  # Default, varies by model
                        region=self.region,
                        pricing=ModelPricing(
                            input_price_per_1k_tokens=0.0,  # Pricing varies by instance type
                            output_price_per_1k_tokens=0.0,
                            currency="USD"
                        ),
                        metadata={
                            'endpoint_arn': endpoint_summary.get('EndpointArn'),
                            'creation_time': endpoint_summary.get('CreationTime').isoformat() if endpoint_summary.get('CreationTime') else None,
                            'last_modified_time': endpoint_summary.get('LastModifiedTime').isoformat() if endpoint_summary.get('LastModifiedTime') else None,
                            'endpoint_config_name': endpoint_info.get('endpoint_config_name'),
                            'instance_type': endpoint_info.get('instance_type'),
                        }
                    )

                    models.append(metadata)

                except Exception as e:
                    # Log error but continue with other endpoints
                    print(f"Warning: Failed to get info for endpoint {endpoint_name}: {str(e)}")
                    continue

            return models

        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', str(e))
            raise RuntimeError(f"Failed to list SageMaker endpoints: {error_message}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during endpoint discovery: {str(e)}") from e

    async def get_model_info(self, model_id: str) -> ModelMetadata:
        """Get metadata for a specific SageMaker endpoint.

        Args:
            model_id: The endpoint name

        Returns:
            ModelMetadata object with endpoint details

        Raises:
            ValueError: If the endpoint is not found
            RuntimeError: If metadata retrieval fails
        """
        try:
            endpoint_info = await self._get_endpoint_info(model_id)
            
            # Create metadata
            metadata = ModelMetadata(
                id=model_id,
                provider=ModelProvider.SAGEMAKER,
                name=model_id,
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=self._map_endpoint_status(endpoint_info.get('endpoint_status')),
                fine_tuning_support=False,
                input_modalities=['text'],
                output_modalities=['text'],
                max_tokens=4096,
                region=self.region,
                pricing=ModelPricing(
                    input_price_per_1k_tokens=0.0,
                    output_price_per_1k_tokens=0.0,
                    currency="USD"
                ),
                metadata={
                    'endpoint_arn': endpoint_info.get('endpoint_arn'),
                    'creation_time': endpoint_info.get('creation_time'),
                    'last_modified_time': endpoint_info.get('last_modified_time'),
                    'endpoint_config_name': endpoint_info.get('endpoint_config_name'),
                    'instance_type': endpoint_info.get('instance_type'),
                }
            )

            return metadata

        except ValueError:
            raise
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ValidationException':
                raise ValueError(f"Endpoint not found: {model_id}") from e
            error_message = e.response.get('Error', {}).get('Message', str(e))
            raise RuntimeError(f"Failed to get endpoint info: {error_message}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error retrieving endpoint info: {str(e)}") from e

    async def validate_connection(self) -> bool:
        """Validate connectivity to the SageMaker endpoint.

        Performs a lightweight check by calling describe_endpoint() to verify
        the endpoint exists and is in service.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sagemaker.describe_endpoint(
                    EndpointName=self.endpoint_name
                )
            )
            
            # Check if endpoint is in service
            status = response.get('EndpointStatus')
            return status == 'InService'
            
        except Exception:
            return False

    def supports_streaming(self) -> bool:
        """Check if this adapter supports streaming responses.

        Returns:
            True if streaming is supported, False otherwise
        """
        # SageMaker supports streaming for some endpoints, but implementation
        # is not complete yet
        return False

    def supports_fine_tuning(self, model_id: str) -> bool:
        """Check if the specified endpoint supports fine-tuning.

        Args:
            model_id: The endpoint name

        Returns:
            False - SageMaker endpoints are already deployed models
        """
        # SageMaker endpoints are already deployed/fine-tuned models
        # Fine-tuning would be done through SageMaker training jobs, not endpoints
        return False

    async def _get_endpoint_info(self, endpoint_name: str) -> dict:
        """Get detailed information about a SageMaker endpoint.

        Args:
            endpoint_name: The endpoint name

        Returns:
            Dictionary containing endpoint details

        Raises:
            ValueError: If endpoint is not found
            RuntimeError: If API call fails
        """
        try:
            loop = asyncio.get_running_loop()
            
            # Get endpoint description
            endpoint_response = await loop.run_in_executor(
                None,
                lambda: self.sagemaker.describe_endpoint(
                    EndpointName=endpoint_name
                )
            )
            
            endpoint_config_name = endpoint_response.get('EndpointConfigName')
            
            # Get endpoint configuration for instance type info
            instance_type = None
            if endpoint_config_name:
                try:
                    config_response = await loop.run_in_executor(
                        None,
                        lambda: self.sagemaker.describe_endpoint_config(
                            EndpointConfigName=endpoint_config_name
                        )
                    )
                    
                    production_variants = config_response.get('ProductionVariants', [])
                    if production_variants:
                        instance_type = production_variants[0].get('InstanceType')
                        
                except Exception as e:
                    # If we can't get config, continue without instance type
                    logger.debug(
                        "Could not describe endpoint config %s: %s",
                        endpoint_config_name, e
                    )
            
            return {
                'endpoint_name': endpoint_name,
                'endpoint_arn': endpoint_response.get('EndpointArn'),
                'endpoint_status': endpoint_response.get('EndpointStatus'),
                'creation_time': endpoint_response.get('CreationTime').isoformat() if endpoint_response.get('CreationTime') else None,
                'last_modified_time': endpoint_response.get('LastModifiedTime').isoformat() if endpoint_response.get('LastModifiedTime') else None,
                'endpoint_config_name': endpoint_config_name,
                'instance_type': instance_type,
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ValidationException':
                raise ValueError(f"Endpoint not found: {endpoint_name}") from e
            raise RuntimeError(f"Failed to describe endpoint: {str(e)}") from e

    def _map_endpoint_status(self, status: str) -> ModelStatus:
        """Map SageMaker endpoint status to ModelStatus enum.

        Args:
            status: SageMaker endpoint status

        Returns:
            Corresponding ModelStatus value
        """
        status_map = {
            'InService': ModelStatus.ACTIVE,
            'Creating': ModelStatus.PROVISIONING,
            'Updating': ModelStatus.PROVISIONING,
            'SystemUpdating': ModelStatus.PROVISIONING,
            'RollingBack': ModelStatus.PROVISIONING,
            'Failed': ModelStatus.FAILED,
            'OutOfService': ModelStatus.INACTIVE,
            'Deleting': ModelStatus.INACTIVE,
        }
        
        return status_map.get(status, ModelStatus.INACTIVE)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        This is a simple approximation. For production use, consider using
        a proper tokenizer for the specific model.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token on average
        return max(1, len(text) // 4)
