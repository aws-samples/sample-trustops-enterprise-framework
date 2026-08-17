"""
AWS Bedrock client wrapper for model inference and fine-tuning.
"""
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

from config.aws_config import config


class BedrockClient:
    """Client wrapper for AWS Bedrock API operations."""
    
    def __init__(self, region: Optional[str] = None):
        """
        Initialize Bedrock client.
        
        Args:
            region: AWS region (defaults to config.region)
        """
        self.region = region or config.region
        session_kwargs = config.get_boto3_session_kwargs()
        if region:
            session_kwargs['region_name'] = region
        
        session = boto3.Session(**session_kwargs)
        self.bedrock_runtime = session.client('bedrock-runtime')
        self.bedrock = session.client('bedrock')
        
        # Model pricing configuration (USD per 1000 tokens)
        self.pricing = {
            'anthropic.claude-v2': {
                'input': config.claude_v2_input_cost,
                'output': config.claude_v2_output_cost
            },
            'anthropic.claude-v2:1': {
                'input': 0.008,
                'output': 0.024
            },
            'anthropic.claude-instant-v1': {
                'input': 0.0008,
                'output': 0.0024
            },
            'anthropic.claude-3-haiku-20240307-v1:0': {
                'input': 0.00025,
                'output': 0.00125
            },
            'anthropic.claude-3-sonnet-20240229-v1:0': {
                'input': 0.003,
                'output': 0.015
            },
            'anthropic.claude-3-opus-20240229-v1:0': {
                'input': 0.015,
                'output': 0.075
            },
            'amazon.titan-text-express-v1': {
                'input': 0.0002,
                'output': 0.0006
            },
            'amazon.titan-text-lite-v1': {
                'input': 0.00015,
                'output': 0.0002
            },
            'meta.llama2-13b-chat-v1': {
                'input': 0.00075,
                'output': 0.001
            },
            'meta.llama2-70b-chat-v1': {
                'input': 0.00195,
                'output': 0.00256
            },
            # Amazon Nova family (per 1000 tokens)
            'amazon.nova-micro-v1:0': {
                'input': 0.000035,
                'output': 0.00014
            },
            'amazon.nova-lite-v1:0': {
                'input': 0.00006,
                'output': 0.00024
            },
            'amazon.nova-pro-v1:0': {
                'input': 0.0008,
                'output': 0.0032
            },
            # Current Claude models (per 1000 tokens)
            'anthropic.claude-haiku-4-5-20251001-v1:0': {
                'input': 0.001,
                'output': 0.005
            },
            'anthropic.claude-sonnet-4-5-20250929-v1:0': {
                'input': 0.003,
                'output': 0.015
            }
        }
    
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """
        Retrieve model configuration from AWS Bedrock.
        
        Args:
            model_id: AWS Bedrock model identifier
            
        Returns:
            Dictionary containing model configuration
            
        Raises:
            ValueError: If model_id is invalid
            ClientError: If AWS API call fails
        """
        try:
            response = self.bedrock.get_foundation_model(modelIdentifier=model_id)
            
            model_details = response.get('modelDetails', {})
            
            return {
                'model_id': model_id,
                'model_arn': model_details.get('modelArn'),
                'model_name': model_details.get('modelName'),
                'provider_name': model_details.get('providerName'),
                'input_modalities': model_details.get('inputModalities', []),
                'output_modalities': model_details.get('outputModalities', []),
                'response_streaming_supported': model_details.get('responseStreamingSupported', False),
                'customizations_supported': model_details.get('customizationsSupported', []),
                'inference_types_supported': model_details.get('inferenceTypesSupported', []),
                'pricing': self.pricing.get(model_id, {'input': 0.0, 'output': 0.0})
            }
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFoundException':
                raise ValueError(f"Invalid model ID: {model_id}") from e
            raise
    
    def invoke_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke a model for inference with error handling.
        
        Args:
            model_id: AWS Bedrock model identifier
            prompt: Input prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing:
                - response_text: Generated text
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - latency_ms: Inference latency in milliseconds
                - cost: Estimated cost in USD
                - model_id: Model identifier used
                
        Raises:
            ValueError: If parameters are invalid
            ClientError: If AWS API call fails
        """
        start_time = time.time()
        
        try:
            # Modern models use the unified Converse API. This covers Amazon Nova,
            # Claude 3 and all newer Claude versions, and other current chat models.
            # Legacy text-completion models (claude-v2, claude-instant) fall through
            # to the provider-specific invoke_model bodies below.
            if self._uses_converse_api(model_id):
                return self._invoke_with_converse_api(
                    model_id, prompt, max_tokens, temperature, top_p, start_time, **kwargs
                )

            # Format request body based on model provider (legacy models)
            if model_id.startswith('anthropic.claude'):
                body = {
                    'prompt': f"\n\nHuman: {prompt}\n\nAssistant:",
                    'max_tokens_to_sample': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    **kwargs
                }
            elif model_id.startswith('amazon.titan'):
                body = {
                    'inputText': prompt,
                    'textGenerationConfig': {
                        'maxTokenCount': max_tokens,
                        'temperature': temperature,
                        'topP': top_p,
                        **kwargs
                    }
                }
            elif model_id.startswith('meta.llama'):
                body = {
                    'prompt': prompt,
                    'max_gen_len': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    **kwargs
                }
            else:
                raise ValueError(f"Unsupported model provider for model: {model_id}")
            
            # Invoke the model
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract response text based on model provider
            if model_id.startswith('anthropic.claude'):
                response_text = response_body.get('completion', '')
                input_tokens = self._count_tokens(prompt)
                output_tokens = self._count_tokens(response_text)
            elif model_id.startswith('amazon.titan'):
                results = response_body.get('results', [{}])
                response_text = results[0].get('outputText', '') if results else ''
                input_tokens = response_body.get('inputTextTokenCount', self._count_tokens(prompt))
                output_tokens = response_body.get('results', [{}])[0].get('tokenCount', self._count_tokens(response_text))
            elif model_id.startswith('meta.llama'):
                response_text = response_body.get('generation', '')
                input_tokens = response_body.get('prompt_token_count', self._count_tokens(prompt))
                output_tokens = response_body.get('generation_token_count', self._count_tokens(response_text))
            else:
                response_text = str(response_body)
                input_tokens = self._count_tokens(prompt)
                output_tokens = self._count_tokens(response_text)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate cost
            cost = self.calculate_cost(model_id, input_tokens, output_tokens)
            
            return {
                'response_text': response_text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'latency_ms': latency_ms,
                'cost': cost,
                'model_id': model_id
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')
            
            # Provide specific error messages
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid request parameters: {error_message}") from e
            elif error_code == 'ResourceNotFoundException':
                raise ValueError(f"Model not found: {model_id}") from e
            elif error_code == 'ThrottlingException':
                raise RuntimeError(f"Rate limit exceeded for model {model_id}. Please retry.") from e
            elif error_code == 'ServiceQuotaExceededException':
                raise RuntimeError(f"Service quota exceeded for model {model_id}") from e
            else:
                raise RuntimeError(f"Bedrock API error: {error_message}") from e
    
    @staticmethod
    def _uses_converse_api(model_id: str) -> bool:
        """Determine whether a model should be invoked via the Converse API.

        The Converse API provides a single request/response schema across
        modern chat models, so we prefer it for everything except the two
        legacy Claude text-completion models that predate it.

        Args:
            model_id: AWS Bedrock model identifier

        Returns:
            True if the model should use the Converse API.
        """
        # Legacy text-completion models that must use the old invoke_model body.
        legacy_prefixes = (
            'anthropic.claude-v2',
            'anthropic.claude-instant',
        )
        if any(model_id.startswith(p) for p in legacy_prefixes):
            return False

        # Modern chat/text models supported by Converse.
        converse_prefixes = (
            'anthropic.claude',   # claude-3 and all newer versions
            'amazon.nova',        # Nova micro/lite/pro
            'meta.llama3',
            'mistral.',
            'cohere.command-r',
        )
        return any(model_id.startswith(p) for p in converse_prefixes)

    def _invoke_with_converse_api(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        start_time: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke Claude 3+ models using the Converse API.
        
        Args:
            model_id: AWS Bedrock model identifier
            prompt: Input prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            start_time: Request start time for latency calculation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with response_text, tokens, latency, cost, model_id
        """
        try:
            # Use Converse API for Claude 3+ models
            response = self.bedrock_runtime.converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": top_p
                }
            )
            
            # Extract response text
            output = response.get('output', {})
            message = output.get('message', {})
            content = message.get('content', [])
            response_text = content[0].get('text', '') if content else ''
            
            # Get token usage
            usage = response.get('usage', {})
            input_tokens = usage.get('inputTokens', self._count_tokens(prompt))
            output_tokens = usage.get('outputTokens', self._count_tokens(response_text))
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate cost
            cost = self.calculate_cost(model_id, input_tokens, output_tokens)
            
            return {
                'response_text': response_text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'latency_ms': latency_ms,
                'cost': cost,
                'model_id': model_id
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')
            
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid request parameters: {error_message}") from e
            elif error_code == 'ResourceNotFoundException':
                raise ValueError(f"Model not found: {model_id}") from e
            elif error_code == 'ThrottlingException':
                raise RuntimeError(f"Rate limit exceeded for model {model_id}. Please retry.") from e
            elif error_code == 'ServiceQuotaExceededException':
                raise RuntimeError(f"Service quota exceeded for model {model_id}") from e
            else:
                raise RuntimeError(f"Bedrock API error: {error_message}") from e
    
    def create_fine_tuning_job(
        self,
        base_model_id: str,
        training_data_s3_uri: str,
        job_name: str,
        output_data_s3_uri: str,
        role_arn: str,
        hyperparameters: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a fine-tuning job via AWS Bedrock.
        
        Args:
            base_model_id: Foundation model identifier to fine-tune
            training_data_s3_uri: S3 URI of training data
            job_name: Unique job identifier
            output_data_s3_uri: S3 URI for output artifacts
            role_arn: IAM role ARN for Bedrock to access S3
            hyperparameters: Optional training hyperparameters
            
        Returns:
            Dictionary containing:
                - job_id: Fine-tuning job identifier
                - job_arn: Job ARN
                - status: Initial job status
                - created_at: Job creation timestamp
                
        Raises:
            ValueError: If parameters are invalid
            ClientError: If AWS API call fails
        """
        try:
            # Default hyperparameters if not provided
            if hyperparameters is None:
                hyperparameters = {
                    'epochCount': '3',
                    'batchSize': '1',
                    'learningRate': '0.00001'
                }
            
            # Create fine-tuning job
            response = self.bedrock.create_model_customization_job(
                jobName=job_name,
                customModelName=f"{job_name}-model",
                roleArn=role_arn,
                baseModelIdentifier=base_model_id,
                trainingDataConfig={
                    's3Uri': training_data_s3_uri
                },
                outputDataConfig={
                    's3Uri': output_data_s3_uri
                },
                hyperParameters=hyperparameters
            )
            
            return {
                'job_id': response['jobArn'],
                'job_arn': response['jobArn'],
                'status': 'InProgress',
                'created_at': datetime.utcnow().isoformat()
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')
            
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid fine-tuning parameters: {error_message}") from e
            elif error_code == 'ResourceNotFoundException':
                raise ValueError(f"Base model not found: {base_model_id}") from e
            elif error_code == 'AccessDeniedException':
                raise RuntimeError(f"Access denied. Check IAM role permissions: {error_message}") from e
            else:
                raise RuntimeError(f"Failed to create fine-tuning job: {error_message}") from e
    
    def get_fine_tuning_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a fine-tuning job.
        
        Args:
            job_id: Fine-tuning job identifier (ARN)
            
        Returns:
            Dictionary containing:
                - job_id: Job identifier
                - status: Current status (InProgress, Completed, Failed, Stopping, Stopped)
                - custom_model_arn: ARN of fine-tuned model (if completed)
                - failure_message: Error message (if failed)
                - training_metrics: Training metrics (if available)
                
        Raises:
            ValueError: If job_id is invalid
            ClientError: If AWS API call fails
        """
        try:
            response = self.bedrock.get_model_customization_job(
                jobIdentifier=job_id
            )
            
            status_map = {
                'InProgress': 'InProgress',
                'Completed': 'Completed',
                'Failed': 'Failed',
                'Stopping': 'Stopping',
                'Stopped': 'Stopped'
            }
            
            status = response.get('status', 'Unknown')
            
            result = {
                'job_id': job_id,
                'status': status_map.get(status, status),
                'custom_model_arn': response.get('outputModelArn'),
                'failure_message': response.get('failureMessage'),
                'training_metrics': response.get('trainingMetrics', {})
            }
            
            return result
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')
            
            if error_code == 'ResourceNotFoundException':
                raise ValueError(f"Fine-tuning job not found: {job_id}") from e
            else:
                raise RuntimeError(f"Failed to get job status: {error_message}") from e
    
    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate the cost of a model inference.
        
        Args:
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        pricing = self.pricing.get(model_id, {'input': 0.0, 'output': 0.0})
        
        input_cost = (input_tokens / 1000.0) * pricing['input']
        output_cost = (output_tokens / 1000.0) * pricing['output']
        
        return input_cost + output_cost
    
    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        This is a simple approximation. For production use, consider using
        a proper tokenizer for the specific model.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token on average
        # This is a rough estimate and varies by model and language
        return max(1, len(text) // 4)
