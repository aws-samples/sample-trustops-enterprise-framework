"""
Fine-tuning orchestrator for managing AWS Bedrock fine-tuning workflows.
"""
import time
from datetime import datetime
from typing import Optional, Dict, Any
from src.aws_clients.bedrock_client import BedrockClient
from src.aws_clients.s3_storage_manager import S3StorageManager
from src.data_models.data_validator import DataValidator
from src.orchestration.workflow_manager import WorkflowManager
from src.data_models.workflow import (
    FineTuningJob,
    DataValidationResult
)
from src.utils.logging_utils import get_logger
from config.aws_config import config


logger = get_logger(__name__)


class FineTuningOrchestrator:
    """
    Manage the fine-tuning pipeline with data quality validation.
    
    This orchestrator handles:
    - Training data validation using DataValidator
    - Fine-tuning job creation via Bedrock API
    - Job status polling with exponential backoff
    - Fine-tuned model metadata storage in DynamoDB
    - Cost calculation and storage
    - Training progress logging to CloudWatch
    """
    
    # Polling configuration
    INITIAL_POLL_DELAY = 30  # seconds
    MAX_POLL_DELAY = 300  # seconds (5 minutes)
    POLL_BACKOFF_MULTIPLIER = 1.5
    MAX_POLL_ATTEMPTS = 200  # ~24 hours with exponential backoff
    
    # Cost configuration (USD)
    TRAINING_COST_PER_HOUR = 10.0  # Approximate cost per training hour
    STORAGE_COST_PER_GB_MONTH = 0.023  # S3 standard storage
    
    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        s3_storage: Optional[S3StorageManager] = None,
        data_validator: Optional[DataValidator] = None,
        workflow_manager: Optional[WorkflowManager] = None
    ):
        """
        Initialize fine-tuning orchestrator.
        
        Args:
            bedrock_client: AWS Bedrock client for fine-tuning operations
            s3_storage: S3 storage manager for training data
            data_validator: Data validator for quality checks
            workflow_manager: Workflow manager for tracking
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        self.s3_storage = s3_storage or S3StorageManager()
        self.data_validator = data_validator or DataValidator()
        self.workflow_manager = workflow_manager or WorkflowManager()
    
    def validate_training_data(
        self,
        training_data_s3_uri: str
    ) -> DataValidationResult:
        """
        Validate training data meets quality standards.
        
        This method:
        1. Downloads training data from S3
        2. Validates format compliance with AWS Bedrock requirements
        3. Checks minimum sample count, prompt-completion pairs
        4. Validates character encoding
        5. Returns validation result with specific errors if validation fails
        
        Args:
            training_data_s3_uri: S3 URI of training data
            
        Returns:
            DataValidationResult with validation status and errors
            
        Raises:
            ValueError: If S3 URI is invalid
            RuntimeError: If data download fails
        """
        try:
            logger.info(f"Validating training data from {training_data_s3_uri}")
            
            # Download training data from S3
            dataset_data = self.s3_storage.download_dataset(training_data_s3_uri)
            content = dataset_data['content']
            
            # Determine format from S3 URI
            data_format = 'jsonl' if training_data_s3_uri.endswith('.jsonl') else 'json'
            
            # Parse content based on format
            import json
            if data_format == 'jsonl':
                examples = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        examples.append(json.loads(line))
            else:
                data = json.loads(content)
                examples = data if isinstance(data, list) else data.get('examples', [])
            
            # Validate training data
            validation_result = self.data_validator.validate_training_data(
                data=examples,
                data_format=data_format
            )
            
            if validation_result.is_valid:
                logger.info(
                    f"Training data validation passed: "
                    f"{validation_result.valid_examples} valid examples"
                )
            else:
                logger.warning(
                    f"Training data validation failed: "
                    f"{len(validation_result.errors)} errors found"
                )
            
            return validation_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise ValueError(f"Invalid training data format: {e}") from e
        except Exception as e:
            logger.error(f"Failed to validate training data: {e}", exc_info=True)
            raise RuntimeError(f"Training data validation failed: {e}") from e
    
    def start_fine_tuning_job(
        self,
        base_model_id: str,
        training_data_s3_uri: str,
        job_name: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None
    ) -> FineTuningJob:
        """
        Initiate fine-tuning job via AWS Bedrock.
        
        This method:
        1. Validates training data before starting job
        2. Creates fine-tuning job via Bedrock API
        3. Stores job metadata in DynamoDB
        4. Logs job creation to CloudWatch
        
        Args:
            base_model_id: Foundation model to fine-tune
            training_data_s3_uri: S3 URI of validated training data
            job_name: Unique job identifier
            hyperparameters: Optional training configuration
            workflow_id: Optional workflow identifier for tracking
            
        Returns:
            FineTuningJob object with job ID and status
            
        Raises:
            ValueError: If validation fails or parameters are invalid
            RuntimeError: If job creation fails
        """
        try:
            logger.info(
                f"Starting fine-tuning job '{job_name}' for model {base_model_id}"
            )
            
            # Step 1: Validate training data
            validation_result = self.validate_training_data(training_data_s3_uri)
            
            if not validation_result.is_valid:
                error_messages = [err.message for err in validation_result.errors]
                raise ValueError(
                    f"Training data validation failed: {'; '.join(error_messages)}"
                )
            
            # Step 2: Prepare output S3 URI
            output_data_s3_uri = (
                f"s3://{config.artifacts_bucket}/"
                f"fine-tuning/{job_name}/output"
            )
            
            # Step 3: Create fine-tuning job via Bedrock
            logger.info("Creating fine-tuning job via Bedrock API")
            job_response = self.bedrock_client.create_fine_tuning_job(
                base_model_id=base_model_id,
                training_data_s3_uri=training_data_s3_uri,
                job_name=job_name,
                output_data_s3_uri=output_data_s3_uri,
                role_arn=config.bedrock_execution_role_arn,
                hyperparameters=hyperparameters
            )
            
            # Step 4: Create FineTuningJob object
            fine_tuning_job = FineTuningJob(
                job_id=job_response['job_id'],
                job_name=job_name,
                base_model_id=base_model_id,
                training_data_s3_uri=training_data_s3_uri,
                status='in_progress',
                hyperparameters=hyperparameters or {},
                training_metrics=None,
                finetuned_model_id=None,
                created_at=datetime.utcnow(),
                completed_at=None,
                error_message=None
            )
            
            # Step 5: Store job metadata in DynamoDB via workflow manager
            if workflow_id:
                self.workflow_manager.update_workflow_status(
                    workflow_id,
                    'running',
                    {
                        'stage': 'fine_tuning_in_progress',
                        'job_id': fine_tuning_job.job_id,
                        'job_name': job_name
                    }
                )
            
            # Step 6: Log job creation to CloudWatch
            logger.info(
                f"Fine-tuning job created successfully: {fine_tuning_job.job_id}"
            )
            
            return fine_tuning_job
            
        except ValueError as e:
            logger.error(f"Invalid parameters for fine-tuning job: {e}")
            if workflow_id:
                self.workflow_manager.update_workflow_status(
                    workflow_id,
                    'failed',
                    {'error': str(e), 'stage': 'fine_tuning_validation'}
                )
            raise
        except Exception as e:
            logger.error(f"Failed to start fine-tuning job: {e}", exc_info=True)
            if workflow_id:
                self.workflow_manager.update_workflow_status(
                    workflow_id,
                    'failed',
                    {'error': str(e), 'stage': 'fine_tuning_creation'}
                )
            raise RuntimeError(f"Failed to start fine-tuning job: {e}") from e
    
    def poll_job_status(
        self,
        job_id: str,
        workflow_id: Optional[str] = None
    ) -> FineTuningJob:
        """
        Check status of running fine-tuning job with exponential backoff.
        
        This method:
        1. Polls job status from Bedrock API
        2. Uses exponential backoff between polls
        3. Logs progress to CloudWatch
        4. Updates workflow status in DynamoDB
        5. Calculates and stores fine-tuning costs when complete
        6. Stores fine-tuned model metadata
        
        Args:
            job_id: Bedrock fine-tuning job identifier
            workflow_id: Optional workflow identifier for tracking
            
        Returns:
            FineTuningJob with current state and metrics
            
        Raises:
            ValueError: If job_id is invalid
            RuntimeError: If polling fails or job fails
            TimeoutError: If max poll attempts exceeded
        """
        try:
            logger.info(f"Polling status for fine-tuning job {job_id}")
            
            poll_delay = self.INITIAL_POLL_DELAY
            attempt = 0
            
            while attempt < self.MAX_POLL_ATTEMPTS:
                attempt += 1
                
                # Get job status from Bedrock
                status_response = self.bedrock_client.get_fine_tuning_job_status(job_id)
                
                status = status_response['status']
                logger.info(
                    f"Fine-tuning job {job_id} status: {status} "
                    f"(attempt {attempt}/{self.MAX_POLL_ATTEMPTS})"
                )
                
                # Update workflow status
                if workflow_id:
                    self.workflow_manager.update_workflow_status(
                        workflow_id,
                        'running',
                        {
                            'stage': 'fine_tuning_in_progress',
                            'job_status': status,
                            'poll_attempt': attempt
                        }
                    )
                
                # Check if job is complete
                if status == 'Completed':
                    logger.info(f"Fine-tuning job {job_id} completed successfully")
                    
                    # Calculate costs
                    training_cost = self._calculate_training_cost(
                        status_response.get('training_metrics', {})
                    )
                    storage_cost = self._calculate_storage_cost()
                    total_cost = training_cost + storage_cost
                    
                    # Create completed FineTuningJob
                    fine_tuning_job = FineTuningJob(
                        job_id=job_id,
                        job_name=f"job-{job_id.split('/')[-1]}",
                        base_model_id='',  # Not in status response
                        training_data_s3_uri='',  # Not in status response
                        status='completed',
                        hyperparameters={},
                        training_metrics=status_response.get(
                            'training_metrics'
                        ),
                        finetuned_model_id=status_response.get(
                            'custom_model_arn'
                        ),
                        created_at=datetime.utcnow(),
                        completed_at=datetime.utcnow(),
                        error_message=None
                    )
                    
                    # Store metadata and costs
                    if workflow_id:
                        self.workflow_manager.update_workflow_status(
                            workflow_id,
                            'completed',
                            {
                                'finetuned_model_id': fine_tuning_job.finetuned_model_id,
                                'training_cost': training_cost,
                                'storage_cost': storage_cost,
                                'total_cost': total_cost,
                                'training_metrics': fine_tuning_job.training_metrics
                            }
                        )
                    
                    logger.info(
                        f"Fine-tuning completed. "
                        f"Model: {fine_tuning_job.finetuned_model_id}, "
                        f"Total cost: ${total_cost:.2f}"
                    )
                    
                    return fine_tuning_job
                
                elif status == 'Failed':
                    error_message = status_response.get(
                        'failure_message', 'Unknown error'
                    )
                    logger.error(
                        f"Fine-tuning job {job_id} failed: {error_message}"
                    )
                    
                    # Create failed FineTuningJob
                    fine_tuning_job = FineTuningJob(
                        job_id=job_id,
                        job_name=f"job-{job_id.split('/')[-1]}",
                        base_model_id='',
                        training_data_s3_uri='',
                        status='failed',
                        hyperparameters={},
                        training_metrics=None,
                        finetuned_model_id=None,
                        created_at=datetime.utcnow(),
                        completed_at=datetime.utcnow(),
                        error_message=error_message
                    )
                    
                    if workflow_id:
                        self.workflow_manager.update_workflow_status(
                            workflow_id,
                            'failed',
                            {'error': error_message, 'stage': 'fine_tuning_failed'}
                        )
                    
                    raise RuntimeError(f"Fine-tuning job failed: {error_message}")
                
                elif status in ['Stopped', 'Stopping']:
                    logger.warning(f"Fine-tuning job {job_id} was stopped")
                    
                    fine_tuning_job = FineTuningJob(
                        job_id=job_id,
                        job_name=f"job-{job_id.split('/')[-1]}",
                        base_model_id='',
                        training_data_s3_uri='',
                        status='failed',
                        hyperparameters={},
                        training_metrics=None,
                        finetuned_model_id=None,
                        created_at=datetime.utcnow(),
                        completed_at=datetime.utcnow(),
                        error_message='Job was stopped'
                    )
                    
                    if workflow_id:
                        self.workflow_manager.update_workflow_status(
                            workflow_id,
                            'failed',
                            {'error': 'Job was stopped', 'stage': 'fine_tuning_stopped'}
                        )
                    
                    raise RuntimeError("Fine-tuning job was stopped")
                
                # Job still in progress, wait before next poll
                logger.info(f"Waiting {poll_delay} seconds before next poll")
                time.sleep(poll_delay)
                
                # Exponential backoff with max delay cap
                poll_delay = min(
                    poll_delay * self.POLL_BACKOFF_MULTIPLIER,
                    self.MAX_POLL_DELAY
                )
            
            # Max attempts exceeded
            logger.error(
                f"Max poll attempts ({self.MAX_POLL_ATTEMPTS}) "
                f"exceeded for job {job_id}"
            )
            if workflow_id:
                self.workflow_manager.update_workflow_status(
                    workflow_id,
                    'failed',
                    {
                        'error': 'Polling timeout exceeded',
                        'stage': 'fine_tuning_timeout'
                    }
                )
            
            raise TimeoutError(
                f"Fine-tuning job polling timeout after "
                f"{self.MAX_POLL_ATTEMPTS} attempts"
            )
            
        except (ValueError, RuntimeError, TimeoutError):
            raise
        except Exception as e:
            logger.error(f"Failed to poll job status: {e}", exc_info=True)
            if workflow_id:
                self.workflow_manager.update_workflow_status(
                    workflow_id,
                    'failed',
                    {'error': str(e), 'stage': 'fine_tuning_polling_error'}
                )
            raise RuntimeError(f"Failed to poll job status: {e}") from e
    
    def _calculate_training_cost(
        self,
        training_metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate training cost based on metrics.
        
        Args:
            training_metrics: Training metrics from Bedrock
            
        Returns:
            Estimated training cost in USD
        """
        # Estimate training time from metrics or use default
        # In production, this would use actual training duration from metrics
        training_hours = training_metrics.get('trainingTimeInHours', 2.0)
        
        return training_hours * self.TRAINING_COST_PER_HOUR
    
    def _calculate_storage_cost(self) -> float:
        """
        Calculate storage cost for training data and model artifacts.
        
        Returns:
            Estimated monthly storage cost in USD
        """
        # Estimate storage size (in production, get actual size from S3)
        # Assuming ~1GB for training data and model artifacts
        storage_gb = 1.0
        
        return storage_gb * self.STORAGE_COST_PER_GB_MONTH
