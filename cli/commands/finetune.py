"""Fine-tuning command."""
import click
import uuid
import time
from cli.utils.output import (
    print_header, print_success, print_error, print_info, 
    print_progress, print_warning
)
from cli.utils.validation import validate_s3_uri, validate_model_id
from cli.config import update_env_from_config


@click.command()
@click.option(
    '--base-model-id',
    required=True,
    help='Base model to fine-tune (e.g., anthropic.claude-v2)'
)
@click.option(
    '--training-data',
    required=True,
    help='S3 URI of training data (s3://bucket/path/training.jsonl)'
)
@click.option(
    '--job-name',
    help='Fine-tuning job name (auto-generated if not provided)'
)
@click.option(
    '--workflow-id',
    help='Optional workflow ID (auto-generated if not provided)'
)
@click.option(
    '--epochs',
    type=int,
    default=3,
    help='Number of training epochs'
)
@click.option(
    '--learning-rate',
    type=float,
    default=0.0001,
    help='Learning rate for training'
)
@click.option(
    '--batch-size',
    type=int,
    default=8,
    help='Training batch size'
)
@click.option(
    '--wait/--no-wait',
    default=True,
    help='Wait for fine-tuning to complete'
)
@click.option(
    '--poll-interval',
    type=int,
    default=60,
    help='Polling interval in seconds (when --wait is enabled)'
)
@click.pass_context
def finetune(ctx, base_model_id, training_data, job_name, workflow_id,
             epochs, learning_rate, batch_size, wait, poll_interval):
    """
    Start a fine-tuning job for a foundation model.
    
    This command validates training data, creates a fine-tuning job via
    AWS Bedrock, and optionally waits for completion while displaying
    progress updates.
    
    Example:
    
        trustops finetune --base-model-id anthropic.claude-v2 \\
            --training-data s3://my-bucket/training.jsonl \\
            --epochs 5
    """
    print_header("Fine-Tuning")
    
    # Update environment from config
    config = ctx.obj.get('config', {})
    update_env_from_config(config)
    
    # Validate inputs
    try:
        validate_model_id(base_model_id)
        validate_s3_uri(training_data)
    except ValueError as e:
        print_error(f"Validation error: {e}")
        raise click.Abort()
    
    # Generate IDs if not provided
    if not job_name:
        job_name = f"finetune-{uuid.uuid4().hex[:8]}"
    if not workflow_id:
        workflow_id = f"finetune-{uuid.uuid4().hex[:8]}"
    
    print_info(f"Base Model: {base_model_id}")
    print_info(f"Training Data: {training_data}")
    print_info(f"Job Name: {job_name}")
    print_info(f"Workflow ID: {workflow_id}")
    print_info(f"Hyperparameters:")
    print_info(f"  Epochs: {epochs}")
    print_info(f"  Learning Rate: {learning_rate}")
    print_info(f"  Batch Size: {batch_size}")
    
    try:
        # Import here to avoid circular dependencies
        from src.orchestration.fine_tuning_orchestrator import (
            FineTuningOrchestrator
        )
        from src.orchestration.workflow_manager import WorkflowManager
        
        # Create workflow
        print_progress("Creating workflow...")
        workflow_manager = WorkflowManager()
        workflow_manager.create_workflow(
            workflow_type='fine-tuning',
            configuration={
                'base_model_id': base_model_id,
                'training_data_s3_uri': training_data,
                'job_name': job_name,
                'hyperparameters': {
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                }
            }
        )
        
        # Start fine-tuning job
        print_progress("Validating training data...")
        orchestrator = FineTuningOrchestrator()
        
        hyperparameters = {
            'epochCount': str(epochs),
            'learningRate': str(learning_rate),
            'batchSize': str(batch_size)
        }
        
        job = orchestrator.start_fine_tuning_job(
            base_model_id=base_model_id,
            training_data_s3_uri=training_data,
            job_name=job_name,
            hyperparameters=hyperparameters,
            workflow_id=workflow_id
        )
        
        print_success(f"Fine-tuning job started: {job.job_id}")
        
        if wait:
            print_info("\nWaiting for fine-tuning to complete...")
            print_info(
                "This may take several hours. "
                "You can press Ctrl+C to stop waiting."
            )
            print_info(
                f"Use 'trustops status {workflow_id}' "
                "to check progress later.\n"
            )
            
            try:
                completed_job = orchestrator.poll_job_status(
                    job_id=job.job_id,
                    workflow_id=workflow_id
                )
                
                print_success("\nFine-tuning completed!")
                print_info(f"Fine-tuned Model ID: "
                          f"{completed_job.finetuned_model_id}")
                
                if completed_job.training_metrics:
                    print_info("\nTraining Metrics:")
                    for key, value in completed_job.training_metrics.items():
                        print_info(f"  {key}: {value}")
                
            except KeyboardInterrupt:
                print_warning(
                    "\nStopped waiting. "
                    "Fine-tuning continues in the background."
                )
                print_info(
                    f"Use 'trustops status {workflow_id}' "
                    "to check progress."
                )
        else:
            print_info(
                f"\nFine-tuning job is running in the background."
            )
            print_info(
                f"Use 'trustops status {workflow_id}' to check progress."
            )
        
    except Exception as e:
        print_error(f"Fine-tuning failed: {e}")
        raise click.Abort()
