"""Baseline evaluation command."""
import click
import uuid
from cli.utils.output import (
    print_header, print_success, print_error, print_info, print_progress
)
from cli.utils.validation import validate_s3_uri, validate_model_id
from cli.config import update_env_from_config


@click.command()
@click.option(
    '--model-id',
    required=True,
    help='AWS Bedrock model identifier (e.g., anthropic.claude-v2)'
)
@click.option(
    '--dataset',
    required=True,
    help='S3 URI of evaluation dataset (s3://bucket/path/dataset.json)'
)
@click.option(
    '--workflow-id',
    help='Optional workflow ID (auto-generated if not provided)'
)
@click.option(
    '--wait/--no-wait',
    default=True,
    help='Wait for evaluation to complete'
)
@click.pass_context
def baseline(ctx, model_id, dataset, workflow_id, wait):
    """
    Run baseline evaluation for a foundation model.
    
    This command evaluates a foundation model's performance before fine-tuning
    to establish a baseline for comparison. It generates responses for all
    evaluation prompts, calculates trust scores, and stores results in S3.
    
    Example:
    
        trustops baseline --model-id anthropic.claude-v2 \\
            --dataset s3://my-bucket/eval-dataset.json
    """
    print_header("Baseline Evaluation")
    
    # Update environment from config
    config = ctx.obj.get('config', {})
    update_env_from_config(config)
    
    # Validate inputs
    try:
        validate_model_id(model_id)
        validate_s3_uri(dataset)
    except ValueError as e:
        print_error(f"Validation error: {e}")
        raise click.Abort()
    
    # Generate workflow ID if not provided
    if not workflow_id:
        workflow_id = f"baseline-{uuid.uuid4().hex[:8]}"
    
    print_info(f"Model ID: {model_id}")
    print_info(f"Dataset: {dataset}")
    print_info(f"Workflow ID: {workflow_id}")
    
    try:
        # Import here to avoid circular dependencies
        from src.orchestration.evaluation_orchestrator import (
            EvaluationOrchestrator
        )
        from src.orchestration.workflow_manager import WorkflowManager
        
        # Create workflow. WorkflowManager generates the canonical workflow ID
        # and persists the manifest in DynamoDB; the orchestrator looks the
        # workflow up by that exact ID, so we must use the returned value
        # rather than the optional user-supplied --workflow-id.
        print_progress("Creating workflow...")
        workflow_manager = WorkflowManager()
        workflow_id = workflow_manager.create_workflow(
            workflow_type='baseline',
            configuration={
                'model_id': model_id,
                'dataset_s3_uri': dataset
            }
        )
        print_info(f"Registered workflow: {workflow_id}")
        
        # Run baseline evaluation
        print_progress("Starting baseline evaluation...")
        orchestrator = EvaluationOrchestrator()
        
        result = orchestrator.run_baseline_evaluation(
            model_id=model_id,
            dataset_s3_uri=dataset,
            workflow_id=workflow_id
        )
        
        # Display results
        print_success("Baseline evaluation completed!")
        print_info(f"\nResults Summary:")
        print_info(f"  Total Examples: {result.metrics.total_examples}")
        print_info(
            f"  Mean Trust Score: {result.metrics.mean_trust_score:.3f}"
        )
        print_info(
            f"  Median Trust Score: {result.metrics.median_trust_score:.3f}"
        )
        print_info(
            f"  Mean Latency: {result.metrics.mean_latency_ms:.1f}ms"
        )
        print_info(
            f"  Total Cost: ${result.metrics.total_cost:.4f}"
        )
        print_info(
            f"  Hallucination Rate: "
            f"{result.metrics.hallucination_rate:.1%}"
        )
        print_info(f"\nResults stored at: {result.results_s3_uri}")
        
        # Display trust score distribution
        print_info("\nTrust Score Distribution:")
        dist = result.metrics.trust_score_distribution
        print_info(f"  High (≥0.8): {dist.get('high', 0)}")
        print_info(f"  Medium (0.6-0.8): {dist.get('medium', 0)}")
        print_info(f"  Low (<0.6): {dist.get('low', 0)}")
        
    except Exception as e:
        print_error(f"Baseline evaluation failed: {e}")
        raise click.Abort()
