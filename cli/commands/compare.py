"""Comparative evaluation command."""
import click
import uuid
from cli.utils.output import (
    print_header, print_success, print_error, print_info, print_progress
)
from cli.utils.validation import validate_s3_uri, validate_model_id
from cli.config import update_env_from_config


@click.command()
@click.option(
    '--baseline-model-id',
    required=True,
    help='Baseline foundation model ID'
)
@click.option(
    '--finetuned-model-id',
    required=True,
    help='Fine-tuned model ID to compare'
)
@click.option(
    '--dataset',
    required=True,
    help='S3 URI of evaluation dataset'
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
def compare(ctx, baseline_model_id, finetuned_model_id, dataset, 
            workflow_id, wait):
    """
    Run comparative evaluation between baseline and fine-tuned models.
    
    This command evaluates both models on the same dataset, calculates
    trust scores, computes semantic similarity, and generates a comparison
    report with improvement metrics.
    
    Example:
    
        trustops compare \\
            --baseline-model-id anthropic.claude-v2 \\
            --finetuned-model-id arn:aws:bedrock:us-east-1:123:model/... \\
            --dataset s3://my-bucket/eval-dataset.json
    """
    print_header("Comparative Evaluation")
    
    # Update environment from config
    config = ctx.obj.get('config', {})
    update_env_from_config(config)
    
    # Validate inputs
    try:
        validate_model_id(baseline_model_id)
        validate_model_id(finetuned_model_id)
        validate_s3_uri(dataset)
    except ValueError as e:
        print_error(f"Validation error: {e}")
        raise click.Abort()
    
    # Generate workflow ID if not provided
    if not workflow_id:
        workflow_id = f"compare-{uuid.uuid4().hex[:8]}"
    
    print_info(f"Baseline Model: {baseline_model_id}")
    print_info(f"Fine-tuned Model: {finetuned_model_id}")
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
            workflow_type='comparative',
            configuration={
                'baseline_model_id': baseline_model_id,
                'finetuned_model_id': finetuned_model_id,
                'dataset_s3_uri': dataset
            }
        )
        print_info(f"Registered workflow: {workflow_id}")
        
        # Run comparative evaluation
        print_progress("Starting comparative evaluation...")
        print_info("This will evaluate both models on the same dataset.\n")
        
        orchestrator = EvaluationOrchestrator()
        
        result = orchestrator.run_comparative_evaluation(
            baseline_model_id=baseline_model_id,
            finetuned_model_id=finetuned_model_id,
            dataset_s3_uri=dataset,
            workflow_id=workflow_id
        )
        
        # Display results
        print_success("Comparative evaluation completed!")
        
        print_info("\n" + "=" * 60)
        print_info("BASELINE MODEL RESULTS")
        print_info("=" * 60)
        print_info(f"Total Examples: {result.baseline_metrics.total_examples}")
        print_info(
            f"Mean Trust Score: "
            f"{result.baseline_metrics.mean_trust_score:.3f}"
        )
        print_info(
            f"Hallucination Rate: "
            f"{result.baseline_metrics.hallucination_rate:.1%}"
        )
        print_info(
            f"Mean Latency: {result.baseline_metrics.mean_latency_ms:.1f}ms"
        )
        print_info(f"Total Cost: ${result.baseline_metrics.total_cost:.4f}")
        
        print_info("\n" + "=" * 60)
        print_info("FINE-TUNED MODEL RESULTS")
        print_info("=" * 60)
        print_info(
            f"Total Examples: {result.finetuned_metrics.total_examples}"
        )
        print_info(
            f"Mean Trust Score: "
            f"{result.finetuned_metrics.mean_trust_score:.3f}"
        )
        print_info(
            f"Hallucination Rate: "
            f"{result.finetuned_metrics.hallucination_rate:.1%}"
        )
        print_info(
            f"Mean Latency: {result.finetuned_metrics.mean_latency_ms:.1f}ms"
        )
        print_info(f"Total Cost: ${result.finetuned_metrics.total_cost:.4f}")
        
        print_info("\n" + "=" * 60)
        print_info("IMPROVEMENT METRICS")
        print_info("=" * 60)
        
        improvement = result.improvement_metrics
        
        # Trust score improvement
        trust_delta = improvement.trust_score_improvement
        trust_symbol = "↑" if trust_delta > 0 else "↓" if trust_delta < 0 else "→"
        print_info(
            f"Trust Score Change: {trust_symbol} "
            f"{abs(trust_delta):.3f} ({trust_delta:+.1%})"
        )
        
        # Hallucination reduction
        hall_delta = improvement.hallucination_reduction
        hall_symbol = "↓" if hall_delta > 0 else "↑" if hall_delta < 0 else "→"
        print_info(
            f"Hallucination Change: {hall_symbol} "
            f"{abs(hall_delta):.3f} ({hall_delta:+.1%})"
        )
        
        # Cost delta
        cost_delta = improvement.cost_delta_per_query
        cost_symbol = "↑" if cost_delta > 0 else "↓" if cost_delta < 0 else "→"
        print_info(
            f"Cost per Query Change: {cost_symbol} "
            f"${abs(cost_delta):.6f} ({improvement.cost_delta_percentage:+.1f}%)"
        )
        
        # Latency delta
        latency_delta = improvement.latency_delta_ms
        latency_symbol = "↑" if latency_delta > 0 else "↓" if latency_delta < 0 else "→"
        print_info(
            f"Latency Change: {latency_symbol} {abs(latency_delta):.1f}ms"
        )
        
        print_info(
            f"\nStatistical Significance: "
            f"{'Yes' if improvement.statistical_significance else 'No'}"
        )
        
        print_info("\n" + "=" * 60)
        print_info("RECOMMENDATION")
        print_info("=" * 60)
        print_info(f"Decision: {improvement.recommendation.upper()}")
        print_info(f"Justification: {improvement.justification}")
        
        print_info(f"\nResults stored at: {result.results_s3_uri}")
        
    except Exception as e:
        print_error(f"Comparative evaluation failed: {e}")
        raise click.Abort()
