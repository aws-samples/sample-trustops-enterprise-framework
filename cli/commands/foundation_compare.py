"""Foundation model comparison command."""
import click
import uuid
from cli.utils.output import (
    print_header, print_success, print_error, print_info, print_progress
)
from cli.utils.validation import validate_s3_uri, validate_model_id
from cli.config import update_env_from_config


@click.command()
@click.option(
    '--model-id-1',
    required=True,
    help='First foundation model ID'
)
@click.option(
    '--model-id-2',
    required=True,
    help='Second foundation model ID'
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
    '--temperature',
    type=float,
    default=0.7,
    help='Inference temperature (default: 0.7)'
)
@click.option(
    '--max-tokens',
    type=int,
    default=2048,
    help='Maximum output tokens (default: 2048)'
)
@click.pass_context
def foundation_compare(ctx, model_id_1, model_id_2, dataset, 
                       workflow_id, temperature, max_tokens):
    """
    Compare two foundation models on the same evaluation dataset.
    
    This command evaluates both models on identical prompts, calculates
    trust scores, detects hallucinations, and generates a comparison
    report with metrics and recommendations.
    
    Example:
    
        trustops foundation-compare \\
            --model-id-1 anthropic.claude-3-haiku-20240307-v1:0 \\
            --model-id-2 anthropic.claude-3-sonnet-20240229-v1:0 \\
            --dataset s3://my-bucket/eval-dataset.json
    """
    print_header("Foundation Model Comparison")
    
    # Update environment from config
    config = ctx.obj.get('config', {})
    update_env_from_config(config)
    
    # Validate inputs
    try:
        validate_model_id(model_id_1)
        validate_model_id(model_id_2)
        validate_s3_uri(dataset)
        
        # Check models are different
        if model_id_1 == model_id_2:
            raise ValueError(
                "Cannot compare identical models. "
                "Please provide two different model IDs."
            )
    except ValueError as e:
        print_error(f"Validation error: {e}")
        raise click.Abort()
    
    # Generate workflow ID if not provided
    if not workflow_id:
        workflow_id = f"foundation-compare-{uuid.uuid4().hex[:8]}"
    
    print_info(f"Model 1: {model_id_1}")
    print_info(f"Model 2: {model_id_2}")
    print_info(f"Dataset: {dataset}")
    print_info(f"Workflow ID: {workflow_id}")
    print_info(f"Temperature: {temperature}")
    print_info(f"Max Tokens: {max_tokens}")
    
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
            workflow_type='foundation_comparison',
            configuration={
                'model_id_1': model_id_1,
                'model_id_2': model_id_2,
                'dataset_s3_uri': dataset,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )
        print_info(f"Registered workflow: {workflow_id}")
        
        # Run foundation model comparison
        print_progress("Starting foundation model comparison...")
        print_info("This will evaluate both models on the same dataset.\n")
        
        orchestrator = EvaluationOrchestrator()
        
        inference_params = {
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        result = orchestrator.run_foundation_model_comparison(
            model_id_1=model_id_1,
            model_id_2=model_id_2,
            dataset_s3_uri=dataset,
            workflow_id=workflow_id,
            inference_params=inference_params
        )
        
        # Display results
        print_success("Foundation model comparison completed!")
        
        print_info("\n" + "=" * 60)
        print_info("MODEL 1 RESULTS")
        print_info("=" * 60)
        print_info(f"Model ID: {model_id_1}")
        print_info(f"Total Examples: {result.baseline_metrics.total_examples}")
        print_info(
            f"Mean Trust Score: {result.baseline_metrics.mean_trust_score:.3f}"
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
        print_info("MODEL 2 RESULTS")
        print_info("=" * 60)
        print_info(f"Model ID: {model_id_2}")
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
        print_info("COMPARISON METRICS")
        print_info("=" * 60)
        
        improvement = result.improvement_metrics
        
        # Trust score improvement
        trust_delta = improvement.trust_score_improvement
        if trust_delta > 0:
            trust_symbol = "↑"
        elif trust_delta < 0:
            trust_symbol = "↓"
        else:
            trust_symbol = "→"
        print_info(
            f"Trust Score Change: {trust_symbol} "
            f"{abs(trust_delta):.3f} ({trust_delta:+.1%})"
        )
        
        # Hallucination reduction
        hall_delta = improvement.hallucination_reduction
        if hall_delta > 0:
            hall_symbol = "↓"
        elif hall_delta < 0:
            hall_symbol = "↑"
        else:
            hall_symbol = "→"
        print_info(
            f"Hallucination Change: {hall_symbol} "
            f"{abs(hall_delta):.3f} ({hall_delta:+.1%})"
        )
        
        # Cost delta
        cost_delta = improvement.cost_delta_per_query
        if cost_delta > 0:
            cost_symbol = "↑"
        elif cost_delta < 0:
            cost_symbol = "↓"
        else:
            cost_symbol = "→"
        print_info(
            f"Cost per Query Change: {cost_symbol} ${abs(cost_delta):.6f} "
            f"({improvement.cost_delta_percentage:+.1f}%)"
        )
        
        # Latency delta
        latency_delta = improvement.latency_delta_ms
        if latency_delta > 0:
            latency_symbol = "↑"
        elif latency_delta < 0:
            latency_symbol = "↓"
        else:
            latency_symbol = "→"
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
        print_error(f"Foundation model comparison failed: {e}")
        raise click.Abort()
