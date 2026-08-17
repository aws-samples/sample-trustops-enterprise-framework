"""Workflow reproduction command."""
import click
from cli.utils.output import (
    print_header, print_success, print_error, print_info, print_progress
)
from cli.config import update_env_from_config


@click.command()
@click.argument('workflow-id')
@click.option(
    '--wait/--no-wait',
    default=True,
    help='Wait for reproduced workflow to complete'
)
@click.pass_context
def reproduce(ctx, workflow_id, wait):
    """
    Reproduce a previous workflow with identical configuration.
    
    This command loads the stored workflow manifest and executes a new
    workflow with the same configuration, dataset, and model settings.
    Useful for validating reproducibility and auditing results.
    
    Example:
    
        trustops reproduce baseline-20240115-abc123
    """
    print_header("Workflow Reproduction")
    
    # Update environment from config
    config = ctx.obj.get('config', {})
    update_env_from_config(config)
    
    print_info(f"Original Workflow ID: {workflow_id}")
    
    try:
        from src.orchestration.workflow_manager import WorkflowManager
        from src.orchestration.evaluation_orchestrator import (
            EvaluationOrchestrator
        )
        from src.orchestration.fine_tuning_orchestrator import (
            FineTuningOrchestrator
        )
        
        workflow_manager = WorkflowManager()
        
        # Get original workflow
        print_progress("Loading original workflow...")
        original_manifest = workflow_manager.get_workflow(workflow_id)
        
        if not original_manifest:
            print_error(f"Workflow {workflow_id} not found.")
            raise click.Abort()
        
        print_info(f"Original Type: {original_manifest.workflow_type}")
        print_info(f"Original Status: {original_manifest.status}")
        
        # Create reproduced workflow
        print_progress("Creating reproduced workflow...")
        new_workflow_id = workflow_manager.reproduce_workflow(
            workflow_id=workflow_id,
            created_by='cli'
        )
        
        print_success(f"New Workflow ID: {new_workflow_id}")
        
        # Execute based on workflow type
        workflow_type = original_manifest.workflow_type
        configuration = original_manifest.configuration
        
        if workflow_type == 'baseline':
            print_progress("Running baseline evaluation...")
            
            orchestrator = EvaluationOrchestrator()
            result = orchestrator.run_baseline_evaluation(
                model_id=configuration['model_id'],
                dataset_s3_uri=configuration['dataset_s3_uri'],
                workflow_id=new_workflow_id
            )
            
            print_success("Baseline evaluation completed!")
            print_info(f"Mean Trust Score: {result.metrics.mean_trust_score:.3f}")
            print_info(f"Results: {result.results_s3_uri}")
        
        elif workflow_type == 'comparative':
            print_progress("Running comparative evaluation...")
            
            orchestrator = EvaluationOrchestrator()
            result = orchestrator.run_comparative_evaluation(
                baseline_model_id=configuration['baseline_model_id'],
                finetuned_model_id=configuration['finetuned_model_id'],
                dataset_s3_uri=configuration['dataset_s3_uri'],
                workflow_id=new_workflow_id
            )
            
            print_success("Comparative evaluation completed!")
            print_info(
                f"Trust Score Improvement: "
                f"{result.improvement_metrics.trust_score_improvement:+.3f}"
            )
            print_info(
                f"Recommendation: "
                f"{result.improvement_metrics.recommendation}"
            )
            print_info(f"Results: {result.results_s3_uri}")
        
        elif workflow_type == 'fine-tuning':
            print_progress("Starting fine-tuning job...")
            
            orchestrator = FineTuningOrchestrator()
            job = orchestrator.start_fine_tuning_job(
                base_model_id=configuration['base_model_id'],
                training_data_s3_uri=configuration['training_data_s3_uri'],
                job_name=configuration['job_name'] + '-reproduced',
                hyperparameters=configuration.get('hyperparameters'),
                workflow_id=new_workflow_id
            )
            
            print_success(f"Fine-tuning job started: {job.job_id}")
            
            if wait:
                print_info("\nWaiting for fine-tuning to complete...")
                print_info(
                    "This may take several hours. "
                    "Press Ctrl+C to stop waiting."
                )
                
                try:
                    completed_job = orchestrator.poll_job_status(
                        job_id=job.job_id,
                        workflow_id=new_workflow_id
                    )
                    
                    print_success("\nFine-tuning completed!")
                    print_info(
                        f"Fine-tuned Model: "
                        f"{completed_job.finetuned_model_id}"
                    )
                
                except KeyboardInterrupt:
                    print_info(
                        "\nStopped waiting. "
                        "Fine-tuning continues in background."
                    )
                    print_info(
                        f"Use 'trustops status {new_workflow_id}' "
                        "to check progress."
                    )
            else:
                print_info(
                    f"Use 'trustops status {new_workflow_id}' "
                    "to check progress."
                )
        
        else:
            print_error(f"Unknown workflow type: {workflow_type}")
            raise click.Abort()
        
        print_info(f"\nReproduced workflow ID: {new_workflow_id}")
        print_info(
            f"Use 'trustops status {new_workflow_id}' "
            "to view full details."
        )
    
    except Exception as e:
        print_error(f"Workflow reproduction failed: {e}")
        raise click.Abort()
