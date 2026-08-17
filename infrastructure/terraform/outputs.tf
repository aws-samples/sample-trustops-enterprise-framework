output "datasets_bucket_name" {
  description = "S3 bucket for datasets"
  value       = aws_s3_bucket.datasets.id
}

output "results_bucket_name" {
  description = "S3 bucket for results"
  value       = aws_s3_bucket.results.id
}

output "artifacts_bucket_name" {
  description = "S3 bucket for artifacts"
  value       = aws_s3_bucket.artifacts.id
}

output "workflows_table_name" {
  description = "DynamoDB table for workflows"
  value       = aws_dynamodb_table.workflows.name
}

output "models_table_name" {
  description = "DynamoDB table for models"
  value       = aws_dynamodb_table.models.name
}

output "knowledge_base_id" {
  description = "Bedrock Knowledge Base ID"
  value       = aws_bedrockagent_knowledge_base.trustops.id
}

output "knowledge_base_data_source_id" {
  description = "Bedrock Knowledge Base data source ID"
  value       = aws_bedrockagent_data_source.s3.data_source_id
}

output "lambda_execution_role_arn" {
  description = "IAM role ARN for Lambda functions"
  value       = aws_iam_role.lambda_execution.arn
}

output "stepfunctions_execution_role_arn" {
  description = "IAM role ARN for Step Functions"
  value       = aws_iam_role.stepfunctions_execution.arn
}

output "evaluation_orchestrator_function_arn" {
  description = "Evaluation orchestrator Lambda function ARN"
  value       = aws_lambda_function.evaluation_orchestrator.arn
}

output "finetuning_orchestrator_function_arn" {
  description = "Fine-tuning orchestrator Lambda function ARN"
  value       = aws_lambda_function.finetuning_orchestrator.arn
}

output "trust_scoring_function_arn" {
  description = "Trust scoring Lambda function ARN"
  value       = aws_lambda_function.trust_scoring.arn
}

output "comparative_evaluation_function_arn" {
  description = "Comparative evaluation Lambda function ARN"
  value       = aws_lambda_function.comparative_evaluation.arn
}

output "evaluate_single_example_function_arn" {
  description = "Evaluate single example Lambda function ARN"
  value       = aws_lambda_function.evaluate_single_example.arn
}

output "aggregate_metrics_function_arn" {
  description = "Aggregate metrics Lambda function ARN"
  value       = aws_lambda_function.aggregate_metrics.arn
}

output "generate_recommendation_function_arn" {
  description = "Generate recommendation Lambda function ARN"
  value       = aws_lambda_function.generate_recommendation.arn
}

output "detect_hallucinations_function_arn" {
  description = "Detect hallucinations Lambda function ARN"
  value       = aws_lambda_function.detect_hallucinations.arn
}

output "baseline_evaluation_state_machine_arn" {
  description = "Baseline evaluation Step Functions state machine ARN"
  value       = aws_sfn_state_machine.baseline_evaluation.arn
}

output "fine_tuning_state_machine_arn" {
  description = "Fine-tuning Step Functions state machine ARN"
  value       = aws_sfn_state_machine.fine_tuning.arn
}

output "comparative_evaluation_state_machine_arn" {
  description = "Comparative evaluation Step Functions state machine ARN"
  value       = aws_sfn_state_machine.comparative_evaluation.arn
}

output "trustops_log_group_name" {
  description = "CloudWatch log group for TrustOps"
  value       = aws_cloudwatch_log_group.trustops.name
}

output "evaluation_log_group_name" {
  description = "CloudWatch log group for evaluation"
  value       = aws_cloudwatch_log_group.evaluation.name
}

output "finetuning_log_group_name" {
  description = "CloudWatch log group for fine-tuning"
  value       = aws_cloudwatch_log_group.finetuning.name
}

output "trustscoring_log_group_name" {
  description = "CloudWatch log group for trust scoring"
  value       = aws_cloudwatch_log_group.trustscoring.name
}

# QuickSight Outputs
output "quicksight_service_role_arn" {
  description = "ARN of the QuickSight service role"
  value       = module.quicksight.quicksight_service_role_arn
}

output "quicksight_athena_workgroup" {
  description = "Name of the Athena workgroup for QuickSight"
  value       = module.quicksight.athena_workgroup_name
}

output "quicksight_glue_database" {
  description = "Name of the Glue catalog database"
  value       = module.quicksight.glue_database_name
}

output "quicksight_data_source_ids" {
  description = "QuickSight data source IDs"
  value       = module.quicksight.data_source_ids
}

output "quicksight_dataset_ids" {
  description = "QuickSight dataset IDs"
  value       = module.quicksight.dataset_ids
}

output "quicksight_glue_tables" {
  description = "Glue catalog table names"
  value       = module.quicksight.glue_table_names
}
