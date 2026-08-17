# QuickSight Module Outputs

output "quicksight_service_role_arn" {
  description = "ARN of the QuickSight service role"
  value       = aws_iam_role.quicksight_service_role.arn
}

output "athena_workgroup_name" {
  description = "Name of the Athena workgroup for QuickSight"
  value       = aws_athena_workgroup.trustops.name
}

output "glue_database_name" {
  description = "Name of the Glue catalog database"
  value       = aws_glue_catalog_database.trustops.name
}

output "data_source_ids" {
  description = "QuickSight data source IDs"
  value = {
    s3_results = aws_quicksight_data_source.s3_results.data_source_id
    athena     = aws_quicksight_data_source.athena.data_source_id
  }
}

output "dataset_ids" {
  description = "QuickSight dataset IDs"
  value = {
    baseline_evaluation    = aws_quicksight_data_set.baseline_evaluation.data_set_id
    comparative_evaluation = aws_quicksight_data_set.comparative_evaluation.data_set_id
    cost_analysis          = aws_quicksight_data_set.cost_analysis.data_set_id
    hallucination_analysis = aws_quicksight_data_set.hallucination_analysis.data_set_id
  }
}

output "glue_table_names" {
  description = "Glue catalog table names"
  value = {
    evaluation_results  = aws_glue_catalog_table.evaluation_results.name
    comparative_results = aws_glue_catalog_table.comparative_results.name
  }
}
