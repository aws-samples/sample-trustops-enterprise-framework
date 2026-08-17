# QuickSight Module Variables

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
}

# Passed from parent module
variable "results_bucket_id" {
  description = "S3 results bucket ID"
  type        = string
}

variable "results_bucket_arn" {
  description = "S3 results bucket ARN"
  type        = string
}

variable "datasets_bucket_id" {
  description = "S3 datasets bucket ID"
  type        = string
}

variable "datasets_bucket_arn" {
  description = "S3 datasets bucket ARN"
  type        = string
}

variable "workflows_table_name" {
  description = "DynamoDB workflows table name"
  type        = string
}

variable "workflows_table_arn" {
  description = "DynamoDB workflows table ARN"
  type        = string
}

variable "models_table_name" {
  description = "DynamoDB models table name"
  type        = string
}

variable "models_table_arn" {
  description = "DynamoDB models table ARN"
  type        = string
}

variable "trustops_log_group_arn" {
  description = "CloudWatch trustops log group ARN"
  type        = string
}

variable "evaluation_log_group_arn" {
  description = "CloudWatch evaluation log group ARN"
  type        = string
}

variable "finetuning_log_group_arn" {
  description = "CloudWatch finetuning log group ARN"
  type        = string
}

variable "trustscoring_log_group_arn" {
  description = "CloudWatch trustscoring log group ARN"
  type        = string
}
