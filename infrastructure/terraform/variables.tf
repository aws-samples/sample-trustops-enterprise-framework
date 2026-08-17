variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "trustops"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "embedding_model_id" {
  description = "Bedrock embedding model for Knowledge Base"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days (365 minimum for audit compliance)"
  type        = number
  default     = 365

  validation {
    condition     = var.log_retention_days >= 365
    error_message = "Log retention must be at least 365 days for audit compliance."
  }
}

variable "trust_score_threshold" {
  description = "Trust score threshold for flagging responses"
  type        = string
  default     = "0.7"
}

variable "lambda_evaluation_timeout" {
  description = "Timeout for evaluation Lambda functions in seconds"
  type        = number
  default     = 900
}

variable "lambda_evaluation_memory" {
  description = "Memory for evaluation Lambda functions in MB"
  type        = number
  default     = 1024
}

variable "lambda_trust_scoring_timeout" {
  description = "Timeout for trust scoring Lambda function in seconds"
  type        = number
  default     = 300
}

variable "lambda_trust_scoring_memory" {
  description = "Memory for trust scoring Lambda function in MB"
  type        = number
  default     = 512
}

variable "alert_email" {
  description = "Email address for CloudWatch alarm notifications"
  type        = string
  default     = ""
}
