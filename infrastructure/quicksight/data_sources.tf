# QuickSight Data Sources Configuration
# This file defines QuickSight data source connections to S3, DynamoDB, and CloudWatch

# QuickSight Service Role
resource "aws_iam_role" "quicksight_service_role" {
  name = "${var.project_name}-quicksight-service-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "quicksight.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role_policy" "quicksight_policy" {
  name = "quicksight-data-access"
  role = aws_iam_role.quicksight_service_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 permissions for reading results and datasets
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          var.results_bucket_arn,
          "${var.results_bucket_arn}/*",
          var.datasets_bucket_arn,
          "${var.datasets_bucket_arn}/*"
        ]
      },
      # DynamoDB permissions for reading workflow and model metadata
      {
        Effect = "Allow"
        Action = [
          "dynamodb:DescribeTable",
          "dynamodb:ListTables",
          "dynamodb:GetItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = [
          var.workflows_table_arn,
          "${var.workflows_table_arn}/index/*",
          var.models_table_arn,
          "${var.models_table_arn}/index/*"
        ]
      },
      # CloudWatch metric reads. These actions operate across the account's
      # metric namespaces and do not support resource-level ARNs, so "*" is
      # the only valid Resource; scope is limited by the namespace condition.
      {
        #checkov:skip=CKV_AWS_355:cloudwatch:GetMetricData/ListMetrics do not
        #  support resource-level permissions; constrained by namespace instead.
        Effect = "Allow"
        Action = [
          "cloudwatch:GetMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics",
          "cloudwatch:DescribeAlarms"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = ["TrustOps", "AWS/Lambda", "AWS/States"]
          }
        }
      },
      # CloudWatch Logs permissions
      {
        Effect = "Allow"
        Action = [
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          "logs:GetLogEvents",
          "logs:FilterLogEvents"
        ]
        Resource = [
          var.trustops_log_group_arn,
          var.evaluation_log_group_arn,
          var.finetuning_log_group_arn,
          var.trustscoring_log_group_arn
        ]
      }
    ]
  })
}

# QuickSight Data Source - S3 Results
resource "aws_quicksight_data_source" "s3_results" {
  data_source_id = "${var.project_name}-s3-results-${var.environment}"
  name           = "TrustOps Evaluation Results (S3)"

  parameters {
    s3 {
      manifest_file_location {
        bucket = var.results_bucket_id
        key    = "quicksight/manifest.json"
      }
    }
  }

  type = "S3"

  permission {
    principal = aws_iam_role.quicksight_service_role.arn
    actions = [
      "quicksight:DescribeDataSource",
      "quicksight:DescribeDataSourcePermissions",
      "quicksight:PassDataSource",
      "quicksight:UpdateDataSource",
      "quicksight:DeleteDataSource",
      "quicksight:UpdateDataSourcePermissions"
    ]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# NOTE: QuickSight has no native DynamoDB connector. Workflow and model
# metadata is surfaced through the Athena data source below, querying the Glue
# tables over the results bucket. The IAM role retains DynamoDB read access for
# other consumers of that metadata.

# QuickSight Data Source - Athena for S3 Query
resource "aws_quicksight_data_source" "athena" {
  data_source_id = "${var.project_name}-athena-${var.environment}"
  name           = "TrustOps Athena Query"

  parameters {
    athena {
      work_group = aws_athena_workgroup.trustops.name
    }
  }

  type = "ATHENA"

  permission {
    principal = aws_iam_role.quicksight_service_role.arn
    actions = [
      "quicksight:DescribeDataSource",
      "quicksight:DescribeDataSourcePermissions",
      "quicksight:PassDataSource",
      "quicksight:UpdateDataSource",
      "quicksight:DeleteDataSource",
      "quicksight:UpdateDataSourcePermissions"
    ]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Athena Workgroup for QuickSight queries
resource "aws_athena_workgroup" "trustops" {
  name = "${var.project_name}-quicksight-${var.environment}"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://${var.results_bucket_id}/athena-results/"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Glue Database for Athena queries
resource "aws_glue_catalog_database" "trustops" {
  name = "${var.project_name}_${var.environment}"

  description = "TrustOps evaluation results database"
}

# Glue Table for evaluation results
resource "aws_glue_catalog_table" "evaluation_results" {
  name          = "evaluation_results"
  database_name = aws_glue_catalog_database.trustops.name

  table_type = "EXTERNAL_TABLE"

  parameters = {
    "classification" = "json"
  }

  storage_descriptor {
    location      = "s3://${var.results_bucket_id}/evaluations/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      serialization_library = "org.openx.data.jsonserde.JsonSerDe"
    }

    columns {
      name = "workflow_id"
      type = "string"
    }

    columns {
      name = "model_id"
      type = "string"
    }

    columns {
      name = "timestamp"
      type = "timestamp"
    }

    columns {
      name = "example_id"
      type = "string"
    }

    columns {
      name = "trust_score"
      type = "double"
    }

    columns {
      name = "trust_score_components"
      type = "struct<context_grounding:double,output_structure:double,uncertainty_indicators:double,factual_consistency:double,response_completeness:double>"
    }

    columns {
      name = "hallucination_rate"
      type = "double"
    }

    columns {
      name = "latency_ms"
      type = "double"
    }

    columns {
      name = "input_tokens"
      type = "int"
    }

    columns {
      name = "output_tokens"
      type = "int"
    }

    columns {
      name = "cost"
      type = "double"
    }

    columns {
      name = "category"
      type = "string"
    }

    columns {
      name = "passed"
      type = "boolean"
    }
  }
}

# Glue Table for comparative results
resource "aws_glue_catalog_table" "comparative_results" {
  name          = "comparative_results"
  database_name = aws_glue_catalog_database.trustops.name

  table_type = "EXTERNAL_TABLE"

  parameters = {
    "classification" = "json"
  }

  storage_descriptor {
    location      = "s3://${var.results_bucket_id}/comparisons/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      serialization_library = "org.openx.data.jsonserde.JsonSerDe"
    }

    columns {
      name = "workflow_id"
      type = "string"
    }

    columns {
      name = "baseline_model_id"
      type = "string"
    }

    columns {
      name = "finetuned_model_id"
      type = "string"
    }

    columns {
      name = "timestamp"
      type = "timestamp"
    }

    columns {
      name = "trust_score_improvement"
      type = "double"
    }

    columns {
      name = "hallucination_reduction"
      type = "double"
    }

    columns {
      name = "latency_delta_ms"
      type = "double"
    }

    columns {
      name = "cost_delta_per_query"
      type = "double"
    }

    columns {
      name = "cost_delta_percentage"
      type = "double"
    }

    columns {
      name = "statistical_significance"
      type = "boolean"
    }

    columns {
      name = "recommendation"
      type = "string"
    }
  }
}
