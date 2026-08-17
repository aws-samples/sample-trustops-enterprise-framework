# QuickSight Dataset Definitions
# This file defines QuickSight datasets for evaluation results with refresh schedules

# Dataset - Baseline Evaluation Metrics
resource "aws_quicksight_data_set" "baseline_evaluation" {
  data_set_id = "${var.project_name}-baseline-evaluation-${var.environment}"
  name        = "Baseline Evaluation Metrics"
  import_mode = "SPICE"

  physical_table_map {
    physical_table_map_id = "evaluation-results"

    relational_table {
      data_source_arn = aws_quicksight_data_source.athena.arn
      catalog         = "AwsDataCatalog"
      schema          = aws_glue_catalog_database.trustops.name
      name            = aws_glue_catalog_table.evaluation_results.name

      input_columns {
        name = "workflow_id"
        type = "STRING"
      }

      input_columns {
        name = "model_id"
        type = "STRING"
      }

      input_columns {
        name = "timestamp"
        type = "DATETIME"
      }

      input_columns {
        name = "example_id"
        type = "STRING"
      }

      input_columns {
        name = "trust_score"
        type = "DECIMAL"
      }

      input_columns {
        name = "hallucination_rate"
        type = "DECIMAL"
      }

      input_columns {
        name = "latency_ms"
        type = "DECIMAL"
      }

      input_columns {
        name = "input_tokens"
        type = "INTEGER"
      }

      input_columns {
        name = "output_tokens"
        type = "INTEGER"
      }

      input_columns {
        name = "cost"
        type = "DECIMAL"
      }

      input_columns {
        name = "category"
        type = "STRING"
      }

      input_columns {
        name = "passed"
        type = "BOOLEAN"
      }
    }
  }

  logical_table_map {
    logical_table_map_id = "baseline-metrics"

    alias = "Baseline Evaluation Metrics"

    source {
      physical_table_id = "evaluation-results"
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "trust_score_category"
          column_id   = "trust-score-category"

          expression = "ifelse({trust_score} >= 0.8, 'High', ifelse({trust_score} >= 0.6, 'Medium', 'Low'))"
        }
      }
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "cost_per_1k_tokens"
          column_id   = "cost-per-1k-tokens"

          expression = "({cost} / ({input_tokens} + {output_tokens})) * 1000"
        }
      }
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "date"
          column_id   = "date"

          expression = "truncDate('DD', {timestamp})"
        }
      }
    }
  }

  permissions {
    principal = aws_iam_role.quicksight_service_role.arn
    actions = [
      "quicksight:DescribeDataSet",
      "quicksight:DescribeDataSetPermissions",
      "quicksight:PassDataSet",
      "quicksight:DescribeIngestion",
      "quicksight:ListIngestions",
      "quicksight:UpdateDataSet",
      "quicksight:DeleteDataSet",
      "quicksight:CreateIngestion",
      "quicksight:CancelIngestion",
      "quicksight:UpdateDataSetPermissions"
    ]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Dataset - Comparative Evaluation Metrics
resource "aws_quicksight_data_set" "comparative_evaluation" {
  data_set_id = "${var.project_name}-comparative-evaluation-${var.environment}"
  name        = "Comparative Evaluation Metrics"
  import_mode = "SPICE"

  physical_table_map {
    physical_table_map_id = "comparative-results"

    relational_table {
      data_source_arn = aws_quicksight_data_source.athena.arn
      catalog         = "AwsDataCatalog"
      schema          = aws_glue_catalog_database.trustops.name
      name            = aws_glue_catalog_table.comparative_results.name

      input_columns {
        name = "workflow_id"
        type = "STRING"
      }

      input_columns {
        name = "baseline_model_id"
        type = "STRING"
      }

      input_columns {
        name = "finetuned_model_id"
        type = "STRING"
      }

      input_columns {
        name = "timestamp"
        type = "DATETIME"
      }

      input_columns {
        name = "trust_score_improvement"
        type = "DECIMAL"
      }

      input_columns {
        name = "hallucination_reduction"
        type = "DECIMAL"
      }

      input_columns {
        name = "latency_delta_ms"
        type = "DECIMAL"
      }

      input_columns {
        name = "cost_delta_per_query"
        type = "DECIMAL"
      }

      input_columns {
        name = "cost_delta_percentage"
        type = "DECIMAL"
      }

      input_columns {
        name = "statistical_significance"
        type = "BOOLEAN"
      }

      input_columns {
        name = "recommendation"
        type = "STRING"
      }
    }
  }

  logical_table_map {
    logical_table_map_id = "comparative-metrics"

    alias = "Comparative Evaluation Metrics"

    source {
      physical_table_id = "comparative-results"
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "improvement_category"
          column_id   = "improvement-category"

          expression = "ifelse({trust_score_improvement} >= 0.1, 'Significant', ifelse({trust_score_improvement} >= 0.05, 'Moderate', 'Minimal'))"
        }
      }
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "date"
          column_id   = "date"

          expression = "truncDate('DD', {timestamp})"
        }
      }
    }
  }

  permissions {
    principal = aws_iam_role.quicksight_service_role.arn
    actions = [
      "quicksight:DescribeDataSet",
      "quicksight:DescribeDataSetPermissions",
      "quicksight:PassDataSet",
      "quicksight:DescribeIngestion",
      "quicksight:ListIngestions",
      "quicksight:UpdateDataSet",
      "quicksight:DeleteDataSet",
      "quicksight:CreateIngestion",
      "quicksight:CancelIngestion",
      "quicksight:UpdateDataSetPermissions"
    ]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Dataset - Cost Analysis
resource "aws_quicksight_data_set" "cost_analysis" {
  data_set_id = "${var.project_name}-cost-analysis-${var.environment}"
  name        = "Cost Analysis"
  import_mode = "SPICE"

  physical_table_map {
    physical_table_map_id = "evaluation-results-cost"

    relational_table {
      data_source_arn = aws_quicksight_data_source.athena.arn
      catalog         = "AwsDataCatalog"
      schema          = aws_glue_catalog_database.trustops.name
      name            = aws_glue_catalog_table.evaluation_results.name

      input_columns {
        name = "workflow_id"
        type = "STRING"
      }

      input_columns {
        name = "model_id"
        type = "STRING"
      }

      input_columns {
        name = "timestamp"
        type = "DATETIME"
      }

      input_columns {
        name = "trust_score"
        type = "DECIMAL"
      }

      input_columns {
        name = "input_tokens"
        type = "INTEGER"
      }

      input_columns {
        name = "output_tokens"
        type = "INTEGER"
      }

      input_columns {
        name = "cost"
        type = "DECIMAL"
      }
    }
  }

  logical_table_map {
    logical_table_map_id = "cost-metrics"

    alias = "Cost Analysis Metrics"

    source {
      physical_table_id = "evaluation-results-cost"
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "cost_per_high_trust_response"
          column_id   = "cost-per-high-trust"

          expression = "ifelse({trust_score} >= 0.8, {cost}, null)"
        }
      }
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "total_tokens"
          column_id   = "total-tokens"

          expression = "{input_tokens} + {output_tokens}"
        }
      }
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "cost_efficiency_score"
          column_id   = "cost-efficiency"

          expression = "{trust_score} / {cost}"
        }
      }
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "date"
          column_id   = "date"

          expression = "truncDate('DD', {timestamp})"
        }
      }
    }
  }

  permissions {
    principal = aws_iam_role.quicksight_service_role.arn
    actions = [
      "quicksight:DescribeDataSet",
      "quicksight:DescribeDataSetPermissions",
      "quicksight:PassDataSet",
      "quicksight:DescribeIngestion",
      "quicksight:ListIngestions",
      "quicksight:UpdateDataSet",
      "quicksight:DeleteDataSet",
      "quicksight:CreateIngestion",
      "quicksight:CancelIngestion",
      "quicksight:UpdateDataSetPermissions"
    ]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Dataset - Hallucination Analysis
resource "aws_quicksight_data_set" "hallucination_analysis" {
  data_set_id = "${var.project_name}-hallucination-analysis-${var.environment}"
  name        = "Hallucination Analysis"
  import_mode = "SPICE"

  physical_table_map {
    physical_table_map_id = "evaluation-results-hallucination"

    relational_table {
      data_source_arn = aws_quicksight_data_source.athena.arn
      catalog         = "AwsDataCatalog"
      schema          = aws_glue_catalog_database.trustops.name
      name            = aws_glue_catalog_table.evaluation_results.name

      input_columns {
        name = "workflow_id"
        type = "STRING"
      }

      input_columns {
        name = "model_id"
        type = "STRING"
      }

      input_columns {
        name = "timestamp"
        type = "DATETIME"
      }

      input_columns {
        name = "hallucination_rate"
        type = "DECIMAL"
      }

      input_columns {
        name = "category"
        type = "STRING"
      }

      input_columns {
        name = "trust_score"
        type = "DECIMAL"
      }
    }
  }

  logical_table_map {
    logical_table_map_id = "hallucination-metrics"

    alias = "Hallucination Analysis Metrics"

    source {
      physical_table_id = "evaluation-results-hallucination"
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "hallucination_severity"
          column_id   = "hallucination-severity"

          expression = "ifelse({hallucination_rate} >= 0.3, 'High', ifelse({hallucination_rate} >= 0.1, 'Medium', 'Low'))"
        }
      }
    }

    data_transforms {
      create_columns_operation {
        columns {
          column_name = "date"
          column_id   = "date"

          expression = "truncDate('DD', {timestamp})"
        }
      }
    }
  }

  permissions {
    principal = aws_iam_role.quicksight_service_role.arn
    actions = [
      "quicksight:DescribeDataSet",
      "quicksight:DescribeDataSetPermissions",
      "quicksight:PassDataSet",
      "quicksight:DescribeIngestion",
      "quicksight:ListIngestions",
      "quicksight:UpdateDataSet",
      "quicksight:DeleteDataSet",
      "quicksight:CreateIngestion",
      "quicksight:CancelIngestion",
      "quicksight:UpdateDataSetPermissions"
    ]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Refresh Schedule for Baseline Evaluation Dataset
resource "aws_quicksight_refresh_schedule" "baseline_evaluation" {
  data_set_id = aws_quicksight_data_set.baseline_evaluation.data_set_id
  schedule_id = "daily-refresh"

  schedule {
    refresh_type = "FULL_REFRESH"

    schedule_frequency {
      interval        = "DAILY"
      time_of_the_day = "06:00"
    }
  }
}

# Refresh Schedule for Comparative Evaluation Dataset
resource "aws_quicksight_refresh_schedule" "comparative_evaluation" {
  data_set_id = aws_quicksight_data_set.comparative_evaluation.data_set_id
  schedule_id = "daily-refresh"

  schedule {
    refresh_type = "FULL_REFRESH"

    schedule_frequency {
      interval        = "DAILY"
      time_of_the_day = "06:00"
    }
  }
}

# Refresh Schedule for Cost Analysis Dataset
resource "aws_quicksight_refresh_schedule" "cost_analysis" {
  data_set_id = aws_quicksight_data_set.cost_analysis.data_set_id
  schedule_id = "daily-refresh"

  schedule {
    refresh_type = "FULL_REFRESH"

    schedule_frequency {
      interval        = "DAILY"
      time_of_the_day = "06:00"
    }
  }
}

# Refresh Schedule for Hallucination Analysis Dataset
resource "aws_quicksight_refresh_schedule" "hallucination_analysis" {
  data_set_id = aws_quicksight_data_set.hallucination_analysis.data_set_id
  schedule_id = "daily-refresh"

  schedule {
    refresh_type = "FULL_REFRESH"

    schedule_frequency {
      interval        = "DAILY"
      time_of_the_day = "06:00"
    }
  }
}
