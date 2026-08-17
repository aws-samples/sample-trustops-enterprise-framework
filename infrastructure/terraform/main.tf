terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# KMS Key for encryption
resource "aws_kms_key" "trustops" {
  description             = "KMS key for TrustOps ${var.environment} encryption"
  enable_key_rotation     = true
  deletion_window_in_days = 30

  # Explicit key policy: account root retains administration, and the AWS
  # services that encrypt TrustOps data are granted only the grant/encrypt
  # operations they need.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "EnableRootAccountAdministration"
        Effect    = "Allow"
        Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid    = "AllowCloudWatchLogsEncryption"
        Effect = "Allow"
        Principal = {
          Service = "logs.${var.aws_region}.amazonaws.com"
        }
        Action = [
          "kms:Encrypt*",
          "kms:Decrypt*",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:Describe*"
        ]
        Resource = "*"
        Condition = {
          ArnLike = {
            "kms:EncryptionContext:aws:logs:arn" = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:*"
          }
        }
      },
      {
        Sid    = "AllowServiceUseViaGrants"
        Effect = "Allow"
        Principal = {
          Service = [
            "s3.amazonaws.com",
            "sns.amazonaws.com",
            "sqs.amazonaws.com",
            "dynamodb.amazonaws.com",
            "lambda.amazonaws.com",
            "states.amazonaws.com"
          ]
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey",
          "kms:CreateGrant"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_kms_alias" "trustops" {
  name          = "alias/${var.project_name}-${var.environment}"
  target_key_id = aws_kms_key.trustops.key_id
}

# SNS Topic for Alarm Notifications
resource "aws_sns_topic" "alarms" {
  name              = "${var.project_name}-alarms-${var.environment}"
  kms_master_key_id = aws_kms_key.trustops.id

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Dead Letter Queue for Lambda
resource "aws_sqs_queue" "lambda_dlq" {
  name                      = "${var.project_name}-lambda-dlq-${var.environment}"
  message_retention_seconds = 1209600
  kms_master_key_id         = aws_kms_key.trustops.id

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# S3 Access Logs Bucket
resource "aws_s3_bucket" "access_logs" {
  #checkov:skip=CKV_AWS_145:S3 server access log delivery does not support
  #  SSE-KMS; this bucket must use SSE-S3. Buckets holding TrustOps data use KMS.
  #checkov:skip=CKV_AWS_144:Cross-region replication is not enabled for this
  #  framework; evaluation data is reproducible from source datasets and CRR
  #  would double storage cost. Enable per-environment if RPO requires it.
  #checkov:skip=CKV2_AWS_62:No event-driven consumers exist for these buckets;
  #  workflows are orchestrated by Step Functions, not S3 notifications.
  bucket = "${var.project_name}-access-logs-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket_public_access_block" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  versioning_configuration {
    status = "Enabled"
  }
}

# S3 server access logging cannot write to a bucket encrypted with SSE-KMS,
# so this bucket uses SSE-S3. All buckets holding TrustOps data use KMS.
resource "aws_s3_bucket_server_side_encryption_configuration" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  rule {
    id     = "expire-old-logs"
    status = "Enabled"

    filter {}

    expiration {
      days = 90
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_ownership_controls" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  rule {
    # BucketOwnerEnforced disables ACLs entirely.
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_policy" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "S3ServerAccessLogsPolicy"
        Effect    = "Allow"
        Principal = { Service = "logging.s3.amazonaws.com" }
        Action    = "s3:PutObject"
        Resource  = "${aws_s3_bucket.access_logs.arn}/*"
        Condition = {
          ArnLike = {
            "aws:SourceArn" = [
              aws_s3_bucket.datasets.arn,
              aws_s3_bucket.results.arn,
              aws_s3_bucket.artifacts.arn
            ]
          }
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      },
      # The deny is evaluated against the log-delivery PutObject above too;
      # S3 server access log delivery uses TLS, so it is unaffected.
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.access_logs.arn,
          "${aws_s3_bucket.access_logs.arn}/*"
        ]
        Condition = {
          Bool = { "aws:SecureTransport" = "false" }
        }
      }
    ]
  })
}

# Deny any request that does not arrive over TLS, so data in transit cannot
# fall back to plaintext HTTP. Mirrors the CloudFormation bucket policies.
resource "aws_s3_bucket_policy" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.datasets.arn,
          "${aws_s3_bucket.datasets.arn}/*"
        ]
        Condition = {
          Bool = { "aws:SecureTransport" = "false" }
        }
      }
    ]
  })
}

resource "aws_s3_bucket_policy" "results" {
  bucket = aws_s3_bucket.results.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.results.arn,
          "${aws_s3_bucket.results.arn}/*"
        ]
        Condition = {
          Bool = { "aws:SecureTransport" = "false" }
        }
      }
    ]
  })
}

resource "aws_s3_bucket_policy" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.artifacts.arn,
          "${aws_s3_bucket.artifacts.arn}/*"
        ]
        Condition = {
          Bool = { "aws:SecureTransport" = "false" }
        }
      }
    ]
  })
}

# S3 Buckets
resource "aws_s3_bucket" "datasets" {
  #checkov:skip=CKV_AWS_144:Cross-region replication is not enabled for this
  #  framework; evaluation data is reproducible from source datasets and CRR
  #  would double storage cost. Enable per-environment if RPO requires it.
  #checkov:skip=CKV2_AWS_62:No event-driven consumers exist for these buckets;
  #  workflows are orchestrated by Step Functions, not S3 notifications.
  bucket = "${var.project_name}-datasets-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.trustops.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_logging" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "datasets/"
}

resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    id     = "archive-old-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket" "results" {
  #checkov:skip=CKV_AWS_144:Cross-region replication is not enabled for this
  #  framework; evaluation data is reproducible from source datasets and CRR
  #  would double storage cost. Enable per-environment if RPO requires it.
  #checkov:skip=CKV2_AWS_62:No event-driven consumers exist for these buckets;
  #  workflows are orchestrated by Step Functions, not S3 notifications.
  bucket = "${var.project_name}-results-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "results" {
  bucket = aws_s3_bucket.results.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "results" {
  bucket = aws_s3_bucket.results.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.trustops.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_logging" "results" {
  bucket = aws_s3_bucket.results.id

  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "results/"
}

resource "aws_s3_bucket_public_access_block" "results" {
  bucket = aws_s3_bucket.results.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "results" {
  bucket = aws_s3_bucket.results.id

  rule {
    id     = "archive-old-results"
    status = "Enabled"

    filter {}

    transition {
      days          = 180
      storage_class = "GLACIER"
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket" "artifacts" {
  #checkov:skip=CKV_AWS_144:Cross-region replication is not enabled for this
  #  framework; evaluation data is reproducible from source datasets and CRR
  #  would double storage cost. Enable per-environment if RPO requires it.
  #checkov:skip=CKV2_AWS_62:No event-driven consumers exist for these buckets;
  #  workflows are orchestrated by Step Functions, not S3 notifications.
  bucket = "${var.project_name}-artifacts-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.trustops.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_logging" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "artifacts/"
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-old-artifact-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# DynamoDB Tables
resource "aws_dynamodb_table" "workflows" {
  name         = "${var.project_name}-workflows-${var.environment}"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "workflow_id"

  attribute {
    name = "workflow_id"
    type = "S"
  }

  attribute {
    name = "created_at"
    type = "S"
  }

  attribute {
    name = "workflow_type"
    type = "S"
  }

  attribute {
    name = "status"
    type = "S"
  }

  global_secondary_index {
    name            = "workflow-type-index"
    hash_key        = "workflow_type"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "status-index"
    hash_key        = "status"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.trustops.arn
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_dynamodb_table" "models" {
  name         = "${var.project_name}-models-${var.environment}"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "model_id"

  attribute {
    name = "model_id"
    type = "S"
  }

  attribute {
    name = "created_at"
    type = "S"
  }

  global_secondary_index {
    name            = "created-at-index"
    hash_key        = "created_at"
    projection_type = "ALL"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.trustops.arn
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "trustops" {
  name              = "/aws/${var.project_name}/${var.environment}"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.trustops.arn

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "evaluation" {
  name              = "/aws/${var.project_name}/${var.environment}/evaluation"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.trustops.arn

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "finetuning" {
  name              = "/aws/${var.project_name}/${var.environment}/finetuning"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.trustops.arn

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "trustscoring" {
  name              = "/aws/${var.project_name}/${var.environment}/trustscoring"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.trustops.arn

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Bedrock Knowledge Base IAM Role
resource "aws_iam_role" "knowledge_base" {
  name = "${var.project_name}-kb-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role_policy" "knowledge_base_policy" {
  name = "knowledge-base-policy"
  role = aws_iam_role.knowledge_base.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel"
        ]
        Resource = "arn:aws:bedrock:${data.aws_region.current.name}::foundation-model/amazon.titan-embed-text-v2:0"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.datasets.arn,
          "${aws_s3_bucket.datasets.arn}/knowledge-base/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = aws_kms_key.trustops.arn
      }
    ]
  })
}

# Bedrock Knowledge Base
resource "aws_bedrockagent_knowledge_base" "trustops" {
  name     = "${var.project_name}-kb-${var.environment}"
  role_arn = aws_iam_role.knowledge_base.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:${data.aws_region.current.name}::foundation-model/amazon.titan-embed-text-v2:0"
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"
    opensearch_serverless_configuration {
      collection_arn    = aws_opensearchserverless_collection.kb_vectors.arn
      vector_index_name = "bedrock-knowledge-base-default-index"
      field_mapping {
        vector_field   = "bedrock-knowledge-base-default-vector"
        text_field     = "AMAZON_BEDROCK_TEXT_CHUNK"
        metadata_field = "AMAZON_BEDROCK_METADATA"
      }
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# OpenSearch Serverless collection (managed by Bedrock KB)
resource "aws_opensearchserverless_security_policy" "kb_encryption" {
  name = "${var.project_name}-kb-enc-${var.environment}"
  type = "encryption"
  policy = jsonencode({
    Rules = [
      {
        ResourceType = "collection"
        Resource     = ["collection/${var.project_name}-kb-${var.environment}"]
      }
    ]
    AWSOwnedKey = true
  })
}

// Accepted security debt SD-7 (see SECURITY.md). AllowFromPublic exposes the
// collection's public endpoint, but access is still gated by IAM through
// aws_opensearchserverless_access_policy.kb_data below - there is no anonymous
// read. Locking this down means provisioning an aoss VPC endpoint and setting
// SourceVPCEs, which also requires attaching the Lambda functions to a VPC.
resource "aws_opensearchserverless_security_policy" "kb_network" {
  name = "${var.project_name}-kb-net-${var.environment}"
  type = "network"
  policy = jsonencode([
    {
      Rules = [
        {
          ResourceType = "collection"
          Resource     = ["collection/${var.project_name}-kb-${var.environment}"]
        },
        {
          ResourceType = "dashboard"
          Resource     = ["collection/${var.project_name}-kb-${var.environment}"]
        }
      ]
      AllowFromPublic = true
    }
  ])
}

resource "aws_opensearchserverless_access_policy" "kb_data" {
  name = "${var.project_name}-kb-data-${var.environment}"
  type = "data"
  policy = jsonencode([
    {
      Rules = [
        {
          ResourceType = "index"
          Resource     = ["index/${var.project_name}-kb-${var.environment}/*"]
          Permission   = ["aoss:CreateIndex", "aoss:UpdateIndex", "aoss:DescribeIndex", "aoss:ReadDocument", "aoss:WriteDocument"]
        },
        {
          ResourceType = "collection"
          Resource     = ["collection/${var.project_name}-kb-${var.environment}"]
          Permission   = ["aoss:CreateCollectionItems", "aoss:DescribeCollectionItems", "aoss:UpdateCollectionItems"]
        }
      ]
      Principal = [
        aws_iam_role.knowledge_base.arn,
        aws_iam_role.lambda_execution.arn
      ]
    }
  ])
}

resource "aws_opensearchserverless_collection" "kb_vectors" {
  name = "${var.project_name}-kb-${var.environment}"
  type = "VECTORSEARCH"

  depends_on = [
    aws_opensearchserverless_security_policy.kb_encryption,
    aws_opensearchserverless_security_policy.kb_network,
    aws_opensearchserverless_access_policy.kb_data
  ]

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Knowledge Base S3 Data Source
resource "aws_bedrockagent_data_source" "s3" {
  name              = "${var.project_name}-kb-s3-${var.environment}"
  knowledge_base_id = aws_bedrockagent_knowledge_base.trustops.id

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn         = aws_s3_bucket.datasets.arn
      inclusion_prefixes = ["knowledge-base/"]
    }
  }
}

# IAM Role for Lambda Functions
resource "aws_iam_role" "lambda_execution" {
  name = "${var.project_name}-lambda-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
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

resource "aws_iam_role_policy" "lambda_policy" {
  name = "trustops-lambda-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 permissions
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetObjectVersion"
        ]
        Resource = [
          aws_s3_bucket.datasets.arn,
          "${aws_s3_bucket.datasets.arn}/*",
          aws_s3_bucket.results.arn,
          "${aws_s3_bucket.results.arn}/*",
          aws_s3_bucket.artifacts.arn,
          "${aws_s3_bucket.artifacts.arn}/*"
        ]
      },
      # DynamoDB permissions
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = [
          aws_dynamodb_table.workflows.arn,
          "${aws_dynamodb_table.workflows.arn}/index/*",
          aws_dynamodb_table.models.arn,
          "${aws_dynamodb_table.models.arn}/index/*"
        ]
      },
      # Bedrock permissions (scoped to foundation and custom models)
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:GetFoundationModel",
          "bedrock:ListFoundationModels"
        ]
        Resource = [
          "arn:aws:bedrock:${data.aws_region.current.name}::foundation-model/*",
          "arn:aws:bedrock:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:custom-model/*",
          "arn:aws:bedrock:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:provisioned-model/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:CreateModelCustomizationJob",
          "bedrock:GetModelCustomizationJob",
          "bedrock:ListModelCustomizationJobs"
        ]
        Resource = [
          "arn:aws:bedrock:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:model-customization-job/*",
          "arn:aws:bedrock:${data.aws_region.current.name}::foundation-model/*"
        ]
      },
      # Bedrock Knowledge Base permissions
      {
        Effect = "Allow"
        Action = [
          "bedrock:Retrieve",
          "bedrock:RetrieveAndGenerate"
        ]
        Resource = aws_bedrockagent_knowledge_base.trustops.arn
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:GetKnowledgeBase",
          "bedrock:ListKnowledgeBases",
          "bedrock:StartIngestionJob",
          "bedrock:GetIngestionJob"
        ]
        Resource = aws_bedrockagent_knowledge_base.trustops.arn
      },
      # S3 delete for knowledge base document management
      {
        Effect = "Allow"
        Action = [
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.datasets.arn}/knowledge-base/*"
      },
      # CloudWatch Logs permissions
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          "${aws_cloudwatch_log_group.trustops.arn}:*",
          "${aws_cloudwatch_log_group.evaluation.arn}:*",
          "${aws_cloudwatch_log_group.finetuning.arn}:*",
          "${aws_cloudwatch_log_group.trustscoring.arn}:*"
        ]
      },
      # X-Ray tracing permissions
      {
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
          "xray:GetSamplingRules",
          "xray:GetSamplingTargets"
        ]
        Resource = "*"
      },
      # DLQ permissions
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage"
        ]
        Resource = aws_sqs_queue.lambda_dlq.arn
      },
      # KMS permissions
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.trustops.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# IAM Role for Step Functions
resource "aws_iam_role" "stepfunctions_execution" {
  name = "${var.project_name}-stepfunctions-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "states.amazonaws.com"
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

resource "aws_iam_role_policy" "stepfunctions_policy" {
  name = "trustops-stepfunctions-policy"
  role = aws_iam_role.stepfunctions_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Lambda invocation permissions
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = "arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:${var.project_name}-*"
      },
      # CloudWatch Logs permissions for Step Functions logging
      # Note: Log delivery APIs are account-level and require Resource: '*'
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogDelivery",
          "logs:GetLogDelivery",
          "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery",
          "logs:ListLogDeliveries",
          "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies",
          "logs:DescribeLogGroups"
        ]
        Resource = "*"
      },
      # X-Ray tracing
      {
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
          "xray:GetSamplingRules",
          "xray:GetSamplingTargets"
        ]
        Resource = "*"
      }
    ]
  })
}

# Lambda Functions
resource "aws_lambda_function" "evaluation_orchestrator" {
  function_name = "${var.project_name}-evaluation-orchestrator-${var.environment}"
  role          = aws_iam_role.lambda_execution.arn
  kms_key_arn   = aws_kms_key.trustops.arn
  #checkov:skip=CKV_AWS_117:TrustOps Lambdas only call public AWS API endpoints
  #  (Bedrock, S3, DynamoDB, Step Functions) and hold no inbound network surface.
  #  VPC attachment would require NAT/VPC endpoints for no isolation benefit.
  #checkov:skip=CKV_AWS_272:Code-signing requires an AWS Signer profile and a
  #  signed artifact pipeline; deployment packages are integrity-checked via S3
  #  versioning and SHA256 checksums instead.
  handler       = "evaluation_handler.lambda_handler"
  runtime       = "python3.11"
  timeout       = 900
  memory_size   = 1024
  architectures = ["arm64"]

  reserved_concurrent_executions = 10

  s3_bucket = aws_s3_bucket.artifacts.id
  s3_key    = "lambda/evaluation_handler.zip"

  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      DATASETS_BUCKET               = aws_s3_bucket.datasets.id
      RESULTS_BUCKET                = aws_s3_bucket.results.id
      WORKFLOWS_TABLE               = aws_dynamodb_table.workflows.name
      MODELS_TABLE                  = aws_dynamodb_table.models.name
      KNOWLEDGE_BASE_ID             = aws_bedrockagent_knowledge_base.trustops.id
      KNOWLEDGE_BASE_DATA_SOURCE_ID = aws_bedrockagent_data_source.s3.data_source_id
      LOG_LEVEL                     = "INFO"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_lambda_function" "finetuning_orchestrator" {
  function_name = "${var.project_name}-finetuning-orchestrator-${var.environment}"
  role          = aws_iam_role.lambda_execution.arn
  kms_key_arn   = aws_kms_key.trustops.arn
  #checkov:skip=CKV_AWS_117:TrustOps Lambdas only call public AWS API endpoints
  #  (Bedrock, S3, DynamoDB, Step Functions) and hold no inbound network surface.
  #  VPC attachment would require NAT/VPC endpoints for no isolation benefit.
  #checkov:skip=CKV_AWS_272:Code-signing requires an AWS Signer profile and a
  #  signed artifact pipeline; deployment packages are integrity-checked via S3
  #  versioning and SHA256 checksums instead.
  handler       = "fine_tuning_handler.lambda_handler"
  runtime       = "python3.11"
  timeout       = 900
  memory_size   = 512
  architectures = ["arm64"]

  reserved_concurrent_executions = 5

  s3_bucket = aws_s3_bucket.artifacts.id
  s3_key    = "lambda/fine_tuning_handler.zip"

  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      DATASETS_BUCKET = aws_s3_bucket.datasets.id
      RESULTS_BUCKET  = aws_s3_bucket.results.id
      WORKFLOWS_TABLE = aws_dynamodb_table.workflows.name
      MODELS_TABLE    = aws_dynamodb_table.models.name
      LOG_LEVEL       = "INFO"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_lambda_function" "trust_scoring" {
  function_name = "${var.project_name}-trust-scoring-${var.environment}"
  role          = aws_iam_role.lambda_execution.arn
  kms_key_arn   = aws_kms_key.trustops.arn
  #checkov:skip=CKV_AWS_117:TrustOps Lambdas only call public AWS API endpoints
  #  (Bedrock, S3, DynamoDB, Step Functions) and hold no inbound network surface.
  #  VPC attachment would require NAT/VPC endpoints for no isolation benefit.
  #checkov:skip=CKV_AWS_272:Code-signing requires an AWS Signer profile and a
  #  signed artifact pipeline; deployment packages are integrity-checked via S3
  #  versioning and SHA256 checksums instead.
  handler       = "trust_scoring_handler.lambda_handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 512
  architectures = ["arm64"]

  reserved_concurrent_executions = 10

  s3_bucket = aws_s3_bucket.artifacts.id
  s3_key    = "lambda/trust_scoring_handler.zip"

  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      KNOWLEDGE_BASE_ID             = aws_bedrockagent_knowledge_base.trustops.id
      KNOWLEDGE_BASE_DATA_SOURCE_ID = aws_bedrockagent_data_source.s3.data_source_id
      TRUST_SCORE_THRESHOLD         = "0.7"
      LOG_LEVEL                     = "INFO"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_lambda_function" "comparative_evaluation" {
  function_name = "${var.project_name}-comparative-evaluation-${var.environment}"
  role          = aws_iam_role.lambda_execution.arn
  kms_key_arn   = aws_kms_key.trustops.arn
  #checkov:skip=CKV_AWS_117:TrustOps Lambdas only call public AWS API endpoints
  #  (Bedrock, S3, DynamoDB, Step Functions) and hold no inbound network surface.
  #  VPC attachment would require NAT/VPC endpoints for no isolation benefit.
  #checkov:skip=CKV_AWS_272:Code-signing requires an AWS Signer profile and a
  #  signed artifact pipeline; deployment packages are integrity-checked via S3
  #  versioning and SHA256 checksums instead.
  handler       = "comparative_evaluation_handler.lambda_handler"
  runtime       = "python3.11"
  timeout       = 900
  memory_size   = 1024
  architectures = ["arm64"]

  reserved_concurrent_executions = 10

  s3_bucket = aws_s3_bucket.artifacts.id
  s3_key    = "lambda/comparative_evaluation_handler.zip"

  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      DATASETS_BUCKET               = aws_s3_bucket.datasets.id
      RESULTS_BUCKET                = aws_s3_bucket.results.id
      WORKFLOWS_TABLE               = aws_dynamodb_table.workflows.name
      MODELS_TABLE                  = aws_dynamodb_table.models.name
      KNOWLEDGE_BASE_ID             = aws_bedrockagent_knowledge_base.trustops.id
      KNOWLEDGE_BASE_DATA_SOURCE_ID = aws_bedrockagent_data_source.s3.data_source_id
      LOG_LEVEL                     = "INFO"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Granular Lambda Handlers for Step Functions Orchestration
resource "aws_lambda_function" "evaluate_single_example" {
  function_name = "${var.project_name}-evaluate-single-example-${var.environment}"
  role          = aws_iam_role.lambda_execution.arn
  kms_key_arn   = aws_kms_key.trustops.arn
  #checkov:skip=CKV_AWS_117:TrustOps Lambdas only call public AWS API endpoints
  #  (Bedrock, S3, DynamoDB, Step Functions) and hold no inbound network surface.
  #  VPC attachment would require NAT/VPC endpoints for no isolation benefit.
  #checkov:skip=CKV_AWS_272:Code-signing requires an AWS Signer profile and a
  #  signed artifact pipeline; deployment packages are integrity-checked via S3
  #  versioning and SHA256 checksums instead.
  handler       = "evaluate_single_example.handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 512
  architectures = ["arm64"]

  reserved_concurrent_executions = 20

  s3_bucket = aws_s3_bucket.artifacts.id
  s3_key    = "lambda/evaluate_single_example.zip"

  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      DATASETS_BUCKET = aws_s3_bucket.datasets.id
      RESULTS_BUCKET  = aws_s3_bucket.results.id
      WORKFLOWS_TABLE = aws_dynamodb_table.workflows.name
      LOG_LEVEL       = "INFO"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_lambda_function" "aggregate_metrics" {
  function_name = "${var.project_name}-aggregate-metrics-${var.environment}"
  role          = aws_iam_role.lambda_execution.arn
  kms_key_arn   = aws_kms_key.trustops.arn
  #checkov:skip=CKV_AWS_117:TrustOps Lambdas only call public AWS API endpoints
  #  (Bedrock, S3, DynamoDB, Step Functions) and hold no inbound network surface.
  #  VPC attachment would require NAT/VPC endpoints for no isolation benefit.
  #checkov:skip=CKV_AWS_272:Code-signing requires an AWS Signer profile and a
  #  signed artifact pipeline; deployment packages are integrity-checked via S3
  #  versioning and SHA256 checksums instead.
  handler       = "aggregate_metrics.handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 512
  architectures = ["arm64"]

  reserved_concurrent_executions = 5

  s3_bucket = aws_s3_bucket.artifacts.id
  s3_key    = "lambda/aggregate_metrics.zip"

  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      RESULTS_BUCKET  = aws_s3_bucket.results.id
      WORKFLOWS_TABLE = aws_dynamodb_table.workflows.name
      LOG_LEVEL       = "INFO"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_lambda_function" "generate_recommendation" {
  function_name = "${var.project_name}-generate-recommendation-${var.environment}"
  role          = aws_iam_role.lambda_execution.arn
  kms_key_arn   = aws_kms_key.trustops.arn
  #checkov:skip=CKV_AWS_117:TrustOps Lambdas only call public AWS API endpoints
  #  (Bedrock, S3, DynamoDB, Step Functions) and hold no inbound network surface.
  #  VPC attachment would require NAT/VPC endpoints for no isolation benefit.
  #checkov:skip=CKV_AWS_272:Code-signing requires an AWS Signer profile and a
  #  signed artifact pipeline; deployment packages are integrity-checked via S3
  #  versioning and SHA256 checksums instead.
  handler       = "generate_recommendation.handler"
  runtime       = "python3.11"
  timeout       = 120
  memory_size   = 256
  architectures = ["arm64"]

  reserved_concurrent_executions = 5

  s3_bucket = aws_s3_bucket.artifacts.id
  s3_key    = "lambda/generate_recommendation.zip"

  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      WORKFLOWS_TABLE = aws_dynamodb_table.workflows.name
      LOG_LEVEL       = "INFO"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_lambda_function" "detect_hallucinations" {
  function_name = "${var.project_name}-detect-hallucinations-${var.environment}"
  role          = aws_iam_role.lambda_execution.arn
  kms_key_arn   = aws_kms_key.trustops.arn
  #checkov:skip=CKV_AWS_117:TrustOps Lambdas only call public AWS API endpoints
  #  (Bedrock, S3, DynamoDB, Step Functions) and hold no inbound network surface.
  #  VPC attachment would require NAT/VPC endpoints for no isolation benefit.
  #checkov:skip=CKV_AWS_272:Code-signing requires an AWS Signer profile and a
  #  signed artifact pipeline; deployment packages are integrity-checked via S3
  #  versioning and SHA256 checksums instead.
  handler       = "detect_hallucinations.handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 512
  architectures = ["arm64"]

  reserved_concurrent_executions = 10

  s3_bucket = aws_s3_bucket.artifacts.id
  s3_key    = "lambda/detect_hallucinations.zip"

  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      KNOWLEDGE_BASE_ID             = aws_bedrockagent_knowledge_base.trustops.id
      KNOWLEDGE_BASE_DATA_SOURCE_ID = aws_bedrockagent_data_source.s3.data_source_id
      RESULTS_BUCKET                = aws_s3_bucket.results.id
      LOG_LEVEL                     = "INFO"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Lambda Permissions for Step Functions (new handlers)
resource "aws_lambda_permission" "evaluate_single_example_invoke" {
  statement_id  = "AllowStepFunctionsInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.evaluate_single_example.function_name
  principal     = "states.amazonaws.com"
  source_arn    = "arn:aws:states:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:stateMachine:${var.project_name}-*"
}

resource "aws_lambda_permission" "aggregate_metrics_invoke" {
  statement_id  = "AllowStepFunctionsInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.aggregate_metrics.function_name
  principal     = "states.amazonaws.com"
  source_arn    = "arn:aws:states:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:stateMachine:${var.project_name}-*"
}

resource "aws_lambda_permission" "generate_recommendation_invoke" {
  statement_id  = "AllowStepFunctionsInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.generate_recommendation.function_name
  principal     = "states.amazonaws.com"
  source_arn    = "arn:aws:states:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:stateMachine:${var.project_name}-*"
}

resource "aws_lambda_permission" "detect_hallucinations_invoke" {
  statement_id  = "AllowStepFunctionsInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.detect_hallucinations.function_name
  principal     = "states.amazonaws.com"
  source_arn    = "arn:aws:states:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:stateMachine:${var.project_name}-*"
}

# Lambda Permissions for Step Functions (existing handlers)
resource "aws_lambda_permission" "evaluation_orchestrator_invoke" {
  statement_id  = "AllowStepFunctionsInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.evaluation_orchestrator.function_name
  principal     = "states.amazonaws.com"
  source_arn    = "arn:aws:states:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:stateMachine:${var.project_name}-*"
}

resource "aws_lambda_permission" "finetuning_orchestrator_invoke" {
  statement_id  = "AllowStepFunctionsInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.finetuning_orchestrator.function_name
  principal     = "states.amazonaws.com"
  source_arn    = "arn:aws:states:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:stateMachine:${var.project_name}-*"
}

resource "aws_lambda_permission" "trust_scoring_invoke" {
  statement_id  = "AllowStepFunctionsInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trust_scoring.function_name
  principal     = "states.amazonaws.com"
  source_arn    = "arn:aws:states:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:stateMachine:${var.project_name}-*"
}

resource "aws_lambda_permission" "comparative_evaluation_invoke" {
  statement_id  = "AllowStepFunctionsInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.comparative_evaluation.function_name
  principal     = "states.amazonaws.com"
  source_arn    = "arn:aws:states:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:stateMachine:${var.project_name}-*"
}

# Step Functions State Machines
resource "aws_sfn_state_machine" "baseline_evaluation" {
  name     = "${var.project_name}-baseline-evaluation-${var.environment}"
  role_arn = aws_iam_role.stepfunctions_execution.arn

  definition = file("${path.module}/../step_functions/baseline_evaluation_workflow.json")

  tracing_configuration {
    enabled = true
  }

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.evaluation.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_sfn_state_machine" "fine_tuning" {
  name     = "${var.project_name}-fine-tuning-${var.environment}"
  role_arn = aws_iam_role.stepfunctions_execution.arn

  definition = file("${path.module}/../step_functions/fine_tuning_workflow.json")

  tracing_configuration {
    enabled = true
  }

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.finetuning.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_sfn_state_machine" "comparative_evaluation" {
  name     = "${var.project_name}-comparative-evaluation-${var.environment}"
  role_arn = aws_iam_role.stepfunctions_execution.arn

  definition = file("${path.module}/../step_functions/comparative_evaluation_workflow.json")

  tracing_configuration {
    enabled = true
  }

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.evaluation.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Include QuickSight resources
module "quicksight" {
  source = "../quicksight"

  project_name = var.project_name
  environment  = var.environment
  aws_region   = var.aws_region

  # Pass required resources
  results_bucket_id          = aws_s3_bucket.results.id
  results_bucket_arn         = aws_s3_bucket.results.arn
  datasets_bucket_id         = aws_s3_bucket.datasets.id
  datasets_bucket_arn        = aws_s3_bucket.datasets.arn
  workflows_table_name       = aws_dynamodb_table.workflows.name
  workflows_table_arn        = aws_dynamodb_table.workflows.arn
  models_table_name          = aws_dynamodb_table.models.name
  models_table_arn           = aws_dynamodb_table.models.arn
  trustops_log_group_arn     = aws_cloudwatch_log_group.trustops.arn
  evaluation_log_group_arn   = aws_cloudwatch_log_group.evaluation.arn
  finetuning_log_group_arn   = aws_cloudwatch_log_group.finetuning.arn
  trustscoring_log_group_arn = aws_cloudwatch_log_group.trustscoring.arn
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "${var.project_name}-high-error-rate-${var.environment}"
  alarm_description   = "Alert when error rate exceeds 5%"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "lambda_timeout" {
  alarm_name          = "${var.project_name}-lambda-timeout-${var.environment}"
  alarm_description   = "Alert when Lambda timeout rate exceeds 10%"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Average"
  threshold           = 850000
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "workflow_failure" {
  alarm_name          = "${var.project_name}-workflow-failure-${var.environment}"
  alarm_description   = "Alert when workflow failure rate exceeds 20%"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ExecutionsFailed"
  namespace           = "AWS/States"
  period              = 3600
  statistic           = "Sum"
  threshold           = 5
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "trust_score_degradation" {
  alarm_name          = "${var.project_name}-trust-score-degradation-${var.environment}"
  alarm_description   = "Alert when mean trust score drops more than 10%"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "TrustScore"
  namespace           = "TrustOps"
  period              = 3600
  statistic           = "Average"
  threshold           = 0.6
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "dlq_messages" {
  alarm_name          = "${var.project_name}-dlq-messages-${var.environment}"
  alarm_description   = "Alert when messages land in the dead letter queue"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]

  dimensions = {
    QueueName = aws_sqs_queue.lambda_dlq.name
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
