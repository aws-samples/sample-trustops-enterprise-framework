# TrustOps AWS Demo - Operational Runbook

## Table of Contents

1. [Deployment Procedures](#deployment-procedures)
2. [Monitoring and Alerting](#monitoring-and-alerting)
3. [Troubleshooting](#troubleshooting)
4. [Backup and Recovery](#backup-and-recovery)
5. [Cost Optimization](#cost-optimization)
6. [Security Best Practices](#security-best-practices)
7. [FAQ](#faq)

## Deployment Procedures

### Prerequisites

Before deploying TrustOps, ensure you have:

- **AWS Account** with appropriate permissions
- **AWS CLI** installed and configured (version 2.x or higher)
- **Python 3.9+** installed
- **Terraform** (optional, for infrastructure as code)
- **Access to AWS Bedrock** (may require requesting access in some regions)

### Required IAM Permissions

The deploying user/role needs permissions for:
- S3 (create buckets, manage objects, enable versioning)
- DynamoDB (create tables, read/write items)
- Lambda (create functions, manage code, configure triggers)
- Step Functions (create state machines, start executions)
- CloudWatch (create log groups, put metrics, create alarms)
- IAM (create roles and policies for Lambda/Step Functions)
- Bedrock (invoke models, create fine-tuning jobs)
- CloudFormation (create and manage stacks)

### Step 1: Clone Repository and Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd trustops-aws-demo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


### Step 2: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Required Variables:**
```bash
AWS_REGION=us-east-1
TRUSTOPS_DATASETS_BUCKET=trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_RESULTS_BUCKET=trustops-results-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_ARTIFACTS_BUCKET=trustops-artifacts-<YOUR_ACCOUNT_ID>-<REGION>
```

**Note:** S3 bucket names must be globally unique. Use a prefix like your organization name.

### Step 3: Deploy Infrastructure

#### Option A: Using CloudFormation (Recommended)

```bash
cd infrastructure

# Deploy the stack
./deploy.sh

# Wait for deployment to complete (5-10 minutes)
# The script will output resource names and ARNs
```

#### Option B: Using Terraform

```bash
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Review planned changes
terraform plan

# Apply changes
terraform apply

# Confirm with 'yes' when prompted
```

### Step 4: Verify Deployment

```bash
# Run verification script
python scripts/verify_aws_setup.py

# Expected output:
# ✓ AWS credentials configured
# ✓ S3 buckets accessible
# ✓ DynamoDB tables exist
# ✓ CloudWatch log groups exist
# ✓ Bedrock accessible
```


### Step 5: Deploy Lambda Functions

```bash
# Package Lambda functions with dependencies
cd lambda_handlers
./package_lambdas.sh

# Deploy Lambda functions
aws lambda update-function-code \
  --function-name trustops-evaluation-handler \
  --zip-file fileb://evaluation_handler.zip

aws lambda update-function-code \
  --function-name trustops-trust-scoring-handler \
  --zip-file fileb://trust_scoring_handler.zip

aws lambda update-function-code \
  --function-name trustops-fine-tuning-handler \
  --zip-file fileb://fine_tuning_handler.zip

aws lambda update-function-code \
  --function-name trustops-comparative-evaluation-handler \
  --zip-file fileb://comparative_evaluation_handler.zip
```

### Step 6: Deploy Step Functions Workflows

```bash
cd step_functions

# Deploy workflows
./deploy_workflows.sh

# Verify workflows
python validate_workflows.py
```

### Step 7: Install CLI (Optional)

```bash
# Install TrustOps CLI
pip install -e .

# Verify installation
trustops --version

# Configure CLI
trustops configure
```

### Step 8: Run Test Evaluation

```bash
# Run demo to verify end-to-end functionality
cd demo
python run_demo.py

# Check results
trustops status --list
```


## Monitoring and Alerting

### CloudWatch Metrics

TrustOps automatically publishes custom metrics to CloudWatch:

#### Trust Score Metrics

| Metric Name | Description | Unit | Namespace |
|-------------|-------------|------|-----------|
| `TrustScore` | Individual trust scores | None (0-1) | TrustOps/Evaluation |
| `MeanTrustScore` | Average trust score per evaluation | None (0-1) | TrustOps/Evaluation |
| `LowTrustResponseCount` | Count of responses below threshold | Count | TrustOps/Evaluation |
| `HallucinationRate` | Percentage of unsupported claims | Percent | TrustOps/Evaluation |

#### Performance Metrics

| Metric Name | Description | Unit | Namespace |
|-------------|-------------|------|-----------|
| `InferenceLatency` | Model inference latency | Milliseconds | TrustOps/Performance |
| `TrustScoringLatency` | Trust score calculation time | Milliseconds | TrustOps/Performance |
| `WorkflowDuration` | Total workflow execution time | Seconds | TrustOps/Workflow |

#### Cost Metrics

| Metric Name | Description | Unit | Namespace |
|-------------|-------------|------|-----------|
| `EvaluationCost` | Cost per evaluation run | USD | TrustOps/Cost |
| `CostPerQuery` | Average cost per query | USD | TrustOps/Cost |
| `TokenUsage` | Total tokens consumed | Count | TrustOps/Cost |

### CloudWatch Alarms

#### Recommended Alarms

**1. High Error Rate Alarm**

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name trustops-high-error-rate \
  --alarm-description "Alert when error rate exceeds 5%" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 0.05 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=FunctionName,Value=trustops-evaluation-handler
```


**2. Trust Score Degradation Alarm**

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name trustops-trust-score-degradation \
  --alarm-description "Alert when mean trust score drops below 0.6" \
  --metric-name MeanTrustScore \
  --namespace TrustOps/Evaluation \
  --statistic Average \
  --period 3600 \
  --evaluation-periods 1 \
  --threshold 0.6 \
  --comparison-operator LessThanThreshold
```

**3. Lambda Timeout Alarm**

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name trustops-lambda-timeouts \
  --alarm-description "Alert when Lambda timeout rate exceeds 10%" \
  --metric-name Duration \
  --namespace AWS/Lambda \
  --statistic Maximum \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 850000 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=FunctionName,Value=trustops-evaluation-handler
```

**4. Cost Anomaly Alarm**

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name trustops-cost-anomaly \
  --alarm-description "Alert when daily cost exceeds $100" \
  --metric-name EvaluationCost \
  --namespace TrustOps/Cost \
  --statistic Sum \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold
```

### CloudWatch Dashboards

Create a dashboard for monitoring:

```bash
aws cloudwatch put-dashboard \
  --dashboard-name TrustOps-Operations \
  --dashboard-body file://cloudwatch-dashboard.json
```


### Log Monitoring

#### Key Log Groups

| Log Group | Purpose | Retention |
|-----------|---------|-----------|
| `/aws/lambda/trustops-evaluation-handler` | Evaluation workflow logs | 30 days |
| `/aws/lambda/trustops-trust-scoring-handler` | Trust scoring logs | 30 days |
| `/aws/lambda/trustops-fine-tuning-handler` | Fine-tuning logs | 90 days |
| `/aws/trustops/workflows` | Workflow audit logs | 365 days |

#### Log Insights Queries

**Query 1: Find Failed Workflows**

```sql
fields @timestamp, workflow_id, error_message
| filter status = "failed"
| sort @timestamp desc
| limit 20
```

**Query 2: Trust Score Distribution**

```sql
fields @timestamp, trust_score
| stats avg(trust_score) as avg_score, 
        min(trust_score) as min_score, 
        max(trust_score) as max_score 
  by bin(5m)
```

**Query 3: High Latency Requests**

```sql
fields @timestamp, workflow_id, latency_ms
| filter latency_ms > 5000
| sort latency_ms desc
| limit 50
```

**Query 4: Cost Analysis**

```sql
fields @timestamp, workflow_id, total_cost
| stats sum(total_cost) as daily_cost by bin(1d)
| sort @timestamp desc
```


## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Bedrock Access Denied

**Symptoms:**
- Error: "AccessDeniedException: User is not authorized to perform: bedrock:InvokeModel"
- Evaluation workflows fail immediately

**Diagnosis:**
```bash
# Check Bedrock access
aws bedrock list-foundation-models --region us-east-1

# Check IAM permissions
aws iam get-role-policy \
  --role-name TrustOpsLambdaExecutionRole \
  --policy-name BedrockAccess
```

**Solutions:**

1. **Request Bedrock Access:**
   - Go to AWS Console → Bedrock → Model access
   - Request access to required models (Claude, Titan, etc.)
   - Wait for approval (can take a few hours)

2. **Update IAM Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:CreateModelCustomizationJob",
        "bedrock:GetModelCustomizationJob"
      ],
      "Resource": "*"
    }
  ]
}
```

3. **Verify Region:**
   - Bedrock is not available in all regions
   - Use us-east-1 or us-west-2 for best availability


#### Issue 2: Lambda Timeout

**Symptoms:**
- Error: "Task timed out after 900.00 seconds"
- Workflows stuck in "running" state
- Incomplete evaluation results

**Diagnosis:**
```bash
# Check Lambda duration metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=trustops-evaluation-handler \
  --start-time 2024-01-15T00:00:00Z \
  --end-time 2024-01-15T23:59:59Z \
  --period 3600 \
  --statistics Maximum
```

**Solutions:**

1. **Increase Lambda Timeout:**
```bash
aws lambda update-function-configuration \
  --function-name trustops-evaluation-handler \
  --timeout 900
```

2. **Increase Lambda Memory:**
```bash
# More memory = more CPU = faster execution
aws lambda update-function-configuration \
  --function-name trustops-evaluation-handler \
  --memory-size 3008
```

3. **Enable Dataset Partitioning:**
   - For large datasets (>100 examples), enable automatic partitioning
   - Set `ENABLE_PARTITIONING=true` in environment variables
   - Adjust `PARTITION_SIZE` based on Lambda timeout

4. **Use Step Functions for Long Workflows:**
   - Step Functions can orchestrate workflows longer than 15 minutes
   - Already configured in `step_functions/baseline_evaluation_workflow.json`


#### Issue 3: S3 Access Denied

**Symptoms:**
- Error: "AccessDenied: Access Denied"
- Cannot upload datasets or retrieve results

**Diagnosis:**
```bash
# Test S3 access
aws s3 ls s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/

# Check bucket policy
aws s3api get-bucket-policy \
  --bucket trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
```

**Solutions:**

1. **Update Lambda Execution Role:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::trustops-*",
        "arn:aws:s3:::trustops-*/*"
      ]
    }
  ]
}
```

2. **Check Bucket Encryption:**
   - If bucket uses KMS encryption, Lambda role needs KMS permissions
   - Add `kms:Decrypt` and `kms:GenerateDataKey` permissions

3. **Verify Bucket Names:**
   - Ensure environment variables match actual bucket names
   - Check for typos in `.env` file


#### Issue 4: Knowledge Base Retrieval Failed

**Symptoms:**
- Error: "Failed to retrieve from Knowledge Base"
- Semantic similarity calculations fail
- Hallucination detection not working

**Diagnosis:**
```bash
# Check the Knowledge Base status
aws bedrock-agent get-knowledge-base \
  --knowledge-base-id "$KNOWLEDGE_BASE_ID"

# Check the most recent ingestion job
aws bedrock-agent list-ingestion-jobs \
  --knowledge-base-id "$KNOWLEDGE_BASE_ID" \
  --data-source-id "$KNOWLEDGE_BASE_DATA_SOURCE_ID"
```

**Solutions:**

1. **Grant the Lambda role retrieve permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:Retrieve",
        "bedrock:GetKnowledgeBase"
      ],
      "Resource": "arn:aws:bedrock:us-east-1:123456789012:knowledge-base/YOUR_KB_ID"
    }
  ]
}
```

2. **Confirm documents were ingested:**
   - Documents must be uploaded under the `knowledge-base/` prefix of the datasets bucket
   - A data source sync must run after upload; `store_document(sync=True)` triggers one
   - A sync in `IN_PROGRESS` means results are not yet queryable

3. **Verify Configuration:**
   - Check `KNOWLEDGE_BASE_ID` and `KNOWLEDGE_BASE_DATA_SOURCE_ID` in `.env`
   - Confirm the Knowledge Base is in the same region as the rest of the stack

4. **Fallback Mode:**
   - With no `KNOWLEDGE_BASE_ID` set, the clients run in mock mode rather than failing
   - Context grounding score defaults to 0.5 (neutral)


#### Issue 5: High Costs

**Symptoms:**
- AWS bill higher than expected
- Cost metrics show unexpected spikes

**Diagnosis:**
```bash
# Check cost metrics
aws cloudwatch get-metric-statistics \
  --namespace TrustOps/Cost \
  --metric-name EvaluationCost \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-31T23:59:59Z \
  --period 86400 \
  --statistics Sum

# Check token usage
aws cloudwatch get-metric-statistics \
  --namespace TrustOps/Cost \
  --metric-name TokenUsage \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-31T23:59:59Z \
  --period 86400 \
  --statistics Sum
```

**Solutions:**

1. **Review Model Selection:**
   - Use Claude Instant instead of Claude v2 for development
   - Claude Instant is 10x cheaper
   - Update `DEFAULT_FOUNDATION_MODEL` in `.env`

2. **Optimize Dataset Size:**
   - Use smaller evaluation datasets during development
   - Sample large datasets for quick iterations

3. **Enable Caching:**
   - Cache evaluation results to avoid re-running
   - Use workflow reproduction for auditing

4. **Set Cost Alerts:**
   - Configure CloudWatch alarms for cost thresholds
   - Review costs daily during active development

5. **Clean Up Resources:**
   - Delete old S3 objects with lifecycle policies
   - Remove unused fine-tuned models
   - Tear down the Bedrock Knowledge Base when retrieval is not in use


#### Issue 6: Data Validation Failures

**Symptoms:**
- Error: "DataValidationError: Invalid dataset format"
- Training data rejected
- Evaluation workflows fail at validation step

**Diagnosis:**
```bash
# Validate dataset locally
python demo/validate_sample_data.py path/to/dataset.json

# Check validation logs
aws logs tail /aws/lambda/trustops-evaluation-handler --follow
```

**Solutions:**

1. **Check Dataset Format:**
   - Evaluation datasets: JSON with array of examples
   - Training datasets: JSONL with one example per line
   - Required fields: `prompt`, `source_documents`, `category`

2. **Example Valid Evaluation Dataset:**
```json
[
  {
    "prompt": "What is the capital of France?",
    "expected_response": "Paris",
    "source_documents": ["France's capital is Paris."],
    "category": "general",
    "metadata": {}
  }
]
```

3. **Example Valid Training Dataset:**
```jsonl
{"prompt": "Question: What is AI?", "completion": "AI is artificial intelligence."}
{"prompt": "Question: What is ML?", "completion": "ML is machine learning."}
```

4. **Common Validation Errors:**
   - Missing required fields → Add all required fields
   - Invalid JSON syntax → Validate with `jq` or JSON validator
   - Empty source_documents → Provide at least one document
   - Insufficient training examples → Minimum 10 examples required


## Backup and Recovery

### Data Backup Strategy

#### S3 Versioning

All S3 buckets have versioning enabled by default:

```bash
# Verify versioning is enabled
aws s3api get-bucket-versioning \
  --bucket trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>

# List object versions
aws s3api list-object-versions \
  --bucket trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION> \
  --prefix evaluation-data/
```

**Recovery:**
```bash
# Restore previous version
aws s3api copy-object \
  --bucket trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION> \
  --copy-source trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/dataset.json?versionId=VERSION_ID \
  --key dataset.json
```

#### DynamoDB Point-in-Time Recovery

Enable PITR for workflow metadata:

```bash
# Enable PITR
aws dynamodb update-continuous-backups \
  --table-name trustops-workflows \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true

# Verify PITR status
aws dynamodb describe-continuous-backups \
  --table-name trustops-workflows
```

**Recovery:**
```bash
# Restore table to specific time
aws dynamodb restore-table-to-point-in-time \
  --source-table-name trustops-workflows \
  --target-table-name trustops-workflows-restored \
  --restore-date-time 2024-01-15T12:00:00Z
```


### Backup Procedures

#### Daily Backup Script

```bash
#!/bin/bash
# daily_backup.sh

BACKUP_DATE=$(date +%Y%m%d)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="${AWS_REGION:-us-east-1}"
# Always qualify the bucket with account and region. S3 bucket names are
# globally unique, so an unqualified short name may already belong to another
# account, and backups would then be written where you cannot see them.
BACKUP_BUCKET="<YOUR_BACKUP_BUCKET>-${ACCOUNT_ID}-${AWS_REGION}"

# Backup S3 datasets
aws s3 sync s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/ \
  s3://${BACKUP_BUCKET}/datasets/${BACKUP_DATE}/ \
  --storage-class GLACIER_IR

# Backup DynamoDB tables
aws dynamodb create-backup \
  --table-name trustops-workflows \
  --backup-name trustops-workflows-${BACKUP_DATE}

aws dynamodb create-backup \
  --table-name trustops-models \
  --backup-name trustops-models-${BACKUP_DATE}

# Export CloudWatch logs
aws logs create-export-task \
  --log-group-name /aws/trustops/workflows \
  --from 0 \
  --to $(date +%s)000 \
  --destination ${BACKUP_BUCKET} \
  --destination-prefix logs/${BACKUP_DATE}/
```

#### Automated Backup with EventBridge

```bash
# Create EventBridge rule for daily backups
aws events put-rule \
  --name trustops-daily-backup \
  --schedule-expression "cron(0 2 * * ? *)" \
  --state ENABLED

# Add Lambda target
aws events put-targets \
  --rule trustops-daily-backup \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:123456789012:function:trustops-backup"
```

### Disaster Recovery

#### Recovery Time Objective (RTO): 4 hours
#### Recovery Point Objective (RPO): 24 hours

**Recovery Steps:**

1. **Restore Infrastructure:**
```bash
cd infrastructure
./deploy.sh
```

2. **Restore S3 Data:**
```bash
aws s3 sync s3://<YOUR_BACKUP_BUCKET>-<YOUR_ACCOUNT_ID>-<REGION>/datasets/latest/ \
  s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/
```

3. **Restore DynamoDB Tables:**
```bash
aws dynamodb restore-table-from-backup \
  --target-table-name trustops-workflows \
  --backup-arn arn:aws:dynamodb:us-east-1:123456789012:table/trustops-workflows/backup/latest
```

4. **Verify Recovery:**
```bash
python scripts/verify_aws_setup.py
```


## Cost Optimization

### Cost Optimization Recommendations

#### 1. Model Selection

| Model | Use Case | Cost | Performance |
|-------|----------|------|-------------|
| Claude Instant | Development, testing | $ | Fast, good quality |
| Claude v2 | Production, high-quality | $$$ | Best quality |
| Llama 2 70B | Cost-sensitive production | $$ | Good quality |
| Titan Text Express | Simple tasks | $ | Basic quality |

**Recommendation:** Use Claude Instant for development, Claude v2 for production.

#### 2. Lambda Optimization

**Right-Size Memory:**
```bash
# Monitor memory usage
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name MemoryUtilization \
  --dimensions Name=FunctionName,Value=trustops-evaluation-handler \
  --start-time 2024-01-15T00:00:00Z \
  --end-time 2024-01-15T23:59:59Z \
  --period 3600 \
  --statistics Average

# Adjust if utilization < 50%
aws lambda update-function-configuration \
  --function-name trustops-evaluation-handler \
  --memory-size 1536  # Reduce from 3008
```

**Use Provisioned Concurrency Sparingly:**
- Only for production workloads with consistent traffic
- Development environments should use on-demand

#### 3. S3 Storage Optimization

**Lifecycle Policies:**
```json
{
  "Rules": [
    {
      "Id": "ArchiveOldResults",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 90,
          "StorageClass": "GLACIER_IR"
        },
        {
          "Days": 365,
          "StorageClass": "DEEP_ARCHIVE"
        }
      ]
    },
    {
      "Id": "DeleteOldVersions",
      "Status": "Enabled",
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      }
    }
  ]
}
```


#### 4. Knowledge Base Optimization

**Development Environment:**
- Leave `KNOWLEDGE_BASE_ID` unset so retrieval runs in mock mode at no cost
- Provision the Knowledge Base only when testing real grounding

**Production Environment:**
- Keep the Knowledge Base in the same region as the Lambda functions
- Batch uploads with `bulk_store_documents()` so one ingestion job covers many
  documents instead of one job per document
- Re-sync on a schedule rather than per write when documents change often

**Tear down when idle:**
```bash
# The vector collection bills OCU-hours even when idle, so destroy the
# Knowledge Base stack when it is not needed.
cd infrastructure/terraform
terraform destroy
```

#### 5. DynamoDB Optimization

**Use On-Demand Billing for Variable Workloads:**
```bash
aws dynamodb update-table \
  --table-name trustops-workflows \
  --billing-mode PAY_PER_REQUEST
```

**Use Provisioned Capacity for Predictable Workloads:**
```bash
aws dynamodb update-table \
  --table-name trustops-workflows \
  --billing-mode PROVISIONED \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

#### 6. CloudWatch Logs Optimization

**Adjust Log Retention:**
```bash
# Reduce retention for non-critical logs
aws logs put-retention-policy \
  --log-group-name /aws/lambda/trustops-evaluation-handler \
  --retention-in-days 7  # Instead of 30
```

**Filter Logs:**
- Only log errors and warnings in production
- Use structured logging for efficient queries
- Avoid logging large payloads


### Cost Monitoring

**Set Up Cost Alerts:**
```bash
# Create budget
aws budgets create-budget \
  --account-id 123456789012 \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json
```

**budget.json:**
```json
{
  "BudgetName": "TrustOps-Monthly",
  "BudgetLimit": {
    "Amount": "500",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
```

**Track Costs by Tag:**
```bash
# Tag resources
aws s3api put-bucket-tagging \
  --bucket trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION> \
  --tagging 'TagSet=[{Key=Project,Value=TrustOps},{Key=Environment,Value=Dev}]'

# View costs by tag in Cost Explorer
```

### Estimated Monthly Costs

| Component | Development | Production |
|-----------|-------------|------------|
| Bedrock (Claude Instant) | $10-50 | $100-500 |
| Bedrock (Claude v2) | - | $500-2000 |
| Lambda | $5-20 | $50-200 |
| S3 | $1-5 | $10-50 |
| DynamoDB | $2-10 | $20-100 |
| Bedrock Knowledge Base | $0 (mock mode) | $200-500 (OpenSearch Serverless OCUs) |
| CloudWatch | $1-5 | $10-50 |
| **Total** | **$19-90** | **$890-3400** |

**Note:** Costs vary significantly based on usage volume. Production estimates assume 10,000-100,000 queries/month.


## Security Best Practices

### IAM Configuration

#### Principle of Least Privilege

**Lambda Execution Role:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:*:*:foundation-model/anthropic.claude-*",
        "arn:aws:bedrock:*:*:foundation-model/amazon.titan-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>-*/*",
        "arn:aws:s3:::trustops-results-<YOUR_ACCOUNT_ID>-<REGION>-*/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query"
      ],
      "Resource": [
        "arn:aws:dynamodb:*:*:table/trustops-workflows",
        "arn:aws:dynamodb:*:*:table/trustops-models"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/lambda/trustops-*"
    }
  ]
}
```

#### Service Control Policies

Restrict Bedrock model access:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": "bedrock:InvokeModel",
      "Resource": "*",
      "Condition": {
        "StringNotLike": {
          "bedrock:ModelId": [
            "anthropic.claude-*",
            "amazon.titan-*"
          ]
        }
      }
    }
  ]
}
```


### Data Encryption

#### Encryption at Rest

**S3 Bucket Encryption:**
```bash
# Enable default encryption with SSE-S3
aws s3api put-bucket-encryption \
  --bucket trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION> \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Or use KMS for more control
aws s3api put-bucket-encryption \
  --bucket trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION> \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "aws:kms",
        "KMSMasterKeyID": "arn:aws:kms:us-east-1:123456789012:key/KEY-ID"
      }
    }]
  }'
```

**DynamoDB Encryption:**
```bash
# Enable encryption at rest
aws dynamodb update-table \
  --table-name trustops-workflows \
  --sse-specification Enabled=true,SSEType=KMS
```

#### Encryption in Transit

- All AWS API calls use HTTPS by default
- Bedrock API enforces TLS 1.2+
- Bedrock Knowledge Base retrieval uses HTTPS connections

### Access Control

#### S3 Bucket Policies

**Block Public Access:**
```bash
aws s3api put-public-access-block \
  --bucket trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION> \
  --public-access-block-configuration \
    BlockPublicAcls=true,\
    IgnorePublicAcls=true,\
    BlockPublicPolicy=true,\
    RestrictPublicBuckets=true
```

**Restrict Access to Specific Roles:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>",
        "arn:aws:s3:::trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/*"
      ],
      "Condition": {
        "StringNotLike": {
          "aws:PrincipalArn": [
            "arn:aws:iam::123456789012:role/TrustOpsLambdaExecutionRole",
            "arn:aws:iam::123456789012:role/TrustOpsAdminRole"
          ]
        }
      }
    }
  ]
}
```


### Audit Logging

#### Enable CloudTrail

```bash
# Create trail for API audit logging.
#
# The destination bucket must be account- and region-qualified. Bucket names
# are globally unique, so an unqualified short name may resolve to a bucket in
# someone else's account — delivering a complete record of your API activity
# to a third party.
aws cloudtrail create-trail \
  --name trustops-audit-trail \
  --s3-bucket-name <YOUR_AUDIT_LOG_BUCKET>-<YOUR_ACCOUNT_ID>-<REGION> \
  --is-multi-region-trail \
  --enable-log-file-validation

# Start logging
aws cloudtrail start-logging \
  --name trustops-audit-trail
```

#### Monitor Sensitive Operations

**CloudWatch Metric Filters:**
```bash
# Alert on unauthorized access attempts
aws logs put-metric-filter \
  --log-group-name /aws/lambda/trustops-evaluation-handler \
  --filter-name UnauthorizedAccess \
  --filter-pattern "[time, request_id, event_type = AccessDenied*, ...]" \
  --metric-transformations \
    metricName=UnauthorizedAccessAttempts,\
    metricNamespace=TrustOps/Security,\
    metricValue=1
```

### Secrets Management

**Store Sensitive Configuration in Secrets Manager:**
```bash
# Store API keys or credentials
aws secretsmanager create-secret \
  --name trustops/bedrock-api-key \
  --secret-string '{"api_key":"your-key-here"}'

# Retrieve in Lambda
import boto3
secrets_client = boto3.client('secretsmanager')
response = secrets_client.get_secret_value(SecretId='trustops/bedrock-api-key')
```

### Network Security

**VPC Configuration (Optional):**
- Deploy Lambda functions in VPC for network isolation
- Use VPC endpoints for AWS services (S3, DynamoDB, Bedrock)
- Configure security groups to restrict traffic

**Example VPC Endpoint:**
```bash
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-12345678 \
  --service-name com.amazonaws.us-east-1.bedrock-runtime \
  --route-table-ids rtb-12345678
```


## FAQ

### General Questions

**Q: What AWS regions support TrustOps?**

A: TrustOps works in any region with AWS Bedrock access. Recommended regions:
- us-east-1 (N. Virginia) - Best model availability
- us-west-2 (Oregon) - Good model availability
- eu-west-1 (Ireland) - For EU deployments

**Q: Can I use TrustOps with non-AWS models?**

A: Currently, TrustOps is designed for AWS Bedrock models. However, the architecture is extensible. You can implement custom model clients by extending the `BedrockClient` interface.

**Q: How long does a typical evaluation take?**

A: Timing depends on dataset size:
- Small (10-50 examples): 2-5 minutes
- Medium (50-200 examples): 5-15 minutes
- Large (200-1000 examples): 15-60 minutes

**Q: What's the maximum dataset size?**

A: No hard limit, but practical limits:
- Single Lambda: ~100 examples (due to 15-minute timeout)
- With partitioning: 1000+ examples (distributed across multiple invocations)

### Cost Questions

**Q: How much does TrustOps cost to run?**

A: Costs vary by usage:
- Development: $50-100/month
- Production (10K queries/month): $500-1000/month
- Production (100K queries/month): $2000-3500/month

Most cost is from Bedrock model inference. See [Cost Optimization](#cost-optimization) section.

**Q: How can I reduce costs?**

A: Key strategies:
1. Use Claude Instant instead of Claude v2 for development
2. Reduce evaluation dataset sizes during testing
3. Leave the Knowledge Base unprovisioned unless real grounding is needed
4. Enable S3 lifecycle policies to archive old data
5. Use on-demand DynamoDB billing for variable workloads


### Technical Questions

**Q: What happens if a Lambda function times out?**

A: Step Functions handle timeouts gracefully:
1. Workflow status updated to "failed"
2. Error logged to CloudWatch
3. Partial results saved (if any)
4. Workflow can be retried or reproduced

**Q: Can I customize trust score weights?**

A: Yes, modify the weights in `src/trust_scoring/trust_scoring_engine.py`:
```python
WEIGHTS = {
    'context_grounding': 0.30,      # Adjust these values
    'output_structure': 0.20,
    'uncertainty_indicators': 0.15,
    'factual_consistency': 0.15,
    'response_completeness': 0.20
}
```
Ensure weights sum to 1.0.

**Q: How do I add a new model?**

A: Update the Bedrock client configuration:
1. Add model pricing to `.env`
2. Update `BedrockClient` to handle new model format
3. Test with sample evaluation

**Q: Can I run TrustOps locally?**

A: Yes, for development:
1. Configure AWS credentials locally
2. Run Python scripts directly (not Lambda)
3. Use LocalStack for local AWS service emulation (optional)

**Q: How do I debug failed workflows?**

A: Check these sources:
1. CloudWatch Logs: `/aws/lambda/trustops-*`
2. DynamoDB: Query `trustops-workflows` table for workflow status
3. S3: Check for partial results in results bucket
4. Step Functions: View execution history in console


### Security Questions

**Q: Is my data encrypted?**

A: Yes:
- S3: Encrypted at rest with SSE-S3 or KMS
- DynamoDB: Encrypted at rest with KMS
- In transit: All API calls use HTTPS/TLS 1.2+

**Q: Who can access my evaluation data?**

A: Only IAM principals with explicit permissions:
- Lambda execution roles (for processing)
- Admin roles (for management)
- No public access (enforced by bucket policies)

**Q: Are model responses logged?**

A: Yes, for audit purposes:
- Full responses stored in S3 (encrypted)
- Metadata logged to CloudWatch
- Audit trail in CloudTrail

**Q: How long is data retained?**

A: Configurable retention:
- S3: Indefinite (with lifecycle policies for archival)
- CloudWatch Logs: 30-365 days (configurable)
- DynamoDB: Indefinite (with PITR for 35 days)

### Operational Questions

**Q: How do I scale TrustOps for production?**

A: Key scaling strategies:
1. Enable dataset partitioning for large evaluations
2. Use provisioned concurrency for Lambda (if needed)
3. Raise the OpenSearch Serverless OCU ceiling backing the Knowledge Base
4. Use DynamoDB auto-scaling or on-demand billing
5. Implement caching for frequently accessed data

**Q: What's the SLA for TrustOps?**

A: TrustOps inherits AWS service SLAs:
- Lambda: 99.95%
- S3: 99.99%
- DynamoDB: 99.99%
- Bedrock: 99.9%

Overall availability depends on your architecture and redundancy.

**Q: How do I monitor system health?**

A: Use CloudWatch dashboards and alarms:
1. Trust score metrics (mean, distribution)
2. Error rates (Lambda, Bedrock API)
3. Latency metrics (inference, trust scoring)
4. Cost metrics (daily spend, cost per query)

See [Monitoring and Alerting](#monitoring-and-alerting) section.

---

**Document Version:** 1.0  
**Last Updated:** 2024-01-15  
**Maintained By:** TrustOps Operations Team  
**Support:** Check CloudWatch logs and AWS documentation for detailed troubleshooting
