# TrustOps Infrastructure

This directory contains Infrastructure as Code (IaC) templates for deploying the TrustOps AWS Demo system.

## Overview

The TrustOps infrastructure includes:

- **S3 Buckets**: Storage for datasets, results, and artifacts with versioning and lifecycle policies
- **DynamoDB Tables**: Workflow and model metadata storage with appropriate indexes
- **Lambda Functions**: Serverless compute for evaluation, fine-tuning, trust scoring, and comparative analysis
- **Step Functions**: Workflow orchestration for baseline evaluation, fine-tuning, and comparative evaluation
- **Bedrock Knowledge Base**: Semantic similarity analysis and vector retrieval, backed by an OpenSearch Serverless collection that Bedrock manages
- **CloudWatch**: Log groups and alarms for monitoring and alerting
- **IAM Roles**: Least-privilege permissions for Lambda and Step Functions
- **QuickSight**: Dashboards for visualization of trust metrics and model comparisons

## Deployment Options

You can deploy the infrastructure using either:

1. **AWS CloudFormation** (recommended for AWS-native workflows)
2. **Terraform** (recommended for multi-cloud or existing Terraform workflows)

## Prerequisites

### Common Requirements

- AWS CLI installed and configured
- AWS account with appropriate permissions
- Python 3.11+ (for Lambda function packaging)
- Valid AWS credentials configured (`aws configure`)

### CloudFormation Specific

- No additional tools required

### Terraform Specific

- Terraform 1.0+ installed

## Quick Start

### Using the Deployment Script (Recommended)

The `deploy.sh` script automates the entire deployment process:

```bash
# Deploy using CloudFormation (default)
./deploy.sh

# Deploy using Terraform
DEPLOYMENT_METHOD=terraform ./deploy.sh

# Deploy to a specific environment
ENVIRONMENT=staging ./deploy.sh

# Deploy to a specific region
AWS_REGION=us-west-2 ./deploy.sh

# Combine options
PROJECT_NAME=mytrustops ENVIRONMENT=prod AWS_REGION=eu-west-1 DEPLOYMENT_METHOD=terraform ./deploy.sh
```

### Manual CloudFormation Deployment

1. **Package Lambda functions**:
   ```bash
   cd ../lambda_handlers
   for handler in evaluation_handler fine_tuning_handler trust_scoring_handler comparative_evaluation_handler; do
       mkdir -p /tmp/${handler}
       cp ${handler}.py /tmp/${handler}/
       cp -r ../src /tmp/${handler}/
       pip install -r ../requirements.txt -t /tmp/${handler}/
       cd /tmp/${handler}
       zip -r ${handler}.zip .
       cd -
   done
   ```

2. **Upload Lambda packages to S3**:
   ```bash
   ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
   ARTIFACTS_BUCKET="trustops-artifacts-dev-${ACCOUNT_ID}"
   
   aws s3 mb s3://${ARTIFACTS_BUCKET}
   aws s3 cp /tmp/evaluation_handler/evaluation_handler.zip s3://${ARTIFACTS_BUCKET}/lambda/
   aws s3 cp /tmp/fine_tuning_handler/fine_tuning_handler.zip s3://${ARTIFACTS_BUCKET}/lambda/
   aws s3 cp /tmp/trust_scoring_handler/trust_scoring_handler.zip s3://${ARTIFACTS_BUCKET}/lambda/
   aws s3 cp /tmp/comparative_evaluation_handler/comparative_evaluation_handler.zip s3://${ARTIFACTS_BUCKET}/lambda/
   ```

3. **Upload Step Functions definitions**:
   ```bash
   aws s3 cp ../step_functions/baseline_evaluation_workflow.json s3://${ARTIFACTS_BUCKET}/step_functions/
   aws s3 cp ../step_functions/fine_tuning_workflow.json s3://${ARTIFACTS_BUCKET}/step_functions/
   aws s3 cp ../step_functions/comparative_evaluation_workflow.json s3://${ARTIFACTS_BUCKET}/step_functions/
   ```

4. **Deploy CloudFormation stack**:
   ```bash
   aws cloudformation create-stack \
       --stack-name trustops-dev \
       --template-body file://cloudformation-template.yaml \
       --parameters \
           ParameterKey=ProjectName,ParameterValue=trustops \
           ParameterKey=Environment,ParameterValue=dev \
       --capabilities CAPABILITY_NAMED_IAM
   
   # Wait for completion
   aws cloudformation wait stack-create-complete --stack-name trustops-dev
   ```

5. **View outputs**:
   ```bash
   aws cloudformation describe-stacks \
       --stack-name trustops-dev \
       --query 'Stacks[0].Outputs' \
       --output table
   ```

### Manual Terraform Deployment

1. **Package Lambda functions** (same as CloudFormation step 1)

2. **Upload Lambda packages** (same as CloudFormation step 2)

3. **Upload Step Functions definitions** (same as CloudFormation step 3)

4. **Initialize Terraform**:
   ```bash
   cd terraform
   terraform init
   ```

5. **Create workspace**:
   ```bash
   terraform workspace new dev
   terraform workspace select dev
   ```

6. **Plan deployment**:
   ```bash
   terraform plan \
       -var="project_name=trustops" \
       -var="environment=dev" \
       -var="aws_region=us-east-1" \
       -out=tfplan
   ```

7. **Apply deployment**:
   ```bash
   terraform apply tfplan
   ```

8. **View outputs**:
   ```bash
   terraform output
   ```

## Configuration

### CloudFormation Parameters

- `ProjectName`: Project name for resource naming (default: `trustops`)
- `Environment`: Environment name - dev, staging, or prod (default: `dev`)

### Terraform Variables

Edit `terraform/variables.tf` or provide via command line:

- `project_name`: Project name for resource naming
- `environment`: Environment name (dev, staging, prod)
- `aws_region`: AWS region for resources
- `embedding_model_id`: Bedrock embedding model used by the Knowledge Base
- `alert_email`: Address subscribed to the CloudWatch alarm topic (optional)
- `log_retention_days`: CloudWatch log retention in days
- `trust_score_threshold`: Trust score threshold for flagging responses
- `lambda_evaluation_timeout`: Timeout for evaluation Lambda functions
- `lambda_evaluation_memory`: Memory for evaluation Lambda functions
- `lambda_trust_scoring_timeout`: Timeout for trust scoring Lambda function
- `lambda_trust_scoring_memory`: Memory for trust scoring Lambda function

## Resource Details

### S3 Buckets

1. **Datasets Bucket** (`{project}-datasets-{env}-{account-id}`)
   - Stores evaluation and training datasets
   - Versioning enabled
   - Lifecycle: Archive old versions to Glacier after 90 days

2. **Results Bucket** (`{project}-results-{env}-{account-id}`)
   - Stores evaluation results and reports
   - Versioning enabled
   - Lifecycle: Archive to Glacier after 180 days

3. **Artifacts Bucket** (`{project}-artifacts-{env}-{account-id}`)
   - Stores Lambda deployment packages and Step Functions definitions
   - Versioning enabled

### DynamoDB Tables

1. **Workflows Table** (`{project}-workflows-{env}`)
   - Primary key: `workflow_id`
   - GSI: `workflow-type-index` (workflow_type, created_at)
   - GSI: `status-index` (status, created_at)
   - Point-in-time recovery enabled

2. **Models Table** (`{project}-models-{env}`)
   - Primary key: `model_id`
   - GSI: `created-at-index` (created_at)
   - Point-in-time recovery enabled

### Lambda Functions

1. **Evaluation Orchestrator** (`{project}-evaluation-orchestrator-{env}`)
   - Runtime: Python 3.11
   - Timeout: 900 seconds (15 minutes)
   - Memory: 1024 MB
   - Purpose: Orchestrate baseline and comparative evaluations

2. **Fine-Tuning Orchestrator** (`{project}-finetuning-orchestrator-{env}`)
   - Runtime: Python 3.11
   - Timeout: 900 seconds
   - Memory: 512 MB
   - Purpose: Manage fine-tuning pipeline

3. **Trust Scoring** (`{project}-trust-scoring-{env}`)
   - Runtime: Python 3.11
   - Timeout: 300 seconds (5 minutes)
   - Memory: 512 MB
   - Purpose: Calculate trust scores for model responses

4. **Comparative Evaluation** (`{project}-comparative-evaluation-{env}`)
   - Runtime: Python 3.11
   - Timeout: 900 seconds
   - Memory: 1024 MB
   - Purpose: Compare baseline and fine-tuned models

5. **Evaluate Single Example** (`{project}-evaluate-single-example-{env}`)
   - Runtime: Python 3.11
   - Timeout: 300 seconds (5 minutes)
   - Memory: 512 MB
   - Purpose: Invoke a single model against a single prompt via Bedrock, return response with latency and token counts

6. **Aggregate Metrics** (`{project}-aggregate-metrics-{env}`)
   - Runtime: Python 3.11
   - Timeout: 300 seconds
   - Memory: 512 MB
   - Purpose: Load per-example results from S3, compute aggregate BaselineMetrics (mean trust score, hallucination rate, latency percentiles, cost)

7. **Generate Recommendation** (`{project}-generate-recommendation-{env}`)
   - Runtime: Python 3.11
   - Timeout: 120 seconds (2 minutes)
   - Memory: 256 MB
   - Purpose: Compare baseline vs fine-tuned metrics, run statistical significance tests, produce DEPLOY/ITERATE/REJECT recommendation with justification

8. **Detect Hallucinations** (`{project}-detect-hallucinations-{env}`)
   - Runtime: Python 3.11
   - Timeout: 300 seconds
   - Memory: 512 MB
   - Purpose: Decompose model responses into claims, evaluate each against source documents, compute per-claim grounding scores and overall hallucination rate

### Step Functions State Machines

1. **Baseline Evaluation** (`{project}-baseline-evaluation-{env}`)
   - Orchestrates: ValidateDataset → InvokeModel (parallel Map) → ScoreTrust → DetectHallucinations → AggregateMetrics → StoreResults
   - Logs to evaluation log group

2. **Fine-Tuning** (`{project}-fine-tuning-{env}`)
   - Orchestrates: BaselineEvaluation → SubmitFineTuning → PollStatus (loop) → FineTunedEvaluation → ComparativeMetrics → GenerateRecommendation → ApprovalGate → StoreResults
   - Produces DEPLOY/ITERATE/REJECT recommendation with statistical significance
   - Logs to fine-tuning log group

3. **Comparative Evaluation** (`{project}-comparative-evaluation-{env}`)
   - Orchestrates: ValidateDataset → ParallelEvaluation (Model A + Model B) → ComparativeMetrics → StatisticalSignificance → ApprovalGate → StoreResults
   - Includes human approval gate before finalizing results
   - Logs to evaluation log group

### Bedrock Knowledge Base

- Knowledge Base name: `{project}-kb-{env}`
- Embedding model: configurable via `embedding_model_id`
- Data source: the `knowledge-base/` prefix of the datasets bucket
- Vector store: an OpenSearch Serverless collection provisioned alongside it,
  with encryption, network, and data access policies applied
- Bedrock generates and stores embeddings; no index is managed directly

### CloudWatch Log Groups

1. `/aws/{project}/{env}` - General TrustOps logs
2. `/aws/{project}/{env}/evaluation` - Evaluation workflow logs
3. `/aws/{project}/{env}/finetuning` - Fine-tuning workflow logs
4. `/aws/{project}/{env}/trustscoring` - Trust scoring logs

Retention: 30 days (configurable)

### CloudWatch Alarms

1. **High Error Rate** - Alerts when Lambda error rate exceeds 5%
2. **Lambda Timeout** - Alerts when Lambda timeout rate exceeds 10%
3. **Workflow Failure** - Alerts when workflow failure rate exceeds 20%
4. **Trust Score Degradation** - Alerts when mean trust score drops below 0.6

## Updating Infrastructure

### CloudFormation

```bash
aws cloudformation update-stack \
    --stack-name trustops-dev \
    --template-body file://cloudformation-template.yaml \
    --parameters \
        ParameterKey=ProjectName,ParameterValue=trustops \
        ParameterKey=Environment,ParameterValue=dev \
    --capabilities CAPABILITY_NAMED_IAM
```

### Terraform

```bash
cd terraform
terraform plan -var="environment=dev"
terraform apply
```

## Destroying Infrastructure

### CloudFormation

```bash
aws cloudformation delete-stack --stack-name trustops-dev
aws cloudformation wait stack-delete-complete --stack-name trustops-dev
```

### Terraform

```bash
cd terraform
terraform destroy -var="environment=dev"
```

**Warning**: This will delete all resources including S3 buckets with data. Ensure you have backups before destroying.

## Troubleshooting

### Lambda Function Deployment Issues

If Lambda functions fail to deploy:

1. Verify the artifacts bucket exists and contains the Lambda packages
2. Check IAM permissions for Lambda execution role
3. Verify the Lambda package structure is correct
4. Check CloudWatch logs for detailed error messages

### Knowledge Base Issues

If the Bedrock Knowledge Base fails to create:

1. Verify Bedrock and the chosen embedding model are available in your region
2. Check service limits for Knowledge Bases and OpenSearch Serverless collections
3. Verify the Knowledge Base IAM role can read the datasets bucket
4. Confirm the data access policy grants the role access to the collection

### Step Functions Issues

If Step Functions fail to execute:

1. Verify Lambda functions are deployed and accessible
2. Check Step Functions execution role has Lambda invoke permissions
3. Review CloudWatch logs for detailed error messages
4. Verify workflow definitions are valid JSON

## Cost Estimation

Approximate monthly costs for dev environment (us-east-1):

- **S3**: $5-20 (depending on data volume)
- **DynamoDB**: $5-10 (on-demand pricing)
- **Lambda**: $10-50 (depending on execution frequency)
- **Bedrock Knowledge Base / OpenSearch Serverless**: $0 when not provisioned;
  OpenSearch Serverless bills a minimum OCU commitment when it is
- **CloudWatch**: $5-10 (logs and metrics)
- **Step Functions**: $1-5 (depending on workflow executions)

**Total estimated cost**: $26-95/month without a Knowledge Base. Bedrock
inference is billed per token on top of this.

Production environments will have higher costs based on usage patterns.

## Security Considerations

- All S3 buckets have public access blocked
- All data is encrypted at rest (AES256)
- The Knowledge Base vector collection is encrypted and reached over HTTPS
- IAM roles follow least-privilege principle
- CloudWatch logs are retained for audit purposes
- DynamoDB has point-in-time recovery enabled

## Deploy / Iterate / Reject Decision

The Step Functions workflows produce a deployment recommendation backed by:

1. **Statistical Significance**: Paired t-test (n >= 20) or Wilcoxon signed-rank (n < 20) on per-example trust scores
2. **Confidence Intervals**: 95% CI via t-distribution
3. **Break-Even Analysis**: Query volume at which the cost premium of a better model is offset by reduced hallucination remediation costs

| Recommendation | Criteria |
|---|---|
| DEPLOY | Trust +10%, hallucination -5%, cost <= +20%, p < 0.05 |
| ITERATE | Moderate improvement or cost too high |
| REJECT | Insufficient improvement |

The approval gate pauses execution for human review before the recommendation is finalized and stored.

## Support

For issues or questions:

1. Check CloudWatch logs for error details
2. Review AWS service quotas and limits
3. Verify IAM permissions are correctly configured
4. Consult the TrustOps documentation

## Requirements Validation

This infrastructure satisfies the following requirements:

- **Requirement 9.6**: Lambda orchestration with IAM roles and least-privilege permissions
- **Requirement 10.1**: S3 buckets with versioning enabled for data storage
- **Requirement 10.6**: Lifecycle policies for archiving old data to Glacier

## Next Steps

After deploying the infrastructure:

1. Upload sample datasets to the datasets bucket
2. Configure the CLI with the deployed resource names
3. Run a baseline evaluation to verify the system
4. Set up QuickSight dashboards for monitoring (see quicksight/README.md)
5. Configure SNS topics for alarm notifications (optional)
