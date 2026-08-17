# TrustOps Enterprise Framework

A trust-first fine-tuning and evaluation framework for enterprise GenAI using AWS services.

> **Important**: This project is provided as a sample/educational implementation
> and is NOT intended for production use without additional security hardening.
> See [SECURITY.md](SECURITY.md) for production recommendations and known limitations.

## Overview

TrustOps provides quantifiable trust scoring, automated model comparison, hallucination detection, and cost-performance optimization visibility through auditable, repeatable evaluation workflows.

## Features

- **Baseline Model Evaluation**: Assess foundation model performance before fine-tuning
- **Fine-Tuning Pipeline**: Validate and fine-tune models with quality-checked training data
- **Comparative Evaluation**: Side-by-side comparison of baseline vs. fine-tuned models
- **Foundation Model Comparison**: Compare any two foundation models to determine which better suits your use case
- **Deploy/Iterate/Reject Decision**: Data-driven deployment recommendations backed by paired statistical significance testing (t-test + Wilcoxon), break-even volume analysis, and configurable thresholds
- **Statistical Significance Testing**: Paired t-test and Wilcoxon signed-rank tests with p-values, confidence intervals, and automatic test selection based on sample size
- **Break-Even Cost Analysis**: Compute the query volume at which a higher-trust model's cost premium pays for itself through reduced hallucination remediation costs
- **Trust Scoring**: Real-time confidence assessment for model outputs
- **Hallucination Detection**: Identify and measure unsupported claims in responses
- **Cost-Performance Visibility**: Track and optimize cost alongside quality metrics
- **Auditable Workflows**: Complete lineage from data to deployment decision

## Project Structure

```
trustops-enterprise-framework/
├── cli/                    # Command-line interface
│   ├── commands/           # CLI command modules
│   └── utils/              # CLI utilities
├── src/                    # Source code
│   ├── adapters/           # Model provider adapters (Bedrock, SageMaker, External API)
│   ├── aws_clients/        # AWS service wrappers (S3, Bedrock, Knowledge Bases)
│   ├── clients/            # Inference client
│   ├── data_models/        # Pydantic data classes and schemas
│   ├── datasets/           # Dataset management, validation, PII detection
│   ├── evaluation/         # Evaluation engine, metrics, comparison, reporting
│   ├── fine_tuning/        # Fine-tuning pipeline, cost estimation, progress
│   ├── hallucination/      # Hallucination detection (claim extraction, evidence search)
│   ├── orchestration/      # Workflow orchestration, statistical testing, break-even analysis, approval gates
│   ├── registry/           # Model registry, discovery, pricing
│   ├── storage/            # Results storage (S3 + DynamoDB), export, versioning
│   ├── trust_scoring/      # Trust scoring engines and custom metrics
│   └── utils/              # Shared utilities (logging, retry, rate limiting)
├── config/                 # Configuration (AWS config, model pricing)
├── dashboard/              # Streamlit dashboard (pages, components, styles)
├── demo/sample_data/       # Sample datasets (QA, summarization, classification, chat)
├── lambda_handlers/        # Serverless Step Functions handlers (evaluate, aggregate, recommend, detect)
├── infrastructure/         # Infrastructure as Code (CloudFormation + Terraform)
├── scripts/                # Setup and verification scripts
└── docs/                   # Documentation
```

## Prerequisites

- Python 3.11+
- AWS Account with Bedrock, S3, DynamoDB, Lambda, Step Functions access
- AWS CLI v2 configured with credentials

## Quick Start

### 1. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install the CLI

```bash
make cli-dev
trustops --version
```

### 3. Configure AWS credentials

```bash
aws configure
```

### 4. Set up environment variables

```bash
cp infrastructure/env.example .env
```

Edit `.env` with your deployed resource names (bucket names, table names, etc.).

### 5. Deploy AWS infrastructure

```bash
cd infrastructure
./deploy.sh
```

This creates S3 buckets, DynamoDB tables, Lambda functions, Step Functions state machines, IAM roles, and CloudWatch alarms. The deployment generates a unique `DeploymentId` each run to avoid resource name collisions.

Semantic retrieval is backed by Amazon Bedrock Knowledge Bases, which is
optional: without `KNOWLEDGE_BASE_ID` set, the retrieval clients run in mock
mode. See [SETUP.md](SETUP.md) for provisioning one.

### 6. Update `.env` with deployed resource names

After deployment completes, the script displays resource names. Update your `.env`:

```bash
# Use the actual bucket/table names from the CloudFormation outputs
TRUSTOPS_DATASETS_BUCKET=trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_RESULTS_BUCKET=trustops-results-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_ARTIFACTS_BUCKET=trustops-artifacts-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_WORKFLOWS_TABLE=trustops-workflows-dev-<deployment-id>
TRUSTOPS_MODELS_TABLE=trustops-models-dev-<deployment-id>
KNOWLEDGE_BASE_ID=<knowledge-base-id>
KNOWLEDGE_BASE_DATA_SOURCE_ID=<data-source-id>
```

### 7. Verify infrastructure

```bash
make verify
```

## Post-Deployment Testing

Once infrastructure is deployed, follow these steps to validate the full framework end-to-end.

### Step 1: Verify AWS connectivity

```bash
trustops auth check
```

### Step 2: Configure the CLI

```bash
trustops configure
# Enter your region, bucket names, and table names from the deployment outputs
```

### Step 3: List available models

```bash
trustops models list --provider bedrock
```

If you get `AccessDeniedException`, you need to request model access in the AWS Console under Bedrock → Model access.

### Step 4: Upload a sample dataset

```bash
trustops datasets upload demo/sample_data/qa_sample.jsonl \
  --name "QA Evaluation Set" \
  --task-type qa
```

Note the `dataset-id` from the output.

### Step 5: Run a baseline evaluation

```bash
trustops evaluate baseline \
  --model-id anthropic.claude-3-haiku-20240307-v1:0 \
  --dataset-id <dataset-id>
```

### Step 6: Test Lambda functions directly

```bash
# Invoke the trust scoring Lambda
aws lambda invoke \
  --function-name trustops-trust-scoring-dev-<deployment-id> \
  --payload '{"response": "AWS Bedrock provides foundation model access.", "prompt": "What is Bedrock?", "source_documents": ["AWS Bedrock is a managed service for foundation models."]}' \
  --region us-east-1 \
  /tmp/trust-output.json

cat /tmp/trust-output.json

# Invoke the evaluation Lambda
aws lambda invoke \
  --function-name trustops-evaluation-orchestrator-dev-<deployment-id> \
  --payload '{"model_id": "anthropic.claude-3-haiku-20240307-v1:0", "dataset_id": "test-dataset", "evaluation_type": "baseline"}' \
  --region us-east-1 \
  /tmp/eval-output.json

cat /tmp/eval-output.json
```

### Step 7: Test Step Functions workflows

```bash
# Start a baseline evaluation workflow
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:<account-id>:stateMachine:trustops-baseline-evaluation-dev-<deployment-id> \
  --input '{"model_id": "anthropic.claude-3-haiku-20240307-v1:0", "dataset_id": "test-dataset", "inference_params": {"max_tokens": 512, "temperature": 0.0}}' \
  --region us-east-1
```

### Step 8: Verify DynamoDB tables

```bash
aws dynamodb scan --table-name trustops-workflows-dev-<deployment-id> --max-items 5 --region us-east-1
aws dynamodb scan --table-name trustops-models-dev-<deployment-id> --max-items 5 --region us-east-1
```

### Step 9: Verify S3 buckets

```bash
aws s3 ls s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/
aws s3 ls s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>/
```

### Step 10: Run the dashboard

```bash
make run-dashboard
# Open http://localhost:8501
```

## CLI Usage

### Baseline Evaluation

```bash
trustops evaluate baseline \
  --model-id anthropic.claude-3-haiku-20240307-v1:0 \
  --dataset-id <dataset-id>
```

### Fine-Tune a Model

```bash
trustops finetune start \
  --model-id amazon.titan-text-express-v1 \
  --dataset-id <training-dataset-id> \
  --job-name my-finetune-job

trustops finetune status --job-id <job-id>
```

### Comparative Evaluation

```bash
trustops evaluate compare \
  --baseline-model-id anthropic.claude-v2 \
  --finetuned-model-id <finetuned-model-arn> \
  --dataset-id <dataset-id>
```

### View Deployment Recommendation

After a comparative evaluation completes, the framework produces a Deploy/Iterate/Reject recommendation:

- **DEPLOY**: Trust improvement >= 10%, hallucination reduction >= 5%, cost increase <= 20%, statistically significant (p < 0.05)
- **ITERATE**: Moderate improvement or high cost — consider additional fine-tuning
- **REJECT**: Insufficient improvement — fine-tuning did not provide meaningful benefits

The recommendation includes:
- Paired t-test and Wilcoxon p-values
- 95% confidence intervals for the mean trust score difference
- Break-even volume (queries needed for cost premium to pay off via reduced hallucination remediation)

### Foundation Model Comparison

```bash
trustops foundation-compare \
  --model-id-1 anthropic.claude-3-sonnet-20240229-v1:0 \
  --model-id-2 anthropic.claude-3-haiku-20240307-v1:0 \
  --dataset s3://your-bucket/eval-data.json
```

### Workflow Management

```bash
trustops workflows list
trustops workflows status --workflow-id <id>
trustops workflows resume --workflow-id <id>
```

### Results

```bash
trustops results get --evaluation-id <eval-id>
trustops results export --evaluation-id <eval-id> --format json
```

For the full CLI reference, see [docs/END_TO_END_RUNBOOK.md](docs/END_TO_END_RUNBOOK.md).

## Testing

```bash
make test              # All tests
make test-unit         # Unit tests
make test-property     # Property-based tests
make test-integration  # Integration tests
make test-cli          # CLI tests
make test-dashboard    # Dashboard tests
make coverage          # Tests with coverage report
```

## Documentation

- [End-to-End Runbook](docs/END_TO_END_RUNBOOK.md) — Complete operational guide with CLI reference
- [Getting Started](docs/GETTING_STARTED.md) — Quick setup walkthrough
- [Framework Overview](docs/FRAMEWORK_OVERVIEW.md) — Architecture and design
- [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md) — Component deep dives
- [Operational Runbook](docs/OPERATIONAL_RUNBOOK.md) — Troubleshooting and operations
- [Infrastructure README](infrastructure/README.md) — Deployment details

## Cleanup

```bash
# Delete the CloudFormation stack (replaces <stack-name> with your actual stack name)
aws cloudformation delete-stack --stack-name trustops-dev-<deployment-id> --region us-east-1

# Empty and delete S3 buckets
aws s3 rb s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION> --force
aws s3 rb s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION> --force
aws s3 rb s3://trustops-artifacts-<YOUR_ACCOUNT_ID>-<REGION> --force
```

## Security

See [SECURITY.md](SECURITY.md) for how to report a vulnerability, the
production hardening checklist, and the known security debt this sample
accepts.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This project follows the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Licensed under the MIT-0 License. See [LICENSE](LICENSE).
