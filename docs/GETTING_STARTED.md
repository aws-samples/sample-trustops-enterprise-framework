# Getting Started with TrustOps Enterprise Framework

## Prerequisites

- Python 3.11+
- AWS account with access to Bedrock, SageMaker, S3, and DynamoDB
- AWS CLI configured (`aws configure`)

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd trustops-enterprise-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
make install

# Install CLI in development mode
make cli-dev
```

## AWS Configuration

1. Copy and edit the environment file:

```bash
cp infrastructure/env.example .env
# Edit .env with your AWS region, S3 bucket names, and DynamoDB table names
```

2. Create DynamoDB tables:

```bash
make create-tables
```

3. Create S3 buckets (replace with your bucket names):

```bash
aws s3 mb s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
aws s3 mb s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>
aws s3 mb s3://trustops-artifacts-<YOUR_ACCOUNT_ID>-<REGION>
```

## Verify Setup

```bash
trustops --help
make verify
```

## Sample Workflow Walkthrough

This walkthrough demonstrates the core evaluate → fine-tune → compare loop.

### 1. Register and discover models

```bash
# List available Bedrock models
trustops models list

# Check model details
trustops models info --model-id anthropic.claude-v2
```

### 2. Upload a dataset

```bash
# Upload a sample QA dataset
trustops datasets upload demo/sample_data/qa_dataset.jsonl \
  --name "QA Evaluation Set" \
  --task-type qa
```

### 3. Run baseline evaluation

```bash
# Evaluate a model against the dataset
trustops evaluate baseline \
  --model-id anthropic.claude-v2 \
  --dataset-id <dataset-id>
```

### 4. Fine-tune a model

```bash
# Start fine-tuning with default hyperparameters
trustops finetune start \
  --model-id amazon.titan-text-express-v1 \
  --dataset-id <dataset-id> \
  --job-name my-first-finetune

# Monitor progress
trustops finetune status --job-id <job-id>
```

### 5. Compare baseline vs fine-tuned

```bash
# Run comparative evaluation
trustops evaluate compare \
  --baseline-model-id anthropic.claude-v2 \
  --finetuned-model-id <finetuned-model-id> \
  --dataset-id <dataset-id>
```

### 6. View results

```bash
# List evaluation results
trustops results list

# Export a report
trustops results export --evaluation-id <eval-id> --format json
```

## Running the Dashboard

```bash
make run-dashboard
# Open http://localhost:8501 in your browser
```

## Running Tests

```bash
make test          # All tests
make test-unit     # Unit tests only
make test-cli      # CLI tests only
make test-dashboard # Dashboard tests only
```
