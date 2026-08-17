# TrustOps CLI

Command-line interface for TrustOps - a trust-first fine-tuning and evaluation framework for enterprise GenAI.

## Installation

### From Source

```bash
# Install in development mode
pip install -e .

# Or install directly
pip install .
```

### Verify Installation

```bash
trustops --version
```

## Quick Start

### 1. Configure TrustOps

Set up your AWS configuration:

```bash
trustops configure
```

You'll be prompted for:
- AWS Region
- S3 buckets for datasets, results, and artifacts
- DynamoDB table name for workflow tracking

Configuration is saved to `~/.trustops/config.json`.

### 2. Run Baseline Evaluation

Evaluate a foundation model before fine-tuning:

```bash
trustops baseline \
  --model-id anthropic.claude-v2 \
  --dataset s3://my-bucket/eval-dataset.json
```

### 3. Fine-Tune a Model

Start a fine-tuning job:

```bash
trustops finetune \
  --base-model-id anthropic.claude-v2 \
  --training-data s3://my-bucket/training.jsonl \
  --epochs 5 \
  --learning-rate 0.0001
```

### 4. Compare Models

Compare baseline and fine-tuned models:

```bash
trustops compare \
  --baseline-model-id anthropic.claude-v2 \
  --finetuned-model-id arn:aws:bedrock:us-east-1:123:model/... \
  --dataset s3://my-bucket/eval-dataset.json
```

### 5. Compare Foundation Models

Compare any two AWS Bedrock foundation models:

```bash
trustops foundation-compare \
  --model-id-1 anthropic.claude-3-haiku-20240307-v1:0 \
  --model-id-2 anthropic.claude-3-sonnet-20240229-v1:0 \
  --dataset s3://my-bucket/eval-dataset.json
```

### 6. Check Status

View workflow status:

```bash
# Check specific workflow
trustops status baseline-20240115-abc123

# List all workflows
trustops status --list

# Filter by type and status
trustops status --list --type baseline --status-filter completed
```

### 7. Reproduce Workflow

Reproduce a previous workflow for auditing:

```bash
trustops reproduce baseline-20240115-abc123
```

## Commands

### When to Use Each Command

**`trustops baseline`** - Evaluate a single foundation model before fine-tuning
- Use when: You want to establish baseline performance metrics for a model
- Output: Trust scores, hallucination rates, costs, and latency for one model

**`trustops compare`** - Compare baseline vs fine-tuned models
- Use when: You've fine-tuned a model and want to see if it improved
- Output: Side-by-side metrics showing improvement from baseline to fine-tuned

**`trustops foundation-compare`** - Compare any two foundation models
- Use when: You're choosing between different foundation models (e.g., Claude vs Titan, Haiku vs Sonnet)
- Output: Side-by-side metrics with recommendation on which model to deploy
- Examples: Claude 3 Haiku vs Sonnet, Claude vs Titan, Claude vs Llama

**`trustops finetune`** - Start a fine-tuning job
- Use when: You want to customize a foundation model for your specific use case
- Output: Fine-tuned model ARN that can be used in comparative evaluation

### `trustops configure`

Configure AWS settings for TrustOps.

**Options:**
- `--region`: AWS region (default: us-east-1)
- `--datasets-bucket`: S3 bucket for datasets
- `--results-bucket`: S3 bucket for results
- `--artifacts-bucket`: S3 bucket for artifacts
- `--workflows-table`: DynamoDB table name

### `trustops baseline`

Run baseline evaluation for a foundation model.

**Options:**
- `--model-id`: AWS Bedrock model identifier (required)
- `--dataset`: S3 URI of evaluation dataset (required)
- `--workflow-id`: Optional workflow ID (auto-generated if not provided)
- `--wait/--no-wait`: Wait for completion (default: wait)

**Example:**
```bash
trustops baseline \
  --model-id anthropic.claude-v2 \
  --dataset s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval.json
```

### `trustops finetune`

Start a fine-tuning job for a foundation model.

**Options:**
- `--base-model-id`: Base model to fine-tune (required)
- `--training-data`: S3 URI of training data (required)
- `--job-name`: Job name (auto-generated if not provided)
- `--workflow-id`: Workflow ID (auto-generated if not provided)
- `--epochs`: Number of training epochs (default: 3)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--batch-size`: Training batch size (default: 8)
- `--wait/--no-wait`: Wait for completion (default: wait)
- `--poll-interval`: Polling interval in seconds (default: 60)

**Example:**
```bash
trustops finetune \
  --base-model-id anthropic.claude-v2 \
  --training-data s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/training.jsonl \
  --epochs 5 \
  --learning-rate 0.0001 \
  --batch-size 16
```

### `trustops compare`

Run comparative evaluation between baseline and fine-tuned models.

**Options:**
- `--baseline-model-id`: Baseline model ID (required)
- `--finetuned-model-id`: Fine-tuned model ID (required)
- `--dataset`: S3 URI of evaluation dataset (required)
- `--workflow-id`: Workflow ID (auto-generated if not provided)
- `--wait/--no-wait`: Wait for completion (default: wait)

**Example:**
```bash
trustops compare \
  --baseline-model-id anthropic.claude-v2 \
  --finetuned-model-id arn:aws:bedrock:us-east-1:123:model/my-model \
  --dataset s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval.json
```

### `trustops foundation-compare`

Compare any two AWS Bedrock foundation models to determine which better suits your use case.

**Options:**
- `--model-id-1`: First foundation model ID (required)
- `--model-id-2`: Second foundation model ID (required)
- `--dataset`: S3 URI of evaluation dataset (required)
- `--workflow-id`: Workflow ID (auto-generated if not provided)
- `--temperature`: Inference temperature (default: 0.7)
- `--max-tokens`: Maximum output tokens (default: 2048)

**Examples:**

Compare Claude 3 Haiku vs Claude 3 Sonnet:
```bash
trustops foundation-compare \
  --model-id-1 anthropic.claude-3-haiku-20240307-v1:0 \
  --model-id-2 anthropic.claude-3-sonnet-20240229-v1:0 \
  --dataset s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval.json
```

Compare Claude vs Titan:
```bash
trustops foundation-compare \
  --model-id-1 anthropic.claude-3-haiku-20240307-v1:0 \
  --model-id-2 amazon.titan-text-express-v1 \
  --dataset s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval.json \
  --temperature 0.5 \
  --max-tokens 1024
```

Compare Claude vs Llama:
```bash
trustops foundation-compare \
  --model-id-1 anthropic.claude-3-haiku-20240307-v1:0 \
  --model-id-2 meta.llama2-70b-chat-v1 \
  --dataset s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval.json
```

### `trustops status`

Check workflow status or list workflows.

**Arguments:**
- `WORKFLOW_ID`: Workflow ID to check (optional)

**Options:**
- `--list`: List all workflows
- `--type`: Filter by workflow type (baseline, comparative, fine-tuning)
- `--status-filter`: Filter by status (created, running, completed, failed)
- `--limit`: Maximum workflows to display (default: 10)

**Examples:**
```bash
# Check specific workflow
trustops status baseline-20240115-abc123

# List all workflows
trustops status --list

# List completed baseline evaluations
trustops status --list --type baseline --status-filter completed

# Show last 20 workflows
trustops status --list --limit 20
```

### `trustops reproduce`

Reproduce a previous workflow with identical configuration.

**Arguments:**
- `WORKFLOW_ID`: Workflow ID to reproduce (required)

**Options:**
- `--wait/--no-wait`: Wait for completion (default: wait)

**Example:**
```bash
trustops reproduce baseline-20240115-abc123
```

## Configuration

### Configuration File

Configuration is stored in `~/.trustops/config.json`:

```json
{
  "aws": {
    "region": "us-east-1",
    "datasets_bucket": "trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>",
    "results_bucket": "trustops-results-<YOUR_ACCOUNT_ID>-<REGION>",
    "artifacts_bucket": "trustops-artifacts-<YOUR_ACCOUNT_ID>-<REGION>",
    "workflows_table": "trustops-workflows"
  },
  "trust_scoring": {
    "threshold": 0.7,
    "hallucination_similarity_threshold": 0.7
  },
  "models": {
    "default_embedding_model": "amazon.titan-embed-text-v1",
    "default_foundation_model": "anthropic.claude-v2"
  }
}
```

### Environment Variables

You can also configure TrustOps using environment variables:

```bash
export AWS_REGION=us-east-1
export TRUSTOPS_DATASETS_BUCKET=trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
export TRUSTOPS_RESULTS_BUCKET=trustops-results-<YOUR_ACCOUNT_ID>-<REGION>
export TRUSTOPS_ARTIFACTS_BUCKET=trustops-artifacts-<YOUR_ACCOUNT_ID>-<REGION>
export TRUSTOPS_WORKFLOWS_TABLE=trustops-workflows
export TRUST_SCORE_THRESHOLD=0.7
export HALLUCINATION_SIMILARITY_THRESHOLD=0.7
```

### Custom Configuration File

Use a custom configuration file:

```bash
trustops --config /path/to/config.json baseline --model-id ...
```

## Dataset Formats

### Evaluation Dataset

JSON format with evaluation examples:

```json
{
  "dataset_id": "eval-001",
  "name": "Customer Support Evaluation",
  "description": "Evaluation dataset for customer support use case",
  "examples": [
    {
      "prompt": "How do I reset my password?",
      "expected_response": "To reset your password...",
      "source_documents": [
        "Password reset instructions: ..."
      ],
      "category": "authentication",
      "metadata": {}
    }
  ],
  "created_at": "2024-01-15T10:00:00Z",
  "version": "1.0"
}
```

### Training Dataset

JSONL format with prompt-completion pairs:

```jsonl
{"prompt": "How do I reset my password?", "completion": "To reset your password..."}
{"prompt": "What are your business hours?", "completion": "Our business hours are..."}
```

## Output

### Baseline Evaluation Output

```
============================================================
  Baseline Evaluation
============================================================

⟳ Creating workflow...
⟳ Starting baseline evaluation...
✓ Baseline evaluation completed!

Results Summary:
  Total Examples: 100
  Mean Trust Score: 0.782
  Median Trust Score: 0.801
  Mean Latency: 1234.5ms
  Total Cost: $0.1234
  Hallucination Rate: 8.5%

Results stored at: s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>/baseline-20240115-abc123/results.json

Trust Score Distribution:
  High (≥0.8): 65
  Medium (0.6-0.8): 28
  Low (<0.6): 7
```

### Comparative Evaluation Output

```
============================================================
  Comparative Evaluation
============================================================

⟳ Creating workflow...
⟳ Starting comparative evaluation...
✓ Comparative evaluation completed!

============================================================
BASELINE MODEL RESULTS
============================================================
Total Examples: 100
Mean Trust Score: 0.782
Hallucination Rate: 8.5%
Mean Latency: 1234.5ms
Total Cost: $0.1234

============================================================
FINE-TUNED MODEL RESULTS
============================================================
Total Examples: 100
Mean Trust Score: 0.856
Hallucination Rate: 3.2%
Mean Latency: 1189.2ms
Total Cost: $0.1456

============================================================
IMPROVEMENT METRICS
============================================================
Trust Score Change: ↑ 0.074 (+9.5%)
Hallucination Change: ↓ 0.053 (-62.4%)
Cost per Query Change: ↑ $0.000222 (+18.0%)
Latency Change: ↓ 45.3ms

Statistical Significance: Yes

============================================================
RECOMMENDATION
============================================================
Decision: DEPLOY
Justification: Significant improvement in trust score and hallucination 
reduction justifies the modest cost increase. The fine-tuned model shows 
strong performance gains across all quality metrics.

Results stored at: s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>/compare-20240115-xyz789/results.json
```

### Foundation Model Comparison Output

```
============================================================
  Foundation Model Comparison
============================================================

Model 1: anthropic.claude-3-haiku-20240307-v1:0
Model 2: anthropic.claude-3-sonnet-20240229-v1:0
Dataset: s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval.json
Workflow ID: foundation-compare-a1b2c3d4
Temperature: 0.7
Max Tokens: 2048

⟳ Creating workflow...
⟳ Starting foundation model comparison...
This will evaluate both models on the same dataset.

✓ Foundation model comparison completed!

============================================================
MODEL 1 RESULTS
============================================================
Model ID: anthropic.claude-3-haiku-20240307-v1:0
Total Examples: 100
Mean Trust Score: 0.812
Hallucination Rate: 6.2%
Mean Latency: 892.3ms
Total Cost: $0.0456

============================================================
MODEL 2 RESULTS
============================================================
Model ID: anthropic.claude-3-sonnet-20240229-v1:0
Total Examples: 100
Mean Trust Score: 0.891
Hallucination Rate: 2.8%
Mean Latency: 1456.7ms
Total Cost: $0.1234

============================================================
COMPARISON METRICS
============================================================
Trust Score Change: ↑ 0.079 (+9.7%)
Hallucination Change: ↓ 0.034 (-54.8%)
Cost per Query Change: ↑ $0.000778 (+170.6%)
Latency Change: ↑ 564.4ms

Statistical Significance: Yes

============================================================
RECOMMENDATION
============================================================
Decision: DEPLOY_MODEL_2
Justification: Model 2 (Claude 3 Sonnet) shows statistically significant 
improvement in trust score (+9.7%) and substantial hallucination reduction 
(-54.8%). While cost per query increases by 170.6% and latency is higher, 
the quality improvements justify the additional cost for use cases where 
accuracy and reliability are critical.

Results stored at: s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>/foundation-compare-a1b2c3d4/results.json
```

## Troubleshooting

### AWS Credentials

Ensure AWS credentials are configured:

```bash
aws configure
```

Or use environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

### S3 Bucket Access

Verify you have access to the configured S3 buckets:

```bash
aws s3 ls s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/
```

### DynamoDB Table

Ensure the workflows table exists:

```bash
aws dynamodb describe-table --table-name trustops-workflows
```

### Model Access

Verify you have access to AWS Bedrock models:

```bash
aws bedrock list-foundation-models
```

## Support

For issues and questions:
- Check the main TrustOps documentation
- Review CloudWatch logs for detailed error information
- Verify AWS service quotas and permissions

## License

See the main TrustOps repository for license information.
