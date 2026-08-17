# TrustOps Enterprise Framework — End-to-End Runbook

A complete guide for setting up, running, and operating the TrustOps Enterprise Framework. This runbook covers every feature with step-by-step instructions, from initial setup through production workflows.

---

## Table of Contents

1. [Prerequisites and Setup](#1-prerequisites-and-setup)
2. [Model Discovery and Management](#2-model-discovery-and-management)
3. [Dataset Management](#3-dataset-management)
4. [Baseline Model Evaluation](#4-baseline-model-evaluation)
5. [Fine-Tuning a Model](#5-fine-tuning-a-model)
6. [Comparative Evaluation (Baseline vs Fine-Tuned)](#6-comparative-evaluation)
7. [Foundation Model Comparison](#7-foundation-model-comparison)
8. [Deploy/Iterate/Reject Decision](#8-deploy-iterate-reject-decision)
9. [Trust Scoring Deep Dive](#9-trust-scoring-deep-dive)
10. [Hallucination Detection](#10-hallucination-detection)
11. [Workflow Orchestration](#11-workflow-orchestration)
12. [Results Storage and Export](#12-results-storage-and-export)
13. [Dashboard Usage](#13-dashboard-usage)
14. [Running Tests](#14-running-tests)
15. [Complete End-to-End Walkthrough](#15-complete-end-to-end-walkthrough)
16. [Troubleshooting Quick Reference](#16-troubleshooting-quick-reference)

---

## 1. Prerequisites and Setup

### System Requirements

- Python 3.11+
- AWS CLI v2 configured
- AWS account with Bedrock, SageMaker, S3, DynamoDB access
- Git

### Step 1: Clone and Install

```bash
git clone <repo-url>
cd trustops-enterprise-framework

python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

make install
make cli-dev
```

Verify the CLI is available:

```bash
trustops --version
# trustops, version 0.1.0
```

### Step 2: Configure Environment

```bash
cp infrastructure/env.example .env
```

Edit `.env` — the key variables to set:

```bash
AWS_REGION=us-east-1
AWS_PROFILE=default

# S3 buckets (must be globally unique)
TRUSTOPS_DATASETS_BUCKET=trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_RESULTS_BUCKET=trustops-results-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_ARTIFACTS_BUCKET=trustops-artifacts-<YOUR_ACCOUNT_ID>-<REGION>

# DynamoDB tables
TRUSTOPS_MODELS_TABLE=trustops-models
TRUSTOPS_DATASETS_TABLE=trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_EVALUATIONS_TABLE=trustops-evaluations
TRUSTOPS_FINETUNING_TABLE=trustops-finetuning
TRUSTOPS_WORKFLOWS_TABLE=trustops-workflows
TRUSTOPS_RESULTS_TABLE=trustops-results-<YOUR_ACCOUNT_ID>-<REGION>

# Default models
DEFAULT_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
DEFAULT_FOUNDATION_MODEL=anthropic.claude-v2
```

### Step 3: Deploy AWS Infrastructure

```bash
# Option A: CloudFormation (recommended)
cd infrastructure && ./deploy.sh

# Option B: Terraform
cd infrastructure/terraform && terraform init && terraform apply
```

Or create resources manually:

```bash
# Create S3 buckets
aws s3 mb s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
aws s3 mb s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>
aws s3 mb s3://trustops-artifacts-<YOUR_ACCOUNT_ID>-<REGION>

# Create DynamoDB tables
make create-tables
```

### Step 4: Configure CLI

```bash
trustops configure
# Follow prompts for region, bucket names, table names
```

### Step 5: Verify

```bash
make verify
# ✓ AWS credentials configured
# ✓ S3 buckets accessible
# ✓ DynamoDB tables exist
# ✓ Bedrock accessible
```

### Step 6: Check Authentication

```bash
trustops auth check
# Displays: identity ARN, account ID, user ID
```

---

## 2. Model Discovery and Management

TrustOps supports three model providers: AWS Bedrock, SageMaker endpoints, and external APIs (OpenAI-compatible).

### List Available Models

```bash
# All models
trustops models list

# Filter by provider
trustops models list --provider bedrock
trustops models list --provider sagemaker
trustops models list --provider external_api

# Filter by capability
trustops models list --capability text_generation
trustops models list --capability embedding
trustops models list --fine-tunable

# JSON output
trustops models list --format json
```

### Get Model Details

```bash
trustops models info --model-id anthropic.claude-3-haiku-20240307-v1:0
# Shows: provider, capabilities, pricing, status, fine-tuning support
```

### Register an External Model

```bash
trustops models register \
  --provider external_api \
  --config '{"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY", "model_name": "gpt-4-turbo"}'
```

### Register a SageMaker Endpoint

```bash
trustops models register \
  --provider sagemaker \
  --config '{"endpoint_name": "my-llama-endpoint", "region": "us-east-1"}'
```

### Model Pricing

Pricing is configured in `config/model_pricing.yaml`. The framework ships with pricing for 50+ models across Bedrock, OpenAI, and SageMaker. To add custom pricing:

```yaml
# config/model_pricing.yaml
bedrock:
  your-custom-model:
    input_price_per_1k_tokens: 0.001
    output_price_per_1k_tokens: 0.002
    currency: USD
```

---

## 3. Dataset Management

### Supported Formats and Task Types

| Format | Extension | Task Types |
|--------|-----------|------------|
| JSONL | `.jsonl` | QA, Summarization, Classification, Text Generation, Chat |
| CSV | `.csv` | QA, Summarization, Classification, Text Generation |
| Parquet | `.parquet` | All |

### Upload a Dataset

```bash
# Auto-detect format and task type
trustops datasets upload demo/sample_data/qa_sample.jsonl \
  --name "QA Evaluation Set"

# Specify task type explicitly
trustops datasets upload my_data.csv \
  --name "Customer Support Classification" \
  --task-type classification
```

### List Datasets

```bash
trustops datasets list
trustops datasets list --task-type qa
trustops datasets list --format jsonl
```

### Analyze Dataset Quality

```bash
trustops datasets analyze --dataset-id <dataset-id>
# Returns: completeness score, diversity score, balance score,
#          token statistics, issues, recommendations
```

### Convert Format

```bash
trustops datasets convert --dataset-id <dataset-id> --format parquet
trustops datasets convert --dataset-id <dataset-id> --format csv
```

### Split Dataset

```bash
trustops datasets split --dataset-id <dataset-id> \
  --train 0.8 --val 0.1 --test 0.1 \
  --stratify-by category
```

### Detect and Mask PII

```bash
# Detect PII (email, phone, SSN, credit card, IP)
trustops datasets analyze --dataset-id <dataset-id> --pii

# Mask PII
trustops datasets upload my_data.jsonl --name "Cleaned Data" --mask-pii
```

### Sample Datasets

The framework ships with sample datasets in `demo/sample_data/`:

| File | Task Type | Records |
|------|-----------|---------|
| `qa_sample.jsonl` | QA | 10 |
| `summarization_sample.jsonl` | Summarization | 10 |
| `classification_sample.jsonl` | Classification | 10 |
| `chat_sample.jsonl` | Chat | 10 |

---

## 4. Baseline Model Evaluation

Baseline evaluation assesses a model's performance before any fine-tuning, establishing reference metrics.

### Run Baseline Evaluation

```bash
trustops evaluate baseline \
  --model-id anthropic.claude-3-haiku-20240307-v1:0 \
  --dataset-id <dataset-id>
```

### With Custom Configuration

```bash
trustops evaluate baseline \
  --model-id anthropic.claude-v2 \
  --dataset-id <dataset-id> \
  --temperature 0.3 \
  --max-tokens 512 \
  --timeout 120
```

### Using the Legacy Command

```bash
trustops baseline \
  --model-id anthropic.claude-v2 \
  --dataset s3://your-bucket/eval-dataset.json
```

### What You Get

The baseline evaluation produces:
- Per-response trust scores (5 dimensions: accuracy, consistency, safety, bias, context grounding)
- Per-response hallucination analysis (claim extraction, evidence matching, grounding scores)
- Aggregate metrics: mean/median trust score, p50/p95/p99 latency, total cost
- Per-category breakdown
- Cost summary (input tokens, output tokens, cost per query)

---

## 5. Fine-Tuning a Model

### Check Eligibility

Not all models support fine-tuning. The framework validates this automatically, but you can check:

```bash
trustops models list --fine-tunable
```

### Start Fine-Tuning

```bash
trustops finetune start \
  --model-id amazon.titan-text-express-v1 \
  --dataset-id <training-dataset-id> \
  --job-name "customer-support-finetune-v1"
```

### With Custom Hyperparameters

```bash
trustops finetune start \
  --model-id amazon.titan-text-express-v1 \
  --dataset-id <training-dataset-id> \
  --job-name "custom-finetune" \
  --learning-rate 0.00005 \
  --epochs 5 \
  --batch-size 16
```

### Monitor Progress

```bash
# Check status
trustops finetune status --job-id <job-id>

# List all jobs
trustops finetune list
trustops finetune list --status training

# Stop a job
trustops finetune stop --job-id <job-id>
```

### Cost Estimation

The framework displays an estimated cost before starting the job. Costs depend on model family, dataset size, and number of epochs.

### What Happens on Completion

1. Fine-tuned model ARN is extracted
2. Model is registered in the Model Registry with training metadata
3. Audit trail records all parameters, metrics, and outcomes

---

## 6. Comparative Evaluation

Compare a baseline model against a fine-tuned model on identical prompts.

### Run Comparison

```bash
trustops evaluate compare \
  --baseline-model-id anthropic.claude-v2 \
  --finetuned-model-id <finetuned-model-arn> \
  --dataset-id <dataset-id>
```

### Using the Legacy Command

```bash
trustops compare \
  --baseline-model-id anthropic.claude-v2 \
  --finetuned-model-id arn:aws:bedrock:us-east-1:123:model/... \
  --dataset s3://your-bucket/eval-dataset.json
```

### What You Get

- Side-by-side trust scores for both models
- Improvement metrics: trust score delta, hallucination reduction, latency delta, cost delta
- Statistical significance (p-value, confidence intervals)
- Deployment recommendation: DEPLOY / ITERATE / REJECT
  - DEPLOY: all thresholds met (trust improvement >= 5%, cost increase <= 20%, hallucination reduction >= 10%)
  - REJECT: trust score decreased or hallucination rate increased
  - ITERATE: otherwise
- Cost-performance analysis (cost per trust score point, break-even volume)
- Per-category breakdown

---

## 7. Foundation Model Comparison

Compare any two foundation models without fine-tuning — useful for model selection.

```bash
trustops foundation-compare \
  --model-id-1 anthropic.claude-3-haiku-20240307-v1:0 \
  --model-id-2 anthropic.claude-3-sonnet-20240229-v1:0 \
  --dataset s3://your-bucket/eval-dataset.json
```

This sends identical prompts to both models concurrently and produces the same comparison report as the comparative evaluation.

---

## 8. Deploy/Iterate/Reject Decision

Every comparative evaluation (baseline vs fine-tuned, or foundation model vs foundation model) culminates in a data-driven deployment recommendation.

### Understanding the Decision

| Decision | When It's Produced | What It Means |
|----------|-------------------|---------------|
| **DEPLOY** | Trust improvement >= 10%, hallucination reduction >= 5%, cost increase <= 20%, p-value < 0.05 | Safe to deploy the better model to production |
| **ITERATE** | Moderate improvement, or cost too high, or not statistically significant | More fine-tuning or optimization needed |
| **REJECT** | Trust improvement < 5% or hallucination reduction negligible | Fine-tuning didn't help; try a different approach |

### Statistical Significance

The framework automatically selects the appropriate test:

- **n >= 20 examples**: Paired t-test (Central Limit Theorem applies)
- **n < 20 examples**: Wilcoxon signed-rank test (non-parametric, no normality assumption)
- **n < 3 examples**: Insufficient data (no test performed, p-value = 1.0)

Both tests are always computed. Results include:
- `p_value_ttest`: p-value from paired t-test
- `p_value_wilcoxon`: p-value from Wilcoxon signed-rank test
- `confidence_interval_lower` / `confidence_interval_upper`: 95% CI for the mean difference
- `test_type_used`: Which test was selected as primary ("ttest" or "wilcoxon")

### Break-Even Volume Analysis

When the better model costs more per query:

```bash
# The framework computes:
break_even_volume = cost_premium_per_query / (hallucination_rate_reduction * remediation_cost)
```

Default remediation cost: **$50 per hallucinated response** (configurable).

Example output:
```
Break-even volume: 6,000 queries
Monthly savings at 10K queries: $2,000
Monthly savings at 100K queries: $20,000
```

### Viewing the Recommendation

After a comparative evaluation completes:

```bash
trustops results get --evaluation-id <eval-id> --format json
```

The `improvement_metrics` object includes:
```json
{
  "trust_score_improvement": 0.13,
  "hallucination_reduction": 0.08,
  "statistical_significance": true,
  "p_value_ttest": 0.003,
  "p_value_wilcoxon": 0.005,
  "confidence_interval_lower": -0.17,
  "confidence_interval_upper": -0.09,
  "test_type_used": "ttest",
  "recommendation": "deploy",
  "justification": "Significant trust improvement (13.0%) and hallucination reduction (8.0%) with acceptable cost increase (4.0%)."
}
```

### Step Functions Workflow

The recommendation is produced by the `generate_recommendation` Lambda handler, orchestrated by Step Functions:

1. Baseline evaluation completes → metrics stored to S3
2. Fine-tuned evaluation completes → metrics stored to S3
3. `GenerateRecommendation` Lambda compares metrics and runs statistical tests
4. **Approval Gate** pauses workflow for human review
5. Upon approval, results are stored with full audit trail

### Customizing Thresholds

The recommendation thresholds can be adjusted by modifying the `_generate_recommendation` function in `src/orchestration/metrics_aggregator.py`:

- Trust score improvement threshold (default: 0.10)
- Hallucination reduction threshold (default: 0.05)
- Cost increase limit (default: 20%)
- Significance level (default: p < 0.05)

---

## 9. Trust Scoring Deep Dive

Trust scores are calculated across 5 dimensions, each scored 0–1:

| Dimension | Default Weight | What It Measures |
|-----------|---------------|------------------|
| Accuracy | 0.25 | Exact match, fuzzy match, semantic similarity vs expected answer |
| Consistency | 0.20 | Response variance across multiple invocations of the same prompt |
| Safety | 0.20 | Harmful content, toxicity, policy violations |
| Bias | 0.15 | Demographic bias, stereotyping, unbalanced treatment |
| Context Grounding | 0.20 | Semantic similarity to source documents (embedding cosine similarity) |

### Customizing Weights

Weights are configurable per evaluation run. They must sum to 1.0.

### Review Flagging

Responses with an overall trust score below the review threshold (default: 0.6) are automatically flagged for human review.

### Custom Metrics

You can register custom scoring functions via the plugin interface:

```python
from src.trust_scoring.custom_metric import CustomMetricBase

class MyCustomScorer(CustomMetricBase):
    def score(self, prompt, response, **kwargs):
        # Your scoring logic
        return 0.85  # Return score in [0, 1]
```

### Trust Score Calibration

Calibrate scores against human-labeled ground truth to validate scoring accuracy:

```python
from src.trust_scoring.trust_scoring_engine import TrustScoringEngine

engine = TrustScoringEngine(...)
result = await engine.calibrate(ground_truth_data)
# Returns correlation coefficient
```

---

## 10. Hallucination Detection

The hallucination detector identifies unsupported claims in model outputs.

### How It Works

1. Response is split into individual claims (sentence segmentation)
2. Each claim is classified as factual or opinion/hedged
3. Factual claims are matched against source documents using embedding cosine similarity
4. Claims below the similarity threshold are flagged as unsupported
5. Hallucination rate = unsupported claims / total factual claims

### Sensitivity Levels

| Level | Threshold | Use Case |
|-------|-----------|----------|
| Strict | 0.8 | Medical, legal, compliance |
| Moderate | 0.6 | General business (default) |
| Lenient | 0.4 | Creative, exploratory |

### Alerts

When hallucination rate exceeds a configurable threshold, the system triggers alerts.

---

## 11. Workflow Orchestration

Workflows chain multiple steps into automated, auditable pipelines.

### Pre-Built Templates

| Template | Steps |
|----------|-------|
| Full Pipeline | Dataset → Baseline Eval → Fine-Tuning → Post-Tuning Eval → Comparison |
| Evaluation Only | Dataset → Baseline Eval |
| Comparison Only | Model A vs Model B comparison |

### Run a Workflow

```bash
trustops workflows run --config workflow_config.json
```

### Monitor Workflows

```bash
# List workflows
trustops workflows list
trustops workflows list --status running

# Check status
trustops workflows status --workflow-id <id>

# Resume a failed workflow
trustops workflows resume --workflow-id <id>

# Cancel a running workflow
trustops workflows cancel --workflow-id <id>

# Reproduce a completed workflow
trustops workflows reproduce --workflow-id <id>
```

### Check Legacy Status

```bash
trustops status <workflow-id>
trustops status --list
trustops status --list --type baseline --status-filter completed
```

### Workflow Features

- Unique workflow IDs (UUID-based) for isolation
- Sequential and parallel step execution
- Approval gates for manual checkpoints
- Retry with exponential backoff on transient failures
- Progress persistence to DynamoDB (resume after disconnection)
- Complete manifest with checksums for all artifacts
- Cost and time estimation before execution
- Concurrent workflow isolation (no cross-workflow interference)

---

## 12. Results Storage and Export

### Retrieve Results

```bash
trustops results get --evaluation-id <eval-id>
```

### Compare Results

```bash
trustops results compare --eval-id-1 <id1> --eval-id-2 <id2>
```

### Export Results

```bash
trustops results export --evaluation-id <eval-id> --format json
trustops results export --evaluation-id <eval-id> --format csv
trustops results export --evaluation-id <eval-id> --format pdf
```

### Storage Architecture

- S3: Full results stored at `s3://{bucket}/trustops/{workflow_id}/{artifact_type}/{timestamp}/`
- DynamoDB: Metadata for fast querying with GSIs
- Versioning: All results are versioned with SHA-256 checksums
- Retention: Configurable archival to S3 Glacier and deletion policies

---

## 13. Dashboard Usage

### Start the Dashboard

```bash
make run-dashboard
# Opens at http://localhost:8501
```

### Dashboard Pages

| Page | Features |
|------|----------|
| Home | Recent workflows, system health, cost summary |
| Models | Browse registry, filter by provider/capability, register external models |
| Datasets | Upload, quality reports, PII detection, version history, lineage graph |
| Evaluation | Baseline evaluation wizard, real-time progress, aggregate metrics drill-down |
| Comparison | Side-by-side metrics, improvement charts, deployment recommendation |
| Fine-Tuning | Step-by-step wizard, cost estimate, real-time loss charts, job history |
| Workflows | Template selector, progress tracker, audit trail viewer |

### Dashboard Components

- 5-dimension radar chart for trust scores
- Hallucination text highlighting with evidence links
- Trust score distribution histograms
- Cost-performance analysis charts
- Real-time progress bars with ETA

---

## 14. Running Tests

```bash
# All tests
make test

# By category
make test-unit          # 100+ unit tests
make test-property      # Property-based tests (Hypothesis)
make test-integration   # End-to-end integration tests
make test-dashboard     # Dashboard page and component tests
make test-cli           # CLI command tests

# With coverage
make coverage

# Verbose
make test-all
```

### Test Categories

| Category | Count | What It Covers |
|----------|-------|----------------|
| Unit | 100+ | All components: adapters, scorers, parsers, calculators, etc. |
| Property | PBT | Data model invariants, score ranges, format round-trips |
| Integration | 5 | E2E workflow, multi-provider, concurrent workflows, failure recovery, degraded mode |
| Dashboard | 20 | Page rendering, components, caching, error handling |
| CLI | 8 | All CLI commands with mocked backends |

---

## 15. Complete End-to-End Walkthrough

This walkthrough demonstrates the full evaluate → fine-tune → compare loop.

### Step 1: Discover Models

```bash
trustops models list --provider bedrock
trustops models info --model-id amazon.titan-text-express-v1
```

### Step 2: Upload Evaluation Dataset

```bash
trustops datasets upload demo/sample_data/qa_sample.jsonl \
  --name "QA Eval Set v1" \
  --task-type qa
# Note the dataset-id from the output
```

### Step 3: Analyze Dataset Quality

```bash
trustops datasets analyze --dataset-id <dataset-id>
```

### Step 4: Run Baseline Evaluation

```bash
trustops evaluate baseline \
  --model-id amazon.titan-text-express-v1 \
  --dataset-id <dataset-id>
# Note the evaluation-id from the output
```

### Step 5: Review Baseline Results

```bash
trustops results get --evaluation-id <eval-id>
trustops results export --evaluation-id <eval-id> --format json
```

### Step 6: Prepare Training Data

```bash
# Upload training dataset (JSONL format for Bedrock fine-tuning)
trustops datasets upload training_data.jsonl \
  --name "Training Data v1" \
  --task-type qa
```

### Step 7: Fine-Tune

```bash
trustops finetune start \
  --model-id amazon.titan-text-express-v1 \
  --dataset-id <training-dataset-id> \
  --job-name "qa-finetune-v1"

# Monitor
trustops finetune status --job-id <job-id>
```

### Step 8: Compare Baseline vs Fine-Tuned

```bash
trustops evaluate compare \
  --baseline-model-id amazon.titan-text-express-v1 \
  --finetuned-model-id <finetuned-model-id> \
  --dataset-id <dataset-id>
```

### Step 9: Review Comparison

```bash
trustops results get --evaluation-id <comparison-eval-id>
# Check: trust score delta, hallucination reduction, deployment recommendation
```

### Step 10: Export Final Report

```bash
trustops results export --evaluation-id <comparison-eval-id> --format json
```

---

## 16. Troubleshooting Quick Reference

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `AccessDeniedException` on Bedrock | Model access not requested | AWS Console → Bedrock → Model access → Request |
| `trustops: command not found` | CLI not installed | `make cli-dev` |
| S3 bucket errors | Bucket doesn't exist or wrong name | Check `.env` bucket names, run `make verify` |
| DynamoDB errors | Tables not created | `make create-tables` |
| Evaluation timeout | Large dataset or slow model | Reduce dataset size or increase `--timeout` |
| High costs | Using expensive model for dev | Switch to Claude Haiku or Titan Lite for development |
| Dataset validation failure | Wrong format or missing fields | Check format requirements in `demo/sample_data/README.md` |
| Dashboard won't start | Streamlit not installed | `pip install streamlit` or `make install` |
| Import errors | Package not installed in dev mode | `make cli-dev` |

For detailed troubleshooting, see `docs/OPERATIONAL_RUNBOOK.md`.

---

## CLI Command Reference

```
trustops --help                    # Show all commands
trustops configure                 # Set up AWS configuration
trustops auth check                # Verify credentials
trustops auth configure            # Interactive credential setup

trustops models list               # List models
trustops models info               # Model details
trustops models register           # Register external model

trustops datasets upload           # Upload dataset
trustops datasets list             # List datasets
trustops datasets analyze          # Quality analysis
trustops datasets convert          # Convert format
trustops datasets split            # Train/val/test split

trustops evaluate baseline         # Baseline evaluation
trustops evaluate compare          # Comparative evaluation

trustops finetune start            # Start fine-tuning
trustops finetune status           # Job status
trustops finetune stop             # Stop job
trustops finetune list             # List jobs

trustops workflows run             # Run workflow
trustops workflows list            # List workflows
trustops workflows status          # Workflow status
trustops workflows resume          # Resume workflow
trustops workflows reproduce       # Reproduce workflow
trustops workflows cancel          # Cancel workflow

trustops results get               # Get results
trustops results export            # Export results
trustops results compare           # Compare results

trustops config show               # Show configuration
trustops config set                # Update configuration

trustops baseline                  # Legacy: baseline evaluation
trustops compare                   # Legacy: comparative evaluation
trustops foundation-compare        # Legacy: foundation model comparison
trustops status                    # Legacy: workflow status
trustops reproduce                 # Legacy: reproduce workflow
```

---

## Makefile Targets

```
make setup             # Create venv and install dependencies
make install           # Install dependencies only
make cli-install       # Install CLI
make cli-dev           # Install CLI in development mode
make test              # Run all tests
make test-unit         # Unit tests
make test-property     # Property-based tests
make test-integration  # Integration tests
make test-dashboard    # Dashboard tests
make test-cli          # CLI tests
make coverage          # Tests with coverage report
make run-dashboard     # Start Streamlit dashboard
make create-tables     # Create DynamoDB tables
make verify            # Verify AWS infrastructure
make deploy            # Deploy infrastructure
make lint              # Run linting
make format            # Format code with black
make clean             # Remove generated files
```
