# TrustOps CLI

Command-line interface for TrustOps — a trust-first LLM evaluation, fine-tuning, and comparison framework for enterprise GenAI on AWS.

## Installation

```bash
pip install -e .        # development mode
trustops --version      # verify
```

## Quick Start

```bash
# 1. Configure AWS
trustops configure

# 2. Check credentials
trustops auth check

# 3. List available models
trustops models list --provider bedrock

# 4. Upload evaluation dataset
trustops datasets upload demo/sample_data/qa_sample.jsonl \
  --name "QA Evaluation Set" --task-type qa

# 5. Run baseline evaluation
trustops baseline \
  --model-id amazon.nova-lite-v1:0 \
  --dataset s3://my-bucket/eval/qa_sample.jsonl --wait

# 6. Compare two models
trustops foundation-compare \
  --model-id-1 amazon.nova-lite-v1:0 \
  --model-id-2 amazon.nova-pro-v1:0 \
  --dataset s3://my-bucket/eval/qa_sample.jsonl
```

## Commands

| Command | Description |
|---------|-------------|
| `trustops auth check` | Verify AWS credentials |
| `trustops auth configure` | Interactive AWS setup wizard |
| `trustops config show` | Display current configuration |
| `trustops config set KEY VALUE` | Update a configuration value |
| `trustops models list` | List available models (with `--provider`, `--fine-tunable` filters) |
| `trustops models info MODEL_ID` | Show model details |
| `trustops models register` | Register an external model |
| `trustops datasets upload FILE` | Upload a dataset with auto-detection |
| `trustops datasets list` | List uploaded datasets |
| `trustops datasets analyze` | Run quality analysis on a dataset |
| `trustops evaluate baseline` | Run baseline evaluation |
| `trustops evaluate compare` | Run comparative evaluation |
| `trustops baseline` | Run baseline with workflow tracking |
| `trustops compare` | Run comparison with workflow tracking |
| `trustops foundation-compare` | Compare two foundation models head-to-head |
| `trustops finetune start` | Start a fine-tuning job (with cost estimate and consent) |
| `trustops finetune status JOB_ID` | Check fine-tuning job status |
| `trustops finetune stop JOB_ID` | Stop a running fine-tuning job |
| `trustops workflows list` | List all workflows |
| `trustops workflows status ID` | Check workflow status |
| `trustops workflows run` | Run a workflow from template |
| `trustops workflows resume ID` | Resume a paused workflow |
| `trustops status --list` | List recent workflows |
| `trustops status WORKFLOW_ID` | Detailed workflow status |
| `trustops results get EVAL_ID` | Retrieve evaluation results |
| `trustops results export EVAL_ID` | Export results (JSON, CSV, PDF) |
| `trustops results compare ID1 ID2` | Compare two evaluations side-by-side |
| `trustops reproduce WORKFLOW_ID` | Reproduce a previous workflow |

## Usage Examples

### Baseline Evaluation

```bash
trustops baseline \
  --model-id amazon.nova-lite-v1:0 \
  --dataset s3://my-bucket/eval-dataset.jsonl \
  --wait
```

### Foundation Model Comparison

```bash
trustops foundation-compare \
  --model-id-1 amazon.nova-lite-v1:0 \
  --model-id-2 amazon.nova-pro-v1:0 \
  --dataset s3://my-bucket/eval-dataset.jsonl \
  --temperature 0.7 --max-tokens 2048
```

### Fine-Tuning

```bash
# Shows cost estimate, asks for confirmation before proceeding
trustops finetune start \
  --model amazon.nova-lite-v1:0 \
  --dataset my-training-data \
  --num-examples 500

# Check job progress
trustops finetune status ft-abc123def456

# Skip confirmation prompt (for automation)
trustops finetune start --model amazon.nova-lite-v1:0 --dataset my-data --yes
```

### Export Results

```bash
# JSON export
trustops results export baseline-20240115-abc123 --export-format json --output report.json

# CSV export
trustops results export baseline-20240115-abc123 --export-format csv --output report.csv

# PDF export
trustops results export baseline-20240115-abc123 --export-format pdf --output report.pdf
```

## Configuration

Configuration is stored at `~/.trustops/config.json`. The CLI also reads environment variables from `.env` at the project root.

Key settings:
- `AWS_REGION` — AWS region for all services
- `TRUSTOPS_DATASETS_BUCKET` — S3 bucket for evaluation datasets
- `TRUSTOPS_RESULTS_BUCKET` — S3 bucket for evaluation results
- `TRUSTOPS_ARTIFACTS_BUCKET` — S3 bucket for workflow artifacts

See `infrastructure/env.example` for the complete list.

## Output Formats

All commands support `--format text|json|yaml` for output formatting:

```bash
trustops models list --format json
trustops status --list --format yaml
```
