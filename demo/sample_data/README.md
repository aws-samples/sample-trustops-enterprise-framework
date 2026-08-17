# TrustOps Sample Datasets

Sample datasets for demos and testing, covering the core task types supported by the TrustOps Dataset Manager.

## Datasets

| File | Task Type | Records | Description |
|------|-----------|---------|-------------|
| `qa_sample.jsonl` | QA | 10 | Question-answering pairs about ML and AWS concepts with context passages |
| `summarization_sample.jsonl` | Summarization | 10 | Document summarization examples across technology, ML, and AWS topics |
| `classification_sample.jsonl` | Classification | 10 | Support ticket classification (bug, feature_request, question, etc.) |
| `chat_sample.jsonl` | Chat | 10 | Multi-turn conversations about TrustOps features and workflows |

## Format

All datasets use JSONL format (one JSON object per line).

### Field Structures

**QA** (`qa_sample.jsonl`):
- `prompt` — The question
- `completion` — The expected answer
- `context` — Reference passage for grounding
- `category` — Topic category

**Summarization** (`summarization_sample.jsonl`):
- `prompt` — Document text with summarization instruction
- `completion` — Reference summary
- `category` — Topic category

**Classification** (`classification_sample.jsonl`):
- `prompt` — Input text to classify
- `completion` — Classification label
- `category` — Functional area

**Chat** (`chat_sample.jsonl`):
- `messages` — Array of `{role, content}` objects (system/user/assistant turns)
- `category` — Conversation topic

## Usage

```bash
# Run baseline evaluation with a sample dataset
trustops evaluate baseline --model <model_id> --dataset demo/sample_data/qa_sample.jsonl

# Analyze dataset quality
trustops datasets analyze demo/sample_data/classification_sample.jsonl
```
