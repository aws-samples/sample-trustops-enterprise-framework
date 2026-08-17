# TrustOps Enterprise Framework

**Trust-first LLM evaluation, fine-tuning, and comparison for AWS.**

Evaluate any model. Quantify trust. Detect hallucinations. Fine-tune with confidence. Deploy with evidence.

---

## What It Does

TrustOps gives you a single platform to answer: *"Can I trust this model for production?"*

It takes you from raw dataset → model evaluation → fine-tuning → side-by-side comparison → deploy/reject decision, with full audit trails and cost tracking at every step.

---

## Core Capabilities

| Capability | What You Get |
|---|---|
| **Multi-Provider Models** | Unified access to AWS Bedrock, SageMaker, and external APIs (OpenAI, etc.) — 50+ models with pricing built in, auto-discovery, health monitoring, TTL-based metadata caching |
| **Dataset Management** | Upload, auto-detect format, validate quality, detect & mask PII, split train/val/test, convert between JSONL/CSV/Parquet, synthetic data generation (paraphrase/diverse/creative), domain templates (customer support, legal, medical, financial), version everything with lineage tracking |
| **Baseline Evaluation** | Run any model against your dataset — get trust scores, hallucination rates, latency percentiles, cost per query, per-category breakdowns, resumable evaluations with checkpoint persistence |
| **Fine-Tuning Pipeline** | Validate eligibility, estimate cost upfront, launch Bedrock or SageMaker jobs, monitor training loss, early stopping, automated hyperparameter tuning, job resume, audit logging, auto-register the result |
| **Comparative Evaluation** | Same prompts, two models, side-by-side — trust score deltas, hallucination reduction, paired statistical significance (t-test + Wilcoxon with automatic test selection), 95% confidence intervals, deploy/iterate/reject recommendation, break-even cost analysis with monthly savings projections |
| **Foundation Model Comparison** | Compare any two foundation models head-to-head (e.g., Haiku vs Sonnet) without fine-tuning — same comparison metrics and recommendation engine |
| **Trust Scoring** | 5-dimension scoring (accuracy, consistency, safety, bias, context grounding) with configurable weights, custom metrics plugin, calibration against ground truth, confidence intervals, review flagging |
| **Hallucination Detection** | Claim extraction → opinion/hedge filtering → evidence matching via embedding cosine similarity → grounding scores — with strict/moderate/lenient sensitivity, text-span highlighting, aggregate metrics by category/model/time, alerting on threshold breach |
| **Semantic Similarity** | Bedrock Knowledge Bases retrieval with managed embeddings (Titan, Cohere), bulk document ingestion, vector search — powers both trust scoring and hallucination detection |
| **Workflow Orchestration** | Chain steps into reproducible pipelines with retry logic, approval gates, parallel execution, progress persistence, cost/time estimation, partial failure handling, failure notifications, concurrent workflow isolation, full manifests with SHA-256 checksums |
| **Results Storage** | Versioned results in S3 (gzip compressed) + DynamoDB, query by any dimension, export to JSON/CSV/PDF, checksum integrity verification, configurable retention policies (Glacier archival), access audit logging |
| **CLI** | Full CLI (`trustops`) for automation and scripting, JSON/YAML output modes, every feature accessible, CI/CD friendly |
| **Streamlit Dashboard** | 7-page web UI — models, datasets, evaluation, comparison, fine-tuning, workflows, home — with radar charts, hallucination highlighting, real-time progress, dark/light theming, degraded mode handling, health check endpoint |
| **QuickSight Analytics** | 4 pre-built QuickSight dashboards (baseline evaluation, comparative evaluation, cost analysis, hallucination analysis) fed via S3 → Glue → Athena pipeline with daily auto-refresh |
| **Containerized Deployment** | Dockerfile + docker-compose for the dashboard, built-in health checks, configurable via environment variables |

---

## The Workflow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Upload  │───▶│ Baseline │───▶│  Fine-   │───▶│ Compare  │───▶│  Deploy  │
│ Dataset  │    │   Eval   │    │  Tune    │    │ Models   │    │ Decision │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │               │
  Quality         Trust           Cost            Delta          DEPLOY /
  Analysis       Scores         Estimate        Metrics        ITERATE /
  PII Scan     Hallucination   Loss Tracking   Significance     REJECT
  Versioning    Cost/Latency   Early Stopping  Cost-Perf
```

---

## Deploy / Iterate / Reject Decision Framework

The final step of every comparative evaluation produces a data-driven deployment recommendation:

| Decision | Criteria | Action |
|----------|----------|--------|
| **DEPLOY** | Trust improvement >= 10% AND hallucination reduction >= 5% AND cost increase <= 20% AND statistically significant (p < 0.05) | Approve for production deployment |
| **ITERATE** | Moderate improvement OR cost too high OR not statistically significant | Additional fine-tuning recommended |
| **REJECT** | Insufficient improvement in trust or hallucination metrics | Fine-tuning did not help; try different approach |

### Statistical Rigor

- **Paired t-test** (n >= 20): Used when sample size supports Central Limit Theorem
- **Wilcoxon signed-rank** (n < 20): Non-parametric test for small sample sizes
- **95% Confidence Intervals**: Computed via t-distribution on the per-example difference vector
- **Both tests always run**: Primary test is selected automatically, but both p-values are reported

### Break-Even Volume Analysis

When comparing a higher-trust model (Model B) that costs more per query against a cheaper model (Model A):

```
break_even_volume = cost_premium_per_query / (hallucination_rate_reduction * remediation_cost_per_hallucination)
```

Default remediation cost: $50/hallucinated response (configurable). The analysis provides:
- Exact break-even query volume
- Monthly savings at 1K, 10K, 100K, 1M queries
- Cost crossover chart data for visualization

---

## Trust Score Dimensions

| Dimension | Weight | Measures |
|---|---|---|
| Accuracy | 25% | Match against expected answers (exact, fuzzy, semantic) |
| Consistency | 20% | Response stability across repeated invocations |
| Safety | 20% | Harmful content, toxicity, policy violations |
| Bias | 15% | Demographic bias, stereotyping, fairness |
| Context Grounding | 20% | Factual support from source documents |

Overall score: weighted combination in **[0, 1]**. Below threshold → flagged for human review.

---

## Supported Models

**Bedrock**: Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku, Claude v2, Titan Text/Embed, Llama 3.1/3/2, Mistral Large/Small/Mixtral, Cohere Command R/R+, AI21 Jamba/Jurassic

**SageMaker**: Any deployed endpoint

**External APIs**: Any OpenAI-compatible API (GPT-4, GPT-3.5, etc.)

---

## Dataset Features

**Formats**: JSONL, CSV, Parquet (auto-detected, bidirectional conversion)

**Task Types**: QA, Summarization, Classification, Text Generation, Chat

**Quality**: Completeness, diversity, balance scoring with actionable recommendations

**PII**: Regex detection (email, phone, SSN, credit card, IP) with mask/redact/remove strategies

**Augmentation**: Synthetic data generation via LLM (paraphrase, diverse, creative strategies)

**Domain Templates**: Pre-built schemas for customer support, legal, medical, financial datasets

**Versioning**: S3-backed with SHA-256 checksums, lineage tracking, usage history

---

## Quick Start

```bash
pip install -e .                    # Install
trustops configure                  # Set up AWS
trustops models list                # See available models
trustops datasets upload data.jsonl # Upload your data
trustops evaluate baseline          # Evaluate a model
trustops finetune start             # Fine-tune it
trustops evaluate compare           # Compare before/after
trustops results export             # Get your report
make run-dashboard                  # Visual dashboard
```

---

## Interfaces

**CLI** — `trustops` command with subcommands for every feature, JSON/YAML output modes, scriptable for CI/CD

**Streamlit Dashboard** — Web UI at `localhost:8501` with 7 pages (Home, Models, Datasets, Evaluation, Comparison, Fine-Tuning, Workflows), interactive charts, radar plots, hallucination highlighting, dark/light mode, degraded mode handling when AWS services are unavailable

**QuickSight Dashboards** — 4 pre-built analytics dashboards (Baseline Evaluation, Comparative Evaluation, Cost Analysis, Hallucination Analysis) with daily auto-refresh via S3 → Glue → Athena pipeline

---

## Infrastructure

**AWS Services**: Bedrock (including Knowledge Bases), SageMaker, S3, DynamoDB, Step Functions, Lambda, CloudWatch, Glue, Athena, QuickSight

**Serverless Evaluation Handlers**: 4 granular Lambda functions for Step Functions orchestration — evaluate single example, aggregate metrics, generate recommendation, detect hallucinations

**Deployment**: CloudFormation or Terraform, single `./deploy.sh` command

**Containerization**: Dockerfile + docker-compose for the dashboard with built-in health checks

**Language**: Python 3.11+

**Testing**: 100+ unit tests, property-based tests (Hypothesis), integration tests, dashboard tests

**Monitoring**: CloudWatch metrics (trust scores, latency, cost, errors), configurable alarms, CloudTrail audit logging

---

## Key Design Principles

- **Provider Abstraction** — one interface for all model providers
- **Trust-First** — every output gets a quantifiable trust score
- **Auditability** — complete lineage with checksums and version tracking
- **Resilience** — retry logic, progress persistence, graceful degradation
- **Extensibility** — plugin architecture for custom metrics and providers
