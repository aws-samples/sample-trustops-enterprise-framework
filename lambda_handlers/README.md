# Lambda Handlers — Dual-Directory Structure

This project contains two separate `lambda_handlers` directories. They are **not duplicates** — each serves a distinct purpose in the deployment architecture.

## `lambda_handlers/` (top-level) — Orchestrator Handlers

These are high-level entry points invoked by API Gateway and Step Functions top-level triggers:

- `evaluation_handler.py` — Orchestrates end-to-end model evaluation workflows
- `fine_tuning_handler.py` — Orchestrates fine-tuning job lifecycle
- `trust_scoring_handler.py` — Orchestrates trust score computation
- `comparative_evaluation_handler.py` — Orchestrates side-by-side model comparisons

## `src/lambda_handlers/` — Granular Step Functions Handlers

These are per-step handlers used **inside** Step Functions state machine workflows:

- `evaluate_single_example.py` — Evaluates a single prompt/response example
- `aggregate_metrics.py` — Aggregates per-example metrics into summary scores
- `generate_recommendation.py` — Generates recommendations from aggregated results
- `detect_hallucinations.py` — Runs hallucination detection on model outputs

## Packaging

Both sets of handlers are packaged separately during deployment. See `infrastructure/deploy.sh` (lines 70–99) which creates individual zip artifacts for each handler, bundling the shared `src/` library into every package.
