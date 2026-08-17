"""Demo data generators for the dashboard."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone


def _ts(days_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def generate_demo_models() -> list[dict]:
    return [
        {"id": "bedrock-claude-3", "provider": "bedrock", "name": "Claude 3 Sonnet",
         "capabilities": ["text_generation", "chat"], "status": "active",
         "fine_tuning_support": True, "max_tokens": 200000},
        {"id": "bedrock-titan-text", "provider": "bedrock", "name": "Titan Text Express",
         "capabilities": ["text_generation"], "status": "active",
         "fine_tuning_support": True, "max_tokens": 8192},
        {"id": "sagemaker-llama2", "provider": "sagemaker", "name": "Llama 2 70B",
         "capabilities": ["text_generation", "chat"], "status": "active",
         "fine_tuning_support": False, "max_tokens": 4096},
        {"id": "external-gpt4", "provider": "external", "name": "GPT-4",
         "capabilities": ["text_generation", "chat"], "status": "inactive",
         "fine_tuning_support": False, "max_tokens": 128000},
    ]


def generate_demo_datasets() -> list[dict]:
    return [
        {"id": "ds-qa-001", "name": "Customer Support QA", "format": "jsonl",
         "task_type": "qa", "row_count": 5000,
         "quality": {"completeness": 0.95, "diversity": 0.82, "balance": 0.78}},
        {"id": "ds-sum-001", "name": "Legal Summarization", "format": "csv",
         "task_type": "summarization", "row_count": 2000,
         "quality": {"completeness": 0.88, "diversity": 0.75, "balance": 0.90}},
        {"id": "ds-cls-001", "name": "Sentiment Classification", "format": "parquet",
         "task_type": "classification", "row_count": 10000,
         "quality": {"completeness": 1.0, "diversity": 0.65, "balance": 0.92}},
    ]


def generate_demo_evaluations() -> list[dict]:
    return [
        {"id": "eval-001", "model_id": "bedrock-claude-3", "dataset_id": "ds-qa-001",
         "mean_trust_score": 0.82, "total_cost": 12.50, "created_at": _ts(2)},
        {"id": "eval-002", "model_id": "bedrock-titan-text", "dataset_id": "ds-qa-001",
         "mean_trust_score": 0.71, "total_cost": 4.30, "created_at": _ts(1)},
        {"id": "eval-003", "model_id": "sagemaker-llama2", "dataset_id": "ds-sum-001",
         "mean_trust_score": 0.76, "total_cost": 8.90, "created_at": _ts(0)},
    ]


def generate_demo_workflows() -> list[dict]:
    return [
        {"id": "wf-001", "name": "Full Pipeline Run", "status": "completed",
         "created_at": _ts(3),
         "steps": [
             {"name": "Dataset Upload", "status": "completed"},
             {"name": "Baseline Evaluation", "status": "completed"},
             {"name": "Fine-Tuning", "status": "completed"},
             {"name": "Post-Tuning Evaluation", "status": "completed"},
             {"name": "Comparison", "status": "completed"},
         ]},
        {"id": "wf-002", "name": "Evaluation Only", "status": "running",
         "created_at": _ts(0),
         "steps": [
             {"name": "Dataset Upload", "status": "completed"},
             {"name": "Baseline Evaluation", "status": "running"},
         ]},
        {"id": "wf-003", "name": "Comparison Run", "status": "failed",
         "created_at": _ts(1),
         "steps": [
             {"name": "Comparison", "status": "failed"},
         ]},
    ]


def generate_demo_fine_tuning_jobs() -> list[dict]:
    return [
        {"id": "ft-001", "model_id": "bedrock-claude-3", "status": "completed",
         "training_metrics": [
             {"epoch": 1, "loss": 2.5, "validation_loss": 2.8},
             {"epoch": 2, "loss": 1.8, "validation_loss": 2.1},
             {"epoch": 3, "loss": 1.2, "validation_loss": 1.5},
         ]},
        {"id": "ft-002", "model_id": "bedrock-titan-text", "status": "running",
         "training_metrics": [
             {"epoch": 1, "loss": 3.1, "validation_loss": 3.4},
         ]},
    ]


def generate_demo_trust_scores() -> dict:
    return {
        "overall": 0.78,
        "dimensions": {
            "accuracy": {"score": 0.82, "weight": 0.25},
            "consistency": {"score": 0.75, "weight": 0.20},
            "safety": {"score": 0.90, "weight": 0.20},
            "bias": {"score": 0.68, "weight": 0.15},
            "context_grounding": {"score": 0.72, "weight": 0.20},
        },
    }


def generate_demo_comparison() -> dict:
    return {
        "model_1": {"id": "bedrock-claude-3", "mean_trust_score": 0.72},
        "model_2": {"id": "ft-bedrock-claude-3", "mean_trust_score": 0.85},
        "improvement": {
            "trust_score_delta": 0.13,
            "hallucination_reduction": 0.08,
            "latency_delta_ms": 50,
            "cost_delta_per_query": 0.002,
            "p_value": 0.003,
        },
        "recommendation": "deploy",
        "justification": "Trust score improved by 18% with minimal cost increase and statistically significant hallucination reduction.",
    }
