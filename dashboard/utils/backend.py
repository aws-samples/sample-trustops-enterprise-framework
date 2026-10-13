"""Backend service connectors for the dashboard.

Each getter returns the corresponding service instance or ``None``
when the service is unavailable (e.g. missing AWS credentials).

The ``list_*`` helpers query AWS directly (S3 for results, DynamoDB for
workflows) so the dashboard can display real evaluation data instead of
falling back to hardcoded demo data.
"""

from __future__ import annotations

import json
from typing import Any, Optional


def get_model_registry() -> Optional[Any]:
    """Return a ModelRegistry instance with Bedrock adapter and discovered models, or None."""
    try:
        import asyncio
        from src.registry.model_registry import ModelRegistry
        from src.data_models.model import ModelProvider

        registry = ModelRegistry()
        try:
            from src.adapters.bedrock_adapter import BedrockAdapter
            adapter = BedrockAdapter(model_id="amazon.nova-lite-v1:0")
            registry.register_adapter(ModelProvider.BEDROCK, adapter)
            # Discover models so list_models() returns real data
            asyncio.run(registry.discover_models())
        except Exception:
            pass
        return registry
    except Exception:
        return None


def get_dataset_manager() -> Optional[Any]:
    """Return a DatasetManager instance or None."""
    try:
        from src.datasets.dataset_manager import DatasetManager
        return DatasetManager()
    except Exception:
        return None


def get_evaluation_engine() -> Optional[Any]:
    """Return an EvaluationEngine instance or None."""
    try:
        from src.evaluation.evaluation_engine import EvaluationEngine
        return EvaluationEngine()
    except Exception:
        return None


def get_fine_tuning_pipeline() -> Optional[Any]:
    """Return a FineTuningPipeline instance or None."""
    try:
        from src.fine_tuning.fine_tuning_pipeline import FineTuningPipeline
        return FineTuningPipeline()
    except Exception:
        return None


def get_workflow_orchestrator() -> Optional[Any]:
    """Return a WorkflowOrchestrator instance or None."""
    try:
        from src.orchestration.workflow_orchestrator import WorkflowOrchestrator
        return WorkflowOrchestrator()
    except Exception:
        return None


def get_results_store() -> Optional[Any]:
    """Return a ResultsStore instance or None."""
    try:
        from src.storage.results_store import ResultsStore
        return ResultsStore()
    except Exception:
        return None


def get_workflow_manager() -> Optional[Any]:
    """Return a WorkflowManager instance or None."""
    try:
        from src.orchestration.workflow_manager import WorkflowManager
        return WorkflowManager()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Direct AWS query helpers for the dashboard pages
# ---------------------------------------------------------------------------

def list_s3_results() -> list[dict]:
    """List evaluation results stored in the results S3 bucket.

    Scans the results bucket for results.json files and returns a list
    of summary dicts the dashboard pages can display.
    """
    try:
        import boto3
        from config.aws_config import config

        s3 = boto3.client("s3", **config.get_boto3_session_kwargs())
        bucket = config.results_bucket

        paginator = s3.get_paginator("list_objects_v2")
        results = []

        for page in paginator.paginate(Bucket=bucket, MaxKeys=100):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("results.json"):
                    try:
                        resp = s3.get_object(Bucket=bucket, Key=key)
                        body = json.loads(resp["Body"].read().decode("utf-8"))
                        body["_s3_key"] = key
                        results.append(body)
                    except Exception:
                        continue

        return results
    except Exception:
        return []


def list_dynamodb_workflows() -> list[dict]:
    """List workflows from the DynamoDB workflows table."""
    try:
        import boto3
        from config.aws_config import config

        dynamodb = boto3.resource("dynamodb", **config.get_boto3_session_kwargs())
        table = dynamodb.Table(config.workflows_table)

        response = table.scan(Limit=50)
        return response.get("Items", [])
    except Exception:
        return []
