"""Backend service connectors for the dashboard.

Each getter returns the corresponding service instance or ``None``
when the service is unavailable (e.g. missing AWS credentials).
"""

from __future__ import annotations

from typing import Any, Optional


def get_model_registry() -> Optional[Any]:
    """Return a ModelRegistry instance or None."""
    try:
        from src.registry.model_registry import ModelRegistry
        return ModelRegistry()
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
