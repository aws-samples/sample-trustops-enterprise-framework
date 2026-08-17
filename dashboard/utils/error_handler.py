"""Degraded mode error handler.

Detects unavailable AWS services, disables affected features,
and shows status banners indicating which features are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    import streamlit as st
except ImportError:  # allow import in test environments without streamlit running
    import types
    st = types.SimpleNamespace(warning=lambda *a, **kw: None, info=lambda *a, **kw: None)


SERVICE_FEATURE_MAP: dict[str, list[str]] = {
    "model_registry": ["Models", "Evaluation", "Comparison", "Fine-Tuning"],
    "dataset_manager": ["Datasets", "Evaluation", "Fine-Tuning"],
    "evaluation_engine": ["Evaluation", "Comparison"],
    "fine_tuning_pipeline": ["Fine-Tuning"],
    "workflow_orchestrator": ["Workflows"],
    "results_store": ["Evaluation", "Comparison", "Workflows"],
}


@dataclass
class ServiceStatus:
    """Status of a backend service."""

    name: str
    available: bool
    error: Optional[str] = None
    affected_features: list[str] = field(default_factory=list)


def check_service_availability() -> dict[str, ServiceStatus]:
    """Probe each backend service and return availability statuses."""
    from dashboard.utils.backend import (
        get_model_registry,
        get_dataset_manager,
        get_evaluation_engine,
        get_fine_tuning_pipeline,
        get_workflow_orchestrator,
        get_results_store,
    )

    _getters = {
        "model_registry": get_model_registry,
        "dataset_manager": get_dataset_manager,
        "evaluation_engine": get_evaluation_engine,
        "fine_tuning_pipeline": get_fine_tuning_pipeline,
        "workflow_orchestrator": get_workflow_orchestrator,
        "results_store": get_results_store,
    }

    statuses: dict[str, ServiceStatus] = {}
    for name, getter in _getters.items():
        features = SERVICE_FEATURE_MAP[name]
        try:
            instance = getter()
            if instance is None:
                statuses[name] = ServiceStatus(
                    name=name, available=False,
                    error="Service returned None",
                    affected_features=features,
                )
            else:
                statuses[name] = ServiceStatus(
                    name=name, available=True,
                    affected_features=features,
                )
        except Exception as exc:
            statuses[name] = ServiceStatus(
                name=name, available=False,
                error=str(exc),
                affected_features=features,
            )
    return statuses


def get_unavailable_features(statuses: dict[str, ServiceStatus]) -> set[str]:
    """Return the set of features that are unavailable given service statuses."""
    unavailable: set[str] = set()
    for status in statuses.values():
        if not status.available:
            unavailable.update(status.affected_features)
    return unavailable


def is_feature_available(feature: str, statuses: dict[str, ServiceStatus]) -> bool:
    """Check whether a specific feature is available."""
    return feature not in get_unavailable_features(statuses)


def render_status_banner(statuses: dict[str, ServiceStatus]) -> None:
    """Render a warning banner when services are degraded."""
    down_services = [s for s in statuses.values() if not s.available]
    if not down_services:
        return

    unavailable = get_unavailable_features(statuses)
    service_names = ", ".join(s.name for s in down_services)
    feature_names = ", ".join(sorted(unavailable))
    st.warning(
        f"⚠️ **Degraded Mode** — The following services are unavailable: {service_names}. "
        f"Affected features: {feature_names}."
    )


def with_degraded_fallback(
    feature: str,
    statuses: dict[str, ServiceStatus],
    fallback_message: Optional[str] = None,
) -> bool:
    """Check feature availability and show fallback message if unavailable.

    Returns ``True`` if the feature is available, ``False`` otherwise.
    """
    if is_feature_available(feature, statuses):
        return True

    msg = fallback_message or f"The {feature} feature is currently unavailable due to a backend service outage."
    st.info(msg)
    return False
