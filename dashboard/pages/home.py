"""Dashboard home page.

Displays an overview of recent workflows, evaluations, models, system
health, and cost summary.  Falls back to demo data when backend services
are unavailable.

Requirements: 10.2
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_home_data() -> dict[str, Any]:
    """Load data for the home page from backends, falling back to demo data."""
    from dashboard.utils.backend import (
        get_model_registry,
        get_dataset_manager,
        get_evaluation_engine,
        get_workflow_orchestrator,
    )
    from dashboard.utils.demo_data import (
        generate_demo_models,
        generate_demo_datasets,
        generate_demo_evaluations,
        generate_demo_workflows,
    )

    data: dict[str, Any] = {"demo_mode": False}

    # Models
    registry = get_model_registry()
    if registry is not None:
        try:
            models = registry.list_models() if hasattr(registry, "list_models") else []
            data["models"] = [m if isinstance(m, dict) else m.__dict__ for m in models] if models else []
        except Exception:
            data["models"] = generate_demo_models()
            data["demo_mode"] = True
    else:
        data["models"] = generate_demo_models()
        data["demo_mode"] = True

    # Datasets
    dm = get_dataset_manager()
    if dm is not None:
        try:
            datasets = dm.list_datasets() if hasattr(dm, "list_datasets") else []
            data["datasets"] = [d if isinstance(d, dict) else d.__dict__ for d in datasets] if datasets else []
        except Exception:
            data["datasets"] = generate_demo_datasets()
            data["demo_mode"] = True
    else:
        data["datasets"] = generate_demo_datasets()
        data["demo_mode"] = True

    # Evaluations
    ee = get_evaluation_engine()
    if ee is not None:
        try:
            evals = ee.list_evaluations() if hasattr(ee, "list_evaluations") else []
            data["evaluations"] = [e if isinstance(e, dict) else e.__dict__ for e in evals] if evals else []
        except Exception:
            data["evaluations"] = generate_demo_evaluations()
            data["demo_mode"] = True
    else:
        data["evaluations"] = generate_demo_evaluations()
        data["demo_mode"] = True

    # Workflows
    wo = get_workflow_orchestrator()
    if wo is not None:
        try:
            wfs = wo.list_workflows() if hasattr(wo, "list_workflows") else []
            data["workflows"] = [w if isinstance(w, dict) else w.__dict__ for w in wfs] if wfs else []
        except Exception:
            data["workflows"] = generate_demo_workflows()
            data["demo_mode"] = True
    else:
        data["workflows"] = generate_demo_workflows()
        data["demo_mode"] = True

    return data


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _count_by_status(items: list[dict], key: str = "status") -> dict[str, int]:
    """Count items grouped by a status field."""
    counts: dict[str, int] = {}
    for item in items:
        status = item.get(key, "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _total_cost(evaluations: list[dict]) -> float:
    """Sum total_cost across evaluations."""
    return sum(e.get("total_cost", 0.0) for e in evaluations)


def _avg_trust_score(evaluations: list[dict]) -> float | None:
    """Average mean_trust_score across evaluations."""
    scores = [e["mean_trust_score"] for e in evaluations if "mean_trust_score" in e]
    return sum(scores) / len(scores) if scores else None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_kpi_row(st: Any, models: list, datasets: list, evaluations: list, workflows: list) -> None:
    """Render the top-level KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    active_models = sum(1 for m in models if m.get("status") == "active")
    with col1:
        st.metric("Registered Models", len(models), f"{active_models} active")

    with col2:
        total_rows = sum(d.get("row_count", 0) for d in datasets)
        st.metric("Datasets", len(datasets), f"{total_rows:,} rows")

    with col3:
        avg_ts = _avg_trust_score(evaluations)
        ts_label = f"{avg_ts:.2f}" if avg_ts is not None else "N/A"
        st.metric("Avg Trust Score", ts_label)

    running_wfs = sum(1 for w in workflows if w.get("status") == "running")
    with col4:
        st.metric("Workflows", len(workflows), f"{running_wfs} running")


def _render_system_health(st: Any, models: list, workflows: list) -> None:
    """Render the system health section."""
    st.subheader("🩺 System Health")

    model_statuses = _count_by_status(models)
    wf_statuses = _count_by_status(workflows)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Status**")
        for status, count in sorted(model_statuses.items()):
            from dashboard.utils.helpers import status_emoji
            st.write(f"{status_emoji(status)} {status.capitalize()}: **{count}**")

    with col2:
        st.markdown("**Workflow Status**")
        for status, count in sorted(wf_statuses.items()):
            from dashboard.utils.helpers import status_emoji
            st.write(f"{status_emoji(status)} {status.capitalize()}: **{count}**")


def _render_cost_summary(st: Any, evaluations: list[dict]) -> None:
    """Render the cost summary section."""
    st.subheader("💰 Cost Summary")

    total = _total_cost(evaluations)
    from dashboard.utils.helpers import format_cost

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Evaluation Cost", format_cost(total))
    with col2:
        avg_cost = total / len(evaluations) if evaluations else 0.0
        st.metric("Avg Cost / Evaluation", format_cost(avg_cost))
    with col3:
        st.metric("Evaluations Run", len(evaluations))


def _render_recent_evaluations(st: Any, evaluations: list[dict]) -> None:
    """Render the recent evaluations table."""
    st.subheader("📈 Recent Evaluations")

    if not evaluations:
        st.info("No evaluations yet. Run a baseline evaluation to get started.")
        return

    from dashboard.utils.helpers import format_cost

    # Show most recent first (up to 10)
    recent = sorted(evaluations, key=lambda e: e.get("created_at", ""), reverse=True)[:10]

    header_cols = st.columns([2, 2, 2, 1.5, 1.5])
    header_cols[0].markdown("**Evaluation ID**")
    header_cols[1].markdown("**Model**")
    header_cols[2].markdown("**Dataset**")
    header_cols[3].markdown("**Trust Score**")
    header_cols[4].markdown("**Cost**")

    for ev in recent:
        row = st.columns([2, 2, 2, 1.5, 1.5])
        row[0].write(ev.get("id", "—"))
        row[1].write(ev.get("model_id", "—"))
        row[2].write(ev.get("dataset_id", "—"))
        ts = ev.get("mean_trust_score")
        row[3].write(f"{ts:.2f}" if ts is not None else "—")
        row[4].write(format_cost(ev.get("total_cost", 0.0)))


def _render_recent_workflows(st: Any, workflows: list[dict]) -> None:
    """Render the recent workflows table."""
    st.subheader("🔄 Recent Workflows")

    if not workflows:
        st.info("No workflows yet. Start a workflow to see it here.")
        return

    from dashboard.utils.helpers import status_emoji as _status_emoji

    recent = sorted(
        workflows, key=lambda w: w.get("created_at", ""), reverse=True,
    )[:10]

    header_cols = st.columns([2, 3, 1.5, 1.5])
    header_cols[0].markdown("**Workflow ID**")
    header_cols[1].markdown("**Name**")
    header_cols[2].markdown("**Status**")
    header_cols[3].markdown("**Steps**")

    for wf in recent:
        row = st.columns([2, 3, 1.5, 1.5])
        row[0].write(wf.get("id", "—"))
        row[1].write(wf.get("name", "—"))
        status = wf.get("status", "unknown")
        row[2].write(f"{_status_emoji(status)} {status.capitalize()}")
        steps = wf.get("steps", [])
        completed = sum(1 for s in steps if s.get("status") == "completed")
        row[3].write(f"{completed}/{len(steps)}")


def _render_model_provider_chart(st: Any, models: list[dict]) -> None:
    """Render a pie chart of models by provider."""
    if not models:
        return

    provider_counts = _count_by_status(models, key="provider")
    labels = list(provider_counts.keys())
    values = list(provider_counts.values())

    fig = go.Figure(data=go.Pie(
        labels=[p.capitalize() for p in labels],
        values=values,
        hole=0.4,
        marker=dict(colors=["#1a73e8", "#2ecc71", "#f39c12", "#e74c3c"]),
    ))
    fig.update_layout(
        title=dict(text="Models by Provider"),
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_trust_score_overview(st: Any, evaluations: list[dict]) -> None:
    """Render a bar chart of trust scores across recent evaluations."""
    scored = [e for e in evaluations if "mean_trust_score" in e]
    if not scored:
        return

    ids = [e.get("id", "?") for e in scored]
    scores = [e["mean_trust_score"] for e in scored]

    colors = ["#2ecc71" if s >= 0.8 else "#f39c12" if s >= 0.6 else "#e74c3c" for s in scores]

    fig = go.Figure(data=go.Bar(x=ids, y=scores, marker_color=colors))
    fig.update_layout(
        title=dict(text="Trust Scores by Evaluation"),
        xaxis=dict(title="Evaluation"),
        yaxis=dict(title="Mean Trust Score", range=[0, 1]),
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_home_page() -> None:
    """Render the home / overview page."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("🏠 TrustOps Dashboard")

    data = _load_home_data()

    if data["demo_mode"]:
        st.info(
            "📋 **Demo Mode** — Showing sample data. Connect backend services "
            "to see live information."
        )

    models = data["models"]
    datasets = data["datasets"]
    evaluations = data["evaluations"]
    workflows = data["workflows"]

    # KPI row
    _render_kpi_row(st, models, datasets, evaluations, workflows)

    st.divider()

    # Charts row
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        _render_model_provider_chart(st, models)
    with chart_col2:
        _render_trust_score_overview(st, evaluations)

    st.divider()

    # System health + cost summary
    health_col, cost_col = st.columns(2)
    with health_col:
        _render_system_health(st, models, workflows)
    with cost_col:
        _render_cost_summary(st, evaluations)

    st.divider()

    # Recent evaluations and workflows
    _render_recent_evaluations(st, evaluations)

    st.divider()

    _render_recent_workflows(st, workflows)
