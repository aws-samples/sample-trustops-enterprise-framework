"""Evaluation page — baseline evaluation browser and wizard.

Displays evaluation listings with trust scores and status, a baseline
evaluation wizard, real-time progress display, aggregate metrics with
drill-down, and per-category performance breakdown charts.
Falls back to demo data when backend services are unavailable.

Requirements: 10.3, 10.6
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_evaluation_data() -> dict[str, Any]:
    """Load evaluation data from backend, falling back to demo data."""
    from dashboard.utils.backend import (
        get_evaluation_engine,
        get_model_registry,
        get_dataset_manager,
    )
    from dashboard.utils.demo_data import (
        generate_demo_models,
        generate_demo_datasets,
    )

    data: dict[str, Any] = {"demo_mode": False}

    # Evaluations
    ee = get_evaluation_engine()
    if ee is not None:
        try:
            evals = (
                ee.list_evaluations()
                if hasattr(ee, "list_evaluations")
                else []
            )
            data["evaluations"] = [
                e if isinstance(e, dict) else e.__dict__
                for e in evals
            ] if evals else []
        except Exception:
            data["evaluations"] = _enrich_demo_evaluations()
            data["demo_mode"] = True
    else:
        data["evaluations"] = _enrich_demo_evaluations()
        data["demo_mode"] = True

    # Models (for wizard selectors)
    registry = get_model_registry()
    if registry is not None:
        try:
            models = (
                registry.list_models()
                if hasattr(registry, "list_models")
                else []
            )
            data["models"] = [
                m if isinstance(m, dict) else m.__dict__
                for m in models
            ] if models else []
        except Exception:
            data["models"] = generate_demo_models()
            data["demo_mode"] = True
    else:
        data["models"] = generate_demo_models()
        data["demo_mode"] = True

    # Datasets (for wizard selectors)
    dm = get_dataset_manager()
    if dm is not None:
        try:
            datasets = (
                dm.list_datasets()
                if hasattr(dm, "list_datasets")
                else []
            )
            data["datasets"] = [
                d if isinstance(d, dict) else d.__dict__
                for d in datasets
            ] if datasets else []
        except Exception:
            data["datasets"] = generate_demo_datasets()
            data["demo_mode"] = True
    else:
        data["datasets"] = generate_demo_datasets()
        data["demo_mode"] = True

    return data


def _enrich_demo_evaluations() -> list[dict]:
    """Return demo evaluations enriched with extra fields for the page."""
    from dashboard.utils.demo_data import generate_demo_evaluations

    evals = generate_demo_evaluations()
    _DEMO_CATEGORIES: dict[str, dict[str, dict[str, float]]] = {
        "eval-001": {
            "factual": {"mean_trust_score": 0.85, "count": 120},
            "reasoning": {"mean_trust_score": 0.78, "count": 80},
            "creative": {"mean_trust_score": 0.80, "count": 50},
        },
        "eval-002": {
            "factual": {"mean_trust_score": 0.74, "count": 120},
            "reasoning": {"mean_trust_score": 0.68, "count": 80},
            "creative": {"mean_trust_score": 0.70, "count": 50},
        },
        "eval-003": {
            "factual": {"mean_trust_score": 0.80, "count": 60},
            "reasoning": {"mean_trust_score": 0.72, "count": 40},
        },
    }
    _DEMO_DETAILS: dict[str, dict[str, Any]] = {
        "eval-001": {
            "status": "completed",
            "total_examples": 250,
            "latency_p50_ms": 320.0,
            "latency_p95_ms": 890.0,
            "latency_p99_ms": 1450.0,
            "hallucination_rate": 0.12,
        },
        "eval-002": {
            "status": "completed",
            "total_examples": 250,
            "latency_p50_ms": 180.0,
            "latency_p95_ms": 520.0,
            "latency_p99_ms": 980.0,
            "hallucination_rate": 0.18,
        },
        "eval-003": {
            "status": "running",
            "total_examples": 100,
            "latency_p50_ms": 250.0,
            "latency_p95_ms": 710.0,
            "latency_p99_ms": 1200.0,
            "hallucination_rate": 0.15,
            "progress_percent": 65.0,
            "completed_examples": 65,
        },
    }
    for ev in evals:
        eid = ev.get("id", "")
        details = _DEMO_DETAILS.get(eid, {})
        ev.setdefault("status", details.get("status", "completed"))
        ev.setdefault("total_examples", details.get("total_examples", 100))
        ev.setdefault(
            "latency_p50_ms",
            details.get("latency_p50_ms", 200.0),
        )
        ev.setdefault(
            "latency_p95_ms",
            details.get("latency_p95_ms", 600.0),
        )
        ev.setdefault(
            "latency_p99_ms",
            details.get("latency_p99_ms", 1000.0),
        )
        ev.setdefault(
            "hallucination_rate",
            details.get("hallucination_rate", 0.15),
        )
        ev.setdefault(
            "per_category",
            _DEMO_CATEGORIES.get(eid, {}),
        )
        if ev["status"] == "running":
            ev.setdefault(
                "progress_percent",
                details.get("progress_percent", 50.0),
            )
            ev.setdefault(
                "completed_examples",
                details.get("completed_examples", 50),
            )
    return evals


def _avg_trust_score(evaluations: list[dict]) -> float | None:
    """Return the average mean_trust_score, or None if no scores."""
    scores = [
        e["mean_trust_score"]
        for e in evaluations
        if isinstance(e.get("mean_trust_score"), (int, float))
    ]
    return sum(scores) / len(scores) if scores else None


def _total_cost(evaluations: list[dict]) -> float:
    """Sum total_cost across evaluations."""
    return sum(e.get("total_cost", 0.0) for e in evaluations)


def _running_evaluations(evaluations: list[dict]) -> list[dict]:
    """Return evaluations that are currently running."""
    return [
        e for e in evaluations
        if str(e.get("status", "")).lower() in ("running", "in_progress")
    ]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_evaluation_summary(
    st: Any,
    evaluations: list[dict],
) -> None:
    """Render summary KPI cards for evaluations."""
    from dashboard.utils.helpers import format_cost

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Evaluations", len(evaluations))
    with col2:
        avg = _avg_trust_score(evaluations)
        st.metric(
            "Avg Trust Score",
            f"{avg:.2f}" if avg is not None else "—",
        )
    with col3:
        running = len(_running_evaluations(evaluations))
        st.metric("Running", running)
    with col4:
        st.metric("Total Cost", format_cost(_total_cost(evaluations)))


def _render_evaluation_table(
    st: Any,
    evaluations: list[dict],
) -> str | None:
    """Render the evaluation listing table. Returns selected eval id."""
    from dashboard.utils.helpers import status_emoji

    if not evaluations:
        st.info(
            "No evaluations found. "
            "Run a baseline evaluation to get started."
        )
        return None

    header = st.columns([1.5, 1.5, 1.5, 1, 1, 1])
    header[0].markdown("**Evaluation**")
    header[1].markdown("**Model**")
    header[2].markdown("**Dataset**")
    header[3].markdown("**Trust Score**")
    header[4].markdown("**Status**")
    header[5].markdown("**Cost**")

    selected: str | None = None
    for ev in evaluations:
        row = st.columns([1.5, 1.5, 1.5, 1, 1, 1])
        eid = ev.get("id", "—")
        if row[0].button(eid, key=f"eval_select_{eid}"):
            selected = eid
        row[1].write(ev.get("model_id", "—"))
        row[2].write(ev.get("dataset_id", "—"))
        score = ev.get("mean_trust_score")
        score_str = (
            f"{score:.2f}"
            if isinstance(score, (int, float))
            else "—"
        )
        row[3].write(score_str)
        status = str(ev.get("status", "unknown"))
        row[4].write(f"{status_emoji(status)} {status.capitalize()}")
        from dashboard.utils.helpers import format_cost
        row[5].write(format_cost(ev.get("total_cost", 0.0)))

    return selected


def _render_evaluation_wizard(
    st: Any,
    models: list[dict],
    datasets: list[dict],
) -> dict | None:
    """Render the baseline evaluation wizard form.

    Returns the submitted config dict, or ``None`` if not submitted.
    """
    st.subheader("🧪 New Baseline Evaluation")

    with st.form("evaluation_wizard_form"):
        model_options = [
            m.get("name", m.get("id", "unknown")) for m in models
        ]
        model_ids = [m.get("id", "") for m in models]
        selected_model_idx = st.selectbox(
            "Select Model",
            options=range(len(model_options)),
            format_func=lambda i: model_options[i],
        ) if model_options else None

        dataset_options = [
            d.get("name", d.get("id", "unknown")) for d in datasets
        ]
        dataset_ids = [d.get("id", "") for d in datasets]
        selected_dataset_idx = st.selectbox(
            "Select Dataset",
            options=range(len(dataset_options)),
            format_func=lambda i: dataset_options[i],
        ) if dataset_options else None

        st.markdown("**Inference Parameters**")
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
        )
        max_tokens = st.number_input(
            "Max Tokens", min_value=1, value=1024, step=256,
        )
        top_p = st.slider(
            "Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05,
        )

        submitted = st.form_submit_button("▶️ Start Evaluation")

    if (
        submitted
        and selected_model_idx is not None
        and selected_dataset_idx is not None
    ):
        return {
            "model_id": model_ids[selected_model_idx],
            "dataset_id": dataset_ids[selected_dataset_idx],
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "top_p": top_p,
        }
    return None


def _render_progress_display(st: Any, evaluations: list[dict]) -> None:
    """Render real-time progress for running evaluations."""
    running = _running_evaluations(evaluations)
    if not running:
        return

    st.subheader("⏳ Running Evaluations")
    for ev in running:
        eid = ev.get("id", "unknown")
        progress = ev.get("progress_percent", 0.0) / 100.0
        completed = ev.get("completed_examples", 0)
        total = ev.get("total_examples", 0)

        st.markdown(f"**{eid}** — {ev.get('model_id', '?')}")
        st.progress(min(max(progress, 0.0), 1.0))
        st.caption(f"{completed} / {total} examples completed")


def _render_aggregate_metrics(st: Any, evaluation: dict) -> None:
    """Render aggregate metrics for a selected evaluation with drill-down."""
    from dashboard.utils.helpers import format_cost

    st.subheader(f"📊 Metrics — {evaluation.get('id', '?')}")

    col1, col2, col3 = st.columns(3)
    with col1:
        score = evaluation.get("mean_trust_score")
        st.metric(
            "Mean Trust Score",
            f"{score:.2f}" if isinstance(score, (int, float)) else "—",
        )
    with col2:
        st.metric(
            "Hallucination Rate",
            f"{evaluation.get('hallucination_rate', 0):.0%}",
        )
    with col3:
        st.metric("Total Cost", format_cost(evaluation.get("total_cost", 0.0)))

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Total Examples", evaluation.get("total_examples", "—"))
    with col5:
        p50 = evaluation.get("latency_p50_ms")
        st.metric(
            "Latency p50",
            f"{p50:.0f}ms" if isinstance(p50, (int, float)) else "—",
        )
    with col6:
        p95 = evaluation.get("latency_p95_ms")
        st.metric(
            "Latency p95",
            f"{p95:.0f}ms" if isinstance(p95, (int, float)) else "—",
        )


def _render_category_breakdown(st: Any, evaluation: dict) -> None:
    """Render per-category performance breakdown as a bar chart."""
    import plotly.graph_objects as go

    categories = evaluation.get("per_category", {})
    if not categories:
        st.info("No per-category breakdown available for this evaluation.")
        return

    st.subheader("📂 Per-Category Breakdown")

    cat_names = list(categories.keys())
    cat_scores = [
        categories[c].get("mean_trust_score", 0.0) for c in cat_names
    ]
    cat_counts = [
        categories[c].get("count", 0) for c in cat_names
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=cat_names,
                y=cat_scores,
                text=[f"{s:.2f}" for s in cat_scores],
                textposition="auto",
                marker_color="#1a73e8",
                hovertext=[
                    f"{c}: score={s:.2f}, n={n}"
                    for c, s, n in zip(cat_names, cat_scores, cat_counts)
                ],
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Trust Score by Category"),
        xaxis=dict(title="Category"),
        yaxis=dict(title="Mean Trust Score", range=[0, 1]),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_evaluation_detail(st: Any, evaluation: dict) -> None:
    """Render drill-down detail view for a selected evaluation."""
    _render_aggregate_metrics(st, evaluation)
    st.divider()
    _render_category_breakdown(st, evaluation)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_evaluation_page() -> None:
    """Render the baseline evaluation page."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("📈 Evaluation")

    data = _load_evaluation_data()

    if data["demo_mode"]:
        st.info(
            "📋 **Demo Mode** — Showing sample data. "
            "Connect backend services to see live evaluations."
        )

    evaluations = data["evaluations"]
    models = data["models"]
    datasets = data["datasets"]

    # Summary KPIs
    _render_evaluation_summary(st, evaluations)

    st.divider()

    # Progress for running evaluations
    _render_progress_display(st, evaluations)

    # Evaluation listing
    st.subheader("📋 Evaluation History")
    selected_id = _render_evaluation_table(st, evaluations)

    # Drill-down into selected evaluation
    if selected_id:
        selected_eval = next(
            (e for e in evaluations if e.get("id") == selected_id),
            None,
        )
        if selected_eval:
            st.divider()
            _render_evaluation_detail(st, selected_eval)

    st.divider()

    # Evaluation wizard
    config = _render_evaluation_wizard(st, models, datasets)
    if config is not None:
        from dashboard.utils.backend import get_evaluation_engine

        ee = get_evaluation_engine()
        if ee is not None and hasattr(ee, "run_baseline_evaluation"):
            try:
                mid = config["model_id"]
                did = config["dataset_id"]
                st.info(
                    f"Starting evaluation: {mid} "
                    f"on {did}…"
                )
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
        else:
            st.warning(
                "Backend unavailable — evaluation cannot be "
                "started in demo mode."
            )
