"""Comparison page — side-by-side model comparison browser.

Displays past comparisons, side-by-side metrics, improvement deltas,
statistical significance indicators, deployment recommendations, and
cost-performance analysis charts.

Demo-mode fallback: When AWS credentials are not configured or backend
services (DynamoDB/S3) are unreachable, the page automatically falls back
to demo data and displays a visible "Demo Mode" indicator. This ensures
the dashboard is always functional for demonstrations and development,
while seamlessly displaying live evaluation results when connected.

Requirements: 10.4
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_comparison_data() -> dict[str, Any]:
    """Load comparison data from backend, falling back to demo data."""
    from dashboard.utils.backend import get_evaluation_engine, get_workflow_manager
    from dashboard.utils.demo_data import generate_demo_models

    data: dict[str, Any] = {"demo_mode": False}

    # Try WorkflowManager first for live DynamoDB data
    wm = get_workflow_manager()
    if wm is not None:
        try:
            history = wm.get_workflow_history(
                filters={"workflow_type": "comparative"}
            )
            if history:
                data["comparisons"] = [
                    _workflow_to_comparison(w) for w in history
                ]
            else:
                data["comparisons"] = _build_demo_comparisons()
                data["demo_mode"] = True
        except Exception:
            data["comparisons"] = _build_demo_comparisons()
            data["demo_mode"] = True
    else:
        # Fall back to EvaluationEngine
        ee = get_evaluation_engine()
        if ee is not None:
            try:
                comparisons = (
                    ee.list_comparisons()
                    if hasattr(ee, "list_comparisons")
                    else []
                )
                data["comparisons"] = [
                    c if isinstance(c, dict) else c.__dict__
                    for c in comparisons
                ] if comparisons else []
            except Exception:
                data["comparisons"] = _build_demo_comparisons()
                data["demo_mode"] = True
        else:
            data["comparisons"] = _build_demo_comparisons()
            data["demo_mode"] = True

    # Models for reference
    from dashboard.utils.backend import get_model_registry
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

    return data


def _workflow_to_comparison(workflow: dict) -> dict:
    """Convert a WorkflowManager history entry to comparison display format."""
    config = workflow.get("configuration", {})
    return {
        "id": workflow.get("workflow_id", ""),
        "model_1_id": config.get("model_a_id", config.get("baseline_model_id", "")),
        "model_2_id": config.get("model_b_id", config.get("finetuned_model_id", "")),
        "dataset_id": config.get("dataset_id", ""),
        "status": workflow.get("status", "unknown"),
        "created_at": workflow.get("created_at", ""),
        "model_1_metrics": config.get("model_a_metrics", {}),
        "model_2_metrics": config.get("model_b_metrics", {}),
        "improvement": config.get("improvement", {}),
        "recommendation": config.get("recommendation", ""),
        "justification": config.get("justification", ""),
    }


def _build_demo_comparisons() -> list[dict]:
    """Build enriched demo comparison data."""
    from dashboard.utils.demo_data import generate_demo_comparison

    base = generate_demo_comparison()
    improvement = base.get("improvement", {})

    comparisons = [
        {
            "id": "cmp-001",
            "model_1_id": base["model_1"]["id"],
            "model_2_id": base["model_2"]["id"],
            "dataset_id": "ds-qa-001",
            "status": "completed",
            "created_at": "2024-01-15T10:30:00Z",
            "model_1_metrics": {
                "mean_trust_score": base["model_1"]["mean_trust_score"],
                "hallucination_rate": 0.18,
                "latency_p50_ms": 320.0,
                "latency_p95_ms": 890.0,
                "total_cost": 12.50,
                "cost_per_query": 0.050,
                "total_examples": 250,
            },
            "model_2_metrics": {
                "mean_trust_score": base["model_2"]["mean_trust_score"],
                "hallucination_rate": 0.10,
                "latency_p50_ms": 370.0,
                "latency_p95_ms": 940.0,
                "total_cost": 13.00,
                "cost_per_query": 0.052,
                "total_examples": 250,
            },
            "improvement": {
                "trust_score_delta": improvement.get("trust_score_delta", 0.13),
                "trust_score_delta_percent": 18.1,
                "hallucination_reduction": improvement.get("hallucination_reduction", 0.08),
                "hallucination_reduction_percent": 44.4,
                "latency_delta_ms": improvement.get("latency_delta_ms", 50),
                "latency_delta_percent": 15.6,
                "cost_delta_per_query": improvement.get("cost_delta_per_query", 0.002),
                "cost_delta_percent": 4.0,
                "p_value": improvement.get("p_value", 0.003),
                "statistical_significance": 0.997,
                "confidence_interval": (-0.17, -0.09),
            },
            "recommendation": base.get("recommendation", "deploy"),
            "justification": base.get("justification", ""),
        },
        {
            "id": "cmp-002",
            "model_1_id": "bedrock-titan-text",
            "model_2_id": "ft-bedrock-titan-text",
            "dataset_id": "ds-qa-001",
            "status": "completed",
            "created_at": "2024-01-14T14:00:00Z",
            "model_1_metrics": {
                "mean_trust_score": 0.71,
                "hallucination_rate": 0.22,
                "latency_p50_ms": 180.0,
                "latency_p95_ms": 520.0,
                "total_cost": 4.30,
                "cost_per_query": 0.017,
                "total_examples": 250,
            },
            "model_2_metrics": {
                "mean_trust_score": 0.74,
                "hallucination_rate": 0.19,
                "latency_p50_ms": 200.0,
                "latency_p95_ms": 560.0,
                "total_cost": 5.10,
                "cost_per_query": 0.020,
                "total_examples": 250,
            },
            "improvement": {
                "trust_score_delta": 0.03,
                "trust_score_delta_percent": 4.2,
                "hallucination_reduction": 0.03,
                "hallucination_reduction_percent": 13.6,
                "latency_delta_ms": 20,
                "latency_delta_percent": 11.1,
                "cost_delta_per_query": 0.003,
                "cost_delta_percent": 18.6,
                "p_value": 0.12,
                "statistical_significance": 0.88,
                "confidence_interval": (-0.07, 0.01),
            },
            "recommendation": "iterate",
            "justification": "Trust score improved marginally but results are not statistically significant. Consider additional fine-tuning epochs.",
        },
    ]
    return comparisons


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_RECOMMENDATION_DISPLAY = {
    "deploy": ("✅", "DEPLOY", "green"),
    "iterate": ("🔄", "ITERATE", "orange"),
    "reject": ("❌", "REJECT", "red"),
}


def _render_comparison_summary(st: Any, comparisons: list[dict]) -> None:
    """Render summary KPI cards for comparisons."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Comparisons", len(comparisons))
    with col2:
        deploy_count = sum(
            1 for c in comparisons
            if str(c.get("recommendation", "")).lower() == "deploy"
        )
        st.metric("Deploy Recommended", deploy_count)
    with col3:
        deltas = [
            c.get("improvement", {}).get("trust_score_delta", 0.0)
            for c in comparisons
            if isinstance(c.get("improvement", {}).get("trust_score_delta"), (int, float))
        ]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
        st.metric("Avg Trust Δ", f"{avg_delta:+.2f}")


def _render_comparison_table(st: Any, comparisons: list[dict]) -> str | None:
    """Render the comparison listing table. Returns selected comparison id."""
    from dashboard.utils.helpers import status_emoji

    if not comparisons:
        st.info("No comparisons found. Run a comparative evaluation to get started.")
        return None

    header = st.columns([1.2, 1.5, 1.5, 1, 1, 1])
    header[0].markdown("**Comparison**")
    header[1].markdown("**Model A**")
    header[2].markdown("**Model B**")
    header[3].markdown("**Trust Δ**")
    header[4].markdown("**Recommendation**")
    header[5].markdown("**Status**")

    selected: str | None = None
    for cmp in comparisons:
        row = st.columns([1.2, 1.5, 1.5, 1, 1, 1])
        cid = cmp.get("id", "—")
        if row[0].button(cid, key=f"cmp_select_{cid}"):
            selected = cid
        row[1].write(cmp.get("model_1_id", "—"))
        row[2].write(cmp.get("model_2_id", "—"))

        delta = cmp.get("improvement", {}).get("trust_score_delta")
        delta_str = f"{delta:+.2f}" if isinstance(delta, (int, float)) else "—"
        row[3].write(delta_str)

        rec = str(cmp.get("recommendation", "unknown")).lower()
        emoji, label, _ = _RECOMMENDATION_DISPLAY.get(rec, ("⚪", rec.upper(), "gray"))
        row[4].write(f"{emoji} {label}")

        status = str(cmp.get("status", "unknown"))
        row[5].write(f"{status_emoji(status)} {status.capitalize()}")

    return selected


def _render_side_by_side_metrics(st: Any, comparison: dict) -> None:
    """Render side-by-side model metrics display."""
    from dashboard.utils.helpers import format_cost

    m1 = comparison.get("model_1_metrics", {})
    m2 = comparison.get("model_2_metrics", {})
    m1_id = comparison.get("model_1_id", "Model A")
    m2_id = comparison.get("model_2_id", "Model B")

    st.subheader("📊 Side-by-Side Metrics")

    col_label, col_a, col_b = st.columns([1.5, 1.5, 1.5])
    col_label.markdown("**Metric**")
    col_a.markdown(f"**{m1_id}**")
    col_b.markdown(f"**{m2_id}**")

    _metrics_rows = [
        ("Trust Score", "mean_trust_score", lambda v: f"{v:.2f}"),
        ("Hallucination Rate", "hallucination_rate", lambda v: f"{v:.0%}"),
        ("Latency p50", "latency_p50_ms", lambda v: f"{v:.0f}ms"),
        ("Latency p95", "latency_p95_ms", lambda v: f"{v:.0f}ms"),
        ("Cost / Query", "cost_per_query", format_cost),
        ("Total Cost", "total_cost", format_cost),
    ]

    for label, key, fmt in _metrics_rows:
        r_label, r_a, r_b = st.columns([1.5, 1.5, 1.5])
        r_label.write(label)
        v1 = m1.get(key)
        v2 = m2.get(key)
        r_a.write(fmt(v1) if isinstance(v1, (int, float)) else "—")
        r_b.write(fmt(v2) if isinstance(v2, (int, float)) else "—")


def _render_improvement_metrics(st: Any, comparison: dict) -> None:
    """Render improvement metrics with deltas and percentages."""
    improvement = comparison.get("improvement", {})
    if not improvement:
        st.info("No improvement metrics available.")
        return

    st.subheader("📈 Improvement Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = improvement.get("trust_score_delta", 0.0)
        pct = improvement.get("trust_score_delta_percent", 0.0)
        st.metric(
            "Trust Score Δ",
            f"{delta:+.3f}",
            delta=f"{pct:+.1f}%",
        )
    with col2:
        red = improvement.get("hallucination_reduction", 0.0)
        red_pct = improvement.get("hallucination_reduction_percent", 0.0)
        st.metric(
            "Hallucination Reduction",
            f"{red:+.3f}",
            delta=f"{red_pct:+.1f}%",
        )
    with col3:
        lat = improvement.get("latency_delta_ms", 0.0)
        lat_pct = improvement.get("latency_delta_percent", 0.0)
        st.metric(
            "Latency Δ",
            f"{lat:+.0f}ms",
            delta=f"{lat_pct:+.1f}%",
            delta_color="inverse",
        )
    with col4:
        cost = improvement.get("cost_delta_per_query", 0.0)
        cost_pct = improvement.get("cost_delta_percent", 0.0)
        st.metric(
            "Cost Δ / Query",
            f"${cost:+.4f}",
            delta=f"{cost_pct:+.1f}%",
            delta_color="inverse",
        )


def _render_statistical_significance(st: Any, comparison: dict) -> None:
    """Render statistical significance indicators."""
    improvement = comparison.get("improvement", {})

    # Support both legacy (p_value) and new (p_value_ttest/p_value_wilcoxon) fields
    p_value_ttest = improvement.get("p_value_ttest")
    p_value_wilcoxon = improvement.get("p_value_wilcoxon")
    test_type = improvement.get("test_type_used", "")
    p_value = improvement.get("p_value")
    ci = improvement.get("confidence_interval")
    ci_lower = improvement.get("confidence_interval_lower")
    ci_upper = improvement.get("confidence_interval_upper")

    # Use the primary p-value for significance determination
    primary_p = None
    if test_type == "ttest" and isinstance(p_value_ttest, (int, float)):
        primary_p = p_value_ttest
    elif test_type == "wilcoxon" and isinstance(p_value_wilcoxon, (int, float)):
        primary_p = p_value_wilcoxon
    elif isinstance(p_value, (int, float)):
        primary_p = p_value

    st.subheader("🔬 Statistical Significance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if primary_p is not None:
            sig_label = "Significant" if primary_p < 0.05 else "Not Significant"
            st.metric("p-value (primary)", f"{primary_p:.4f}")
            st.caption(f"{'✅' if primary_p < 0.05 else '⚠️'} {sig_label}")
        else:
            st.metric("p-value", "—")

    with col2:
        if test_type:
            display_type = "Paired t-test" if test_type == "ttest" else "Wilcoxon" if test_type == "wilcoxon" else test_type
            st.metric("Test Used", display_type)
        else:
            significance = improvement.get("statistical_significance")
            if isinstance(significance, (int, float)):
                st.metric("Confidence Level", f"{significance:.1%}")
            else:
                st.metric("Test Used", "—")

    with col3:
        if isinstance(ci_lower, (int, float)) and isinstance(ci_upper, (int, float)):
            st.metric("95% CI", f"[{ci_lower:+.3f}, {ci_upper:+.3f}]")
        elif isinstance(ci, (list, tuple)) and len(ci) == 2:
            st.metric("95% CI", f"[{ci[0]:+.3f}, {ci[1]:+.3f}]")
        else:
            st.metric("95% CI", "—")

    with col4:
        if isinstance(p_value_ttest, (int, float)) and isinstance(p_value_wilcoxon, (int, float)):
            other_p = p_value_wilcoxon if test_type == "ttest" else p_value_ttest
            other_label = "Wilcoxon" if test_type == "ttest" else "t-test"
            st.metric(f"p-value ({other_label})", f"{other_p:.4f}")
        else:
            st.metric("Secondary test", "—")


def _render_deployment_recommendation(st: Any, comparison: dict) -> None:
    """Render deployment recommendation with justification."""
    rec = str(comparison.get("recommendation", "unknown")).lower()
    justification = comparison.get("justification", "No justification provided.")

    emoji, label, color = _RECOMMENDATION_DISPLAY.get(
        rec, ("⚪", rec.upper(), "gray"),
    )

    st.subheader("🚀 Deployment Recommendation")
    st.markdown(f"### {emoji} {label}")
    st.info(justification)


def _render_cost_performance_chart(st: Any, comparison: dict) -> None:
    """Render cost-performance analysis chart."""
    import plotly.graph_objects as go

    m1 = comparison.get("model_1_metrics", {})
    m2 = comparison.get("model_2_metrics", {})
    m1_id = comparison.get("model_1_id", "Model A")
    m2_id = comparison.get("model_2_id", "Model B")

    m1_trust = m1.get("mean_trust_score")
    m2_trust = m2.get("mean_trust_score")
    m1_cost = m1.get("cost_per_query")
    m2_cost = m2.get("cost_per_query")

    if not all(
        isinstance(v, (int, float))
        for v in [m1_trust, m2_trust, m1_cost, m2_cost]
    ):
        st.info("Insufficient data for cost-performance chart.")
        return

    st.subheader("💰 Cost-Performance Analysis")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=m1_id,
        x=["Trust Score", "Cost/Query ($)", "Hallucination Rate"],
        y=[
            m1_trust,
            m1_cost,
            m1.get("hallucination_rate", 0),
        ],
        marker_color="#1a73e8",
        text=[
            f"{m1_trust:.2f}",
            f"${m1_cost:.4f}",
            f"{m1.get('hallucination_rate', 0):.0%}",
        ],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        name=m2_id,
        x=["Trust Score", "Cost/Query ($)", "Hallucination Rate"],
        y=[
            m2_trust,
            m2_cost,
            m2.get("hallucination_rate", 0),
        ],
        marker_color="#34a853",
        text=[
            f"{m2_trust:.2f}",
            f"${m2_cost:.4f}",
            f"{m2.get('hallucination_rate', 0):.0%}",
        ],
        textposition="auto",
    ))

    fig.update_layout(
        title=dict(text="Cost-Performance Comparison"),
        barmode="group",
        yaxis=dict(title="Value"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cost per trust point summary
    m1_cpt = m1_cost / m1_trust if m1_trust > 0 else 0
    m2_cpt = m2_cost / m2_trust if m2_trust > 0 else 0
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{m1_id} — Cost/Trust Point", f"${m1_cpt:.4f}")
    with col2:
        st.metric(f"{m2_id} — Cost/Trust Point", f"${m2_cpt:.4f}")


def _render_comparison_detail(st: Any, comparison: dict) -> None:
    """Render full detail view for a selected comparison."""
    _render_side_by_side_metrics(st, comparison)
    st.divider()
    _render_improvement_metrics(st, comparison)
    st.divider()
    _render_statistical_significance(st, comparison)
    st.divider()
    _render_deployment_recommendation(st, comparison)
    st.divider()
    _render_cost_performance_chart(st, comparison)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_comparison_page() -> None:
    """Render the model comparison page."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("⚖️ Comparison")

    data = _load_comparison_data()

    # Connection status indicator
    if data["demo_mode"]:
        st.sidebar.warning("🟡 Demo Mode")
        st.info(
            "📋 **Demo Mode** — Showing sample data. "
            "Connect backend services to see live comparisons."
        )
    else:
        st.sidebar.success("🟢 Live")

    comparisons = data["comparisons"]

    # Summary KPIs
    _render_comparison_summary(st, comparisons)

    st.divider()

    # Comparison listing
    st.subheader("📋 Comparison History")
    selected_id = _render_comparison_table(st, comparisons)

    # Drill-down into selected comparison
    if selected_id:
        selected_cmp = next(
            (c for c in comparisons if c.get("id") == selected_id),
            None,
        )
        if selected_cmp:
            st.divider()
            _render_comparison_detail(st, selected_cmp)
