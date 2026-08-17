"""Workflows page — listing, template selector, progress, and audit trail.

Displays workflow listings with status and step progress, a template
selector for starting new workflows, execution progress with step-by-step
visualization, workflow history browser with filtering, and an audit trail
viewer. Falls back to demo data when backend services are unavailable.

Requirements: 10.13
"""

from __future__ import annotations

from typing import Any


WORKFLOW_TEMPLATES: dict[str, dict] = {
    "Full Pipeline": {
        "id": "full_pipeline",
        "description": (
            "Complete pipeline: dataset upload, baseline "
            "evaluation, fine-tuning, post-tuning "
            "evaluation, and comparison."
        ),
        "steps": [
            "Dataset Upload",
            "Baseline Evaluation",
            "Fine-Tuning",
            "Post-Tuning Evaluation",
            "Comparison",
        ],
    },
    "Evaluation Only": {
        "id": "evaluation_only",
        "description": (
            "Run baseline evaluation on a dataset "
            "without fine-tuning."
        ),
        "steps": [
            "Dataset Upload",
            "Baseline Evaluation",
        ],
    },
    "Comparison Only": {
        "id": "comparison_only",
        "description": (
            "Compare two models without fine-tuning."
        ),
        "steps": ["Comparison"],
    },
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_workflows_data() -> dict[str, Any]:
    """Load workflow data from backend, falling back to demo."""
    from dashboard.utils.backend import (
        get_workflow_orchestrator,
    )

    data: dict[str, Any] = {"demo_mode": False}

    orchestrator = get_workflow_orchestrator()
    if orchestrator is not None:
        try:
            workflows = (
                orchestrator.list_workflows()
                if hasattr(orchestrator, "list_workflows")
                else []
            )
            data["workflows"] = [
                w if isinstance(w, dict) else w.__dict__
                for w in workflows
            ] if workflows else []
        except Exception:
            data["workflows"] = _build_demo_workflows()
            data["demo_mode"] = True
    else:
        data["workflows"] = _build_demo_workflows()
        data["demo_mode"] = True

    return data


def _build_demo_workflows() -> list[dict]:
    """Build enriched demo workflow data with audit events."""
    from dashboard.utils.demo_data import (
        generate_demo_workflows,
    )

    workflows = generate_demo_workflows()
    _EXTRA: dict[str, dict[str, Any]] = {
        "wf-001": _demo_wf001_extra(),
        "wf-002": _demo_wf002_extra(),
        "wf-003": _demo_wf003_extra(),
    }
    for wf in workflows:
        wid = wf.get("id", "")
        extra = _EXTRA.get(wid, {})
        for key, val in extra.items():
            wf.setdefault(key, val)
    return workflows


def _demo_wf001_extra() -> dict[str, Any]:
    """Extra fields for demo workflow wf-001."""
    return {
        "template": "Full Pipeline",
        "duration_seconds": 18720.0,
        "total_cost": 57.50,
        "model_id": "bedrock-claude-3",
        "dataset_id": "ds-qa-001",
        "completed_at": "2024-01-12T14:30:00Z",
        "audit_trail": [
            {
                "timestamp": "2024-01-12T08:00:00Z",
                "step": "Dataset Upload",
                "event": "started",
                "detail": "Uploading ds-qa-001",
            },
            {
                "timestamp": "2024-01-12T08:02:30Z",
                "step": "Dataset Upload",
                "event": "completed",
                "detail": "5000 rows validated",
            },
            {
                "timestamp": "2024-01-12T08:03:00Z",
                "step": "Baseline Evaluation",
                "event": "started",
                "detail": "Evaluating bedrock-claude-3",
            },
            {
                "timestamp": "2024-01-12T09:15:00Z",
                "step": "Baseline Evaluation",
                "event": "completed",
                "detail": "Trust score: 0.72",
            },
            {
                "timestamp": "2024-01-12T09:16:00Z",
                "step": "Fine-Tuning",
                "event": "started",
                "detail": "Job ft-001 submitted",
            },
            {
                "timestamp": "2024-01-12T13:00:00Z",
                "step": "Fine-Tuning",
                "event": "completed",
                "detail": "Final loss: 1.2",
            },
            {
                "timestamp": "2024-01-12T13:01:00Z",
                "step": "Post-Tuning Evaluation",
                "event": "started",
                "detail": "Evaluating ft-claude3-qa-v1",
            },
            {
                "timestamp": "2024-01-12T14:00:00Z",
                "step": "Post-Tuning Evaluation",
                "event": "completed",
                "detail": "Trust score: 0.85",
            },
            {
                "timestamp": "2024-01-12T14:01:00Z",
                "step": "Comparison",
                "event": "started",
                "detail": (
                    "Comparing baseline vs fine-tuned"
                ),
            },
            {
                "timestamp": "2024-01-12T14:30:00Z",
                "step": "Comparison",
                "event": "completed",
                "detail": "Recommendation: DEPLOY",
            },
        ],
    }


def _demo_wf002_extra() -> dict[str, Any]:
    """Extra fields for demo workflow wf-002."""
    return {
        "template": "Evaluation Only",
        "duration_seconds": None,
        "total_cost": 4.30,
        "model_id": "bedrock-titan-text",
        "dataset_id": "ds-qa-001",
        "completed_at": None,
        "progress_percent": 60.0,
        "audit_trail": [
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "step": "Dataset Upload",
                "event": "started",
                "detail": "Uploading ds-qa-001",
            },
            {
                "timestamp": "2024-01-15T10:01:00Z",
                "step": "Dataset Upload",
                "event": "completed",
                "detail": "5000 rows validated",
            },
            {
                "timestamp": "2024-01-15T10:02:00Z",
                "step": "Baseline Evaluation",
                "event": "started",
                "detail": (
                    "Evaluating bedrock-titan-text"
                ),
            },
        ],
    }


def _demo_wf003_extra() -> dict[str, Any]:
    """Extra fields for demo workflow wf-003."""
    return {
        "template": "Comparison Only",
        "duration_seconds": 120.0,
        "total_cost": 0.0,
        "model_id": "bedrock-claude-3",
        "dataset_id": "ds-sum-001",
        "completed_at": None,
        "error_message": (
            "Model bedrock-titan-text unavailable"
        ),
        "audit_trail": [
            {
                "timestamp": "2024-01-14T09:00:00Z",
                "step": "Comparison",
                "event": "started",
                "detail": (
                    "Comparing bedrock-claude-3 "
                    "vs bedrock-titan-text"
                ),
            },
            {
                "timestamp": "2024-01-14T09:02:00Z",
                "step": "Comparison",
                "event": "failed",
                "detail": (
                    "Model bedrock-titan-text "
                    "unavailable"
                ),
            },
        ],
    }


def _step_progress(steps: list[dict]) -> tuple[int, int]:
    """Return (completed_count, total_count) for steps."""
    total = len(steps)
    completed = sum(
        1 for s in steps
        if str(s.get("status", "")).lower() == "completed"
    )
    return completed, total


def _running_workflows(
    workflows: list[dict],
) -> list[dict]:
    """Return workflows that are currently running."""
    return [
        w for w in workflows
        if str(w.get("status", "")).lower()
        in ("running", "in_progress")
    ]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_workflow_summary(
    st: Any,
    workflows: list[dict],
) -> None:
    """Render summary KPI cards for workflows."""
    from dashboard.utils.helpers import format_cost

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Workflows", len(workflows))
    with col2:
        completed = sum(
            1 for w in workflows
            if str(w.get("status", "")).lower()
            == "completed"
        )
        st.metric("Completed", completed)
    with col3:
        running = len(_running_workflows(workflows))
        st.metric("Running", running)
    with col4:
        total_cost = sum(
            w.get("total_cost", 0.0) or 0.0
            for w in workflows
        )
        st.metric("Total Cost", format_cost(total_cost))


def _render_workflow_table(
    st: Any,
    workflows: list[dict],
    status_filter: str,
) -> str | None:
    """Render workflow history table. Returns selected id."""
    from dashboard.utils.helpers import status_emoji

    filtered = workflows
    if status_filter != "All":
        filtered = [
            w for w in workflows
            if str(w.get("status", "")).lower()
            == status_filter.lower()
        ]

    if not filtered:
        st.info("No workflows match the current filter.")
        return None

    header = st.columns([1, 2, 1.2, 1.2, 1])
    header[0].markdown("**ID**")
    header[1].markdown("**Name**")
    header[2].markdown("**Template**")
    header[3].markdown("**Status**")
    header[4].markdown("**Steps**")

    selected: str | None = None
    for wf in filtered:
        row = st.columns([1, 2, 1.2, 1.2, 1])
        wid = wf.get("id", "—")
        if row[0].button(wid, key=f"wf_sel_{wid}"):
            selected = wid
        row[1].write(wf.get("name", "—"))
        row[2].write(wf.get("template", "—"))
        status = str(wf.get("status", "unknown"))
        row[3].write(
            f"{status_emoji(status)} "
            f"{status.capitalize()}"
        )
        steps = wf.get("steps", [])
        done, total = _step_progress(steps)
        row[4].write(f"{done}/{total}")

    return selected


def _render_step_visualization(
    st: Any,
    steps: list[dict],
) -> None:
    """Render step-by-step status visualization."""
    from dashboard.utils.helpers import status_emoji

    if not steps:
        st.info("No steps defined for this workflow.")
        return

    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        name = step.get("name", f"Step {i + 1}")
        status = str(step.get("status", "pending"))
        emoji = status_emoji(status)
        with cols[i]:
            st.markdown(
                f"**{emoji} {name}**\n\n"
                f"{status.capitalize()}"
            )


def _render_execution_progress(
    st: Any,
    workflows: list[dict],
) -> None:
    """Render execution progress for running workflows."""
    running = _running_workflows(workflows)
    if not running:
        return

    st.subheader("⏳ Running Workflows")
    for wf in running:
        wid = wf.get("id", "unknown")
        name = wf.get("name", "Unnamed")
        steps = wf.get("steps", [])
        done, total = _step_progress(steps)
        progress = wf.get("progress_percent")
        if progress is None and total > 0:
            progress = (done / total) * 100.0

        st.markdown(f"**{wid}** — {name}")
        pct = min(max((progress or 0.0) / 100.0, 0.0), 1.0)
        st.progress(pct)
        st.caption(
            f"Steps completed: {done} / {total}"
        )
        _render_step_visualization(st, steps)


def _render_template_selector(st: Any) -> dict | None:
    """Render workflow template selector and configurator.

    Returns the selected template config, or ``None``.
    """
    st.subheader("🚀 Start New Workflow")

    names = list(WORKFLOW_TEMPLATES.keys())

    with st.form("workflow_template_form"):
        sel_idx = st.selectbox(
            "Workflow Template",
            options=range(len(names)),
            format_func=lambda i: names[i],
        )

        tmpl = WORKFLOW_TEMPLATES[names[sel_idx]]
        st.markdown(f"_{tmpl['description']}_")
        st.markdown(
            "**Steps:** "
            + " → ".join(tmpl["steps"])
        )

        workflow_name = st.text_input(
            "Workflow Name",
            value=f"New {names[sel_idx]} Run",
        )

        submitted = st.form_submit_button(
            "▶️ Start Workflow"
        )

    if submitted:
        return {
            "template_id": tmpl["id"],
            "template_name": names[sel_idx],
            "workflow_name": workflow_name,
            "steps": tmpl["steps"],
        }
    return None


def _render_audit_trail(
    st: Any,
    audit_trail: list[dict],
) -> None:
    """Render audit trail with step events and timestamps."""
    from dashboard.utils.helpers import status_emoji

    if not audit_trail:
        st.info("No audit events recorded.")
        return

    st.markdown("**Audit Trail**")

    header = st.columns([1.5, 1.5, 1, 2.5])
    header[0].markdown("**Timestamp**")
    header[1].markdown("**Step**")
    header[2].markdown("**Event**")
    header[3].markdown("**Detail**")

    for event in audit_trail:
        row = st.columns([1.5, 1.5, 1, 2.5])
        ts = event.get("timestamp", "—")
        if isinstance(ts, str) and "T" in ts:
            ts = ts.replace("T", " ").split(".")[0]
        row[0].write(ts)
        row[1].write(event.get("step", "—"))
        evt = str(event.get("event", "unknown"))
        row[2].write(
            f"{status_emoji(evt)} {evt.capitalize()}"
        )
        row[3].write(event.get("detail", "—"))


def _render_workflow_detail(
    st: Any,
    workflow: dict,
) -> None:
    """Render detail view for a selected workflow."""
    from dashboard.utils.helpers import (
        format_cost,
        format_duration,
        status_emoji,
    )

    wid = workflow.get("id", "?")
    status = str(workflow.get("status", "unknown"))

    st.subheader(
        f"📋 Workflow Detail — {wid} "
        f"{status_emoji(status)} {status.capitalize()}"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Template",
            workflow.get("template", "—"),
        )
    with col2:
        st.metric(
            "Model",
            workflow.get("model_id", "—"),
        )
    with col3:
        cost = workflow.get("total_cost")
        st.metric(
            "Cost",
            format_cost(cost)
            if isinstance(cost, (int, float))
            else "—",
        )
    with col4:
        dur = workflow.get("duration_seconds")
        st.metric(
            "Duration",
            format_duration(dur)
            if isinstance(dur, (int, float))
            else "—",
        )

    error = workflow.get("error_message")
    if error:
        st.error(f"Error: {error}")

    st.divider()
    st.markdown("**Step Progress**")
    steps = workflow.get("steps", [])
    _render_step_visualization(st, steps)

    audit_trail = workflow.get("audit_trail", [])
    if audit_trail:
        st.divider()
        _render_audit_trail(st, audit_trail)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_workflows_page() -> None:
    """Render the workflow management page."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("🔄 Workflows")

    data = _load_workflows_data()

    if data["demo_mode"]:
        st.info(
            "📋 **Demo Mode** — Showing sample data. "
            "Connect backend services to see live "
            "workflows."
        )

    workflows = data["workflows"]

    # Summary KPIs
    _render_workflow_summary(st, workflows)

    st.divider()

    # Execution progress for running workflows
    _render_execution_progress(st, workflows)

    # Workflow history with filtering
    st.subheader("📋 Workflow History")
    statuses = [
        "All", "Completed", "Running",
        "Failed", "Pending",
    ]
    status_filter = st.selectbox(
        "Filter by Status",
        options=statuses,
        key="wf_status_filter",
    )
    selected_id = _render_workflow_table(
        st, workflows, status_filter,
    )

    # Drill-down into selected workflow
    if selected_id:
        selected_wf = next(
            (
                w for w in workflows
                if w.get("id") == selected_id
            ),
            None,
        )
        if selected_wf:
            st.divider()
            _render_workflow_detail(st, selected_wf)

    st.divider()

    # Template selector
    config = _render_template_selector(st)
    if config is not None:
        from dashboard.utils.backend import (
            get_workflow_orchestrator,
        )

        orchestrator = get_workflow_orchestrator()
        if orchestrator is not None and hasattr(
            orchestrator, "start_workflow",
        ):
            try:
                name = config["workflow_name"]
                tmpl = config["template_name"]
                st.info(
                    f"Starting workflow: {name} "
                    f"({tmpl})…"
                )
            except Exception as exc:
                st.error(
                    f"Workflow start failed: {exc}"
                )
        else:
            st.warning(
                "Backend unavailable — workflows "
                "cannot be started in demo mode."
            )
