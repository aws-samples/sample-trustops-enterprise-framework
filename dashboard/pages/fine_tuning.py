"""Fine-tuning page — fine-tuning wizard, job history, and training progress.

Displays fine-tuning job listings with status and metrics, a wizard form
for configuring new jobs (model/dataset/hyperparameters), cost estimates,
real-time training progress with loss charts, and job history.
Falls back to demo data when backend services are unavailable.

Requirements: 10.5
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_fine_tuning_data() -> dict[str, Any]:
    """Load fine-tuning data from backend, falling back to demo."""
    from dashboard.utils.backend import (
        get_fine_tuning_pipeline,
        get_model_registry,
        get_dataset_manager,
    )
    from dashboard.utils.demo_data import (
        generate_demo_models,
        generate_demo_datasets,
    )

    data: dict[str, Any] = {"demo_mode": False}

    # Fine-tuning jobs
    pipeline = get_fine_tuning_pipeline()
    if pipeline is not None:
        try:
            jobs = (
                pipeline.list_jobs()
                if hasattr(pipeline, "list_jobs")
                else []
            )
            data["jobs"] = [
                j if isinstance(j, dict) else j.__dict__
                for j in jobs
            ] if jobs else []
        except Exception:
            data["jobs"] = _build_demo_jobs()
            data["demo_mode"] = True
    else:
        data["jobs"] = _build_demo_jobs()
        data["demo_mode"] = True

    # Models (for wizard selectors — only fine-tunable)
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


def _build_demo_jobs() -> list[dict]:
    """Build enriched demo fine-tuning job data."""
    from dashboard.utils.demo_data import generate_demo_fine_tuning_jobs

    jobs = generate_demo_fine_tuning_jobs()
    _EXTRA: dict[str, dict[str, Any]] = {
        "ft-001": {
            "base_model_name": "Claude 3 Sonnet",
            "dataset_id": "ds-qa-001",
            "dataset_name": "Customer Support QA",
            "created_at": "2024-01-12T08:00:00Z",
            "completed_at": "2024-01-12T14:30:00Z",
            "estimated_cost": 45.00,
            "actual_cost": 42.50,
            "hyperparameters": {
                "learning_rate": 1e-5,
                "epochs": 3,
                "batch_size": 8,
                "warmup_steps": 100,
            },
            "finetuned_model_id": "ft-claude3-qa-v1",
        },
        "ft-002": {
            "base_model_name": "Titan Text Express",
            "dataset_id": "ds-sum-001",
            "dataset_name": "Legal Summarization",
            "created_at": "2024-01-15T10:00:00Z",
            "completed_at": None,
            "estimated_cost": 22.00,
            "actual_cost": None,
            "hyperparameters": {
                "learning_rate": 2e-5,
                "epochs": 5,
                "batch_size": 4,
                "warmup_steps": 50,
            },
            "finetuned_model_id": None,
            "progress_percent": 35.0,
            "current_epoch": 2,
            "total_epochs": 5,
        },
    }
    for job in jobs:
        jid = job.get("id", "")
        extra = _EXTRA.get(jid, {})
        for key, val in extra.items():
            job.setdefault(key, val)
    return jobs


def _running_jobs(jobs: list[dict]) -> list[dict]:
    """Return jobs that are currently running/training."""
    return [
        j for j in jobs
        if str(j.get("status", "")).lower()
        in ("running", "training", "in_progress")
    ]


def _estimate_cost(
    model_id: str,
    dataset_rows: int,
    epochs: int,
) -> dict[str, Any]:
    """Compute a simple cost estimate for a fine-tuning job."""
    _BASE_COSTS: dict[str, float] = {
        "bedrock-claude-3": 8.0,
        "bedrock-titan-text": 3.0,
    }
    base = _BASE_COSTS.get(model_id, 5.0)
    row_factor = dataset_rows / 1000.0
    estimated = base * row_factor * epochs
    return {
        "estimated_training_cost": round(estimated, 2),
        "estimated_duration_hours": round(
            estimated / 10.0, 1,
        ),
        "cost_breakdown": {
            "compute": round(estimated * 0.7, 2),
            "storage": round(estimated * 0.2, 2),
            "data_transfer": round(estimated * 0.1, 2),
        },
        "currency": "USD",
        "confidence": (
            "high" if model_id in _BASE_COSTS else "medium"
        ),
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_job_summary(st: Any, jobs: list[dict]) -> None:
    """Render summary KPI cards for fine-tuning jobs."""
    from dashboard.utils.helpers import format_cost

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Jobs", len(jobs))
    with col2:
        completed = sum(
            1 for j in jobs
            if str(j.get("status", "")).lower() == "completed"
        )
        st.metric("Completed", completed)
    with col3:
        running = len(_running_jobs(jobs))
        st.metric("Running", running)
    with col4:
        total_cost = sum(
            j.get("actual_cost", 0.0) or 0.0
            for j in jobs
        )
        st.metric("Total Cost", format_cost(total_cost))


def _render_job_table(st: Any, jobs: list[dict]) -> str | None:
    """Render the job history table. Returns selected job id."""
    from dashboard.utils.helpers import status_emoji, format_cost

    if not jobs:
        st.info(
            "No fine-tuning jobs found. "
            "Use the wizard below to start one."
        )
        return None

    header = st.columns([1.2, 1.5, 1.5, 1, 1, 1])
    header[0].markdown("**Job ID**")
    header[1].markdown("**Base Model**")
    header[2].markdown("**Dataset**")
    header[3].markdown("**Status**")
    header[4].markdown("**Cost**")
    header[5].markdown("**Final Loss**")

    selected: str | None = None
    for job in jobs:
        row = st.columns([1.2, 1.5, 1.5, 1, 1, 1])
        jid = job.get("id", "—")
        if row[0].button(jid, key=f"ft_select_{jid}"):
            selected = jid
        row[1].write(
            job.get("base_model_name", job.get("model_id", "—"))
        )
        row[2].write(
            job.get("dataset_name", job.get("dataset_id", "—"))
        )
        status = str(job.get("status", "unknown"))
        row[3].write(
            f"{status_emoji(status)} {status.capitalize()}"
        )
        cost = job.get("actual_cost") or job.get("estimated_cost")
        row[4].write(
            format_cost(cost) if isinstance(cost, (int, float))
            else "—"
        )
        metrics = job.get("training_metrics", [])
        if metrics:
            last_loss = metrics[-1].get("loss")
            row[5].write(
                f"{last_loss:.3f}"
                if isinstance(last_loss, (int, float))
                else "—"
            )
        else:
            row[5].write("—")

    return selected


def _render_wizard(
    st: Any,
    models: list[dict],
    datasets: list[dict],
) -> dict | None:
    """Render the fine-tuning wizard form.

    Returns the submitted config dict, or ``None`` if not submitted.
    """
    st.subheader("🧙 Fine-Tuning Wizard")

    ft_models = [
        m for m in models if m.get("fine_tuning_support")
    ]
    if not ft_models:
        st.warning("No fine-tunable models available.")
        return None

    with st.form("fine_tuning_wizard_form"):
        # Step 1: Model selection
        st.markdown("**Step 1 — Select Base Model**")
        model_names = [
            m.get("name", m.get("id", "unknown"))
            for m in ft_models
        ]
        model_ids = [m.get("id", "") for m in ft_models]
        sel_model_idx = st.selectbox(
            "Base Model",
            options=range(len(model_names)),
            format_func=lambda i: model_names[i],
        )

        # Step 2: Dataset selection
        st.markdown("**Step 2 — Select Training Dataset**")
        ds_names = [
            d.get("name", d.get("id", "unknown"))
            for d in datasets
        ]
        ds_ids = [d.get("id", "") for d in datasets]
        sel_ds_idx = st.selectbox(
            "Training Dataset",
            options=range(len(ds_names)),
            format_func=lambda i: ds_names[i],
        ) if ds_names else None

        # Step 3: Hyperparameters
        st.markdown("**Step 3 — Configure Hyperparameters**")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
            value=1e-5,
        )
        epochs = st.slider(
            "Epochs", min_value=1, max_value=10, value=3,
        )
        batch_size = st.select_slider(
            "Batch Size",
            options=[2, 4, 8, 16, 32],
            value=8,
        )
        warmup_steps = st.number_input(
            "Warmup Steps",
            min_value=0, value=100, step=50,
        )

        submitted = st.form_submit_button(
            "📊 Estimate Cost & Submit"
        )

    if (
        submitted
        and sel_model_idx is not None
        and sel_ds_idx is not None
    ):
        model_id = model_ids[sel_model_idx]
        dataset_id = ds_ids[sel_ds_idx]
        ds_rows = datasets[sel_ds_idx].get("row_count", 1000)
        return {
            "model_id": model_id,
            "model_name": model_names[sel_model_idx],
            "dataset_id": dataset_id,
            "dataset_name": ds_names[sel_ds_idx],
            "dataset_rows": ds_rows,
            "hyperparameters": {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "warmup_steps": int(warmup_steps),
            },
        }
    return None


def _render_cost_estimate(st: Any, config: dict) -> None:
    """Render cost estimate display before job submission."""
    from dashboard.utils.helpers import format_cost

    estimate = _estimate_cost(
        config["model_id"],
        config.get("dataset_rows", 1000),
        config["hyperparameters"]["epochs"],
    )

    st.subheader("💰 Cost Estimate")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Estimated Cost",
            format_cost(estimate["estimated_training_cost"]),
        )
    with col2:
        st.metric(
            "Est. Duration",
            f"{estimate['estimated_duration_hours']}h",
        )
    with col3:
        st.metric("Confidence", estimate["confidence"].capitalize())

    breakdown = estimate.get("cost_breakdown", {})
    if breakdown:
        st.markdown("**Cost Breakdown**")
        for component, cost in breakdown.items():
            st.write(
                f"  • {component.replace('_', ' ').title()}: "
                f"{format_cost(cost)}"
            )


def _render_training_progress(st: Any, jobs: list[dict]) -> None:
    """Render real-time training progress for running jobs."""
    running = _running_jobs(jobs)
    if not running:
        return

    st.subheader("⏳ Training in Progress")
    for job in running:
        jid = job.get("id", "unknown")
        model_name = job.get(
            "base_model_name", job.get("model_id", "?"),
        )
        progress = job.get("progress_percent", 0.0) / 100.0
        current_epoch = job.get("current_epoch", 0)
        total_epochs = job.get("total_epochs", 0)

        st.markdown(f"**{jid}** — {model_name}")
        st.progress(min(max(progress, 0.0), 1.0))
        st.caption(
            f"Epoch {current_epoch} / {total_epochs}"
        )

        # Loss chart for running job
        metrics = job.get("training_metrics", [])
        if metrics:
            _render_loss_chart(st, metrics, jid)


def _render_loss_chart(
    st: Any,
    metrics: list[dict],
    job_id: str = "",
) -> None:
    """Render a Plotly line chart of training/validation loss."""
    epochs = [m.get("epoch", i + 1) for i, m in enumerate(metrics)]
    train_loss = [m.get("loss", 0.0) for m in metrics]
    val_loss = [m.get("validation_loss") for m in metrics]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode="lines+markers",
        name="Training Loss",
        line=dict(color="#1a73e8", width=2),
        marker=dict(size=6),
    ))

    # Only add validation loss if present
    if any(v is not None for v in val_loss):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=[v if v is not None else None for v in val_loss],
            mode="lines+markers",
            name="Validation Loss",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            marker=dict(size=6),
        ))

    title_suffix = f" — {job_id}" if job_id else ""
    fig.update_layout(
        title=dict(text=f"Training Loss{title_suffix}"),
        xaxis=dict(title="Epoch", dtick=1),
        yaxis=dict(title="Loss"),
        height=350,
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_job_detail(st: Any, job: dict) -> None:
    """Render detail view for a selected fine-tuning job."""
    from dashboard.utils.helpers import format_cost, status_emoji

    jid = job.get("id", "?")
    status = str(job.get("status", "unknown"))

    st.subheader(
        f"📋 Job Detail — {jid} "
        f"{status_emoji(status)} {status.capitalize()}"
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        model_name = job.get(
            "base_model_name", job.get("model_id", "—"),
        )
        st.metric("Base Model", model_name)
    with col2:
        ds_name = job.get(
            "dataset_name", job.get("dataset_id", "—"),
        )
        st.metric("Dataset", ds_name)
    with col3:
        cost = job.get("actual_cost") or job.get("estimated_cost")
        st.metric(
            "Cost",
            format_cost(cost)
            if isinstance(cost, (int, float)) else "—",
        )
    with col4:
        ft_model = job.get("finetuned_model_id")
        st.metric(
            "Fine-Tuned Model",
            ft_model if ft_model else "—",
        )

    # Hyperparameters
    hp = job.get("hyperparameters", {})
    if hp:
        st.markdown("**Hyperparameters**")
        hp_cols = st.columns(4)
        hp_items = list(hp.items())
        for i, (key, val) in enumerate(hp_items):
            label = key.replace("_", " ").title()
            hp_cols[i % 4].write(f"{label}: **{val}**")

    # Loss chart
    metrics = job.get("training_metrics", [])
    if metrics:
        st.divider()
        _render_loss_chart(st, metrics, jid)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_fine_tuning_page() -> None:
    """Render the fine-tuning management page."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("🔧 Fine-Tuning")

    data = _load_fine_tuning_data()

    if data["demo_mode"]:
        st.info(
            "📋 **Demo Mode** — Showing sample data. "
            "Connect backend services to see live jobs."
        )

    jobs = data["jobs"]
    models = data["models"]
    datasets = data["datasets"]

    # Summary KPIs
    _render_job_summary(st, jobs)

    st.divider()

    # Training progress for running jobs
    _render_training_progress(st, jobs)

    # Job history
    st.subheader("📋 Job History")
    selected_id = _render_job_table(st, jobs)

    # Drill-down into selected job
    if selected_id:
        selected_job = next(
            (j for j in jobs if j.get("id") == selected_id),
            None,
        )
        if selected_job:
            st.divider()
            _render_job_detail(st, selected_job)

    st.divider()

    # Fine-tuning wizard
    config = _render_wizard(st, models, datasets)
    if config is not None:
        _render_cost_estimate(st, config)

        from dashboard.utils.backend import get_fine_tuning_pipeline

        pipeline = get_fine_tuning_pipeline()
        if pipeline is not None and hasattr(
            pipeline, "start_fine_tuning",
        ):
            try:
                mid = config["model_name"]
                did = config["dataset_name"]
                st.info(
                    f"Starting fine-tuning: {mid} "
                    f"on {did}…"
                )
            except Exception as exc:
                st.error(f"Fine-tuning failed: {exc}")
        else:
            st.warning(
                "Backend unavailable — fine-tuning "
                "cannot be started in demo mode."
            )
