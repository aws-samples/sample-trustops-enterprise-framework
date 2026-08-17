"""Datasets page — dataset browser, upload, quality reports, PII, versioning.

Displays registered datasets with quality scores, upload form,
quality report visualization, PII detection results, version history,
and lineage display.  Falls back to demo data when backend services
are unavailable.

Requirements: 10.14
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_datasets_data() -> dict[str, Any]:
    """Load dataset data from backend, falling back to demo data."""
    from dashboard.utils.backend import get_dataset_manager
    from dashboard.utils.demo_data import generate_demo_datasets

    data: dict[str, Any] = {"demo_mode": False}

    dm = get_dataset_manager()
    if dm is not None:
        try:
            datasets = dm.list_datasets() if hasattr(dm, "list_datasets") else []
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


def _get_quality_scores(dataset: dict) -> dict[str, float]:
    """Extract quality scores from a dataset dict."""
    quality = dataset.get("quality")
    if isinstance(quality, dict) and quality:
        return {
            "completeness": quality.get("completeness", 0.0),
            "diversity": quality.get("diversity", 0.0),
            "balance": quality.get("balance", 0.0),
        }
    # Handle quality_report style
    report = dataset.get("quality_report")
    if isinstance(report, dict) and report:
        return {
            "completeness": report.get(
                "completeness_score", 0.0
            ),
            "diversity": report.get(
                "diversity_score", 0.0
            ),
            "balance": report.get("balance_score", 0.0),
        }
    return {
        "completeness": 0.0,
        "diversity": 0.0,
        "balance": 0.0,
    }


def _avg_quality(datasets: list[dict], key: str) -> float | None:
    """Average a quality metric across datasets."""
    scores = [_get_quality_scores(d).get(key, 0.0) for d in datasets]
    return sum(scores) / len(scores) if scores else None


def _demo_pii_results() -> list[dict]:
    """Return demo PII detection results."""
    return [
        {"row": 12, "field": "prompt", "pii_type": "email", "value": "[EMAIL]"},
        {"row": 45, "field": "completion", "pii_type": "phone", "value": "[PHONE]"},
        {"row": 78, "field": "prompt", "pii_type": "ssn", "value": "[SSN]"},
        {"row": 134, "field": "completion", "pii_type": "credit_card", "value": "[CREDIT_CARD]"},
    ]


def _demo_version_history() -> list[dict]:
    """Return demo version history."""
    return [
        {"version": "v3", "created_at": "2024-01-15T10:30:00Z", "row_count": 5000,
         "changes": "Added 500 new QA pairs", "checksum": "abc123"},
        {"version": "v2", "created_at": "2024-01-10T08:00:00Z", "row_count": 4500,
         "changes": "PII masking applied", "checksum": "def456"},
        {"version": "v1", "created_at": "2024-01-05T14:00:00Z", "row_count": 4500,
         "changes": "Initial upload", "checksum": "ghi789"},
    ]


def _demo_lineage() -> list[dict]:
    """Return demo lineage data."""
    return [
        {"step": "Upload", "source": "raw_customer_qa.jsonl", "timestamp": "2024-01-05"},
        {"step": "PII Detection", "source": "ds-qa-001 v1", "timestamp": "2024-01-08"},
        {"step": "PII Masking", "source": "ds-qa-001 v1", "timestamp": "2024-01-10"},
        {"step": "Augmentation", "source": "ds-qa-001 v2", "timestamp": "2024-01-15"},
    ]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_dataset_summary(st: Any, datasets: list[dict]) -> None:
    """Render summary KPI cards for datasets."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Datasets", len(datasets))
    with col2:
        total_rows = sum(d.get("row_count", 0) for d in datasets)
        st.metric("Total Rows", f"{total_rows:,}")
    with col3:
        avg_comp = _avg_quality(datasets, "completeness")
        st.metric("Avg Completeness", f"{avg_comp:.0%}" if avg_comp is not None else "N/A")
    with col4:
        formats = set(d.get("format", "unknown") for d in datasets)
        st.metric("Formats", len(formats))


def _render_dataset_table(st: Any, datasets: list[dict]) -> str | None:
    """Render the dataset listing table. Returns selected dataset ID or None."""
    st.subheader("📋 Dataset Registry")

    if not datasets:
        st.info("No datasets available. Upload a dataset to get started.")
        return None

    header = st.columns([2, 1.5, 1.5, 1, 1, 1, 1])
    header[0].markdown("**Name**")
    header[1].markdown("**Task Type**")
    header[2].markdown("**Format**")
    header[3].markdown("**Rows**")
    header[4].markdown("**Completeness**")
    header[5].markdown("**Diversity**")
    header[6].markdown("**Balance**")

    for d in datasets:
        row = st.columns([2, 1.5, 1.5, 1, 1, 1, 1])
        row[0].write(d.get("name", d.get("id", "—")))
        row[1].write(d.get("task_type", "—"))
        row[2].write(d.get("format", "—").upper())
        rc = d.get("row_count", 0)
        row[3].write(f"{rc:,}" if isinstance(rc, (int, float)) else "—")

        quality = _get_quality_scores(d)
        row[4].write(f"{quality['completeness']:.0%}")
        row[5].write(f"{quality['diversity']:.0%}")
        row[6].write(f"{quality['balance']:.0%}")

    # Dataset selector for detail views
    dataset_ids = [d.get("id", d.get("name", f"ds-{i}")) for i, d in enumerate(datasets)]
    dataset_names = [d.get("name", d.get("id", f"Dataset {i}")) for i, d in enumerate(datasets)]
    options = dict(zip(dataset_names, dataset_ids))

    selected_name = st.selectbox("Select dataset for details", options=["None"] + list(options.keys()))
    if selected_name and selected_name != "None":
        return options[selected_name]
    return None


def _render_upload_form(st: Any) -> dict | None:
    """Render dataset upload form. Returns upload data or None."""
    st.subheader("📤 Upload Dataset")

    with st.form("upload_dataset_form"):
        uploaded_file = st.file_uploader(
            "Drag and drop or browse for a dataset file",
            type=["jsonl", "csv", "parquet"],
            help="Supported formats: JSONL, CSV, Parquet",
        )
        name = st.text_input("Dataset Name")
        description = st.text_input("Description (optional)")
        task_type = st.selectbox(
            "Task Type",
            options=["auto-detect", "qa", "summarization", "classification", "text_generation", "chat"],
            index=0,
        )
        submitted = st.form_submit_button("Upload Dataset")

    if submitted and uploaded_file and name:
        return {
            "file": uploaded_file,
            "name": name,
            "description": description,
            "task_type": None if task_type == "auto-detect" else task_type,
        }
    return None


def _render_quality_report(st: Any, dataset: dict) -> None:
    """Render quality report with interactive bar chart."""
    st.subheader("📊 Quality Report")

    quality = _get_quality_scores(dataset)
    dimensions = list(quality.keys())
    scores = list(quality.values())

    colors = [
        "#2ecc71" if s >= 0.8 else "#f39c12" if s >= 0.6 else "#e74c3c"
        for s in scores
    ]

    fig = go.Figure(data=go.Bar(
        x=[d.capitalize() for d in dimensions],
        y=scores,
        marker_color=colors,
        text=[f"{s:.0%}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=f"Quality Scores — {dataset.get('name', 'Dataset')}"),
        yaxis=dict(title="Score", range=[0, 1.1]),
        height=350,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    recs = []
    if quality["completeness"] < 0.9:
        recs.append("⚠️ Completeness below 90% — check for missing fields in your dataset.")
    if quality["diversity"] < 0.7:
        recs.append("⚠️ Low diversity — consider adding more varied prompts or using data augmentation.")
    if quality["balance"] < 0.7:
        recs.append("⚠️ Imbalanced categories — consider resampling or adding examples to underrepresented categories.")
    if recs:
        st.markdown("**Recommendations:**")
        for r in recs:
            st.write(r)
    else:
        st.success("✅ Dataset quality looks good across all dimensions.")


def _render_pii_results(st: Any, dataset: dict) -> None:
    """Render PII detection results."""
    st.subheader("🔒 PII Detection Results")

    pii_results = dataset.get("pii_results") or _demo_pii_results()

    if not pii_results:
        st.success("✅ No PII detected in this dataset.")
        return

    st.warning(f"⚠️ Found **{len(pii_results)}** PII instance(s) in this dataset.")

    header = st.columns([1, 1.5, 1.5, 2])
    header[0].markdown("**Row**")
    header[1].markdown("**Field**")
    header[2].markdown("**PII Type**")
    header[3].markdown("**Masked Value**")

    for pii in pii_results:
        row = st.columns([1, 1.5, 1.5, 2])
        row[0].write(str(pii.get("row", "—")))
        row[1].write(pii.get("field", "—"))
        row[2].write(pii.get("pii_type", "—").upper())
        row[3].write(pii.get("value", "—"))

    # PII type breakdown chart
    type_counts: dict[str, int] = {}
    for pii in pii_results:
        pt = pii.get("pii_type", "unknown")
        type_counts[pt] = type_counts.get(pt, 0) + 1

    if type_counts:
        fig = go.Figure(data=go.Pie(
            labels=[t.upper() for t in type_counts.keys()],
            values=list(type_counts.values()),
            hole=0.4,
            marker=dict(colors=["#e74c3c", "#f39c12", "#3498db", "#9b59b6", "#1abc9c"]),
        ))
        fig.update_layout(
            title=dict(text="PII Types Found"),
            height=300,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_version_history(st: Any, dataset: dict) -> None:
    """Render dataset version history browser."""
    st.subheader("📜 Version History")

    versions = dataset.get("versions") or _demo_version_history()

    if not versions:
        st.info("No version history available for this dataset.")
        return

    for v in versions:
        version_label = v.get("version", "—")
        created = v.get("created_at", "—")
        rows = v.get("row_count", "—")
        changes = v.get("changes", "—")
        checksum = v.get("checksum", "—")

        with st.expander(f"**{version_label}** — {created}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Rows:** {rows:,}" if isinstance(rows, int) else f"**Rows:** {rows}")
                st.write(f"**Checksum:** `{checksum}`")
            with col2:
                st.write(f"**Changes:** {changes}")


def _render_lineage(st: Any, dataset: dict) -> None:
    """Render dataset lineage graph."""
    st.subheader("🔗 Dataset Lineage")

    lineage = dataset.get("lineage") or _demo_lineage()

    if not lineage:
        st.info("No lineage information available for this dataset.")
        return

    # Render as a step-by-step flow
    for i, step in enumerate(lineage):
        step_name = step.get("step", "—")
        source = step.get("source", "—")
        ts = step.get("timestamp", "—")

        col1, col2, col3 = st.columns([1, 2, 1.5])
        col1.write(f"**Step {i + 1}**")
        col2.write(f"{step_name}: `{source}`")
        col3.write(ts)

        if i < len(lineage) - 1:
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;⬇️")


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_datasets_page() -> None:
    """Render the dataset management page."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("📊 Datasets")

    data = _load_datasets_data()

    if data["demo_mode"]:
        st.info(
            "📋 **Demo Mode** — Showing sample data. "
            "Connect backend services to see live datasets."
        )

    datasets = data["datasets"]

    # Summary KPIs
    _render_dataset_summary(st, datasets)

    st.divider()

    # Dataset table with selector
    selected_id = _render_dataset_table(st, datasets)

    st.divider()

    # Upload form
    upload_data = _render_upload_form(st)
    if upload_data is not None:
        from dashboard.utils.backend import get_dataset_manager

        dm = get_dataset_manager()
        if dm is not None and hasattr(dm, "upload_dataset"):
            try:
                dm.upload_dataset(
                    file_path=upload_data["file"],
                    name=upload_data["name"],
                    task_type=upload_data["task_type"],
                    description=upload_data["description"],
                )
                st.success(f"✅ Dataset **{upload_data['name']}** uploaded successfully.")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
        else:
            st.warning(
                "Backend unavailable — dataset upload "
                "is not possible in demo mode."
            )

    # Detail views for selected dataset
    if selected_id:
        selected_dataset = next(
            (d for d in datasets if d.get("id") == selected_id),
            None,
        )
        if selected_dataset:
            st.divider()
            _render_quality_report(st, selected_dataset)

            st.divider()
            _render_pii_results(st, selected_dataset)

            st.divider()
            _render_version_history(st, selected_dataset)

            st.divider()
            _render_lineage(st, selected_dataset)
