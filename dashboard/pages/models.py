"""Models page — model registry browser.

Displays registered models with health status, filtering by provider /
capability / status, and a registration form for external API models.
Falls back to demo data when backend services are unavailable.

Requirements: 10.3
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_models_data() -> dict[str, Any]:
    """Load model data from backend, falling back to demo data."""
    from dashboard.utils.backend import get_model_registry
    from dashboard.utils.demo_data import generate_demo_models

    data: dict[str, Any] = {"demo_mode": False}

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


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _unique_values(models: list[dict], key: str) -> list[str]:
    """Return sorted unique values for *key* across models."""
    values: set[str] = set()
    for m in models:
        val = m.get(key)
        if val is not None:
            if isinstance(val, list):
                values.update(str(v) for v in val)
            else:
                values.add(str(val))
    return sorted(values)


def _filter_models(
    models: list[dict],
    provider: str | None = None,
    capability: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Filter models by provider, capability, and/or status."""
    filtered = models
    if provider:
        filtered = [
            m for m in filtered
            if str(m.get("provider", "")).lower() == provider.lower()
        ]
    if capability:
        filtered = [
            m for m in filtered
            if capability in (m.get("capabilities") or [])
        ]
    if status:
        filtered = [
            m for m in filtered
            if str(m.get("status", "")).lower() == status.lower()
        ]
    return filtered


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_filters(
    st: Any,
    models: list[dict],
) -> dict[str, str | None]:
    """Render sidebar filter widgets and return selected values."""
    st.sidebar.subheader("🔍 Filter Models")

    providers = _unique_values(models, "provider")
    selected_provider = st.sidebar.selectbox(
        "Provider",
        options=["All"] + providers,
        index=0,
    )

    capabilities = _unique_values(models, "capabilities")
    selected_capability = st.sidebar.selectbox(
        "Capability",
        options=["All"] + capabilities,
        index=0,
    )

    statuses = _unique_values(models, "status")
    selected_status = st.sidebar.selectbox(
        "Status",
        options=["All"] + statuses,
        index=0,
    )

    return {
        "provider": (
            None if selected_provider == "All"
            else selected_provider
        ),
        "capability": (
            None if selected_capability == "All"
            else selected_capability
        ),
        "status": (
            None if selected_status == "All"
            else selected_status
        ),
    }


def _render_model_table(st: Any, models: list[dict]) -> None:
    """Render the model registry table with health status."""
    from dashboard.utils.helpers import status_emoji

    if not models:
        st.info("No models match the current filters.")
        return

    header = st.columns([2, 1.5, 2, 1.5, 1.5, 1])
    header[0].markdown("**Model**")
    header[1].markdown("**Provider**")
    header[2].markdown("**Capabilities**")
    header[3].markdown("**Status**")
    header[4].markdown("**Max Tokens**")
    header[5].markdown("**Fine-Tune**")

    for m in models:
        row = st.columns([2, 1.5, 2, 1.5, 1.5, 1])
        row[0].write(m.get("name", m.get("id", "—")))
        row[1].write(
            str(m.get("provider", "—")).capitalize()
        )
        caps = m.get("capabilities") or []
        row[2].write(", ".join(caps) if caps else "—")
        model_status = str(m.get("status", "unknown"))
        row[3].write(
            f"{status_emoji(model_status)} "
            f"{model_status.capitalize()}"
        )
        row[4].write(
            f"{m.get('max_tokens', '—'):,}"
            if isinstance(m.get("max_tokens"), (int, float))
            else "—"
        )
        ft = m.get("fine_tuning_support", False)
        row[5].write("✅" if ft else "—")


def _render_model_summary(
    st: Any,
    models: list[dict],
    filtered: list[dict],
) -> None:
    """Render summary KPI cards for the model registry."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Models", len(models))
    with col2:
        active = sum(
            1 for m in models
            if m.get("status") == "active"
        )
        st.metric("Active", active)
    with col3:
        ft_count = sum(
            1 for m in models
            if m.get("fine_tuning_support")
        )
        st.metric("Fine-Tunable", ft_count)
    with col4:
        st.metric("Showing", len(filtered))


def _render_registration_form(st: Any) -> dict | None:
    """Render a registration form for external API models.

    Returns the submitted form data dict, or ``None`` if the form
    was not submitted.
    """
    st.subheader("➕ Register External API Model")

    with st.form("register_model_form"):
        name = st.text_input("Model Name")
        api_url = st.text_input("API Base URL")
        api_key = st.text_input(
            "API Key", type="password",
        )
        capabilities = st.multiselect(
            "Capabilities",
            options=[
                "text_generation",
                "chat",
                "embedding",
                "completion",
            ],
            default=["text_generation"],
        )
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            value=4096,
            step=1024,
        )
        submitted = st.form_submit_button("Register Model")

    if submitted and name and api_url:
        return {
            "name": name,
            "api_url": api_url,
            "api_key": api_key,
            "capabilities": capabilities,
            "max_tokens": int(max_tokens),
            "provider": "external_api",
            "status": "active",
        }
    return None


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_models_page() -> None:
    """Render the model registry browser page."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("🤖 Model Registry")

    data = _load_models_data()

    if data["demo_mode"]:
        st.info(
            "📋 **Demo Mode** — Showing sample data. "
            "Connect backend services to see live models."
        )

    models = data["models"]

    # Sidebar filters
    filters = _render_filters(st, models)
    filtered = _filter_models(
        models,
        provider=filters["provider"],
        capability=filters["capability"],
        status=filters["status"],
    )

    # Summary KPIs
    _render_model_summary(st, models, filtered)

    st.divider()

    # Model table
    st.subheader("📋 Registered Models")
    _render_model_table(st, filtered)

    st.divider()

    # Registration form
    result = _render_registration_form(st)
    if result is not None:
        from dashboard.utils.backend import get_model_registry

        registry = get_model_registry()
        if registry is not None and hasattr(registry, "register_model"):
            try:
                registry.register_model(result)
                st.success(
                    f"✅ Model **{result['name']}** registered."
                )
            except Exception as exc:
                st.error(f"Registration failed: {exc}")
        else:
            st.warning(
                "Backend unavailable — model registration "
                "is not possible in demo mode."
            )
