"""TrustOps Enterprise Dashboard — main entry point."""

from __future__ import annotations

import logging

from dashboard.pages.home import render_home_page
from dashboard.pages.models import render_models_page
from dashboard.pages.datasets import render_datasets_page
from dashboard.pages.evaluation import render_evaluation_page
from dashboard.pages.comparison import render_comparison_page
from dashboard.pages.fine_tuning import render_fine_tuning_page
from dashboard.pages.workflows import render_workflows_page

logger = logging.getLogger(__name__)

PAGE_RENDERERS: dict[str, callable] = {
    "Home": render_home_page,
    "Models": render_models_page,
    "Datasets": render_datasets_page,
    "Evaluation": render_evaluation_page,
    "Comparison": render_comparison_page,
    "Fine-Tuning": render_fine_tuning_page,
    "Workflows": render_workflows_page,
}


def main() -> None:
    """Run the Streamlit dashboard."""
    import streamlit as st
    from dashboard.config import get_config

    config = get_config()
    st.set_page_config(
        page_title=config.page_title,
        page_icon=config.page_icon,
        layout=config.layout,
    )

    page = st.sidebar.radio("Navigation", list(PAGE_RENDERERS.keys()))
    try:
        PAGE_RENDERERS[page]()
    except Exception as e:
        logger.exception("Error rendering page %s", page)
        st.error(f"An error occurred while loading the {page} page. Please try again.")


if __name__ == "__main__":
    main()
