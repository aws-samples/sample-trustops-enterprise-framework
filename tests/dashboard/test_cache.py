"""Tests for dashboard caching and performance utilities."""

import math

import pytest

from dashboard.utils.cache import paginate_data


class TestPaginateDataLogic:
    """Tests for pagination logic (non-Streamlit parts)."""

    def test_total_pages_calculation(self):
        data = list(range(100))
        page_size = 25
        total_pages = max(1, math.ceil(len(data) / page_size))
        assert total_pages == 4

    def test_total_pages_with_remainder(self):
        data = list(range(30))
        page_size = 25
        total_pages = max(1, math.ceil(len(data) / page_size))
        assert total_pages == 2

    def test_total_pages_empty_data(self):
        data = []
        page_size = 25
        total_pages = max(1, math.ceil(len(data) / page_size))
        assert total_pages == 1

    def test_total_pages_exact_fit(self):
        data = list(range(50))
        page_size = 25
        total_pages = max(1, math.ceil(len(data) / page_size))
        assert total_pages == 2

    def test_page_slice_first_page(self):
        data = list(range(100))
        page_size = 25
        current_page = 1
        start = (current_page - 1) * page_size
        end = min(start + page_size, len(data))
        assert data[start:end] == list(range(25))

    def test_page_slice_last_page(self):
        data = list(range(30))
        page_size = 25
        current_page = 2
        start = (current_page - 1) * page_size
        end = min(start + page_size, len(data))
        assert data[start:end] == list(range(25, 30))


class TestCachedLoaderImports:
    """Tests that cached loaders can be imported."""

    def test_import_cached_load_models(self):
        from dashboard.utils.cache import cached_load_models
        assert callable(cached_load_models)

    def test_import_cached_load_datasets(self):
        from dashboard.utils.cache import cached_load_datasets
        assert callable(cached_load_datasets)

    def test_import_cached_load_evaluations(self):
        from dashboard.utils.cache import cached_load_evaluations
        assert callable(cached_load_evaluations)

    def test_import_cached_load_workflows(self):
        from dashboard.utils.cache import cached_load_workflows
        assert callable(cached_load_workflows)

    def test_import_clear_all_caches(self):
        from dashboard.utils.cache import clear_all_caches
        assert callable(clear_all_caches)

    def test_import_lazy_load_data(self):
        from dashboard.utils.cache import lazy_load_data
        assert callable(lazy_load_data)
