"""Tests for the CostCalculator.

Requirements: 3.9, 3.11
"""

import pytest

from src.data_models.model import ModelPricing
from src.evaluation.batch_runner import BatchItemResult
from src.evaluation.cost_calculator import CostCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pricing(inp: float = 0.003, out: float = 0.015) -> ModelPricing:
    """Create a ModelPricing with per-1k-token prices."""
    return ModelPricing(
        input_price_per_1k_tokens=inp,
        output_price_per_1k_tokens=out,
    )


def _item(
    index: int = 0,
    input_tokens: int = 100,
    output_tokens: int = 50,
    success: bool = True,
) -> BatchItemResult:
    return BatchItemResult(
        index=index,
        prompt=f"prompt-{index}",
        response_text="resp" if success else "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        success=success,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCostCalculatorQueryCost:
    """Tests for calculate_query_cost."""

    def test_basic_cost(self):
        calc = CostCalculator(
            model_id="test-model",
            pricing=_pricing(inp=0.003, out=0.015),
        )
        # 1000 input tokens * 0.003/1000 + 500 output * 0.015/1000
        cost = calc.calculate_query_cost(
            input_tokens=1000, output_tokens=500
        )
        assert cost == pytest.approx(0.003 + 0.0075)

    def test_zero_tokens(self):
        calc = CostCalculator(
            model_id="test-model",
            pricing=_pricing(),
        )
        assert calc.calculate_query_cost(0, 0) == 0.0

    def test_no_pricing_returns_zero(self):
        """When pricing is explicitly absent, costs are 0."""
        # Pass a broken loader that raises to simulate missing config
        class _BrokenLoader:
            def get_model_pricing(self, *a, **kw):
                raise FileNotFoundError("no config")

        calc = CostCalculator(
            model_id="nonexistent-model",
            pricing=None,
            pricing_loader=_BrokenLoader(),
        )
        assert calc.calculate_query_cost(1000, 500) == 0.0


class TestCostCalculatorBatchCost:
    """Tests for calculate_batch_cost."""

    def test_empty_batch(self):
        calc = CostCalculator(
            model_id="m", pricing=_pricing()
        )
        result = calc.calculate_batch_cost([])
        assert result.total_cost == 0.0
        assert result.total_input_tokens == 0

    def test_single_item(self):
        calc = CostCalculator(
            model_id="m",
            pricing=_pricing(inp=1.0, out=2.0),
        )
        items = [_item(input_tokens=1000, output_tokens=500)]
        result = calc.calculate_batch_cost(items)
        # input: 1000 * 1.0/1000 = 1.0
        # output: 500 * 2.0/1000 = 1.0
        assert result.input_cost == pytest.approx(1.0)
        assert result.output_cost == pytest.approx(1.0)
        assert result.total_cost == pytest.approx(2.0)
        assert result.cost_per_query == pytest.approx(2.0)
        assert result.total_input_tokens == 1000
        assert result.total_output_tokens == 500

    def test_multiple_items(self):
        calc = CostCalculator(
            model_id="m",
            pricing=_pricing(inp=0.002, out=0.004),
        )
        items = [
            _item(0, input_tokens=100, output_tokens=50),
            _item(1, input_tokens=200, output_tokens=100),
        ]
        result = calc.calculate_batch_cost(items)
        # input: 300 * 0.002/1000 = 0.0006
        # output: 150 * 0.004/1000 = 0.0006
        assert result.total_cost == pytest.approx(0.0012)
        assert result.cost_per_query == pytest.approx(0.0006)

    def test_failed_items_excluded(self):
        calc = CostCalculator(
            model_id="m",
            pricing=_pricing(inp=1.0, out=1.0),
        )
        items = [
            _item(0, input_tokens=100, output_tokens=50, success=True),
            _item(1, input_tokens=999, output_tokens=999, success=False),
        ]
        result = calc.calculate_batch_cost(items)
        # Only the successful item counts
        assert result.total_input_tokens == 100
        assert result.total_output_tokens == 50
        assert result.cost_per_query == result.total_cost

    def test_all_failed(self):
        calc = CostCalculator(
            model_id="m",
            pricing=_pricing(),
        )
        items = [_item(success=False), _item(index=1, success=False)]
        result = calc.calculate_batch_cost(items)
        assert result.total_cost == 0.0
        assert result.cost_per_query == 0.0

    def test_currency_propagated(self):
        pricing = ModelPricing(
            input_price_per_1k_tokens=0.001,
            output_price_per_1k_tokens=0.002,
            currency="EUR",
        )
        calc = CostCalculator(
            model_id="m", pricing=pricing
        )
        result = calc.calculate_batch_cost([_item()])
        assert result.currency == "EUR"


class TestCostCalculatorPricingResolution:
    """Tests for pricing resolution from loader."""

    def test_explicit_pricing_used(self):
        p = _pricing(inp=0.005, out=0.010)
        calc = CostCalculator(
            model_id="m", pricing=p
        )
        assert calc.input_price_per_token == pytest.approx(
            0.005 / 1000
        )
        assert calc.output_price_per_token == pytest.approx(
            0.010 / 1000
        )

    def test_no_pricing_graceful(self):
        """When pricing loader fails, costs are 0."""
        class _BrokenLoader:
            def get_model_pricing(self, *a, **kw):
                raise FileNotFoundError("no config")

        calc = CostCalculator(
            model_id="totally-unknown-model-xyz",
            pricing=None,
            pricing_loader=_BrokenLoader(),
        )
        assert calc.input_price_per_token == 0.0
        assert calc.output_price_per_token == 0.0
