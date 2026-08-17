"""
Dedicated Cost Calculator for the Evaluation Engine.

Calculates inference costs using model pricing configuration.
Supports per-query cost calculation and batch cost breakdowns.

Requirements: 3.9, 3.11
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.data_models.model import ModelPricing, ModelProvider
from src.evaluation.batch_runner import BatchItemResult
from src.registry.pricing_loader import PricingLoader

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Breakdown of inference costs.

    Attributes:
        total_cost: Total cost (input + output).
        input_cost: Cost from input tokens only.
        output_cost: Cost from output tokens only.
        cost_per_query: Average cost per query.
        total_input_tokens: Sum of all input tokens.
        total_output_tokens: Sum of all output tokens.
        currency: Currency code.
    """

    total_cost: float = 0.0
    input_cost: float = 0.0
    output_cost: float = 0.0
    cost_per_query: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    currency: str = "USD"


class CostCalculator:
    """Calculates inference costs using model pricing configuration.

    Can load pricing from the YAML config via PricingLoader or
    accept explicit ModelPricing. Falls back to zero cost with a
    warning when pricing is unavailable.

    Usage::

        calc = CostCalculator(model_id="anthropic.claude-3-haiku-20240307-v1:0",
                              provider=ModelProvider.BEDROCK)
        query_cost = calc.calculate_query_cost(input_tokens=500, output_tokens=200)
        breakdown = calc.calculate_batch_cost(batch_items)

    Requirements: 3.9, 3.11
    """

    def __init__(
        self,
        model_id: str,
        provider: ModelProvider = ModelProvider.BEDROCK,
        pricing: Optional[ModelPricing] = None,
        pricing_loader: Optional[PricingLoader] = None,
    ):
        """
        Args:
            model_id: The model identifier for pricing lookup.
            provider: The model provider.
            pricing: Explicit pricing to use. If provided, skips
                config lookup.
            pricing_loader: Optional PricingLoader instance. If None
                and pricing is not provided, a default loader is created.
        """
        self.model_id = model_id
        self.provider = provider
        self._pricing = self._resolve_pricing(
            pricing, pricing_loader
        )

    def _resolve_pricing(
        self,
        explicit_pricing: Optional[ModelPricing],
        loader: Optional[PricingLoader],
    ) -> Optional[ModelPricing]:
        """Resolve pricing from explicit value or config loader."""
        if explicit_pricing is not None:
            return explicit_pricing

        try:
            if loader is None:
                loader = PricingLoader()
            return loader.get_model_pricing(
                self.model_id, self.provider
            )
        except Exception as exc:
            logger.warning(
                "Could not load pricing for model %s: %s. "
                "Costs will be reported as 0.",
                self.model_id,
                exc,
            )
            return None

    @property
    def input_price_per_token(self) -> float:
        """Per-token input price (converted from per-1k)."""
        if self._pricing is None:
            return 0.0
        return self._pricing.input_price_per_1k_tokens / 1000.0

    @property
    def output_price_per_token(self) -> float:
        """Per-token output price (converted from per-1k)."""
        if self._pricing is None:
            return 0.0
        return self._pricing.output_price_per_1k_tokens / 1000.0

    def calculate_query_cost(
        self, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for a single query.

        Formula: input_tokens × input_price + output_tokens × output_price

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in the configured currency.
        """
        return (
            input_tokens * self.input_price_per_token
            + output_tokens * self.output_price_per_token
        )

    def calculate_batch_cost(
        self, items: list[BatchItemResult]
    ) -> CostBreakdown:
        """Calculate total cost and breakdown for a batch of results.

        Only successful items are included in the cost calculation.

        Args:
            items: List of BatchItemResult from a batch run.

        Returns:
            CostBreakdown with totals and per-query average.
        """
        if not items:
            return CostBreakdown(
                currency=self._pricing.currency
                if self._pricing
                else "USD"
            )

        successful = [i for i in items if i.success]
        total_in = sum(i.input_tokens for i in successful)
        total_out = sum(i.output_tokens for i in successful)

        input_cost = total_in * self.input_price_per_token
        output_cost = total_out * self.output_price_per_token
        total_cost = input_cost + output_cost
        cpq = total_cost / len(successful) if successful else 0.0

        return CostBreakdown(
            total_cost=total_cost,
            input_cost=input_cost,
            output_cost=output_cost,
            cost_per_query=cpq,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            currency=self._pricing.currency
            if self._pricing
            else "USD",
        )
