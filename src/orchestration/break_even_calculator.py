"""Break-even volume analysis for model cost-performance comparison."""

from dataclasses import dataclass, field
from math import ceil


@dataclass
class BreakEvenResult:
    break_even_volume: int | None
    has_break_even: bool
    immediate_advantage: bool
    monthly_savings_at_volume: dict[str, float] = field(default_factory=dict)
    cost_crossover_chart_data: list[tuple[int, float, float]] = field(
        default_factory=list
    )
    remediation_cost_per_hallucination: float = 50.0
    premium_per_query: float = 0.0
    hallucination_rate_difference: float = 0.0


class BreakEvenCalculator:

    def __init__(self, remediation_cost_per_hallucination: float = 50.0):
        if remediation_cost_per_hallucination < 0:
            raise ValueError("Costs and rates must be non-negative")
        self.remediation_cost = remediation_cost_per_hallucination

    def compute_break_even(
        self,
        cost_per_query_a: float,
        cost_per_query_b: float,
        hallucination_rate_a: float,
        hallucination_rate_b: float,
    ) -> BreakEvenResult:
        if any(
            v < 0
            for v in [
                cost_per_query_a,
                cost_per_query_b,
                hallucination_rate_a,
                hallucination_rate_b,
            ]
        ):
            raise ValueError("Costs and rates must be non-negative")

        premium_per_query = cost_per_query_b - cost_per_query_a
        hallucination_rate_diff = hallucination_rate_a - hallucination_rate_b

        chart_volumes = [
            0, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000
        ]
        chart_data = [
            (
                v,
                v * (cost_per_query_a + hallucination_rate_a * self.remediation_cost),
                v * (cost_per_query_b + hallucination_rate_b * self.remediation_cost),
            )
            for v in chart_volumes
        ]

        savings_volumes = {"1K": 1000, "10K": 10000, "100K": 100000, "1M": 1000000}

        # No hallucination advantage
        if hallucination_rate_diff <= 0:
            monthly_savings = {
                k: v * (hallucination_rate_diff * self.remediation_cost - premium_per_query)
                for k, v in savings_volumes.items()
            }
            return BreakEvenResult(
                break_even_volume=None,
                has_break_even=False,
                immediate_advantage=False,
                monthly_savings_at_volume=monthly_savings,
                cost_crossover_chart_data=chart_data,
                remediation_cost_per_hallucination=self.remediation_cost,
                premium_per_query=premium_per_query,
                hallucination_rate_difference=hallucination_rate_diff,
            )

        # Immediate advantage (model B is cheaper or equal AND better)
        if premium_per_query <= 0:
            monthly_savings = {
                k: v * (hallucination_rate_diff * self.remediation_cost - premium_per_query)
                for k, v in savings_volumes.items()
            }
            return BreakEvenResult(
                break_even_volume=0,
                has_break_even=True,
                immediate_advantage=True,
                monthly_savings_at_volume=monthly_savings,
                cost_crossover_chart_data=chart_data,
                remediation_cost_per_hallucination=self.remediation_cost,
                premium_per_query=premium_per_query,
                hallucination_rate_difference=hallucination_rate_diff,
            )

        # Normal case: compute break-even
        savings_per_query = hallucination_rate_diff * self.remediation_cost
        break_even = ceil(premium_per_query / savings_per_query)

        monthly_savings = {
            k: v * (savings_per_query - premium_per_query)
            for k, v in savings_volumes.items()
        }

        return BreakEvenResult(
            break_even_volume=break_even,
            has_break_even=True,
            immediate_advantage=False,
            monthly_savings_at_volume=monthly_savings,
            cost_crossover_chart_data=chart_data,
            remediation_cost_per_hallucination=self.remediation_cost,
            premium_per_query=premium_per_query,
            hallucination_rate_difference=hallucination_rate_diff,
        )
