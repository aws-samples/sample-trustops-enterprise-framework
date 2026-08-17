"""
Financial dataset template.

Requirement 2.19: Financial domain-specific template

WARNING: For healthcare/financial/legal applications, AI outputs from models
trained with this template MUST NOT be used as sole decision-makers.
All outputs require review by licensed professionals.
Enable Amazon Bedrock Guardrails for content filtering in production.

This template defines dataset structure and validation only. It does not make
a model suitable for regulated financial advice or automated credit, risk, or
trading decisions. See SECURITY.md for the responsible-AI guidance.
"""

from typing import Optional

from src.data_models.dataset import DatasetTaskType
from .base_template import (
    DatasetTemplate,
    FieldDefinition,
    PIISensitivity,
    QualityThresholds,
    ValidationRule,
)


class FinancialTemplate(DatasetTemplate):
    """Template for financial analysis and risk assessment datasets."""

    def _initialize_template(self) -> None:
        """Initialize financial template."""
        self._task_type = DatasetTaskType.TEXT_GENERATION

        # Define fields
        self._fields = [
            FieldDefinition(
                name="analysis_id",
                data_type="str",
                required=True,
                description="Unique analysis identifier",
                pii_sensitivity=PIISensitivity.NONE,
                min_length=1,
                max_length=100,
            ),
            FieldDefinition(
                name="analysis_type",
                data_type="str",
                required=True,
                description="Type of financial analysis",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=[
                    "risk_assessment",
                    "investment_analysis",
                    "credit_evaluation",
                    "fraud_detection",
                    "market_analysis",
                    "portfolio_review",
                    "compliance",
                    "transaction_analysis",
                ],
            ),
            FieldDefinition(
                name="financial_data",
                data_type="str",
                required=True,
                description="Financial data or scenario description",
                pii_sensitivity=PIISensitivity.HIGH,
                min_length=50,
                max_length=10000,
            ),
            FieldDefinition(
                name="prompt",
                data_type="str",
                required=True,
                description="Analysis prompt or question",
                pii_sensitivity=PIISensitivity.LOW,
                min_length=10,
                max_length=2000,
            ),
            FieldDefinition(
                name="expected_analysis",
                data_type="str",
                required=True,
                description="Expected financial analysis or recommendation",
                pii_sensitivity=PIISensitivity.MEDIUM,
                min_length=50,
                max_length=10000,
            ),
            FieldDefinition(
                name="risk_level",
                data_type="str",
                required=False,
                description="Risk level assessment",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=["low", "medium", "high", "critical"],
            ),
            FieldDefinition(
                name="asset_class",
                data_type="str",
                required=False,
                description="Asset class being analyzed",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=[
                    "equities",
                    "fixed_income",
                    "commodities",
                    "real_estate",
                    "derivatives",
                    "cash",
                    "alternative",
                ],
            ),
            FieldDefinition(
                name="regulatory_framework",
                data_type="str",
                required=False,
                description="Applicable regulatory framework",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=[
                    "SEC",
                    "FINRA",
                    "Basel_III",
                    "MiFID_II",
                    "Dodd_Frank",
                    "SOX",
                    "AML",
                ],
            ),
            FieldDefinition(
                name="time_horizon",
                data_type="str",
                required=False,
                description="Analysis time horizon",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=["short_term", "medium_term", "long_term"],
            ),
            FieldDefinition(
                name="confidence_level",
                data_type="str",
                required=False,
                description="Confidence level of the analysis",
                pii_sensitivity=PIISensitivity.NONE,
                allowed_values=["low", "medium", "high"],
            ),
        ]

        # Define validation rules
        self._validation_rules = [
            ValidationRule(
                name="financial_terminology",
                description="Should contain appropriate financial terminology",
                validator=self._validate_financial_terminology,
            ),
            ValidationRule(
                name="quantitative_analysis",
                description="Analysis should include quantitative elements",
                validator=self._validate_quantitative_analysis,
            ),
            ValidationRule(
                name="risk_disclosure",
                description="Should include appropriate risk disclosures",
                validator=self._validate_risk_disclosure,
            ),
        ]

        # Set quality thresholds
        self._quality_thresholds = QualityThresholds(
            min_completeness=0.95,
            min_diversity=0.75,
            min_balance=0.65,
            min_examples=150,
        )

        # Example data
        self._example_data = [
            {
                "analysis_id": "FIN-2024-001",
                "analysis_type": "risk_assessment",
                "financial_data": "Company XYZ has the following financial metrics: Revenue $500M (up 15% YoY), EBITDA margin 22%, Debt-to-Equity ratio 1.8, Current ratio 1.2, Interest coverage ratio 4.5x. The company operates in the technology sector with 60% revenue from subscription services. Recent expansion into international markets (25% of revenue). Cash position $80M.",
                "prompt": "Assess the financial risk profile of this company and provide a risk rating with justification.",
                "expected_analysis": "Risk Assessment: MEDIUM-HIGH. Positive factors: Strong revenue growth (15% YoY) indicates market demand. Healthy EBITDA margin (22%) shows operational efficiency. Interest coverage ratio (4.5x) demonstrates ability to service debt. Subscription revenue (60%) provides recurring income stability. Risk factors: Elevated Debt-to-Equity ratio (1.8) indicates high leverage, increasing financial risk. Current ratio (1.2) is adequate but not strong, suggesting potential liquidity constraints. International expansion (25% revenue) introduces currency and geopolitical risks. Technology sector exposure brings competitive and disruption risks. Recommendation: Monitor debt levels closely. Maintain cash reserves for debt service. Diversify revenue streams to reduce concentration risk. Consider deleveraging if market conditions deteriorate.",
                "risk_level": "medium",
                "asset_class": "equities",
                "time_horizon": "medium_term",
                "confidence_level": "high",
            },
            {
                "analysis_id": "FIN-2024-002",
                "analysis_type": "investment_analysis",
                "financial_data": "Municipal bond: AAA-rated, 10-year maturity, 3.5% coupon, issued by State Infrastructure Authority. Tax-exempt for federal income tax. Current market price: $102 (par value $100). Yield to maturity: 3.2%. Comparable Treasury yield: 4.0%.",
                "prompt": "Evaluate this municipal bond as an investment for a high-income investor in the 35% tax bracket.",
                "expected_analysis": "Investment Analysis: FAVORABLE. Tax-equivalent yield calculation: 3.2% / (1 - 0.35) = 4.92% tax-equivalent yield, which exceeds the comparable Treasury yield of 4.0% by 92 basis points. This represents attractive value for the high-income investor. Credit quality: AAA rating indicates minimal default risk. The State Infrastructure Authority typically has strong credit fundamentals backed by essential infrastructure revenues. Price consideration: Trading at $102 (2% premium to par) reflects strong demand and quality. The premium is reasonable given the credit quality and tax benefits. Risks: Interest rate risk over 10-year duration. Potential tax law changes affecting municipal bond exemption. State fiscal health should be monitored. Recommendation: SUITABLE for high-income investors seeking tax-advantaged income. The tax-equivalent yield premium justifies the investment. Consider as part of diversified fixed-income allocation. Monitor state fiscal conditions and interest rate environment.",
                "risk_level": "low",
                "asset_class": "fixed_income",
                "time_horizon": "long_term",
                "confidence_level": "high",
            },
        ]

    def _validate_financial_terminology(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate presence of financial terminology."""
        if "expected_analysis" not in example:
            return True, None

        analysis = example["expected_analysis"].lower()

        # Check for financial terms
        financial_terms = [
            "risk", "return", "yield", "revenue", "profit", "loss", "margin",
            "ratio", "valuation", "asset", "liability", "equity", "cash flow",
            "investment", "portfolio", "diversification"
        ]

        found_terms = sum(1 for term in financial_terms if term in analysis)
        if found_terms < 3:
            return False, "Analysis lacks sufficient financial terminology"

        return True, None

    def _validate_quantitative_analysis(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate analysis includes quantitative elements."""
        if "expected_analysis" not in example:
            return True, None

        analysis = example["expected_analysis"]

        # Check for numbers, percentages, or ratios
        import re
        has_numbers = bool(re.search(r'\d+\.?\d*%?', analysis))
        has_currency = bool(re.search(r'\$\d+', analysis))

        if not (has_numbers or has_currency):
            return False, "Financial analysis should include quantitative data"

        return True, None

    def _validate_risk_disclosure(self, example: dict) -> tuple[bool, Optional[str]]:
        """Validate appropriate risk disclosures."""
        if "expected_analysis" not in example:
            return True, None

        analysis = example["expected_analysis"].lower()

        # Check for risk-related language
        risk_terms = ["risk", "risks", "caution", "consider", "monitor", "potential"]
        found_risk_terms = sum(1 for term in risk_terms if term in analysis)

        if found_risk_terms < 1:
            return False, "Financial analysis should include risk considerations"

        return True, None
