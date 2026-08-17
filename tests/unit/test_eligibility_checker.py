"""
Unit tests for the fine-tuning eligibility checker.

Tests cover capability checks, status checks, fine-tuning support flag,
and combinations of multiple failing conditions.

Requirements: 4.1
"""

from src.data_models.model import (
    ModelCapability,
    ModelMetadata,
    ModelProvider,
    ModelStatus,
)
from src.fine_tuning.eligibility_checker import (
    EligibilityResult,
    check_fine_tuning_eligibility,
)


# --- Helpers ---

def _make_model(
    model_id: str = "model-001",
    capabilities: list[ModelCapability] | None = None,
    status: ModelStatus = ModelStatus.ACTIVE,
    fine_tuning_support: bool = True,
) -> ModelMetadata:
    if capabilities is None:
        capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.FINE_TUNABLE]
    return ModelMetadata(
        id=model_id,
        provider=ModelProvider.BEDROCK,
        name="Test Model",
        capabilities=capabilities,
        status=status,
        fine_tuning_support=fine_tuning_support,
    )


# --- EligibilityResult tests ---

class TestEligibilityResult:
    def test_eligible_result_defaults(self):
        result = EligibilityResult(is_eligible=True)
        assert result.is_eligible is True
        assert result.reasons == []

    def test_ineligible_result_with_reasons(self):
        result = EligibilityResult(is_eligible=False, reasons=["reason1", "reason2"])
        assert result.is_eligible is False
        assert len(result.reasons) == 2


# --- Fully eligible model ---

class TestEligibleModel:
    def test_model_with_all_conditions_met(self):
        model = _make_model()
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is True
        assert result.reasons == []

    def test_model_with_many_capabilities_including_fine_tunable(self):
        model = _make_model(
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CHAT,
                ModelCapability.FINE_TUNABLE,
                ModelCapability.COMPLETION,
            ]
        )
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is True
        assert result.reasons == []


# --- Missing FINE_TUNABLE capability ---

class TestMissingCapability:
    def test_no_fine_tunable_capability(self):
        model = _make_model(capabilities=[ModelCapability.TEXT_GENERATION])
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is False
        assert len(result.reasons) == 1
        assert "FINE_TUNABLE" in result.reasons[0]

    def test_empty_capabilities(self):
        model = _make_model(capabilities=[])
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is False
        assert any("FINE_TUNABLE" in r for r in result.reasons)

    def test_embedding_only_capability(self):
        model = _make_model(capabilities=[ModelCapability.EMBEDDING])
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is False
        assert any("FINE_TUNABLE" in r for r in result.reasons)


# --- Model not active ---

class TestModelNotActive:
    def test_inactive_model(self):
        model = _make_model(status=ModelStatus.INACTIVE)
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is False
        assert any("not active" in r for r in result.reasons)
        assert any("inactive" in r for r in result.reasons)

    def test_provisioning_model(self):
        model = _make_model(status=ModelStatus.PROVISIONING)
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is False
        assert any("provisioning" in r for r in result.reasons)

    def test_failed_model(self):
        model = _make_model(status=ModelStatus.FAILED)
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is False
        assert any("failed" in r for r in result.reasons)


# --- Fine-tuning support disabled ---

class TestFineTuningSupportDisabled:
    def test_fine_tuning_support_false(self):
        model = _make_model(fine_tuning_support=False)
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is False
        assert len(result.reasons) == 1
        assert "fine-tuning support" in result.reasons[0]


# --- Multiple failing conditions ---

class TestMultipleFailures:
    def test_no_capability_and_inactive(self):
        model = _make_model(
            capabilities=[ModelCapability.TEXT_GENERATION],
            status=ModelStatus.INACTIVE,
        )
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is False
        assert len(result.reasons) == 2

    def test_all_three_conditions_fail(self):
        model = _make_model(
            capabilities=[ModelCapability.EMBEDDING],
            status=ModelStatus.FAILED,
            fine_tuning_support=False,
        )
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is False
        assert len(result.reasons) == 3
        assert any("FINE_TUNABLE" in r for r in result.reasons)
        assert any("not active" in r for r in result.reasons)
        assert any("fine-tuning support" in r for r in result.reasons)

    def test_inactive_and_no_fine_tuning_support(self):
        model = _make_model(
            status=ModelStatus.PROVISIONING,
            fine_tuning_support=False,
        )
        result = check_fine_tuning_eligibility(model)
        assert result.is_eligible is False
        assert len(result.reasons) == 2


# --- Reason messages contain model ID ---

class TestReasonMessages:
    def test_reasons_include_model_id(self):
        model = _make_model(
            model_id="my-special-model",
            capabilities=[],
            status=ModelStatus.INACTIVE,
            fine_tuning_support=False,
        )
        result = check_fine_tuning_eligibility(model)
        for reason in result.reasons:
            assert "my-special-model" in reason
