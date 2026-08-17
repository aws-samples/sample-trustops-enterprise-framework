"""
Unit tests for the model-dataset compatibility checker.

Tests cover task type compatibility, format compatibility,
model availability checks, and structured validation results.

Requirements: 3.1, 3.2, 3.4
"""

from datetime import datetime

from src.data_models.dataset import (
    DatasetFormat,
    DatasetMetadata,
    DatasetTaskType,
    TokenStatistics,
)
from src.data_models.model import (
    ModelCapability,
    ModelMetadata,
    ModelProvider,
    ModelStatus,
)
from src.evaluation.compatibility_checker import (
    CompatibilityResult,
    check_model_dataset_compatibility,
    check_task_type_support,
)


# --- Fixtures ---

def _make_token_stats() -> TokenStatistics:
    return TokenStatistics(
        total_tokens=1000,
        min_tokens=10,
        max_tokens=200,
        avg_tokens=50.0,
        p95_tokens=180,
        prompt_tokens=600,
        completion_tokens=400,
    )


def _make_dataset(
    task_type: DatasetTaskType = DatasetTaskType.QA,
    fmt: DatasetFormat = DatasetFormat.JSONL,
    dataset_id: str = "ds-001",
) -> DatasetMetadata:
    return DatasetMetadata(
        id=dataset_id,
        name="Test Dataset",
        format=fmt,
        task_type=task_type,
        version="1.0",
        s3_uri="s3://bucket/datasets/test",
        row_count=100,
        token_stats=_make_token_stats(),
        created_at=datetime.now(),
        checksum="abc123",
    )


def _make_model(
    capabilities: list[ModelCapability] | None = None,
    status: ModelStatus = ModelStatus.ACTIVE,
    model_id: str = "model-001",
) -> ModelMetadata:
    if capabilities is None:
        capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.CHAT]
    return ModelMetadata(
        id=model_id,
        provider=ModelProvider.BEDROCK,
        name="Test Model",
        capabilities=capabilities,
        status=status,
    )


# --- CompatibilityResult tests ---

class TestCompatibilityResult:
    def test_compatible_result(self):
        result = CompatibilityResult(is_compatible=True, model_id="m1", dataset_id="d1")
        assert result.is_compatible is True
        assert result.reasons == []
        assert result.model_id == "m1"
        assert result.dataset_id == "d1"

    def test_incompatible_result(self):
        result = CompatibilityResult(
            is_compatible=False,
            reasons=["reason1", "reason2"],
            model_id="m1",
        )
        assert result.is_compatible is False
        assert len(result.reasons) == 2
        assert result.dataset_id is None


# --- Task type compatibility tests ---

class TestTaskTypeCompatibility:
    def test_text_gen_model_supports_qa(self):
        model = _make_model(capabilities=[ModelCapability.TEXT_GENERATION])
        dataset = _make_dataset(task_type=DatasetTaskType.QA)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True
        assert result.reasons == []

    def test_chat_model_supports_qa(self):
        model = _make_model(capabilities=[ModelCapability.CHAT])
        dataset = _make_dataset(task_type=DatasetTaskType.QA)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True

    def test_chat_model_supports_summarization(self):
        model = _make_model(capabilities=[ModelCapability.CHAT])
        dataset = _make_dataset(task_type=DatasetTaskType.SUMMARIZATION)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True

    def test_text_gen_model_supports_classification(self):
        model = _make_model(capabilities=[ModelCapability.TEXT_GENERATION])
        dataset = _make_dataset(task_type=DatasetTaskType.CLASSIFICATION)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True

    def test_completion_model_supports_text_generation(self):
        model = _make_model(capabilities=[ModelCapability.COMPLETION])
        dataset = _make_dataset(task_type=DatasetTaskType.TEXT_GENERATION)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True

    def test_chat_model_supports_chat(self):
        model = _make_model(capabilities=[ModelCapability.CHAT])
        dataset = _make_dataset(task_type=DatasetTaskType.CHAT)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True

    def test_embedding_only_model_fails_qa(self):
        model = _make_model(capabilities=[ModelCapability.EMBEDDING])
        dataset = _make_dataset(task_type=DatasetTaskType.QA)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is False
        assert len(result.reasons) == 1
        assert "does not support task type" in result.reasons[0]
        assert "qa" in result.reasons[0]

    def test_embedding_only_model_fails_chat(self):
        model = _make_model(capabilities=[ModelCapability.EMBEDDING])
        dataset = _make_dataset(task_type=DatasetTaskType.CHAT)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is False
        assert "chat" in result.reasons[0]

    def test_completion_model_fails_chat(self):
        """COMPLETION alone can't satisfy CHAT task."""
        model = _make_model(capabilities=[ModelCapability.COMPLETION])
        dataset = _make_dataset(task_type=DatasetTaskType.CHAT)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is False

    def test_no_capabilities_fails(self):
        model = _make_model(capabilities=[])
        dataset = _make_dataset(task_type=DatasetTaskType.QA)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is False
        assert "none" in result.reasons[0]

    def test_custom_task_type_flexible(self):
        """CUSTOM task type accepts text_generation, chat, or completion."""
        model = _make_model(capabilities=[ModelCapability.TEXT_GENERATION])
        dataset = _make_dataset(task_type=DatasetTaskType.CUSTOM)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True


# --- Format compatibility tests ---

class TestFormatCompatibility:
    def test_jsonl_format_supported(self):
        model = _make_model()
        dataset = _make_dataset(fmt=DatasetFormat.JSONL)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True

    def test_csv_format_supported(self):
        model = _make_model()
        dataset = _make_dataset(fmt=DatasetFormat.CSV)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True

    def test_parquet_format_supported(self):
        model = _make_model()
        dataset = _make_dataset(fmt=DatasetFormat.PARQUET)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True

    def test_huggingface_format_unsupported(self):
        model = _make_model()
        dataset = _make_dataset(fmt=DatasetFormat.HUGGINGFACE)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is False
        assert len(result.reasons) == 1
        assert "huggingface" in result.reasons[0]
        assert "not supported" in result.reasons[0]


# --- Model availability tests ---

class TestModelAvailability:
    def test_active_model_passes(self):
        model = _make_model(status=ModelStatus.ACTIVE)
        dataset = _make_dataset()
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is True

    def test_inactive_model_fails(self):
        model = _make_model(status=ModelStatus.INACTIVE)
        dataset = _make_dataset()
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is False
        assert any("not available" in r for r in result.reasons)
        assert any("inactive" in r for r in result.reasons)

    def test_provisioning_model_fails(self):
        model = _make_model(status=ModelStatus.PROVISIONING)
        dataset = _make_dataset()
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is False
        assert any("provisioning" in r for r in result.reasons)

    def test_failed_model_fails(self):
        model = _make_model(status=ModelStatus.FAILED)
        dataset = _make_dataset()
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is False
        assert any("failed" in r for r in result.reasons)


# --- Multiple incompatibilities ---

class TestMultipleIncompatibilities:
    def test_inactive_and_wrong_capability(self):
        model = _make_model(
            capabilities=[ModelCapability.EMBEDDING],
            status=ModelStatus.INACTIVE,
        )
        dataset = _make_dataset(task_type=DatasetTaskType.CHAT)
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is False
        assert len(result.reasons) == 2

    def test_inactive_wrong_capability_and_bad_format(self):
        model = _make_model(
            capabilities=[ModelCapability.EMBEDDING],
            status=ModelStatus.FAILED,
        )
        dataset = _make_dataset(
            task_type=DatasetTaskType.CHAT,
            fmt=DatasetFormat.HUGGINGFACE,
        )
        result = check_model_dataset_compatibility(model, dataset)
        assert result.is_compatible is False
        assert len(result.reasons) == 3


# --- Result metadata tests ---

class TestResultMetadata:
    def test_result_contains_model_id(self):
        model = _make_model(model_id="my-model")
        dataset = _make_dataset(dataset_id="my-dataset")
        result = check_model_dataset_compatibility(model, dataset)
        assert result.model_id == "my-model"
        assert result.dataset_id == "my-dataset"


# --- check_task_type_support tests ---

class TestCheckTaskTypeSupport:
    def test_supports_task_type(self):
        model = _make_model(capabilities=[ModelCapability.CHAT])
        result = check_task_type_support(model, DatasetTaskType.CHAT)
        assert result.is_compatible is True
        assert result.dataset_id is None

    def test_does_not_support_task_type(self):
        model = _make_model(capabilities=[ModelCapability.EMBEDDING])
        result = check_task_type_support(model, DatasetTaskType.CHAT)
        assert result.is_compatible is False

    def test_inactive_model_with_task_type_support(self):
        model = _make_model(
            capabilities=[ModelCapability.CHAT],
            status=ModelStatus.INACTIVE,
        )
        result = check_task_type_support(model, DatasetTaskType.CHAT)
        assert result.is_compatible is False
        assert any("not available" in r for r in result.reasons)
