"""
Unit tests for synthetic data generator.

Tests the SyntheticDataGenerator class and its methods for generating
paraphrased and varied examples for dataset augmentation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.datasets.synthetic_generator import (
    SyntheticDataGenerator,
    generate_synthetic_data,
)
from src.adapters.base_adapter import InferenceRequest, InferenceResponse
from src.data_models.dataset import DatasetTaskType


@pytest.fixture
def mock_inference_client():
    """Create a mock InferenceClient."""
    client = MagicMock()
    client.invoke = AsyncMock()
    return client


@pytest.fixture
def sample_examples():
    """Create sample dataset examples."""
    return [
        {
            "prompt": "What is the capital of France?",
            "completion": "The capital of France is Paris.",
            "category": "geography",
        },
        {
            "prompt": "Explain photosynthesis.",
            "completion": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "category": "science",
        },
    ]


@pytest.fixture
def qa_examples():
    """Create QA task examples."""
    return [
        {
            "question": "What is Python?",
            "answer": "Python is a high-level programming language.",
        },
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI.",
        },
    ]


class TestSyntheticDataGenerator:
    """Test suite for SyntheticDataGenerator."""

    def test_init(self, mock_inference_client):
        """Test generator initialization."""
        generator = SyntheticDataGenerator(
            mock_inference_client, model_id="test-model"
        )

        assert generator.inference_client == mock_inference_client
        assert generator.model_id == "test-model"

    def test_init_default_model(self, mock_inference_client):
        """Test generator initialization with default model."""
        generator = SyntheticDataGenerator(mock_inference_client)

        assert generator.model_id == "anthropic.claude-3-haiku-20240307-v1:0"

    @pytest.mark.asyncio
    async def test_generate_synthetic_empty_examples(self, mock_inference_client):
        """Test that empty examples raises ValueError."""
        generator = SyntheticDataGenerator(mock_inference_client)

        with pytest.raises(ValueError, match="Cannot generate synthetic data from empty dataset"):
            await generator.generate_synthetic([])

    @pytest.mark.asyncio
    async def test_generate_synthetic_invalid_factor(
        self, mock_inference_client, sample_examples
    ):
        """Test that augmentation factor < 1.0 raises ValueError."""
        generator = SyntheticDataGenerator(mock_inference_client)

        with pytest.raises(ValueError, match="Augmentation factor must be >= 1.0"):
            await generator.generate_synthetic(sample_examples, augmentation_factor=0.5)

    @pytest.mark.asyncio
    async def test_generate_synthetic_factor_one(
        self, mock_inference_client, sample_examples
    ):
        """Test that augmentation factor of 1.0 returns empty list."""
        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator.generate_synthetic(
            sample_examples, augmentation_factor=1.0
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_generate_synthetic_paraphrase(
        self, mock_inference_client, sample_examples
    ):
        """Test paraphrase strategy generates synthetic examples."""
        # Mock LLM response
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="What is France's capital city?",
            input_tokens=20,
            output_tokens=10,
            latency_ms=100,
            model_id="test-model",
            finish_reason="stop",
        )

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator.generate_synthetic(
            sample_examples, augmentation_factor=1.5, strategy="paraphrase"
        )

        # Should generate 1 synthetic example (50% of 2 = 1)
        assert len(result) == 1
        assert result[0]["_synthetic"] is True
        assert result[0]["_augmentation_strategy"] == "paraphrase"
        assert "prompt" in result[0]

    @pytest.mark.asyncio
    async def test_generate_synthetic_diverse(
        self, mock_inference_client, sample_examples
    ):
        """Test diverse strategy generates varied examples."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Could you tell me which city serves as France's capital?",
            input_tokens=20,
            output_tokens=15,
            latency_ms=100,
            model_id="test-model",
            finish_reason="stop",
        )

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator.generate_synthetic(
            sample_examples, augmentation_factor=2.0, strategy="diverse"
        )

        # Should generate 2 synthetic examples (100% of 2 = 2)
        assert len(result) == 2
        assert all(ex["_synthetic"] is True for ex in result)
        assert all(ex["_augmentation_strategy"] == "diverse" for ex in result)

    @pytest.mark.asyncio
    async def test_generate_synthetic_creative(
        self, mock_inference_client, sample_examples
    ):
        """Test creative strategy generates creative variations."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="If you were planning a trip to France, which city would be the seat of government?",
            input_tokens=20,
            output_tokens=20,
            latency_ms=100,
            model_id="test-model",
            finish_reason="stop",
        )

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator.generate_synthetic(
            sample_examples, augmentation_factor=1.5, strategy="creative"
        )

        assert len(result) == 1
        assert result[0]["_synthetic"] is True
        assert result[0]["_augmentation_strategy"] == "creative"

    @pytest.mark.asyncio
    async def test_generate_synthetic_unknown_strategy(
        self, mock_inference_client, sample_examples
    ):
        """Test that unknown strategy raises ValueError."""
        generator = SyntheticDataGenerator(mock_inference_client)

        with pytest.raises(ValueError, match="Unknown strategy"):
            await generator.generate_synthetic(
                sample_examples, strategy="unknown"
            )

    @pytest.mark.asyncio
    async def test_paraphrase_example_with_prompt(
        self, mock_inference_client, sample_examples
    ):
        """Test paraphrasing an example with prompt field."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="What is France's capital city?",
            input_tokens=20,
            output_tokens=10,
            latency_ms=100,
            model_id="test-model",
            finish_reason="stop",
        )

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator._paraphrase_example(sample_examples[0])

        assert result is not None
        assert result["prompt"] == "What is France's capital city?"
        assert result["_synthetic"] is True
        assert result["_augmentation_strategy"] == "paraphrase"
        assert result["category"] == "geography"  # Preserved

    @pytest.mark.asyncio
    async def test_paraphrase_example_with_question(
        self, mock_inference_client, qa_examples
    ):
        """Test paraphrasing an example with question field."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Can you explain what Python is?",
            input_tokens=20,
            output_tokens=10,
            latency_ms=100,
            model_id="test-model",
            finish_reason="stop",
        )

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator._paraphrase_example(qa_examples[0])

        assert result is not None
        assert result["question"] == "Can you explain what Python is?"
        assert result["_synthetic"] is True

    @pytest.mark.asyncio
    async def test_paraphrase_example_no_prompt(self, mock_inference_client):
        """Test paraphrasing an example without prompt field returns None."""
        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator._paraphrase_example({"category": "test"})

        assert result is None

    @pytest.mark.asyncio
    async def test_paraphrase_example_with_completion(
        self, mock_inference_client, sample_examples
    ):
        """Test paraphrasing both prompt and completion."""
        # Mock two responses: one for prompt, one for completion
        mock_inference_client.invoke.side_effect = [
            InferenceResponse(
                text="What is France's capital city?",
                input_tokens=20,
                output_tokens=10,
                latency_ms=100,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="Paris is the capital of France.",
                input_tokens=20,
                output_tokens=10,
                latency_ms=100,
                model_id="test-model",
                finish_reason="stop",
            ),
        ]

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator._paraphrase_example(sample_examples[0])

        assert result is not None
        assert result["prompt"] == "What is France's capital city?"
        assert result["completion"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_paraphrase_example_llm_error(
        self, mock_inference_client, sample_examples
    ):
        """Test that LLM errors are handled gracefully."""
        mock_inference_client.invoke.side_effect = RuntimeError("LLM error")

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator._paraphrase_example(sample_examples[0])

        assert result is None

    @pytest.mark.asyncio
    async def test_create_diverse_variation(
        self, mock_inference_client, sample_examples
    ):
        """Test creating diverse variation."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Could you tell me the capital of France?",
            input_tokens=20,
            output_tokens=10,
            latency_ms=100,
            model_id="test-model",
            finish_reason="stop",
        )

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator._create_diverse_variation(sample_examples[0])

        assert result is not None
        assert result["prompt"] == "Could you tell me the capital of France?"
        assert result["_synthetic"] is True
        assert result["_augmentation_strategy"] == "diverse"

    @pytest.mark.asyncio
    async def test_create_creative_variation(
        self, mock_inference_client, sample_examples
    ):
        """Test creating creative variation."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="If you were traveling to France, which city would you visit as the capital?",
            input_tokens=20,
            output_tokens=20,
            latency_ms=100,
            model_id="test-model",
            finish_reason="stop",
        )

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator._create_creative_variation(sample_examples[0])

        assert result is not None
        assert "capital" in result["prompt"].lower()
        assert result["_synthetic"] is True
        assert result["_augmentation_strategy"] == "creative"

    def test_extract_field_found(self, sample_examples):
        """Test extracting an existing field."""
        generator = SyntheticDataGenerator(MagicMock())

        result = generator._extract_field(
            sample_examples[0], ["prompt", "question"]
        )

        assert result == "What is the capital of France?"

    def test_extract_field_not_found(self, sample_examples):
        """Test extracting a non-existent field returns None."""
        generator = SyntheticDataGenerator(MagicMock())

        result = generator._extract_field(
            sample_examples[0], ["missing", "notfound"]
        )

        assert result is None

    def test_extract_field_empty_value(self):
        """Test extracting an empty field returns None."""
        generator = SyntheticDataGenerator(MagicMock())

        result = generator._extract_field(
            {"prompt": ""}, ["prompt"]
        )

        assert result is None

    def test_build_paraphrase_prompt_no_task_type(self):
        """Test building paraphrase prompt without task type."""
        generator = SyntheticDataGenerator(MagicMock())

        prompt = generator._build_paraphrase_prompt()

        assert "paraphrases text" in prompt
        assert "exact meaning" in prompt

    def test_build_paraphrase_prompt_qa_task(self):
        """Test building paraphrase prompt for QA task."""
        generator = SyntheticDataGenerator(MagicMock())

        prompt = generator._build_paraphrase_prompt(DatasetTaskType.QA)

        assert "questions" in prompt

    def test_build_paraphrase_prompt_summarization_task(self):
        """Test building paraphrase prompt for summarization task."""
        generator = SyntheticDataGenerator(MagicMock())

        prompt = generator._build_paraphrase_prompt(DatasetTaskType.SUMMARIZATION)

        assert "summarize" in prompt

    def test_build_diverse_prompt(self):
        """Test building diverse variation prompt."""
        generator = SyntheticDataGenerator(MagicMock())

        prompt = generator._build_diverse_prompt()

        assert "diverse variations" in prompt
        assert "perspective" in prompt or "formality" in prompt

    def test_build_creative_prompt(self):
        """Test building creative variation prompt."""
        generator = SyntheticDataGenerator(MagicMock())

        prompt = generator._build_creative_prompt()

        assert "creative variations" in prompt
        assert "scenarios" in prompt or "contexts" in prompt

    @pytest.mark.asyncio
    async def test_generate_synthetic_cycles_through_examples(
        self, mock_inference_client, sample_examples
    ):
        """Test that generation cycles through examples when factor > 2.0."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Paraphrased text",
            input_tokens=20,
            output_tokens=10,
            latency_ms=100,
            model_id="test-model",
            finish_reason="stop",
        )

        generator = SyntheticDataGenerator(mock_inference_client)

        # Generate 3x the original (6 examples from 2 originals)
        result = await generator.generate_synthetic(
            sample_examples, augmentation_factor=3.0, strategy="paraphrase"
        )

        # Should generate 4 synthetic examples (200% of 2 = 4)
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_generate_synthetic_handles_partial_failures(
        self, mock_inference_client, sample_examples
    ):
        """Test that partial failures don't stop generation."""
        # First call succeeds, second fails
        mock_inference_client.invoke.side_effect = [
            InferenceResponse(
                text="Paraphrased text",
                input_tokens=20,
                output_tokens=10,
                latency_ms=100,
                model_id="test-model",
                finish_reason="stop",
            ),
            RuntimeError("LLM error"),
        ]

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator.generate_synthetic(
            sample_examples, augmentation_factor=2.0, strategy="paraphrase"
        )

        # Should have 1 successful example (second failed)
        assert len(result) == 1
        assert result[0]["_synthetic"] is True

    @pytest.mark.asyncio
    async def test_generate_synthetic_with_task_type(
        self, mock_inference_client, sample_examples
    ):
        """Test generation with task type context."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Paraphrased question",
            input_tokens=20,
            output_tokens=10,
            latency_ms=100,
            model_id="test-model",
            finish_reason="stop",
        )

        generator = SyntheticDataGenerator(mock_inference_client)

        result = await generator.generate_synthetic(
            sample_examples,
            augmentation_factor=1.5,
            strategy="paraphrase",
            task_type=DatasetTaskType.QA,
        )

        assert len(result) == 1
        assert result[0]["_synthetic"] is True


class TestGenerateSyntheticDataFunction:
    """Test suite for generate_synthetic_data convenience function."""

    @pytest.mark.asyncio
    async def test_generate_synthetic_data_function(
        self, mock_inference_client, sample_examples
    ):
        """Test convenience function works correctly."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Paraphrased text",
            input_tokens=20,
            output_tokens=10,
            latency_ms=100,
            model_id="test-model",
            finish_reason="stop",
        )

        result = await generate_synthetic_data(
            sample_examples,
            mock_inference_client,
            augmentation_factor=1.5,
            strategy="paraphrase",
        )

        assert len(result) == 1
        assert result[0]["_synthetic"] is True

    @pytest.mark.asyncio
    async def test_generate_synthetic_data_with_custom_model(
        self, mock_inference_client, sample_examples
    ):
        """Test convenience function with custom model."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Paraphrased text",
            input_tokens=20,
            output_tokens=10,
            latency_ms=100,
            model_id="custom-model",
            finish_reason="stop",
        )

        result = await generate_synthetic_data(
            sample_examples,
            mock_inference_client,
            model_id="custom-model",
        )

        # Verify the custom model was used
        assert mock_inference_client.invoke.called
