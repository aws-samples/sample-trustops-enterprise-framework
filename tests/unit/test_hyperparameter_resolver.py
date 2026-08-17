"""
Unit tests for the hyperparameter defaults resolver.

Tests cover dataset scale classification, per-model-family defaults,
dataset-size-based selection, and fallback behavior.

Requirements: 4.5, 4.9
"""

from src.fine_tuning.hyperparameter_resolver import (
    DatasetScale,
    classify_dataset_scale,
    resolve_hyperparameters,
)


# --- Dataset scale classification ---


class TestClassifyDatasetScale:
    def test_small_dataset(self):
        assert classify_dataset_scale(100) == DatasetScale.SMALL
        assert classify_dataset_scale(999) == DatasetScale.SMALL

    def test_medium_dataset(self):
        assert classify_dataset_scale(1000) == DatasetScale.MEDIUM
        assert classify_dataset_scale(5000) == DatasetScale.MEDIUM
        assert classify_dataset_scale(10000) == DatasetScale.MEDIUM

    def test_large_dataset(self):
        assert classify_dataset_scale(10001) == DatasetScale.LARGE
        assert classify_dataset_scale(100000) == DatasetScale.LARGE

    def test_zero_is_small(self):
        assert classify_dataset_scale(0) == DatasetScale.SMALL

    def test_boundary_999(self):
        assert classify_dataset_scale(999) == DatasetScale.SMALL

    def test_boundary_1000(self):
        assert classify_dataset_scale(1000) == DatasetScale.MEDIUM

    def test_boundary_10000(self):
        assert classify_dataset_scale(10000) == DatasetScale.MEDIUM

    def test_boundary_10001(self):
        assert classify_dataset_scale(10001) == DatasetScale.LARGE


# --- Claude hyperparameters ---


class TestClaudeHyperparameters:
    def test_small_dataset(self):
        hp = resolve_hyperparameters("anthropic.claude-v2", 500)
        assert hp.learning_rate == 1e-5
        assert hp.epochs == 4
        assert hp.batch_size == 4
        assert hp.warmup_steps == 50
        assert hp.weight_decay == 0.01

    def test_medium_dataset(self):
        hp = resolve_hyperparameters("anthropic.claude-v2", 5000)
        assert hp.learning_rate == 5e-6
        assert hp.epochs == 3
        assert hp.batch_size == 8
        assert hp.warmup_steps == 100

    def test_large_dataset(self):
        hp = resolve_hyperparameters("anthropic.claude-v2", 50000)
        assert hp.learning_rate == 2e-6
        assert hp.epochs == 2
        assert hp.batch_size == 16
        assert hp.warmup_steps == 200
        assert hp.weight_decay == 0.005


# --- Titan hyperparameters ---


class TestTitanHyperparameters:
    def test_small_dataset(self):
        hp = resolve_hyperparameters("amazon.titan-text-v1", 500)
        assert hp.learning_rate == 2e-5
        assert hp.epochs == 5
        assert hp.batch_size == 4

    def test_medium_dataset(self):
        hp = resolve_hyperparameters("amazon.titan-text-v1", 5000)
        assert hp.learning_rate == 1e-5
        assert hp.epochs == 3

    def test_large_dataset(self):
        hp = resolve_hyperparameters("amazon.titan-text-v1", 50000)
        assert hp.learning_rate == 5e-6
        assert hp.epochs == 2
        assert hp.batch_size == 16


# --- Llama hyperparameters ---


class TestLlamaHyperparameters:
    def test_small_dataset(self):
        hp = resolve_hyperparameters("meta.llama3-8b-v1", 500)
        assert hp.learning_rate == 2e-5
        assert hp.epochs == 4

    def test_medium_dataset(self):
        hp = resolve_hyperparameters("meta.llama3-8b-v1", 5000)
        assert hp.learning_rate == 1e-5
        assert hp.epochs == 3

    def test_large_dataset(self):
        hp = resolve_hyperparameters("meta.llama3-8b-v1", 50000)
        assert hp.learning_rate == 5e-6
        assert hp.epochs == 2


# --- Unknown model fallback ---


class TestUnknownModelFallback:
    def test_unknown_model_small(self):
        hp = resolve_hyperparameters("unknown-model", 500)
        assert hp.learning_rate == 1e-5
        assert hp.epochs == 4
        assert hp.batch_size == 4

    def test_unknown_model_medium(self):
        hp = resolve_hyperparameters("unknown-model", 5000)
        assert hp.learning_rate == 5e-6
        assert hp.epochs == 3

    def test_unknown_model_large(self):
        hp = resolve_hyperparameters("unknown-model", 50000)
        assert hp.learning_rate == 2e-6
        assert hp.epochs == 2


# --- Return type ---


class TestReturnType:
    def test_returns_hyperparameter_config(self):
        hp = resolve_hyperparameters("anthropic.claude-v2", 500)
        # Verify it's a proper HyperparameterConfig with all expected fields
        assert hasattr(hp, "learning_rate")
        assert hasattr(hp, "epochs")
        assert hasattr(hp, "batch_size")
        assert hasattr(hp, "warmup_steps")
        assert hasattr(hp, "weight_decay")
        assert hasattr(hp, "max_seq_length")

    def test_larger_datasets_get_lower_learning_rate(self):
        hp_small = resolve_hyperparameters("anthropic.claude-v2", 100)
        hp_large = resolve_hyperparameters("anthropic.claude-v2", 50000)
        assert hp_large.learning_rate < hp_small.learning_rate

    def test_larger_datasets_get_fewer_epochs(self):
        hp_small = resolve_hyperparameters("anthropic.claude-v2", 100)
        hp_large = resolve_hyperparameters("anthropic.claude-v2", 50000)
        assert hp_large.epochs < hp_small.epochs

    def test_larger_datasets_get_bigger_batch_size(self):
        hp_small = resolve_hyperparameters("anthropic.claude-v2", 100)
        hp_large = resolve_hyperparameters("anthropic.claude-v2", 50000)
        assert hp_large.batch_size > hp_small.batch_size
