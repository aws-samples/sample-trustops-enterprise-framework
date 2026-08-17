"""
Unit tests for dataset splitter.

Tests the DatasetSplitter class for train/validation/test splitting
with random and stratified sampling.

Requirement 2.13: Test train/validation/test splitter
"""

import pytest

from src.datasets.splitter import DatasetSplitter, split_dataset


class TestDatasetSplitter:
    """Test suite for DatasetSplitter class."""

    def test_basic_split_default_ratios(self):
        """Test basic split with default 80/10/10 ratios."""
        examples = [{"id": i, "text": f"example {i}"} for i in range(100)]

        splitter = DatasetSplitter(random_seed=42)
        train, val, test = splitter.split(examples)

        # Check sizes are approximately correct
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

        # Check total count preserved
        assert len(train) + len(val) + len(test) == 100

    def test_custom_ratios(self):
        """Test split with custom ratios."""
        examples = [{"id": i} for i in range(100)]

        splitter = DatasetSplitter(random_seed=42)
        train, val, test = splitter.split(
            examples, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15
        )

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_split_preserves_all_examples(self):
        """Test that split preserves all examples without duplication."""
        examples = [{"id": i, "value": i * 2} for i in range(50)]

        splitter = DatasetSplitter(random_seed=42)
        train, val, test = splitter.split(examples)

        # Collect all IDs from splits
        all_ids = set()
        for split in [train, val, test]:
            for ex in split:
                all_ids.add(ex["id"])

        # Check all original IDs are present exactly once
        assert len(all_ids) == 50
        assert all_ids == set(range(50))

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same split."""
        examples = [{"id": i} for i in range(100)]

        splitter1 = DatasetSplitter(random_seed=123)
        train1, val1, test1 = splitter1.split(examples)

        splitter2 = DatasetSplitter(random_seed=123)
        train2, val2, test2 = splitter2.split(examples)

        # Check splits are identical
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_different_seeds_produce_different_splits(self):
        """Test that different seeds produce different splits."""
        examples = [{"id": i} for i in range(100)]

        splitter1 = DatasetSplitter(random_seed=42)
        train1, _, _ = splitter1.split(examples)

        splitter2 = DatasetSplitter(random_seed=99)
        train2, _, _ = splitter2.split(examples)

        # Splits should be different (very unlikely to be identical)
        assert train1 != train2

    def test_stratified_split_maintains_category_balance(self):
        """Test stratified split maintains category proportions."""
        # Create dataset with 3 categories: 60% A, 30% B, 10% C
        examples = (
            [{"id": i, "category": "A"} for i in range(60)]
            + [{"id": i + 60, "category": "B"} for i in range(30)]
            + [{"id": i + 90, "category": "C"} for i in range(10)]
        )

        splitter = DatasetSplitter(random_seed=42)
        train, val, test = splitter.split(examples, stratify_by="category")

        # Count categories in each split
        def count_categories(split):
            counts = {"A": 0, "B": 0, "C": 0}
            for ex in split:
                counts[ex["category"]] += 1
            return counts

        train_counts = count_categories(train)
        val_counts = count_categories(val)
        test_counts = count_categories(test)

        # Check proportions are maintained (approximately)
        # Train should have ~48 A, ~24 B, ~8 C
        assert 46 <= train_counts["A"] <= 50
        assert 22 <= train_counts["B"] <= 26
        assert 6 <= train_counts["C"] <= 10

        # Validation should have ~6 A, ~3 B, ~1 C
        assert 4 <= val_counts["A"] <= 8
        assert 2 <= val_counts["B"] <= 4
        assert 0 <= val_counts["C"] <= 2

        # Test should have ~6 A, ~3 B, ~1 C
        assert 4 <= test_counts["A"] <= 8
        assert 2 <= test_counts["B"] <= 4
        assert 0 <= test_counts["C"] <= 2

    def test_stratified_split_with_label_field(self):
        """Test stratified split works with 'label' field."""
        examples = [
            {"id": i, "label": "positive" if i % 2 == 0 else "negative"}
            for i in range(100)
        ]

        splitter = DatasetSplitter(random_seed=42)
        train, val, test = splitter.split(examples, stratify_by="label")

        # Count labels in train split
        train_positive = sum(1 for ex in train if ex["label"] == "positive")
        train_negative = sum(1 for ex in train if ex["label"] == "negative")

        # Should be approximately 50/50
        assert 38 <= train_positive <= 42
        assert 38 <= train_negative <= 42

    def test_empty_dataset_raises_error(self):
        """Test that empty dataset raises ValueError."""
        splitter = DatasetSplitter()

        with pytest.raises(ValueError, match="Cannot split empty dataset"):
            splitter.split([])

    def test_invalid_ratios_sum_raises_error(self):
        """Test that ratios not summing to 1.0 raises ValueError."""
        examples = [{"id": i} for i in range(10)]
        splitter = DatasetSplitter()

        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            splitter.split(examples, train_ratio=0.5, validation_ratio=0.3, test_ratio=0.3)

    def test_negative_ratio_raises_error(self):
        """Test that negative ratios raise ValueError."""
        examples = [{"id": i} for i in range(10)]
        splitter = DatasetSplitter()

        with pytest.raises(ValueError, match="All ratios must be non-negative"):
            splitter.split(examples, train_ratio=-0.1, validation_ratio=0.6, test_ratio=0.5)

    def test_all_zero_ratios_raises_error(self):
        """Test that all zero ratios raise ValueError."""
        examples = [{"id": i} for i in range(10)]
        splitter = DatasetSplitter()

        with pytest.raises(ValueError, match="At least one ratio must be positive"):
            splitter.split(examples, train_ratio=0.0, validation_ratio=0.0, test_ratio=0.0)

    def test_stratify_by_missing_field_raises_error(self):
        """Test that stratifying by missing field raises ValueError."""
        examples = [{"id": i, "text": f"example {i}"} for i in range(10)]
        splitter = DatasetSplitter()

        with pytest.raises(ValueError, match="Stratification field.*not found"):
            splitter.split(examples, stratify_by="category")

    def test_split_with_only_train(self):
        """Test split with 100% train ratio."""
        examples = [{"id": i} for i in range(100)]
        splitter = DatasetSplitter(random_seed=42)

        train, val, test = splitter.split(
            examples, train_ratio=1.0, validation_ratio=0.0, test_ratio=0.0
        )

        assert len(train) == 100
        assert len(val) == 0
        assert len(test) == 0

    def test_split_with_only_validation(self):
        """Test split with 100% validation ratio."""
        examples = [{"id": i} for i in range(100)]
        splitter = DatasetSplitter(random_seed=42)

        train, val, test = splitter.split(
            examples, train_ratio=0.0, validation_ratio=1.0, test_ratio=0.0
        )

        assert len(train) == 0
        assert len(val) == 100
        assert len(test) == 0

    def test_split_with_only_test(self):
        """Test split with 100% test ratio."""
        examples = [{"id": i} for i in range(100)]
        splitter = DatasetSplitter(random_seed=42)

        train, val, test = splitter.split(
            examples, train_ratio=0.0, validation_ratio=0.0, test_ratio=1.0
        )

        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 100

    def test_small_dataset_split(self):
        """Test split with very small dataset."""
        examples = [{"id": i} for i in range(5)]
        splitter = DatasetSplitter(random_seed=42)

        train, val, test = splitter.split(examples)

        # With 5 examples and 80/10/10: train=4, val=0, test=0 (due to int rounding)
        assert len(train) + len(val) + len(test) == 5

    def test_get_split_statistics(self):
        """Test split statistics calculation."""
        examples = [{"id": i, "category": "A" if i < 50 else "B"} for i in range(100)]
        splitter = DatasetSplitter(random_seed=42)

        train, val, test = splitter.split(examples, stratify_by="category")
        stats = splitter.get_split_statistics(train, val, test, stratify_by="category")

        assert stats["total_examples"] == 100
        assert stats["train_size"] == len(train)
        assert stats["validation_size"] == len(val)
        assert stats["test_size"] == len(test)
        assert 0.79 <= stats["train_ratio"] <= 0.81
        assert 0.09 <= stats["validation_ratio"] <= 0.11
        assert 0.09 <= stats["test_ratio"] <= 0.11

        # Check category distributions are present
        assert "category_distributions" in stats
        assert "train" in stats["category_distributions"]
        assert "validation" in stats["category_distributions"]
        assert "test" in stats["category_distributions"]

    def test_get_split_statistics_empty_splits(self):
        """Test statistics with empty splits."""
        splitter = DatasetSplitter()
        stats = splitter.get_split_statistics([], [], [])

        assert stats["total_examples"] == 0
        assert stats["train_size"] == 0
        assert stats["validation_size"] == 0
        assert stats["test_size"] == 0

    def test_stratified_split_with_alternative_field_names(self):
        """Test that stratification works with alternative field names."""
        # Use 'class' field instead of 'category'
        examples = [{"id": i, "class": "A" if i < 50 else "B"} for i in range(100)]
        splitter = DatasetSplitter(random_seed=42)

        # Request stratification by 'category' but examples have 'class'
        # Should fall back to 'class' field
        train, val, test = splitter.split(examples, stratify_by="category")

        # Should successfully split
        assert len(train) + len(val) + len(test) == 100

    def test_convenience_function(self):
        """Test the convenience split_dataset function."""
        examples = [{"id": i, "category": "A" if i % 2 == 0 else "B"} for i in range(100)]

        train, val, test = split_dataset(
            examples,
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            stratify_by="category",
            random_seed=42,
        )

        # With stratified sampling, exact counts may vary slightly due to rounding per category
        assert 68 <= len(train) <= 72
        assert 13 <= len(val) <= 17
        assert 13 <= len(test) <= 17
        assert len(train) + len(val) + len(test) == 100

    def test_stratified_split_with_many_categories(self):
        """Test stratified split with many categories."""
        # Create dataset with 10 categories, 10 examples each
        examples = [
            {"id": i, "category": f"cat_{i // 10}"} for i in range(100)
        ]

        splitter = DatasetSplitter(random_seed=42)
        train, val, test = splitter.split(examples, stratify_by="category")

        # Count categories in train split
        train_categories = set(ex["category"] for ex in train)

        # All categories should be represented in train split
        assert len(train_categories) == 10

    def test_stratified_split_with_imbalanced_categories(self):
        """Test stratified split with highly imbalanced categories."""
        # 90 examples of category A, 10 of category B
        examples = (
            [{"id": i, "category": "A"} for i in range(90)]
            + [{"id": i + 90, "category": "B"} for i in range(10)]
        )

        splitter = DatasetSplitter(random_seed=42)
        train, val, test = splitter.split(examples, stratify_by="category")

        # Count B examples in each split
        train_b = sum(1 for ex in train if ex["category"] == "B")
        val_b = sum(1 for ex in val if ex["category"] == "B")
        test_b = sum(1 for ex in test if ex["category"] == "B")

        # Should have approximately 8, 1, 1 B examples
        assert train_b >= 7
        assert val_b + test_b >= 1  # At least some B in val or test

    def test_split_ratios_with_floating_point_precision(self):
        """Test that ratios with floating point precision work correctly."""
        examples = [{"id": i} for i in range(100)]
        splitter = DatasetSplitter(random_seed=42)

        # Ratios that sum to 1.0 but with floating point representation
        train, val, test = splitter.split(
            examples, train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1
        )

        assert len(train) == 70
        assert len(val) == 20
        assert len(test) == 10

    def test_split_with_none_seed(self):
        """Test split with None seed (non-deterministic)."""
        examples = [{"id": i} for i in range(100)]

        splitter = DatasetSplitter(random_seed=None)
        train, val, test = splitter.split(examples)

        # Should still produce valid split
        assert len(train) + len(val) + len(test) == 100

    def test_stratified_split_preserves_all_examples(self):
        """Test that stratified split preserves all examples."""
        examples = [
            {"id": i, "category": f"cat_{i % 5}", "value": i * 2}
            for i in range(100)
        ]

        splitter = DatasetSplitter(random_seed=42)
        train, val, test = splitter.split(examples, stratify_by="category")

        # Collect all IDs
        all_ids = set()
        for split in [train, val, test]:
            for ex in split:
                all_ids.add(ex["id"])

        # Check all IDs present
        assert len(all_ids) == 100
        assert all_ids == set(range(100))

    def test_category_distribution_calculation(self):
        """Test category distribution calculation."""
        examples = [
            {"id": i, "category": "A" if i < 30 else "B" if i < 70 else "C"}
            for i in range(100)
        ]

        splitter = DatasetSplitter(random_seed=42)
        train, val, test = splitter.split(examples, stratify_by="category")

        stats = splitter.get_split_statistics(train, val, test, stratify_by="category")

        # Check distributions
        train_dist = stats["category_distributions"]["train"]
        assert "A" in train_dist
        assert "B" in train_dist
        assert "C" in train_dist

        # Check total counts match
        total_a = (
            train_dist.get("A", 0)
            + stats["category_distributions"]["validation"].get("A", 0)
            + stats["category_distributions"]["test"].get("A", 0)
        )
        assert total_a == 30

    def test_split_with_equal_three_way_split(self):
        """Test split with equal 33.33% ratios."""
        examples = [{"id": i} for i in range(99)]  # Divisible by 3
        splitter = DatasetSplitter(random_seed=42)

        train, val, test = splitter.split(
            examples,
            train_ratio=1/3,
            validation_ratio=1/3,
            test_ratio=1/3,
        )

        # Should be approximately equal
        assert len(train) == 33
        assert len(val) == 33
        assert len(test) == 33
