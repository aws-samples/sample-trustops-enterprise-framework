"""
Dataset splitter for train/validation/test splits.

This module provides functionality to split datasets into train, validation,
and test sets with configurable ratios and stratified sampling support.

Requirement 2.13: Implement train/validation/test splitter with stratified sampling
"""

import random
from collections import defaultdict
from typing import Any, Optional

import numpy as np


class DatasetSplitter:
    """Splits datasets into train/validation/test sets with stratification support."""

    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize the dataset splitter.

        Args:
            random_seed: Random seed for reproducibility. Default is 42.
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def split(
        self,
        examples: list[dict[str, Any]],
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            examples: List of dataset examples to split
            train_ratio: Ratio of examples for training set (default: 0.8)
            validation_ratio: Ratio of examples for validation set (default: 0.1)
            test_ratio: Ratio of examples for test set (default: 0.1)
            stratify_by: Optional field name to stratify by (e.g., 'category', 'label')

        Returns:
            Tuple of (train_examples, validation_examples, test_examples)

        Raises:
            ValueError: If ratios don't sum to 1.0 or if examples list is empty
        """
        # Validate inputs
        if not examples:
            raise ValueError("Cannot split empty dataset")

        # Validate individual ratios are non-negative
        if train_ratio < 0 or validation_ratio < 0 or test_ratio < 0:
            raise ValueError("All ratios must be non-negative")

        # Validate at least one ratio is positive
        if train_ratio == 0 and validation_ratio == 0 and test_ratio == 0:
            raise ValueError("At least one ratio must be positive")

        # Validate ratios sum to 1.0 (with small tolerance for floating point)
        total_ratio = train_ratio + validation_ratio + test_ratio
        if not (0.999 <= total_ratio <= 1.001):
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio:.4f} "
                f"(train={train_ratio}, validation={validation_ratio}, test={test_ratio})"
            )

        # Perform split
        if stratify_by:
            return self._stratified_split(
                examples, train_ratio, validation_ratio, test_ratio, stratify_by
            )
        else:
            return self._random_split(
                examples, train_ratio, validation_ratio, test_ratio
            )

    def _random_split(
        self,
        examples: list[dict[str, Any]],
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Perform random split without stratification.

        Args:
            examples: List of dataset examples
            train_ratio: Ratio for training set
            validation_ratio: Ratio for validation set
            test_ratio: Ratio for test set

        Returns:
            Tuple of (train, validation, test) splits
        """
        # Shuffle examples
        shuffled = examples.copy()
        random.shuffle(shuffled)

        # Calculate split indices
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * validation_ratio)

        # Split
        train = shuffled[:train_end]
        validation = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        return train, validation, test

    def _stratified_split(
        self,
        examples: list[dict[str, Any]],
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        stratify_by: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Perform stratified split to maintain category balance across splits.

        Args:
            examples: List of dataset examples
            train_ratio: Ratio for training set
            validation_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            stratify_by: Field name to stratify by

        Returns:
            Tuple of (train, validation, test) splits

        Raises:
            ValueError: If stratify_by field is missing in examples
        """
        # Group examples by category
        category_groups = defaultdict(list)

        for example in examples:
            # Extract category value
            category = self._extract_category(example, stratify_by)

            if category is None:
                raise ValueError(
                    f"Stratification field '{stratify_by}' not found or empty in example: {example}"
                )

            category_groups[category].append(example)

        # Split each category proportionally
        train_split = []
        validation_split = []
        test_split = []

        for category, category_examples in category_groups.items():
            # Shuffle examples within category
            shuffled = category_examples.copy()
            random.shuffle(shuffled)

            # Calculate split indices for this category
            n = len(shuffled)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * validation_ratio)

            # Split this category
            train_split.extend(shuffled[:train_end])
            validation_split.extend(shuffled[train_end:val_end])
            test_split.extend(shuffled[val_end:])

        # Shuffle the final splits to mix categories
        random.shuffle(train_split)
        random.shuffle(validation_split)
        random.shuffle(test_split)

        return train_split, validation_split, test_split

    def _extract_category(
        self, example: dict[str, Any], stratify_by: str
    ) -> Optional[str]:
        """
        Extract category value from an example.

        Tries multiple common field names if the exact field is not found.

        Args:
            example: Dataset example
            stratify_by: Primary field name to look for

        Returns:
            Category value as string, or None if not found
        """
        # Try exact field name first
        if stratify_by in example and example[stratify_by] is not None:
            return str(example[stratify_by])

        # Try common alternative field names
        alternatives = ["category", "label", "class", "type"]

        for alt in alternatives:
            if alt in example and example[alt] is not None:
                return str(example[alt])

        return None

    def get_split_statistics(
        self,
        train: list[dict[str, Any]],
        validation: list[dict[str, Any]],
        test: list[dict[str, Any]],
        stratify_by: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Calculate statistics about the split.

        Args:
            train: Training set examples
            validation: Validation set examples
            test: Test set examples
            stratify_by: Optional field name that was used for stratification

        Returns:
            Dictionary with split statistics including sizes, ratios, and category distributions
        """
        total = len(train) + len(validation) + len(test)

        if total == 0:
            return {
                "total_examples": 0,
                "train_size": 0,
                "validation_size": 0,
                "test_size": 0,
                "train_ratio": 0.0,
                "validation_ratio": 0.0,
                "test_ratio": 0.0,
            }

        stats = {
            "total_examples": total,
            "train_size": len(train),
            "validation_size": len(validation),
            "test_size": len(test),
            "train_ratio": len(train) / total,
            "validation_ratio": len(validation) / total,
            "test_ratio": len(test) / total,
        }

        # Add category distribution if stratification was used
        if stratify_by:
            stats["category_distributions"] = {
                "train": self._get_category_distribution(train, stratify_by),
                "validation": self._get_category_distribution(validation, stratify_by),
                "test": self._get_category_distribution(test, stratify_by),
            }

        return stats

    def _get_category_distribution(
        self, examples: list[dict[str, Any]], stratify_by: str
    ) -> dict[str, int]:
        """
        Get category distribution for a set of examples.

        Args:
            examples: List of examples
            stratify_by: Field name to extract categories from

        Returns:
            Dictionary mapping category values to counts
        """
        distribution = defaultdict(int)

        for example in examples:
            category = self._extract_category(example, stratify_by)
            if category:
                distribution[category] += 1

        return dict(distribution)


def split_dataset(
    examples: list[dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_by: Optional[str] = None,
    random_seed: Optional[int] = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Convenience function to split a dataset.

    Args:
        examples: List of dataset examples to split
        train_ratio: Ratio of examples for training set (default: 0.8)
        validation_ratio: Ratio of examples for validation set (default: 0.1)
        test_ratio: Ratio of examples for test set (default: 0.1)
        stratify_by: Optional field name to stratify by (e.g., 'category', 'label')
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_examples, validation_examples, test_examples)

    Raises:
        ValueError: If ratios don't sum to 1.0 or if examples list is empty
    """
    splitter = DatasetSplitter(random_seed=random_seed)
    return splitter.split(examples, train_ratio, validation_ratio, test_ratio, stratify_by)
