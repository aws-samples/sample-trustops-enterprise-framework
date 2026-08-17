"""
Dataset quality analyzer.

This module provides quality analysis for datasets including completeness,
diversity, balance, and token statistics calculations.

Requirement 2.5: Implement dataset quality analyzer
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from src.data_models.dataset import (
    DatasetFormat,
    DatasetQualityReport,
    DatasetTaskType,
    QualityIssue,
    TokenStatistics,
)


class QualityAnalyzer:
    """Analyzes dataset quality across multiple dimensions."""

    def __init__(self):
        """Initialize the quality analyzer."""
        pass

    def analyze(
        self,
        examples: list[dict[str, Any]],
        task_type: DatasetTaskType,
        format_type: DatasetFormat,
    ) -> DatasetQualityReport:
        """
        Analyze dataset quality and generate a comprehensive report.

        Args:
            examples: List of dataset examples
            task_type: Task type of the dataset
            format_type: Format of the dataset

        Returns:
            DatasetQualityReport with quality metrics and recommendations

        Raises:
            ValueError: If examples list is empty
        """
        if not examples:
            raise ValueError("Cannot analyze empty dataset")

        # Calculate quality metrics
        completeness_score = self._calculate_completeness(examples, task_type)
        diversity_score = self._calculate_diversity(examples)
        balance_score = self._calculate_balance(examples)
        token_stats = self._calculate_token_statistics(examples)

        # Detect quality issues
        issues = self._detect_issues(
            examples, completeness_score, diversity_score, balance_score
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            completeness_score, diversity_score, balance_score, token_stats
        )

        return DatasetQualityReport(
            completeness_score=completeness_score,
            diversity_score=diversity_score,
            balance_score=balance_score,
            token_stats=token_stats,
            issues=issues,
            recommendations=recommendations,
        )

    def _calculate_completeness(
        self, examples: list[dict[str, Any]], task_type: DatasetTaskType
    ) -> float:
        """
        Calculate completeness score based on missing fields ratio.

        Completeness = 1 - (missing_field_count / total_expected_fields)

        Args:
            examples: List of dataset examples
            task_type: Task type to determine expected fields

        Returns:
            Completeness score in range [0, 1]
        """
        # Define expected fields per task type
        expected_fields = {
            DatasetTaskType.QA: ["prompt", "completion"],
            DatasetTaskType.SUMMARIZATION: ["prompt", "completion"],
            DatasetTaskType.CLASSIFICATION: ["prompt", "completion"],
            DatasetTaskType.TEXT_GENERATION: ["prompt", "completion"],
            DatasetTaskType.CHAT: ["messages"],
            DatasetTaskType.CUSTOM: ["prompt"],
        }

        required_fields = expected_fields.get(task_type, ["prompt"])
        total_expected = len(examples) * len(required_fields)
        missing_count = 0

        for example in examples:
            for field in required_fields:
                if field not in example or not example[field]:
                    missing_count += 1

        if total_expected == 0:
            return 1.0

        completeness = 1.0 - (missing_count / total_expected)
        return max(0.0, min(1.0, completeness))

    def _calculate_diversity(self, examples: list[dict[str, Any]]) -> float:
        """
        Calculate diversity score based on unique prompts ratio.

        Diversity = unique_prompts / total_prompts

        Args:
            examples: List of dataset examples

        Returns:
            Diversity score in range [0, 1]
        """
        prompts = []

        for example in examples:
            # Extract prompt text from various possible fields
            prompt_text = None

            if "prompt" in example:
                prompt_text = example["prompt"]
            elif "question" in example:
                prompt_text = example["question"]
            elif "text" in example:
                prompt_text = example["text"]
            elif "input" in example:
                prompt_text = example["input"]
            elif "messages" in example and isinstance(example["messages"], list):
                # For chat format, use first message as prompt
                if example["messages"]:
                    first_msg = example["messages"][0]
                    if isinstance(first_msg, dict) and "content" in first_msg:
                        prompt_text = first_msg["content"]

            if prompt_text:
                # Normalize prompt text for comparison
                normalized = self._normalize_text(str(prompt_text))
                prompts.append(normalized)

        if not prompts:
            return 0.0

        unique_prompts = len(set(prompts))
        total_prompts = len(prompts)

        diversity = unique_prompts / total_prompts
        return max(0.0, min(1.0, diversity))

    def _calculate_balance(self, examples: list[dict[str, Any]]) -> float:
        """
        Calculate balance score based on category distribution evenness.

        Uses Shannon entropy normalized to [0, 1]:
        Balance = H(categories) / log(num_categories)

        Where H is Shannon entropy. A perfectly balanced distribution has score 1.0.

        Args:
            examples: List of dataset examples

        Returns:
            Balance score in range [0, 1]
        """
        categories = []

        for example in examples:
            # Extract category from various possible fields
            category = example.get("category") or example.get("label") or example.get("class")

            if category:
                categories.append(str(category))

        # If no categories found, return 1.0 (perfectly balanced - no imbalance)
        if not categories:
            return 1.0

        # Count category frequencies
        category_counts = Counter(categories)
        total = len(categories)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in category_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log(p)

        # Normalize by maximum possible entropy (log of number of categories)
        num_categories = len(category_counts)
        if num_categories <= 1:
            return 1.0

        max_entropy = np.log(num_categories)
        balance = entropy / max_entropy

        return max(0.0, min(1.0, balance))

    def _calculate_token_statistics(
        self, examples: list[dict[str, Any]]
    ) -> TokenStatistics:
        """
        Calculate token statistics for prompts and completions.

        Args:
            examples: List of dataset examples

        Returns:
            TokenStatistics with min, max, avg, p95 for prompts and completions
        """
        prompt_tokens_list = []
        completion_tokens_list = []

        for example in examples:
            # Extract prompt
            prompt_text = self._extract_prompt(example)
            if prompt_text:
                prompt_tokens = self._count_tokens(prompt_text)
                prompt_tokens_list.append(prompt_tokens)

            # Extract completion
            completion_text = self._extract_completion(example)
            if completion_text:
                completion_tokens = self._count_tokens(completion_text)
                completion_tokens_list.append(completion_tokens)

        # Combine all tokens
        all_tokens = prompt_tokens_list + completion_tokens_list

        if not all_tokens:
            # Return zero statistics if no tokens found
            return TokenStatistics(
                total_tokens=0,
                min_tokens=0,
                max_tokens=0,
                avg_tokens=0.0,
                p95_tokens=0,
                prompt_tokens=0,
                completion_tokens=0,
            )

        # Calculate statistics
        total_tokens = sum(all_tokens)
        min_tokens = int(np.min(all_tokens))
        max_tokens = int(np.max(all_tokens))
        avg_tokens = float(np.mean(all_tokens))
        p95_tokens = int(np.percentile(all_tokens, 95))
        total_prompt_tokens = sum(prompt_tokens_list)
        total_completion_tokens = sum(completion_tokens_list)

        return TokenStatistics(
            total_tokens=total_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            avg_tokens=avg_tokens,
            p95_tokens=p95_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )

    def _extract_prompt(self, example: dict[str, Any]) -> Optional[str]:
        """Extract prompt text from an example."""
        if "prompt" in example:
            return str(example["prompt"])
        elif "question" in example:
            return str(example["question"])
        elif "text" in example:
            return str(example["text"])
        elif "input" in example:
            return str(example["input"])
        elif "messages" in example and isinstance(example["messages"], list):
            if example["messages"]:
                first_msg = example["messages"][0]
                if isinstance(first_msg, dict) and "content" in first_msg:
                    return str(first_msg["content"])
        return None

    def _extract_completion(self, example: dict[str, Any]) -> Optional[str]:
        """Extract completion text from an example."""
        if "completion" in example:
            return str(example["completion"])
        elif "answer" in example:
            return str(example["answer"])
        elif "response" in example:
            return str(example["response"])
        elif "output" in example:
            return str(example["output"])
        elif "summary" in example:
            return str(example["summary"])
        elif "messages" in example and isinstance(example["messages"], list):
            # For chat format, concatenate all assistant messages
            completions = []
            for msg in example["messages"]:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    if "content" in msg:
                        completions.append(str(msg["content"]))
            if completions:
                return " ".join(completions)
        return None

    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a simple approximation: tokens ≈ words * 1.3
        This is a rough estimate; actual tokenization would require model-specific tokenizers.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Simple word-based approximation
        words = len(text.split())
        # Multiply by 1.3 as a rough approximation (words to tokens ratio)
        tokens = int(words * 1.3)
        return max(1, tokens)  # Ensure at least 1 token

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _detect_issues(
        self,
        examples: list[dict[str, Any]],
        completeness_score: float,
        diversity_score: float,
        balance_score: float,
    ) -> list[QualityIssue]:
        """
        Detect quality issues in the dataset.

        Args:
            examples: List of dataset examples
            completeness_score: Calculated completeness score
            diversity_score: Calculated diversity score
            balance_score: Calculated balance score

        Returns:
            List of detected quality issues
        """
        issues = []

        # Check completeness
        if completeness_score < 0.9:
            severity = "error" if completeness_score < 0.7 else "warning"
            issues.append(
                QualityIssue(
                    severity=severity,
                    category="completeness",
                    message=f"Dataset has low completeness score ({completeness_score:.2f}). "
                    f"Some examples are missing required fields.",
                )
            )

        # Check diversity
        if diversity_score < 0.5:
            severity = "warning" if diversity_score < 0.3 else "info"
            issues.append(
                QualityIssue(
                    severity=severity,
                    category="diversity",
                    message=f"Dataset has low diversity score ({diversity_score:.2f}). "
                    f"Many prompts are duplicates or very similar.",
                )
            )

        # Check balance
        if balance_score < 0.7:
            severity = "warning" if balance_score < 0.5 else "info"
            issues.append(
                QualityIssue(
                    severity=severity,
                    category="balance",
                    message=f"Dataset has low balance score ({balance_score:.2f}). "
                    f"Categories are not evenly distributed.",
                )
            )

        # Check dataset size
        if len(examples) < 100:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="size",
                    message=f"Dataset is small ({len(examples)} examples). "
                    f"Consider adding more examples for better model training.",
                )
            )

        return issues

    def _generate_recommendations(
        self,
        completeness_score: float,
        diversity_score: float,
        balance_score: float,
        token_stats: TokenStatistics,
    ) -> list[str]:
        """
        Generate actionable recommendations based on quality metrics.

        This method provides specific, actionable suggestions with severity
        levels to help improve dataset quality across all dimensions.

        Args:
            completeness_score: Calculated completeness score
            diversity_score: Calculated diversity score
            balance_score: Calculated balance score
            token_stats: Calculated token statistics

        Returns:
            List of recommendations with severity indicators
        """
        recommendations = []

        # Completeness recommendations (critical for model training)
        if completeness_score < 0.7:
            recommendations.append(
                "[CRITICAL] Completeness is very low ({:.0%}). "
                "Immediately review and fill in missing required fields. "
                "Missing data can severely impact model training quality."
                .format(completeness_score)
            )
        elif completeness_score < 0.9:
            recommendations.append(
                "[WARNING] Some examples are missing required fields "
                "({:.0%} complete). Review dataset and fill in missing "
                "prompt or completion fields to improve training quality."
                .format(completeness_score)
            )

        # Diversity recommendations (affects model generalization)
        if diversity_score < 0.3:
            recommendations.append(
                "[CRITICAL] Diversity is very low ({:.0%}). "
                "Most prompts are duplicates. Add unique examples or use "
                "data augmentation techniques (paraphrasing, synonym "
                "replacement) to increase variety."
                .format(diversity_score)
            )
        elif diversity_score < 0.5:
            recommendations.append(
                "[WARNING] Diversity is below optimal ({:.0%}). "
                "Consider adding more unique prompts or using synthetic "
                "data generation to improve model generalization."
                .format(diversity_score)
            )
        elif diversity_score < 0.7:
            recommendations.append(
                "[INFO] Diversity could be improved ({:.0%}). "
                "Adding more varied examples may help model performance "
                "on edge cases."
                .format(diversity_score)
            )

        # Balance recommendations (prevents category bias)
        if balance_score < 0.5:
            recommendations.append(
                "[CRITICAL] Categories are highly imbalanced ({:.0%}). "
                "Add examples to underrepresented categories or use "
                "stratified sampling to prevent model bias toward "
                "majority classes."
                .format(balance_score)
            )
        elif balance_score < 0.7:
            recommendations.append(
                "[WARNING] Category distribution is uneven ({:.0%}). "
                "Consider balancing by adding examples to smaller "
                "categories or using weighted sampling during training."
                .format(balance_score)
            )
        elif balance_score < 0.85:
            recommendations.append(
                "[INFO] Category balance could be improved ({:.0%}). "
                "More even distribution may help model performance "
                "across all categories."
                .format(balance_score)
            )

        # Token length recommendations (affects model performance)
        if token_stats.avg_tokens < 10:
            recommendations.append(
                "[WARNING] Average token count is very low "
                "({:.1f} tokens). Add more detailed prompts and "
                "completions to provide sufficient context for "
                "model learning."
                .format(token_stats.avg_tokens)
            )
        elif token_stats.avg_tokens < 20:
            recommendations.append(
                "[INFO] Average token count is low ({:.1f} tokens). "
                "Consider adding more detail to improve model "
                "understanding."
                .format(token_stats.avg_tokens)
            )

        # Maximum token recommendations (context window limits)
        if token_stats.max_tokens > 8000:
            recommendations.append(
                "[CRITICAL] Some examples exceed 8000 tokens "
                "(max: {}). Most models have context limits of "
                "4K-8K tokens. Split or truncate long examples to "
                "prevent training failures."
                .format(token_stats.max_tokens)
            )
        elif token_stats.max_tokens > 4000:
            recommendations.append(
                "[WARNING] Some examples are very long (max: {} tokens). "
                "Verify they fit within your model's context window "
                "(typically 4K-8K tokens). Consider splitting or "
                "truncating if needed."
                .format(token_stats.max_tokens)
            )
        elif token_stats.max_tokens > 2000:
            recommendations.append(
                "[INFO] Some examples are lengthy (max: {} tokens). "
                "Ensure they fit your model's context window and "
                "consider if shorter examples would be more effective."
                .format(token_stats.max_tokens)
            )

        # Minimum token recommendations (too short examples)
        if token_stats.min_tokens < 3 and token_stats.total_tokens > 0:
            recommendations.append(
                "[WARNING] Some examples are very short (min: {} tokens). "
                "Very brief examples may not provide enough context for "
                "effective model learning. Review and expand if needed."
                .format(token_stats.min_tokens)
            )

        # Token variance recommendations (consistency)
        if token_stats.total_tokens > 0:
            token_range = token_stats.max_tokens - token_stats.min_tokens
            if token_range > 5000:
                recommendations.append(
                    "[WARNING] Large variance in example lengths "
                    "(range: {} tokens). Consider normalizing example "
                    "lengths for more consistent training batches."
                    .format(token_range)
                )

        # Token imbalance recommendations (prompt vs completion ratio)
        if token_stats.prompt_tokens > 0 and token_stats.completion_tokens > 0:
            ratio = token_stats.prompt_tokens / token_stats.completion_tokens
            if ratio >= 10:
                recommendations.append(
                    "[WARNING] Prompts are much longer than completions "
                    "(ratio: {:.1f}:1). Consider more concise prompts or "
                    "more detailed completions for better training balance."
                    .format(ratio)
                )
            elif ratio >= 5:
                recommendations.append(
                    "[INFO] Prompts are notably longer than completions "
                    "(ratio: {:.1f}:1). Verify this matches your use case "
                    "requirements."
                    .format(ratio)
                )
            elif ratio <= 0.1:
                recommendations.append(
                    "[WARNING] Completions are much longer than prompts "
                    "(ratio: 1:{:.1f}). Ensure prompts provide sufficient "
                    "context and instructions for the task."
                    .format(1/ratio)
                )
            elif ratio <= 0.2:
                recommendations.append(
                    "[INFO] Completions are notably longer than prompts "
                    "(ratio: 1:{:.1f}). Verify this matches your use case "
                    "requirements."
                    .format(1/ratio)
                )

        # P95 token recommendations (outlier detection)
        if token_stats.p95_tokens > token_stats.avg_tokens * 3:
            recommendations.append(
                "[INFO] 95th percentile token count ({}) is much higher "
                "than average ({:.1f}). You may have outlier examples "
                "that could affect training. Review longest examples."
                .format(token_stats.p95_tokens, token_stats.avg_tokens)
            )

        # If no issues, provide positive feedback
        if not recommendations:
            recommendations.append(
                "[SUCCESS] Dataset quality is excellent! All metrics are "
                "within optimal ranges. Your dataset is ready for model "
                "training or evaluation."
            )

        return recommendations


def analyze_dataset_file(
    file_path: Union[str, Path],
    task_type: DatasetTaskType,
    format_type: DatasetFormat,
) -> DatasetQualityReport:
    """
    Analyze a dataset file and generate a quality report.

    Args:
        file_path: Path to the dataset file
        task_type: Task type of the dataset
        format_type: Format of the dataset

    Returns:
        DatasetQualityReport with quality metrics

    Raises:
        ValueError: If file cannot be read or parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    # Parse file based on format
    examples = []

    if format_type == DatasetFormat.JSONL:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    elif format_type == DatasetFormat.CSV:
        import csv

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            examples = list(reader)

    elif format_type == DatasetFormat.PARQUET:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(str(file_path))
            examples = table.to_pylist()
        except ImportError:
            raise ValueError("pyarrow is required to read Parquet files")

    else:
        raise ValueError(f"Unsupported format: {format_type}")

    if not examples:
        raise ValueError(f"No examples found in {file_path}")

    # Analyze quality
    analyzer = QualityAnalyzer()
    return analyzer.analyze(examples, task_type, format_type)
