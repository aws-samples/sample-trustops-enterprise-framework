"""
Unit tests for the dataset quality analyzer.

Tests the quality analysis functionality including completeness, diversity,
balance, and token statistics calculations.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data_models.dataset import DatasetFormat, DatasetTaskType
from src.datasets.quality_analyzer import QualityAnalyzer, analyze_dataset_file


class TestQualityAnalyzer:
    """Test suite for QualityAnalyzer class."""

    def test_analyze_empty_dataset_raises_error(self):
        """Test that analyzing an empty dataset raises ValueError."""
        analyzer = QualityAnalyzer()

        with pytest.raises(ValueError, match="Cannot analyze empty dataset"):
            analyzer.analyze([], DatasetTaskType.QA, DatasetFormat.JSONL)

    def test_calculate_completeness_perfect_score(self):
        """Test completeness calculation with all required fields present."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "What is AI?", "completion": "Artificial Intelligence"},
            {"prompt": "What is ML?", "completion": "Machine Learning"},
        ]

        score = analyzer._calculate_completeness(examples, DatasetTaskType.QA)

        assert score == 1.0

    def test_calculate_completeness_missing_fields(self):
        """Test completeness calculation with missing fields."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "What is AI?", "completion": "Artificial Intelligence"},
            {"prompt": "What is ML?"},  # Missing completion
            {"completion": "Deep Learning"},  # Missing prompt
        ]

        score = analyzer._calculate_completeness(examples, DatasetTaskType.QA)

        # 2 missing fields out of 6 total expected (3 examples * 2 fields)
        expected = 1.0 - (2 / 6)
        assert abs(score - expected) < 0.01

    def test_calculate_completeness_empty_fields(self):
        """Test completeness calculation with empty string fields."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "What is AI?", "completion": ""},  # Empty completion
            {"prompt": "", "completion": "ML"},  # Empty prompt
        ]

        score = analyzer._calculate_completeness(examples, DatasetTaskType.QA)

        # 2 empty fields out of 4 total expected
        expected = 1.0 - (2 / 4)
        assert abs(score - expected) < 0.01

    def test_calculate_diversity_all_unique(self):
        """Test diversity calculation with all unique prompts."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "What is AI?"},
            {"prompt": "What is ML?"},
            {"prompt": "What is DL?"},
        ]

        score = analyzer._calculate_diversity(examples)

        assert score == 1.0

    def test_calculate_diversity_all_duplicates(self):
        """Test diversity calculation with all duplicate prompts."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "What is AI?"},
            {"prompt": "What is AI?"},
            {"prompt": "What is AI?"},
        ]

        score = analyzer._calculate_diversity(examples)

        # Only 1 unique prompt out of 3 total
        expected = 1 / 3
        assert abs(score - expected) < 0.01

    def test_calculate_diversity_mixed(self):
        """Test diversity calculation with mixed unique and duplicate prompts."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "What is AI?"},
            {"prompt": "What is ML?"},
            {"prompt": "What is AI?"},  # Duplicate
            {"prompt": "What is DL?"},
        ]

        score = analyzer._calculate_diversity(examples)

        # 3 unique prompts out of 4 total
        expected = 3 / 4
        assert abs(score - expected) < 0.01

    def test_calculate_diversity_case_insensitive(self):
        """Test that diversity calculation is case-insensitive."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "What is AI?"},
            {"prompt": "WHAT IS AI?"},  # Same as first, different case
            {"prompt": "what is ai?"},  # Same as first, different case
        ]

        score = analyzer._calculate_diversity(examples)

        # All are the same when normalized
        expected = 1 / 3
        assert abs(score - expected) < 0.01

    def test_calculate_diversity_no_prompts(self):
        """Test diversity calculation when no prompts are found."""
        analyzer = QualityAnalyzer()
        examples = [
            {"completion": "Answer 1"},
            {"completion": "Answer 2"},
        ]

        score = analyzer._calculate_diversity(examples)

        assert score == 0.0

    def test_calculate_diversity_chat_format(self):
        """Test diversity calculation with chat format."""
        analyzer = QualityAnalyzer()
        examples = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "Hi there"}]},
            {"messages": [{"role": "user", "content": "Hello"}]},  # Duplicate
        ]

        score = analyzer._calculate_diversity(examples)

        # 2 unique prompts out of 3 total
        expected = 2 / 3
        assert abs(score - expected) < 0.01

    def test_calculate_balance_no_categories(self):
        """Test balance calculation when no categories are present."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "What is AI?"},
            {"prompt": "What is ML?"},
        ]

        score = analyzer._calculate_balance(examples)

        # No categories = perfectly balanced
        assert score == 1.0

    def test_calculate_balance_perfect(self):
        """Test balance calculation with perfectly balanced categories."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "Q1", "category": "A"},
            {"prompt": "Q2", "category": "B"},
            {"prompt": "Q3", "category": "C"},
            {"prompt": "Q4", "category": "A"},
            {"prompt": "Q5", "category": "B"},
            {"prompt": "Q6", "category": "C"},
        ]

        score = analyzer._calculate_balance(examples)

        # Perfectly balanced (2 examples per category)
        assert score > 0.99  # Should be very close to 1.0

    def test_calculate_balance_imbalanced(self):
        """Test balance calculation with imbalanced categories."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "Q1", "category": "A"},
            {"prompt": "Q2", "category": "A"},
            {"prompt": "Q3", "category": "A"},
            {"prompt": "Q4", "category": "A"},
            {"prompt": "Q5", "category": "B"},  # Only 1 in category B
        ]

        score = analyzer._calculate_balance(examples)

        # Imbalanced distribution should have lower score
        assert score < 0.8

    def test_calculate_balance_single_category(self):
        """Test balance calculation with only one category."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "Q1", "category": "A"},
            {"prompt": "Q2", "category": "A"},
            {"prompt": "Q3", "category": "A"},
        ]

        score = analyzer._calculate_balance(examples)

        # Single category = perfectly balanced
        assert score == 1.0

    def test_calculate_token_statistics(self):
        """Test token statistics calculation."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "Short prompt", "completion": "Short answer"},
            {"prompt": "This is a longer prompt with more words", "completion": "Longer answer here"},
            {"prompt": "Medium length", "completion": "Medium answer"},
        ]

        stats = analyzer._calculate_token_statistics(examples)

        assert stats.total_tokens > 0
        assert stats.min_tokens > 0
        assert stats.max_tokens >= stats.min_tokens
        assert stats.avg_tokens > 0
        assert stats.p95_tokens >= stats.avg_tokens
        assert stats.prompt_tokens > 0
        assert stats.completion_tokens > 0

    def test_calculate_token_statistics_empty(self):
        """Test token statistics with no text content."""
        analyzer = QualityAnalyzer()
        examples = [
            {"category": "A"},
            {"category": "B"},
        ]

        stats = analyzer._calculate_token_statistics(examples)

        assert stats.total_tokens == 0
        assert stats.min_tokens == 0
        assert stats.max_tokens == 0
        assert stats.avg_tokens == 0.0
        assert stats.p95_tokens == 0
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0

    def test_count_tokens(self):
        """Test token counting approximation."""
        analyzer = QualityAnalyzer()

        # Test various text lengths
        assert analyzer._count_tokens("hello") >= 1
        assert analyzer._count_tokens("hello world") >= 2
        assert analyzer._count_tokens("") >= 1  # Minimum 1 token

        # Longer text should have more tokens
        short_text = "hello"
        long_text = "hello world this is a longer text"
        assert analyzer._count_tokens(long_text) > analyzer._count_tokens(short_text)

    def test_normalize_text(self):
        """Test text normalization."""
        analyzer = QualityAnalyzer()

        assert analyzer._normalize_text("Hello World") == "hello world"
        assert analyzer._normalize_text("  Extra   Spaces  ") == "extra spaces"
        assert analyzer._normalize_text("UPPERCASE") == "uppercase"

    def test_detect_issues_high_quality(self):
        """Test issue detection with high-quality dataset."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": f"Question {i}", "completion": f"Answer {i}", "category": f"Cat{i % 3}"}
            for i in range(200)
        ]

        issues = analyzer._detect_issues(examples, 1.0, 0.9, 0.9)

        # High quality dataset should have no or minimal issues
        assert len(issues) <= 1  # Might have info-level issues

    def test_detect_issues_low_completeness(self):
        """Test issue detection with low completeness."""
        analyzer = QualityAnalyzer()
        examples = [{"prompt": "Q"}] * 10

        issues = analyzer._detect_issues(examples, 0.5, 0.9, 0.9)

        # Should detect completeness issue
        completeness_issues = [i for i in issues if i.category == "completeness"]
        assert len(completeness_issues) > 0
        assert completeness_issues[0].severity in ["error", "warning"]

    def test_detect_issues_low_diversity(self):
        """Test issue detection with low diversity."""
        analyzer = QualityAnalyzer()
        examples = [{"prompt": "Q"}] * 10

        issues = analyzer._detect_issues(examples, 0.9, 0.2, 0.9)

        # Should detect diversity issue
        diversity_issues = [i for i in issues if i.category == "diversity"]
        assert len(diversity_issues) > 0

    def test_detect_issues_small_dataset(self):
        """Test issue detection with small dataset."""
        analyzer = QualityAnalyzer()
        examples = [{"prompt": f"Q{i}", "completion": f"A{i}"} for i in range(50)]

        issues = analyzer._detect_issues(examples, 0.9, 0.9, 0.9)

        # Should detect size issue
        size_issues = [i for i in issues if i.category == "size"]
        assert len(size_issues) > 0

    def test_generate_recommendations_high_quality(self):
        """Test recommendation generation for high-quality dataset."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        token_stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=100,
            avg_tokens=50.0,
            p95_tokens=90,
            prompt_tokens=5000,
            completion_tokens=5000,
        )

        recommendations = analyzer._generate_recommendations(
            0.95, 0.85, 0.85, token_stats
        )

        # Should have positive feedback with SUCCESS indicator
        assert any("[SUCCESS]" in r for r in recommendations)
        assert any("excellent" in r.lower() or "ready" in r.lower()
                   for r in recommendations)

    def test_generate_recommendations_low_completeness(self):
        """Test recommendation generation for low completeness."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        token_stats = TokenStatistics(
            total_tokens=1000,
            min_tokens=10,
            max_tokens=100,
            avg_tokens=50.0,
            p95_tokens=90,
            prompt_tokens=500,
            completion_tokens=500,
        )

        recommendations = analyzer._generate_recommendations(
            0.7, 0.9, 0.9, token_stats
        )

        # Should recommend filling missing fields with WARNING
        assert any("[WARNING]" in r and "missing" in r.lower()
                   for r in recommendations)

    def test_generate_recommendations_low_diversity(self):
        """Test recommendation generation for low diversity."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        token_stats = TokenStatistics(
            total_tokens=1000,
            min_tokens=10,
            max_tokens=100,
            avg_tokens=50.0,
            p95_tokens=90,
            prompt_tokens=500,
            completion_tokens=500,
        )

        recommendations = analyzer._generate_recommendations(
            0.9, 0.2, 0.9, token_stats  # Changed from 0.3 to 0.2
        )

        # Should recommend increasing diversity with CRITICAL
        assert any("[CRITICAL]" in r and "diversity" in r.lower()
                   for r in recommendations)

    def test_generate_recommendations_token_imbalance(self):
        """Test recommendation generation for token imbalance."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        # Prompts much longer than completions
        token_stats = TokenStatistics(
            total_tokens=6000,
            min_tokens=10,
            max_tokens=100,
            avg_tokens=50.0,
            p95_tokens=90,
            prompt_tokens=5000,
            completion_tokens=1000,
        )

        recommendations = analyzer._generate_recommendations(
            0.9, 0.9, 0.9, token_stats
        )

        # Should recommend balancing prompts and completions
        assert any(("prompt" in r.lower() and "completion" in r.lower())
                   for r in recommendations)

    def test_generate_recommendations_critical_completeness(self):
        """Test CRITICAL recommendation for very low completeness."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        token_stats = TokenStatistics(
            total_tokens=1000,
            min_tokens=10,
            max_tokens=100,
            avg_tokens=50.0,
            p95_tokens=90,
            prompt_tokens=500,
            completion_tokens=500,
        )

        recommendations = analyzer._generate_recommendations(
            0.5, 0.9, 0.9, token_stats
        )

        # Should have CRITICAL severity for very low completeness
        assert any("[CRITICAL]" in r and "completeness" in r.lower()
                   for r in recommendations)

    def test_generate_recommendations_critical_balance(self):
        """Test CRITICAL recommendation for highly imbalanced dataset."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        token_stats = TokenStatistics(
            total_tokens=1000,
            min_tokens=10,
            max_tokens=100,
            avg_tokens=50.0,
            p95_tokens=90,
            prompt_tokens=500,
            completion_tokens=500,
        )

        recommendations = analyzer._generate_recommendations(
            0.9, 0.9, 0.4, token_stats
        )

        # Should have CRITICAL severity for highly imbalanced
        assert any("[CRITICAL]" in r and "imbalanced" in r.lower()
                   for r in recommendations)

    def test_generate_recommendations_very_long_examples(self):
        """Test recommendation for examples exceeding context limits."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        token_stats = TokenStatistics(
            total_tokens=50000,
            min_tokens=10,
            max_tokens=9000,  # Exceeds typical 8K limit
            avg_tokens=500.0,
            p95_tokens=8500,
            prompt_tokens=25000,
            completion_tokens=25000,
        )

        recommendations = analyzer._generate_recommendations(
            0.9, 0.9, 0.9, token_stats
        )

        # Should have CRITICAL warning about exceeding context limits
        assert any("[CRITICAL]" in r and "8000" in r
                   for r in recommendations)

    def test_generate_recommendations_very_short_examples(self):
        """Test recommendation for very short examples."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        token_stats = TokenStatistics(
            total_tokens=100,
            min_tokens=1,  # Very short
            max_tokens=50,
            avg_tokens=10.0,
            p95_tokens=45,
            prompt_tokens=50,
            completion_tokens=50,
        )

        recommendations = analyzer._generate_recommendations(
            0.9, 0.9, 0.9, token_stats
        )

        # Should warn about very short examples
        assert any("[WARNING]" in r and "short" in r.lower()
                   for r in recommendations)

    def test_generate_recommendations_large_variance(self):
        """Test recommendation for large token variance."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        token_stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=6000,  # Large range
            avg_tokens=500.0,
            p95_tokens=5500,
            prompt_tokens=5000,
            completion_tokens=5000,
        )

        recommendations = analyzer._generate_recommendations(
            0.9, 0.9, 0.9, token_stats
        )

        # Should warn about large variance
        assert any("[WARNING]" in r and "variance" in r.lower()
                   for r in recommendations)

    def test_generate_recommendations_info_level(self):
        """Test INFO level recommendations for minor issues."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        token_stats = TokenStatistics(
            total_tokens=5000,
            min_tokens=10,
            max_tokens=150,
            avg_tokens=50.0,
            p95_tokens=120,
            prompt_tokens=2500,
            completion_tokens=2500,
        )

        recommendations = analyzer._generate_recommendations(
            0.9, 0.65, 0.82, token_stats  # Slightly below optimal
        )

        # Should have INFO level recommendations
        assert any("[INFO]" in r for r in recommendations)

    def test_generate_recommendations_multiple_issues(self):
        """Test that multiple issues generate multiple recommendations."""
        analyzer = QualityAnalyzer()
        from src.data_models.dataset import TokenStatistics

        token_stats = TokenStatistics(
            total_tokens=1000,
            min_tokens=2,
            max_tokens=5000,
            avg_tokens=8.0,  # Very low
            p95_tokens=4500,
            prompt_tokens=800,
            completion_tokens=200,  # Imbalanced
        )

        recommendations = analyzer._generate_recommendations(
            0.6,  # Low completeness
            0.3,  # Low diversity
            0.4,  # Low balance
            token_stats
        )

        # Should have multiple recommendations for multiple issues
        assert len(recommendations) >= 4
        # Should have CRITICAL recommendations
        assert any("[CRITICAL]" in r for r in recommendations)
        # Should have WARNING recommendations
        assert any("[WARNING]" in r for r in recommendations)

    def test_analyze_complete_workflow(self):
        """Test complete analysis workflow."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": f"Question {i}", "completion": f"Answer {i}", "category": f"Cat{i % 3}"}
            for i in range(150)
        ]

        report = analyzer.analyze(examples, DatasetTaskType.QA, DatasetFormat.JSONL)

        # Verify report structure
        assert 0.0 <= report.completeness_score <= 1.0
        assert 0.0 <= report.diversity_score <= 1.0
        assert 0.0 <= report.balance_score <= 1.0
        assert report.token_stats.total_tokens > 0
        assert isinstance(report.issues, list)
        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) > 0


class TestAnalyzeDatasetFile:
    """Test suite for analyze_dataset_file function."""

    def test_analyze_jsonl_file(self):
        """Test analyzing a JSONL file."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10):
                json.dump(
                    {"prompt": f"Question {i}", "completion": f"Answer {i}", "category": f"Cat{i % 3}"},
                    f,
                )
                f.write("\n")
            temp_path = f.name

        try:
            report = analyze_dataset_file(
                temp_path, DatasetTaskType.QA, DatasetFormat.JSONL
            )

            assert 0.0 <= report.completeness_score <= 1.0
            assert 0.0 <= report.diversity_score <= 1.0
            assert 0.0 <= report.balance_score <= 1.0
            assert report.token_stats.total_tokens > 0
        finally:
            Path(temp_path).unlink()

    def test_analyze_csv_file(self):
        """Test analyzing a CSV file."""
        import csv

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "completion", "category"])
            writer.writeheader()
            for i in range(10):
                writer.writerow(
                    {"prompt": f"Question {i}", "completion": f"Answer {i}", "category": f"Cat{i % 3}"}
                )
            temp_path = f.name

        try:
            report = analyze_dataset_file(
                temp_path, DatasetTaskType.QA, DatasetFormat.CSV
            )

            assert 0.0 <= report.completeness_score <= 1.0
            assert 0.0 <= report.diversity_score <= 1.0
            assert report.token_stats.total_tokens > 0
        finally:
            Path(temp_path).unlink()

    def test_analyze_nonexistent_file(self):
        """Test analyzing a file that doesn't exist."""
        with pytest.raises(ValueError, match="File not found"):
            analyze_dataset_file(
                "/nonexistent/path.jsonl", DatasetTaskType.QA, DatasetFormat.JSONL
            )

    def test_analyze_empty_file(self):
        """Test analyzing an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="No examples found"):
                analyze_dataset_file(
                    temp_path, DatasetTaskType.QA, DatasetFormat.JSONL
                )
        finally:
            Path(temp_path).unlink()

    def test_analyze_unsupported_format(self):
        """Test analyzing with unsupported format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                analyze_dataset_file(
                    temp_path, DatasetTaskType.QA, DatasetFormat.HUGGINGFACE
                )
        finally:
            Path(temp_path).unlink()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_example_dataset(self):
        """Test analyzing a dataset with only one example."""
        analyzer = QualityAnalyzer()
        examples = [{"prompt": "Question", "completion": "Answer"}]

        report = analyzer.analyze(examples, DatasetTaskType.QA, DatasetFormat.JSONL)

        # Should not crash and return valid scores
        assert 0.0 <= report.completeness_score <= 1.0
        assert 0.0 <= report.diversity_score <= 1.0
        assert 0.0 <= report.balance_score <= 1.0

    def test_very_long_text(self):
        """Test with very long text content."""
        analyzer = QualityAnalyzer()
        long_text = " ".join(["word"] * 10000)
        examples = [{"prompt": long_text, "completion": long_text}]

        report = analyzer.analyze(examples, DatasetTaskType.QA, DatasetFormat.JSONL)

        # Should handle long text without crashing
        assert report.token_stats.max_tokens > 1000

    def test_special_characters(self):
        """Test with special characters in text."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "What is 2+2?", "completion": "4"},
            {"prompt": "Hello! How are you?", "completion": "I'm fine, thanks!"},
            {"prompt": "Test @#$%^&*()", "completion": "Special chars"},
        ]

        report = analyzer.analyze(examples, DatasetTaskType.QA, DatasetFormat.JSONL)

        # Should handle special characters
        assert report.diversity_score > 0.0

    def test_unicode_text(self):
        """Test with Unicode characters."""
        analyzer = QualityAnalyzer()
        examples = [
            {"prompt": "你好", "completion": "世界"},
            {"prompt": "Привет", "completion": "мир"},
            {"prompt": "مرحبا", "completion": "عالم"},
        ]

        report = analyzer.analyze(examples, DatasetTaskType.QA, DatasetFormat.JSONL)

        # Should handle Unicode
        assert report.diversity_score > 0.0
        assert report.token_stats.total_tokens > 0
