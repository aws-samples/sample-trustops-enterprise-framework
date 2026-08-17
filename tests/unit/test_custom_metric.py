"""
Unit tests for Custom Metric Plugin Interface.

This module tests the custom metric base class, registry, and integration
with the trust scoring system.

Requirements: 5.6, 5.11
"""

import pytest

from src.data_models.trust_score import DimensionScore, TrustDimension
from src.trust_scoring.custom_metric import (
    CustomMetric,
    CustomMetricRegistry,
    get_global_registry,
    register_custom_metric,
)


# Test implementation of CustomMetric
class TestCustomMetric(CustomMetric):
    """Test implementation of a custom metric."""
    
    async def calculate(
        self,
        prompt: str,
        response: str,
        expected_response: str = None,
        source_documents: list[str] = None,
        model_id: str = None,
        **kwargs
    ) -> DimensionScore:
        """Simple test implementation that scores based on response length."""
        # Score based on response length (longer = better, up to 100 chars)
        score = min(len(response) / 100.0, 1.0)
        
        return DimensionScore(
            dimension=TrustDimension.ACCURACY,  # Using existing dimension for testing
            score=score,
            confidence=0.9,
            details={
                "method": "length_based",
                "response_length": len(response)
            },
            checks_passed=["length_check"] if score > 0.5 else [],
            checks_failed=[] if score > 0.5 else ["length_check"]
        )


class ConfigurableCustomMetric(CustomMetric):
    """Test implementation with configuration."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.threshold = self.config.get('threshold', 0.5)
        self.multiplier = self.config.get('multiplier', 1.0)
    
    async def calculate(
        self,
        prompt: str,
        response: str,
        expected_response: str = None,
        source_documents: list[str] = None,
        model_id: str = None,
        **kwargs
    ) -> DimensionScore:
        """Configurable scoring based on word count."""
        word_count = len(response.split())
        score = min((word_count * self.multiplier) / 50.0, 1.0)
        
        return DimensionScore(
            dimension=TrustDimension.ACCURACY,
            score=score,
            confidence=0.8,
            details={
                "method": "word_count",
                "word_count": word_count,
                "threshold": self.threshold,
                "multiplier": self.multiplier
            },
            checks_passed=["threshold_met"] if score >= self.threshold else [],
            checks_failed=[] if score >= self.threshold else ["threshold_not_met"]
        )
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        return (
            0.0 <= self.threshold <= 1.0 and
            self.multiplier > 0.0
        )


class InvalidCustomMetric:
    """Invalid metric that doesn't extend CustomMetric."""
    pass


# Test CustomMetric base class
class TestCustomMetricBase:
    """Test the CustomMetric abstract base class."""
    
    def test_custom_metric_calculate_not_implemented(self):
        """Test that abstract class cannot be instantiated without calculate()."""
        
        # Create a minimal subclass that doesn't implement calculate
        class IncompleteMetric(CustomMetric):
            pass
        
        # Should raise TypeError when trying to instantiate
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            metric = IncompleteMetric()
    
    @pytest.mark.asyncio
    async def test_custom_metric_calculate_implementation(self):
        """Test that custom metric can be implemented and called."""
        metric = TestCustomMetric()
        
        result = await metric.calculate(
            prompt="What is AI?",
            response="Artificial Intelligence is a field of computer science."
        )
        
        assert isinstance(result, DimensionScore)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.details["method"] == "length_based"
    
    def test_custom_metric_get_name_default(self):
        """Test default get_name() returns class name."""
        metric = TestCustomMetric()
        assert metric.get_name() == "TestCustomMetric"
    
    def test_custom_metric_get_name_override(self):
        """Test get_name() can be overridden."""
        
        class NamedMetric(CustomMetric):
            async def calculate(self, prompt: str, response: str, **kwargs) -> DimensionScore:
                pass
            
            def get_name(self) -> str:
                return "My Custom Metric"
        
        metric = NamedMetric()
        assert metric.get_name() == "My Custom Metric"
    
    def test_custom_metric_get_description_default(self):
        """Test default get_description()."""
        metric = TestCustomMetric()
        description = metric.get_description()
        assert "TestCustomMetric" in description
    
    def test_custom_metric_validate_config_default(self):
        """Test default validate_config() returns True."""
        metric = TestCustomMetric()
        assert metric.validate_config() is True
    
    def test_custom_metric_config_initialization(self):
        """Test custom metric initialization with config."""
        config = {"threshold": 0.7, "multiplier": 2.0}
        metric = ConfigurableCustomMetric(config=config)
        
        assert metric.config == config
        assert metric.threshold == 0.7
        assert metric.multiplier == 2.0
    
    def test_custom_metric_config_defaults(self):
        """Test custom metric uses defaults when config not provided."""
        metric = ConfigurableCustomMetric()
        
        assert metric.threshold == 0.5
        assert metric.multiplier == 1.0
    
    @pytest.mark.asyncio
    async def test_custom_metric_with_all_parameters(self):
        """Test custom metric receives all standard parameters."""
        metric = TestCustomMetric()
        
        result = await metric.calculate(
            prompt="Test prompt",
            response="Test response",
            expected_response="Expected answer",
            source_documents=["doc1", "doc2"],
            model_id="test-model",
            custom_param="custom_value"
        )
        
        assert isinstance(result, DimensionScore)
        assert result.score >= 0.0


# Test CustomMetricRegistry
class TestCustomMetricRegistry:
    """Test the CustomMetricRegistry class."""
    
    def test_registry_initialization(self):
        """Test registry initializes empty."""
        registry = CustomMetricRegistry()
        assert registry.list_metrics() == []
    
    def test_register_custom_metric(self):
        """Test registering a custom metric."""
        registry = CustomMetricRegistry()
        registry.register("test_metric", TestCustomMetric)
        
        assert registry.is_registered("test_metric")
        assert "test_metric" in registry.list_metrics()
    
    def test_register_duplicate_name_raises_error(self):
        """Test registering duplicate name raises ValueError."""
        registry = CustomMetricRegistry()
        registry.register("test_metric", TestCustomMetric)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test_metric", ConfigurableCustomMetric)
    
    def test_register_invalid_class_raises_error(self):
        """Test registering non-CustomMetric class raises ValueError."""
        registry = CustomMetricRegistry()
        
        with pytest.raises(ValueError, match="must extend CustomMetric"):
            registry.register("invalid", InvalidCustomMetric)
    
    def test_unregister_custom_metric(self):
        """Test unregistering a custom metric."""
        registry = CustomMetricRegistry()
        registry.register("test_metric", TestCustomMetric)
        
        assert registry.is_registered("test_metric")
        
        registry.unregister("test_metric")
        
        assert not registry.is_registered("test_metric")
        assert "test_metric" not in registry.list_metrics()
    
    def test_unregister_nonexistent_raises_error(self):
        """Test unregistering non-existent metric raises KeyError."""
        registry = CustomMetricRegistry()
        
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("nonexistent")
    
    def test_get_registered_metric(self):
        """Test getting a registered metric class."""
        registry = CustomMetricRegistry()
        registry.register("test_metric", TestCustomMetric)
        
        metric_class = registry.get("test_metric")
        
        assert metric_class == TestCustomMetric
    
    def test_get_nonexistent_raises_error(self):
        """Test getting non-existent metric raises KeyError."""
        registry = CustomMetricRegistry()
        
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")
    
    def test_instantiate_metric(self):
        """Test instantiating a registered metric."""
        registry = CustomMetricRegistry()
        registry.register("test_metric", TestCustomMetric)
        
        metric = registry.instantiate("test_metric")
        
        assert isinstance(metric, TestCustomMetric)
        assert isinstance(metric, CustomMetric)
    
    def test_instantiate_metric_with_config(self):
        """Test instantiating a metric with configuration."""
        registry = CustomMetricRegistry()
        registry.register("configurable", ConfigurableCustomMetric)
        
        config = {"threshold": 0.8, "multiplier": 1.5}
        metric = registry.instantiate("configurable", config=config)
        
        assert isinstance(metric, ConfigurableCustomMetric)
        assert metric.threshold == 0.8
        assert metric.multiplier == 1.5
    
    def test_list_metrics_multiple(self):
        """Test listing multiple registered metrics."""
        registry = CustomMetricRegistry()
        registry.register("metric1", TestCustomMetric)
        registry.register("metric2", ConfigurableCustomMetric)
        
        metrics = registry.list_metrics()
        
        assert len(metrics) == 2
        assert "metric1" in metrics
        assert "metric2" in metrics
    
    def test_is_registered_true(self):
        """Test is_registered returns True for registered metric."""
        registry = CustomMetricRegistry()
        registry.register("test_metric", TestCustomMetric)
        
        assert registry.is_registered("test_metric") is True
    
    def test_is_registered_false(self):
        """Test is_registered returns False for non-registered metric."""
        registry = CustomMetricRegistry()
        
        assert registry.is_registered("nonexistent") is False
    
    def test_clear_registry(self):
        """Test clearing all registered metrics."""
        registry = CustomMetricRegistry()
        registry.register("metric1", TestCustomMetric)
        registry.register("metric2", ConfigurableCustomMetric)
        
        assert len(registry.list_metrics()) == 2
        
        registry.clear()
        
        assert len(registry.list_metrics()) == 0
        assert not registry.is_registered("metric1")
        assert not registry.is_registered("metric2")


# Test global registry functions
class TestGlobalRegistry:
    """Test global registry convenience functions."""
    
    def setup_method(self):
        """Clear global registry before each test."""
        get_global_registry().clear()
    
    def teardown_method(self):
        """Clear global registry after each test."""
        get_global_registry().clear()
    
    def test_register_custom_metric_global(self):
        """Test registering metric in global registry."""
        register_custom_metric("global_test", TestCustomMetric)
        
        registry = get_global_registry()
        assert registry.is_registered("global_test")
    
    def test_get_global_registry_returns_same_instance(self):
        """Test get_global_registry returns singleton instance."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        
        assert registry1 is registry2
    
    def test_global_registry_persistence(self):
        """Test global registry persists across calls."""
        register_custom_metric("persistent", TestCustomMetric)
        
        # Get registry again
        registry = get_global_registry()
        
        assert registry.is_registered("persistent")


# Integration tests
class TestCustomMetricIntegration:
    """Integration tests for custom metrics."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_custom_metric_usage(self):
        """Test complete workflow: register, instantiate, and use custom metric."""
        # Create registry
        registry = CustomMetricRegistry()
        
        # Register metric
        registry.register("length_scorer", TestCustomMetric)
        
        # Instantiate metric
        metric = registry.instantiate("length_scorer")
        
        # Use metric
        result = await metric.calculate(
            prompt="What is machine learning?",
            response="Machine learning is a subset of artificial intelligence."
        )
        
        # Verify result
        assert isinstance(result, DimensionScore)
        assert 0.0 <= result.score <= 1.0
        assert result.details["method"] == "length_based"
    
    @pytest.mark.asyncio
    async def test_multiple_custom_metrics(self):
        """Test using multiple custom metrics."""
        registry = CustomMetricRegistry()
        
        # Register multiple metrics
        registry.register("length", TestCustomMetric)
        registry.register("word_count", ConfigurableCustomMetric)
        
        # Instantiate both
        length_metric = registry.instantiate("length")
        word_metric = registry.instantiate("word_count", {"threshold": 0.6})
        
        # Use both on same response
        response = "This is a test response with multiple words."
        
        length_result = await length_metric.calculate(
            prompt="test",
            response=response
        )
        
        word_result = await word_metric.calculate(
            prompt="test",
            response=response
        )
        
        # Both should return valid scores
        assert 0.0 <= length_result.score <= 1.0
        assert 0.0 <= word_result.score <= 1.0
    
    def test_custom_metric_validation(self):
        """Test custom metric configuration validation."""
        # Valid config
        valid_metric = ConfigurableCustomMetric({
            "threshold": 0.7,
            "multiplier": 1.5
        })
        assert valid_metric.validate_config() is True
        
        # Invalid config (threshold out of range)
        invalid_metric = ConfigurableCustomMetric({
            "threshold": 1.5,  # > 1.0
            "multiplier": 1.0
        })
        assert invalid_metric.validate_config() is False
        
        # Invalid config (negative multiplier)
        invalid_metric2 = ConfigurableCustomMetric({
            "threshold": 0.5,
            "multiplier": -1.0
        })
        assert invalid_metric2.validate_config() is False


# Edge case tests
class TestCustomMetricEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_custom_metric_empty_response(self):
        """Test custom metric with empty response."""
        metric = TestCustomMetric()
        
        result = await metric.calculate(
            prompt="test",
            response=""
        )
        
        assert result.score == 0.0
    
    @pytest.mark.asyncio
    async def test_custom_metric_very_long_response(self):
        """Test custom metric with very long response."""
        metric = TestCustomMetric()
        
        long_response = "word " * 1000  # 5000 characters
        
        result = await metric.calculate(
            prompt="test",
            response=long_response
        )
        
        # Score should be capped at 1.0
        assert result.score == 1.0
    
    @pytest.mark.asyncio
    async def test_custom_metric_with_none_optional_params(self):
        """Test custom metric handles None for optional parameters."""
        metric = TestCustomMetric()
        
        result = await metric.calculate(
            prompt="test",
            response="test response",
            expected_response=None,
            source_documents=None,
            model_id=None
        )
        
        assert isinstance(result, DimensionScore)
    
    def test_registry_error_messages_helpful(self):
        """Test registry error messages are helpful."""
        registry = CustomMetricRegistry()
        
        # Try to get non-existent metric
        try:
            registry.get("nonexistent")
        except KeyError as e:
            assert "nonexistent" in str(e)
            assert "Available metrics" in str(e)
