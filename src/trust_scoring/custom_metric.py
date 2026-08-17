"""
Custom Metric Plugin Interface for Trust Scoring Engine.

This module provides a plugin architecture for extending the trust scoring
system with custom user-defined metrics. Users can implement custom scoring
functions and register them with the TrustScoringEngine to be included in
the overall trust score calculation.

Requirements: 5.6, 5.11
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.data_models.trust_score import DimensionScore, TrustDimension


class CustomMetric(ABC):
    """
    Abstract base class for custom trust metrics.
    
    Users can extend this class to implement custom scoring logic that
    integrates with the TrustScoringEngine. Custom metrics are calculated
    alongside the five standard dimensions and can be weighted into the
    overall trust score.
    
    Example:
        ```python
        class DomainSpecificMetric(CustomMetric):
            def __init__(self, config: dict):
                super().__init__(config)
                self.threshold = config.get('threshold', 0.7)
            
            async def calculate(
                self,
                prompt: str,
                response: str,
                **kwargs
            ) -> DimensionScore:
                # Custom scoring logic
                score = self._analyze_domain_compliance(response)
                
                return DimensionScore(
                    dimension=TrustDimension.CUSTOM,
                    score=score,
                    confidence=0.9,
                    details={'method': 'domain_analysis'},
                    checks_passed=['domain_compliant'] if score > self.threshold else [],
                    checks_failed=[] if score > self.threshold else ['domain_non_compliant']
                )
        ```
    
    Requirement 5.11: Allow users to register custom scoring functions
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the custom metric.
        
        Args:
            config: Configuration dictionary for the custom metric.
                    Can contain any parameters needed by the implementation.
        """
        self.config = config or {}
    
    @abstractmethod
    async def calculate(
        self,
        prompt: str,
        response: str,
        expected_response: Optional[str] = None,
        source_documents: Optional[list[str]] = None,
        model_id: Optional[str] = None,
        **kwargs
    ) -> DimensionScore:
        """
        Calculate the custom metric score.
        
        This method must be implemented by subclasses to provide custom
        scoring logic. The method receives the same inputs as the standard
        trust scoring dimensions.
        
        Args:
            prompt: The original prompt/query
            response: The model's response text
            expected_response: Expected answer for comparison (optional)
            source_documents: Context documents for grounding check (optional)
            model_id: Model ID for model-specific scoring (optional)
            **kwargs: Additional custom parameters
            
        Returns:
            DimensionScore with the custom metric score in [0, 1] range.
            The dimension field can use TrustDimension.CUSTOM or a custom
            string identifier.
            
        Raises:
            NotImplementedError: If not implemented by subclass
            
        Requirement 5.11: Custom scoring function interface
        """
        raise NotImplementedError(
            "Custom metrics must implement the calculate() method"
        )
    
    def get_name(self) -> str:
        """
        Get the name of this custom metric.
        
        By default, returns the class name. Can be overridden to provide
        a more descriptive name.
        
        Returns:
            Name of the custom metric
        """
        return self.__class__.__name__
    
    def get_description(self) -> str:
        """
        Get a description of what this custom metric measures.
        
        Can be overridden to provide detailed documentation of the metric.
        
        Returns:
            Description of the custom metric
        """
        return f"Custom metric: {self.get_name()}"
    
    def validate_config(self) -> bool:
        """
        Validate the configuration for this custom metric.
        
        Can be overridden to implement custom validation logic.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        return True


class CustomMetricRegistry:
    """
    Registry for managing custom metrics in the TrustScoringEngine.
    
    This class maintains a collection of registered custom metrics and
    provides methods for registration, retrieval, and instantiation.
    
    Requirement 5.11: Custom metric plugin interface
    """
    
    def __init__(self):
        """Initialize the custom metric registry."""
        self._metrics: dict[str, type[CustomMetric]] = {}
    
    def register(
        self,
        name: str,
        metric_class: type[CustomMetric]
    ) -> None:
        """
        Register a custom metric class.
        
        Args:
            name: Unique identifier for the metric
            metric_class: Class that extends CustomMetric
            
        Raises:
            ValueError: If name is already registered or metric_class
                       is not a subclass of CustomMetric
        """
        if name in self._metrics:
            raise ValueError(
                f"Custom metric '{name}' is already registered. "
                "Use a different name or unregister the existing metric first."
            )
        
        if not issubclass(metric_class, CustomMetric):
            raise ValueError(
                f"Metric class must extend CustomMetric, got {metric_class}"
            )
        
        self._metrics[name] = metric_class
    
    def unregister(self, name: str) -> None:
        """
        Unregister a custom metric.
        
        Args:
            name: Name of the metric to unregister
            
        Raises:
            KeyError: If metric name is not registered
        """
        if name not in self._metrics:
            raise KeyError(f"Custom metric '{name}' is not registered")
        
        del self._metrics[name]
    
    def get(self, name: str) -> type[CustomMetric]:
        """
        Get a registered custom metric class.
        
        Args:
            name: Name of the metric
            
        Returns:
            The custom metric class
            
        Raises:
            KeyError: If metric name is not registered
        """
        if name not in self._metrics:
            raise KeyError(
                f"Custom metric '{name}' is not registered. "
                f"Available metrics: {list(self._metrics.keys())}"
            )
        
        return self._metrics[name]
    
    def instantiate(
        self,
        name: str,
        config: Optional[dict] = None
    ) -> CustomMetric:
        """
        Instantiate a registered custom metric.
        
        Args:
            name: Name of the metric
            config: Configuration dictionary for the metric
            
        Returns:
            Instance of the custom metric
            
        Raises:
            KeyError: If metric name is not registered
        """
        metric_class = self.get(name)
        return metric_class(config=config)
    
    def list_metrics(self) -> list[str]:
        """
        List all registered custom metric names.
        
        Returns:
            List of registered metric names
        """
        return list(self._metrics.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a metric is registered.
        
        Args:
            name: Name of the metric
            
        Returns:
            True if registered, False otherwise
        """
        return name in self._metrics
    
    def clear(self) -> None:
        """Clear all registered metrics."""
        self._metrics.clear()


# Global registry instance
_global_registry = CustomMetricRegistry()


def register_custom_metric(
    name: str,
    metric_class: type[CustomMetric]
) -> None:
    """
    Register a custom metric in the global registry.
    
    This is a convenience function for registering metrics without
    directly accessing the global registry.
    
    Args:
        name: Unique identifier for the metric
        metric_class: Class that extends CustomMetric
        
    Example:
        ```python
        register_custom_metric('domain_compliance', DomainComplianceMetric)
        ```
    """
    _global_registry.register(name, metric_class)


def get_global_registry() -> CustomMetricRegistry:
    """
    Get the global custom metric registry.
    
    Returns:
        The global CustomMetricRegistry instance
    """
    return _global_registry
