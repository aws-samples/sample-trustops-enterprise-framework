"""Integration tests for ModelRegistry with PricingLoader.

Requirements: 1.4, 1.14
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from src.registry.model_registry import ModelRegistry
from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
)
from src.adapters.base_adapter import BaseModelAdapter


class MockAdapter(BaseModelAdapter):
    """Mock adapter for testing."""
    
    def __init__(self, models):
        self.models = models
    
    async def invoke(self, request):
        pass
    
    async def invoke_stream(self, request):
        pass
    
    async def list_models(self):
        return self.models
    
    async def get_model_info(self, model_id):
        for model in self.models:
            if model.id == model_id:
                return model
        raise ValueError(f"Model not found: {model_id}")
    
    async def validate_connection(self):
        return True
    
    def supports_streaming(self):
        return False
    
    def supports_fine_tuning(self, model_id):
        return False


class TestModelRegistryPricing:
    """Test suite for ModelRegistry pricing integration."""
    
    @pytest.fixture
    def sample_pricing_yaml(self):
        """Create a sample pricing YAML file for testing."""
        yaml_content = """
bedrock:
  test-model-1:
    input_price_per_1k_tokens: 0.001
    output_price_per_1k_tokens: 0.002
    fine_tuning_price_per_1k_tokens: null
    currency: USD
  
  test-model-2:
    input_price_per_1k_tokens: 0.003
    output_price_per_1k_tokens: 0.009
    fine_tuning_price_per_1k_tokens: 0.015
    currency: USD

default:
  input_price_per_1k_tokens: 0.001
  output_price_per_1k_tokens: 0.002
  fine_tuning_price_per_1k_tokens: null
  currency: USD
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        yield temp_path
        
        Path(temp_path).unlink()
    
    @pytest.mark.asyncio
    async def test_discover_models_applies_pricing(self, sample_pricing_yaml):
        """Test that discover_models applies pricing to discovered models."""
        # Create registry with pricing config
        registry = ModelRegistry(pricing_config_path=sample_pricing_yaml)
        
        # Create mock models without pricing
        mock_models = [
            ModelMetadata(
                id="test-model-1",
                provider=ModelProvider.BEDROCK,
                name="Test Model 1",
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=False,
                max_tokens=4096,
                region="us-east-1"
            ),
            ModelMetadata(
                id="test-model-2",
                provider=ModelProvider.BEDROCK,
                name="Test Model 2",
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=True,
                max_tokens=8192,
                region="us-east-1"
            )
        ]
        
        # Register mock adapter
        adapter = MockAdapter(mock_models)
        registry.register_adapter(ModelProvider.BEDROCK, adapter)
        
        # Discover models
        discovered = await registry.discover_models()
        
        # Verify pricing was applied
        assert len(discovered) == 2
        
        model1 = next(m for m in discovered if m.id == "test-model-1")
        assert model1.pricing is not None
        assert model1.pricing.input_price_per_1k_tokens == 0.001
        assert model1.pricing.output_price_per_1k_tokens == 0.002
        assert model1.pricing.fine_tuning_price_per_1k_tokens is None
        
        model2 = next(m for m in discovered if m.id == "test-model-2")
        assert model2.pricing is not None
        assert model2.pricing.input_price_per_1k_tokens == 0.003
        assert model2.pricing.output_price_per_1k_tokens == 0.009
        assert model2.pricing.fine_tuning_price_per_1k_tokens == 0.015
    
    @pytest.mark.asyncio
    async def test_discover_models_applies_default_pricing(self, sample_pricing_yaml):
        """Test that unknown models get default pricing."""
        registry = ModelRegistry(pricing_config_path=sample_pricing_yaml)
        
        # Create model not in pricing config
        mock_models = [
            ModelMetadata(
                id="unknown-model",
                provider=ModelProvider.BEDROCK,
                name="Unknown Model",
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=False,
                max_tokens=4096,
                region="us-east-1"
            )
        ]
        
        adapter = MockAdapter(mock_models)
        registry.register_adapter(ModelProvider.BEDROCK, adapter)
        
        discovered = await registry.discover_models()
        
        # Verify default pricing was applied
        assert len(discovered) == 1
        model = discovered[0]
        assert model.pricing is not None
        assert model.pricing.input_price_per_1k_tokens == 0.001
        assert model.pricing.output_price_per_1k_tokens == 0.002
    
    def test_get_pricing_loader(self, sample_pricing_yaml):
        """Test accessing the pricing loader from registry."""
        registry = ModelRegistry(pricing_config_path=sample_pricing_yaml)
        
        loader = registry.get_pricing_loader()
        assert loader is not None
        
        # Verify loader works
        pricing = loader.get_model_pricing("test-model-1", ModelProvider.BEDROCK)
        assert pricing is not None
        assert pricing.input_price_per_1k_tokens == 0.001
    
    @pytest.mark.asyncio
    async def test_reload_pricing_config(self, sample_pricing_yaml):
        """Test reloading pricing configuration updates cached models."""
        registry = ModelRegistry(pricing_config_path=sample_pricing_yaml)
        
        # Create and register a model
        mock_models = [
            ModelMetadata(
                id="test-model-1",
                provider=ModelProvider.BEDROCK,
                name="Test Model 1",
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=False,
                max_tokens=4096,
                region="us-east-1"
            )
        ]
        
        adapter = MockAdapter(mock_models)
        registry.register_adapter(ModelProvider.BEDROCK, adapter)
        
        # Discover models
        await registry.discover_models()
        
        # Get model and verify pricing
        model = await registry.get_model("test-model-1")
        assert model.pricing is not None
        original_price = model.pricing.input_price_per_1k_tokens
        
        # Reload pricing config
        registry.reload_pricing_config()
        
        # Get model again and verify pricing is still applied
        model = await registry.get_model("test-model-1")
        assert model.pricing is not None
        assert model.pricing.input_price_per_1k_tokens == original_price
    
    @pytest.mark.asyncio
    async def test_pricing_persists_in_cache(self, sample_pricing_yaml):
        """Test that pricing information persists in cache."""
        registry = ModelRegistry(
            pricing_config_path=sample_pricing_yaml,
            cache_ttl_seconds=3600
        )
        
        mock_models = [
            ModelMetadata(
                id="test-model-1",
                provider=ModelProvider.BEDROCK,
                name="Test Model 1",
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=False,
                max_tokens=4096,
                region="us-east-1"
            )
        ]
        
        adapter = MockAdapter(mock_models)
        registry.register_adapter(ModelProvider.BEDROCK, adapter)
        
        # Discover models
        await registry.discover_models()
        
        # Get model from cache (should have pricing)
        model = await registry.get_model("test-model-1")
        assert model.pricing is not None
        assert model.pricing.input_price_per_1k_tokens == 0.001
        
        # Get again (from cache)
        model2 = await registry.get_model("test-model-1")
        assert model2.pricing is not None
        assert model2.pricing.input_price_per_1k_tokens == 0.001
    
    def test_default_pricing_config_path(self):
        """Test that registry uses default pricing config path."""
        registry = ModelRegistry()
        
        loader = registry.get_pricing_loader()
        assert loader is not None
        assert loader.config_path.name == "model_pricing.yaml"
    
    @pytest.mark.asyncio
    async def test_list_models_includes_pricing(self, sample_pricing_yaml):
        """Test that list_models returns models with pricing."""
        registry = ModelRegistry(pricing_config_path=sample_pricing_yaml)
        
        mock_models = [
            ModelMetadata(
                id="test-model-1",
                provider=ModelProvider.BEDROCK,
                name="Test Model 1",
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=False,
                max_tokens=4096,
                region="us-east-1"
            ),
            ModelMetadata(
                id="test-model-2",
                provider=ModelProvider.BEDROCK,
                name="Test Model 2",
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=True,
                max_tokens=8192,
                region="us-east-1"
            )
        ]
        
        adapter = MockAdapter(mock_models)
        registry.register_adapter(ModelProvider.BEDROCK, adapter)
        
        # Discover models
        await registry.discover_models()
        
        # List all models
        models = await registry.list_models()
        
        # Verify all models have pricing
        assert len(models) == 2
        assert all(model.pricing is not None for model in models)
