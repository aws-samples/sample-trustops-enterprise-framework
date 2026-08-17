"""Unit tests for the PricingLoader.

Requirements: 1.4, 1.14
"""

import pytest
import tempfile
from pathlib import Path

from src.registry.pricing_loader import PricingLoader
from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
    ModelPricing
)


class TestPricingLoader:
    """Test suite for PricingLoader."""
    
    @pytest.fixture
    def sample_pricing_yaml(self):
        """Create a sample pricing YAML file for testing."""
        yaml_content = """
bedrock:
  anthropic.claude-3-haiku-20240307-v1:0:
    input_price_per_1k_tokens: 0.00025
    output_price_per_1k_tokens: 0.00125
    fine_tuning_price_per_1k_tokens: null
    currency: USD
  
  amazon.titan-text-express-v1:
    input_price_per_1k_tokens: 0.0002
    output_price_per_1k_tokens: 0.0006
    fine_tuning_price_per_1k_tokens: 0.008
    currency: USD

external_api:
  gpt-3.5-turbo:
    input_price_per_1k_tokens: 0.0005
    output_price_per_1k_tokens: 0.0015
    fine_tuning_price_per_1k_tokens: 0.008
    currency: USD

sagemaker:
  default:
    input_price_per_1k_tokens: 0.001
    output_price_per_1k_tokens: 0.002
    fine_tuning_price_per_1k_tokens: 0.005
    currency: USD

default:
  input_price_per_1k_tokens: 0.001
  output_price_per_1k_tokens: 0.002
  fine_tuning_price_per_1k_tokens: null
  currency: USD
"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_load_pricing_config(self, sample_pricing_yaml):
        """Test loading pricing configuration from YAML file."""
        loader = PricingLoader(sample_pricing_yaml)
        pricing_data = loader.load_pricing_config()
        
        assert pricing_data is not None
        assert "bedrock" in pricing_data
        assert "external_api" in pricing_data
        assert "sagemaker" in pricing_data
        assert "default" in pricing_data
    
    def test_load_pricing_config_file_not_found(self):
        """Test error handling when config file doesn't exist."""
        loader = PricingLoader("/nonexistent/path/pricing.yaml")
        
        with pytest.raises(FileNotFoundError):
            loader.load_pricing_config()
    
    def test_get_model_pricing_bedrock(self, sample_pricing_yaml):
        """Test retrieving pricing for a Bedrock model."""
        loader = PricingLoader(sample_pricing_yaml)
        
        pricing = loader.get_model_pricing(
            "anthropic.claude-3-haiku-20240307-v1:0",
            ModelProvider.BEDROCK
        )
        
        assert pricing is not None
        assert isinstance(pricing, ModelPricing)
        assert pricing.input_price_per_1k_tokens == 0.00025
        assert pricing.output_price_per_1k_tokens == 0.00125
        assert pricing.fine_tuning_price_per_1k_tokens is None
        assert pricing.currency == "USD"
    
    def test_get_model_pricing_with_fine_tuning(self, sample_pricing_yaml):
        """Test retrieving pricing for a model with fine-tuning support."""
        loader = PricingLoader(sample_pricing_yaml)
        
        pricing = loader.get_model_pricing(
            "amazon.titan-text-express-v1",
            ModelProvider.BEDROCK
        )
        
        assert pricing is not None
        assert pricing.fine_tuning_price_per_1k_tokens == 0.008
    
    def test_get_model_pricing_external_api(self, sample_pricing_yaml):
        """Test retrieving pricing for an external API model."""
        loader = PricingLoader(sample_pricing_yaml)
        
        pricing = loader.get_model_pricing(
            "gpt-3.5-turbo",
            ModelProvider.EXTERNAL_API
        )
        
        assert pricing is not None
        assert pricing.input_price_per_1k_tokens == 0.0005
        assert pricing.output_price_per_1k_tokens == 0.0015
    
    def test_get_model_pricing_sagemaker_default(self, sample_pricing_yaml):
        """Test retrieving default pricing for SageMaker models."""
        loader = PricingLoader(sample_pricing_yaml)
        
        pricing = loader.get_model_pricing(
            "custom-sagemaker-model",
            ModelProvider.SAGEMAKER
        )
        
        assert pricing is not None
        assert pricing.input_price_per_1k_tokens == 0.001
        assert pricing.output_price_per_1k_tokens == 0.002
        assert pricing.fine_tuning_price_per_1k_tokens == 0.005
    
    def test_get_model_pricing_fallback_to_default(self, sample_pricing_yaml):
        """Test fallback to default pricing for unknown models."""
        loader = PricingLoader(sample_pricing_yaml)
        
        pricing = loader.get_model_pricing(
            "unknown-model-id",
            ModelProvider.BEDROCK
        )
        
        assert pricing is not None
        assert pricing.input_price_per_1k_tokens == 0.001
        assert pricing.output_price_per_1k_tokens == 0.002
    
    def test_apply_pricing_to_metadata(self, sample_pricing_yaml):
        """Test applying pricing to model metadata."""
        loader = PricingLoader(sample_pricing_yaml)
        
        metadata = ModelMetadata(
            id="anthropic.claude-3-haiku-20240307-v1:0",
            provider=ModelProvider.BEDROCK,
            name="Claude 3 Haiku",
            capabilities=[ModelCapability.TEXT_GENERATION],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1"
        )
        
        updated_metadata = loader.apply_pricing_to_metadata(metadata)
        
        assert updated_metadata.pricing is not None
        assert updated_metadata.pricing.input_price_per_1k_tokens == 0.00025
        assert updated_metadata.pricing.output_price_per_1k_tokens == 0.00125
    
    def test_apply_pricing_to_models(self, sample_pricing_yaml):
        """Test applying pricing to multiple models."""
        loader = PricingLoader(sample_pricing_yaml)
        
        models = [
            ModelMetadata(
                id="anthropic.claude-3-haiku-20240307-v1:0",
                provider=ModelProvider.BEDROCK,
                name="Claude 3 Haiku",
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=False,
                max_tokens=4096,
                region="us-east-1"
            ),
            ModelMetadata(
                id="gpt-3.5-turbo",
                provider=ModelProvider.EXTERNAL_API,
                name="GPT-3.5 Turbo",
                capabilities=[ModelCapability.CHAT],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=True,
                max_tokens=4096,
                region="us-east-1"
            )
        ]
        
        updated_models = loader.apply_pricing_to_models(models)
        
        assert len(updated_models) == 2
        assert all(model.pricing is not None for model in updated_models)
        assert updated_models[0].pricing.input_price_per_1k_tokens == 0.00025
        assert updated_models[1].pricing.input_price_per_1k_tokens == 0.0005
    
    def test_reload_config(self, sample_pricing_yaml):
        """Test reloading pricing configuration."""
        loader = PricingLoader(sample_pricing_yaml)
        
        # Load initial config
        initial_data = loader.get_pricing_data()
        assert initial_data is not None
        
        # Reload config
        loader.reload_config()
        reloaded_data = loader.get_pricing_data()
        
        assert reloaded_data is not None
        assert reloaded_data == initial_data
    
    def test_get_all_model_ids(self, sample_pricing_yaml):
        """Test retrieving all model IDs with pricing."""
        loader = PricingLoader(sample_pricing_yaml)
        
        # Get all model IDs
        all_ids = loader.get_all_model_ids()
        assert len(all_ids) > 0
        assert "anthropic.claude-3-haiku-20240307-v1:0" in all_ids
        assert "gpt-3.5-turbo" in all_ids
    
    def test_get_all_model_ids_filtered_by_provider(self, sample_pricing_yaml):
        """Test retrieving model IDs filtered by provider."""
        loader = PricingLoader(sample_pricing_yaml)
        
        # Get Bedrock model IDs
        bedrock_ids = loader.get_all_model_ids(ModelProvider.BEDROCK)
        assert len(bedrock_ids) == 2
        assert "anthropic.claude-3-haiku-20240307-v1:0" in bedrock_ids
        assert "amazon.titan-text-express-v1" in bedrock_ids
        
        # Get External API model IDs
        external_ids = loader.get_all_model_ids(ModelProvider.EXTERNAL_API)
        assert len(external_ids) == 1
        assert "gpt-3.5-turbo" in external_ids
    
    def test_pricing_data_caching(self, sample_pricing_yaml):
        """Test that pricing data is cached after first load."""
        loader = PricingLoader(sample_pricing_yaml)
        
        # First call loads from file
        data1 = loader.get_pricing_data()
        
        # Second call should use cached data
        data2 = loader.get_pricing_data()
        
        assert data1 is data2  # Same object reference
    
    def test_default_config_path(self):
        """Test that default config path is set correctly."""
        loader = PricingLoader()
        
        # Check that config_path points to config/model_pricing.yaml
        assert loader.config_path.name == "model_pricing.yaml"
        assert "config" in str(loader.config_path)
