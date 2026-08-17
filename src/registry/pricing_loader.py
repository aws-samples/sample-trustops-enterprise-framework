"""Model pricing configuration loader.

This module provides functionality to load model pricing information from
YAML configuration files and apply it to model metadata.

Requirements: 1.4, 1.14
"""

import os
from pathlib import Path
from typing import Optional, Dict
import yaml

from src.data_models.model import ModelMetadata, ModelPricing, ModelProvider


class PricingLoader:
    """Loader for model pricing configuration.
    
    This class loads pricing information from a YAML configuration file
    and provides methods to retrieve pricing for specific models.
    
    Requirements:
        - 1.4: Maintain pricing metadata for each model
        - 1.14: Load pricing configuration from JSON/YAML config file
    
    Attributes:
        config_path: Path to the pricing configuration file
        _pricing_data: Cached pricing data loaded from the config file
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the PricingLoader.
        
        Args:
            config_path: Path to the pricing YAML file. If None, uses default
                        location at config/model_pricing.yaml
        """
        if config_path is None:
            # Default to config/model_pricing.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "model_pricing.yaml"
        
        self.config_path = Path(config_path)
        self._pricing_data: Optional[Dict] = None
    
    def load_pricing_config(self) -> Dict:
        """Load pricing configuration from YAML file.
        
        Returns:
            Dictionary containing pricing data organized by provider
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the config file is invalid YAML
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Pricing configuration file not found: {self.config_path}"
            )
        
        with open(self.config_path, 'r') as f:
            self._pricing_data = yaml.safe_load(f)
        
        return self._pricing_data
    
    def get_pricing_data(self) -> Dict:
        """Get cached pricing data, loading if necessary.
        
        Returns:
            Dictionary containing pricing data
        """
        if self._pricing_data is None:
            self.load_pricing_config()
        return self._pricing_data
    
    def get_model_pricing(
        self,
        model_id: str,
        provider: ModelProvider
    ) -> Optional[ModelPricing]:
        """Get pricing information for a specific model.
        
        Args:
            model_id: The unique identifier of the model
            provider: The provider of the model
            
        Returns:
            ModelPricing object if found, None otherwise
        """
        pricing_data = self.get_pricing_data()
        
        # Map provider enum to config section
        provider_key = self._get_provider_key(provider)
        
        # Look up pricing in provider section
        provider_pricing = pricing_data.get(provider_key, {})
        model_pricing_dict = provider_pricing.get(model_id)
        
        # If not found, try default for provider
        if model_pricing_dict is None and provider == ModelProvider.SAGEMAKER:
            model_pricing_dict = provider_pricing.get("default")
        
        # If still not found, use global default
        if model_pricing_dict is None:
            model_pricing_dict = pricing_data.get("default")
        
        # Convert to ModelPricing object
        if model_pricing_dict:
            return ModelPricing(
                input_price_per_1k_tokens=model_pricing_dict["input_price_per_1k_tokens"],
                output_price_per_1k_tokens=model_pricing_dict["output_price_per_1k_tokens"],
                fine_tuning_price_per_1k_tokens=model_pricing_dict.get("fine_tuning_price_per_1k_tokens"),
                currency=model_pricing_dict.get("currency", "USD")
            )
        
        return None
    
    def _get_provider_key(self, provider: ModelProvider) -> str:
        """Map ModelProvider enum to config file key.
        
        Args:
            provider: The ModelProvider enum value
            
        Returns:
            String key for the config file
        """
        if provider == ModelProvider.BEDROCK:
            return "bedrock"
        elif provider == ModelProvider.SAGEMAKER:
            return "sagemaker"
        elif provider == ModelProvider.EXTERNAL_API:
            return "external_api"
        else:
            return "default"
    
    def apply_pricing_to_metadata(
        self,
        metadata: ModelMetadata
    ) -> ModelMetadata:
        """Apply pricing information to model metadata.
        
        This method looks up pricing for the model and updates the
        metadata object with the pricing information.
        
        Args:
            metadata: The ModelMetadata object to update
            
        Returns:
            Updated ModelMetadata object with pricing information
        """
        pricing = self.get_model_pricing(metadata.id, metadata.provider)
        if pricing:
            metadata.pricing = pricing
        return metadata
    
    def apply_pricing_to_models(
        self,
        models: list[ModelMetadata]
    ) -> list[ModelMetadata]:
        """Apply pricing information to a list of models.
        
        Args:
            models: List of ModelMetadata objects to update
            
        Returns:
            List of updated ModelMetadata objects with pricing information
        """
        return [self.apply_pricing_to_metadata(model) for model in models]
    
    def reload_config(self) -> None:
        """Reload pricing configuration from file.
        
        This method clears the cache and reloads the configuration,
        useful when the config file has been updated.
        """
        self._pricing_data = None
        self.load_pricing_config()
    
    def get_all_model_ids(self, provider: Optional[ModelProvider] = None) -> list[str]:
        """Get all model IDs that have pricing configured.
        
        Args:
            provider: Optional provider filter
            
        Returns:
            List of model IDs with pricing information
        """
        pricing_data = self.get_pricing_data()
        model_ids = []
        
        if provider:
            provider_key = self._get_provider_key(provider)
            provider_pricing = pricing_data.get(provider_key, {})
            model_ids = [
                model_id for model_id in provider_pricing.keys()
                if model_id != "default" and model_id != "note"
            ]
        else:
            # Get all model IDs from all providers
            for provider_key in ["bedrock", "external_api", "sagemaker"]:
                provider_pricing = pricing_data.get(provider_key, {})
                model_ids.extend([
                    model_id for model_id in provider_pricing.keys()
                    if model_id != "default" and model_id != "note"
                ])
        
        return model_ids
