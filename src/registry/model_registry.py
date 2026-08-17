"""Model Registry for centralized model discovery and metadata management.

This module implements the ModelRegistry class which provides a unified
interface for discovering, registering, and managing models from multiple
providers (AWS Bedrock, SageMaker, External APIs).

Requirements: 1.1, 1.4, 1.5, 1.9, 1.10
"""

from datetime import datetime
from typing import Optional
from collections import defaultdict

from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
)
from src.adapters.base_adapter import BaseModelAdapter
from src.registry.pricing_loader import PricingLoader


class ModelRegistry:
    """Central registry for all available models.

    The ModelRegistry maintains an in-memory cache of model metadata with
    optional DynamoDB persistence. It provides methods for model discovery,
    registration, querying, and health checking across multiple providers.

    Requirements:
        - 1.1: Automatic discovery of AWS Bedrock models
        - 1.4: Maintain metadata including provider, capabilities, pricing
        - 1.5: Track model availability status
        - 1.9: Support filtering by provider, capability, status
        - 1.10: Cache model metadata with configurable TTL

    Attributes:
        cache_ttl_seconds: Time-to-live for cached model metadata
        _models: In-memory cache of model metadata by model ID
        _adapters: Registry of adapter instances by provider
        _cache_timestamps: Timestamps for cache invalidation
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 3600,
        dynamodb_table_name: Optional[str] = None,
        enable_persistence: bool = False,
        pricing_config_path: Optional[str] = None
    ):
        """Initialize the ModelRegistry.

        Args:
            cache_ttl_seconds: TTL for cached metadata (default: 1 hour)
            dynamodb_table_name: Name of DynamoDB table for persistence
            enable_persistence: Whether to enable DynamoDB persistence
            pricing_config_path: Path to pricing configuration file
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self.dynamodb_table_name = dynamodb_table_name
        self.enable_persistence = enable_persistence

        # In-memory cache: model_id -> ModelMetadata
        self._models: dict[str, ModelMetadata] = {}

        # Adapter registry: provider -> adapter instance
        self._adapters: dict[ModelProvider, BaseModelAdapter] = {}

        # Cache timestamps: model_id -> datetime
        self._cache_timestamps: dict[str, datetime] = {}

        # Provider-specific adapter instances
        self._provider_adapters: dict[
            ModelProvider, list[BaseModelAdapter]
        ] = defaultdict(list)

        # Cache statistics
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        
        # Pricing loader (Requirement 1.14)
        self._pricing_loader = PricingLoader(pricing_config_path)
    
    def _is_cache_valid(self, model_id: str) -> bool:
        """Check if cached metadata is still valid.

        Args:
            model_id: The model ID to check

        Returns:
            True if cache is valid, False if expired or not cached
        """
        if model_id not in self._cache_timestamps:
            return False

        timestamp = self._cache_timestamps[model_id]
        age = datetime.now() - timestamp
        return age.total_seconds() < self.cache_ttl_seconds

    def _invalidate_cache(self, model_id: str) -> None:
        """Invalidate cache for a specific model.

        Args:
            model_id: The model ID to invalidate
        """
        if model_id in self._cache_timestamps:
            del self._cache_timestamps[model_id]

    def _update_cache(self, metadata: ModelMetadata) -> None:
        """Update cache with new metadata.

        Args:
            metadata: The model metadata to cache
        """
        self._models[metadata.id] = metadata
        self._cache_timestamps[metadata.id] = datetime.now()

    def register_adapter(
        self,
        provider: ModelProvider,
        adapter: BaseModelAdapter
    ) -> None:
        """Register an adapter for a specific provider.

        Args:
            provider: The model provider
            adapter: The adapter instance
        """
        self._provider_adapters[provider].append(adapter)
    
    async def discover_models(self) -> list[ModelMetadata]:
        """Discover all available models from all registered providers.

        This method queries all registered adapters to discover available
        models and updates the registry cache. It also applies pricing
        information from the pricing configuration.

        Returns:
            List of discovered ModelMetadata objects

        Raises:
            RuntimeError: If model discovery fails for all providers
        """
        discovered_models = []
        errors = []

        for provider, adapters in self._provider_adapters.items():
            for adapter in adapters:
                try:
                    models = await adapter.list_models()
                    for model in models:
                        # Apply pricing information (Requirement 1.14)
                        model = self._pricing_loader.apply_pricing_to_metadata(model)
                        self._update_cache(model)
                        discovered_models.append(model)
                except Exception as e:
                    errors.append(f"{provider}: {str(e)}")

        if not discovered_models and errors:
            error_msg = (
                "Failed to discover models from all providers: "
                f"{'; '.join(errors)}"
            )
            raise RuntimeError(error_msg)

        return discovered_models
    
    async def register_model(self, metadata: ModelMetadata) -> None:
        """Register a new model in the registry.

        Args:
            metadata: The model metadata to register
        """
        # Update cache
        self._update_cache(metadata)

        # TODO: Persist to DynamoDB if enabled
        if self.enable_persistence and self.dynamodb_table_name:
            # Placeholder for DynamoDB persistence
            pass

    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID.

        This method first checks the cache. If the cached entry is valid,
        it returns immediately (cache hit). If the cache is expired or
        missing, it attempts to fetch from adapters (cache miss).

        Args:
            model_id: The unique identifier of the model

        Returns:
            ModelMetadata if found, None otherwise
        """
        # Check cache validity
        if self._is_cache_valid(model_id):
            self._cache_hits += 1
            return self._models.get(model_id)

        # Cache miss or expired
        self._cache_misses += 1

        # Try to refresh from adapters
        for provider, adapters in self._provider_adapters.items():
            for adapter in adapters:
                try:
                    metadata = await adapter.get_model_info(model_id)
                    self._update_cache(metadata)
                    return metadata
                except (ValueError, RuntimeError):
                    # Model not found in this adapter, try next
                    continue

        # Return cached value even if expired, or None
        return self._models.get(model_id)
    
    async def list_models(
        self,
        provider: Optional[ModelProvider] = None,
        capability: Optional[ModelCapability] = None,
        status: Optional[ModelStatus] = None,
        fine_tunable_only: bool = False
    ) -> list[ModelMetadata]:
        """List models with optional filtering.

        Args:
            provider: Filter by provider (optional)
            capability: Filter by capability (optional)
            status: Filter by status (optional)
            fine_tunable_only: Only return fine-tunable models

        Returns:
            List of ModelMetadata matching the filters
        """
        models = list(self._models.values())

        # Apply filters
        if provider is not None:
            models = [m for m in models if m.provider == provider]

        if capability is not None:
            models = [m for m in models if capability in m.capabilities]

        if status is not None:
            models = [m for m in models if m.status == status]

        if fine_tunable_only:
            models = [m for m in models if m.fine_tuning_support]

        return models

    async def update_model_status(
        self,
        model_id: str,
        status: ModelStatus
    ) -> None:
        """Update model availability status.

        Args:
            model_id: The model ID to update
            status: The new status

        Raises:
            ValueError: If model not found
        """
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")

        # Update status
        self._models[model_id].status = status

        # Invalidate cache to force refresh on next access
        self._invalidate_cache(model_id)

        # TODO: Persist to DynamoDB if enabled
        if self.enable_persistence and self.dynamodb_table_name:
            # Placeholder for DynamoDB persistence
            pass
    
    async def health_check(self, model_id: str) -> bool:
        """Check if a model is healthy and available.

        This method attempts to validate connectivity to the model's
        provider and updates the model status accordingly.

        Args:
            model_id: The model ID to check

        Returns:
            True if model is healthy, False otherwise
        """
        metadata = await self.get_model(model_id)
        if not metadata:
            return False

        # Find the appropriate adapter
        adapters = self._provider_adapters.get(metadata.provider, [])

        for adapter in adapters:
            try:
                is_healthy = await adapter.validate_connection()

                # Update health check timestamp
                metadata.last_health_check = datetime.now()

                # Update status based on health check
                if is_healthy:
                    if metadata.status != ModelStatus.ACTIVE:
                        await self.update_model_status(
                            model_id, ModelStatus.ACTIVE
                        )
                else:
                    if metadata.status == ModelStatus.ACTIVE:
                        await self.update_model_status(
                            model_id, ModelStatus.INACTIVE
                        )

                return is_healthy
            except Exception:
                # Health check failed
                if metadata.status == ModelStatus.ACTIVE:
                    await self.update_model_status(
                        model_id, ModelStatus.INACTIVE
                    )
                return False

        return False

    def get_adapter(self, model_id: str) -> Optional[BaseModelAdapter]:
        """Get the appropriate adapter for a model.

        Args:
            model_id: The model ID

        Returns:
            BaseModelAdapter instance if found, None otherwise
        """
        metadata = self._models.get(model_id)
        if not metadata:
            return None

        # Return the first adapter for this provider
        adapters = self._provider_adapters.get(metadata.provider, [])
        return adapters[0] if adapters else None

    async def clear_cache(self) -> None:
        """Clear all cached model metadata."""
        self._models.clear()
        self._cache_timestamps.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    async def refresh_model_cache(self, model_id: str) -> Optional[ModelMetadata]:
        """Manually refresh cache for a specific model.

        This method forces a cache refresh by fetching the latest metadata
        from adapters, regardless of cache validity.

        Args:
            model_id: The model ID to refresh

        Returns:
            Updated ModelMetadata if found, None otherwise
        """
        # Invalidate existing cache
        self._invalidate_cache(model_id)

        # Fetch fresh data from adapters
        for provider, adapters in self._provider_adapters.items():
            for adapter in adapters:
                try:
                    metadata = await adapter.get_model_info(model_id)
                    self._update_cache(metadata)
                    return metadata
                except (ValueError, RuntimeError):
                    # Model not found in this adapter, try next
                    continue

        return None

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including:
            - total_models: Total number of cached models
            - valid_cache_entries: Number of valid (non-expired) entries
            - expired_cache_entries: Number of expired entries
            - cache_ttl_seconds: Configured TTL
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            - hit_rate: Cache hit rate (0-1)
        """
        total_models = len(self._models)
        valid_cache = sum(
            1 for model_id in self._models
            if self._is_cache_valid(model_id)
        )
        expired_cache = total_models - valid_cache

        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_requests if total_requests > 0 else 0.0
        )

        return {
            "total_models": total_models,
            "valid_cache_entries": valid_cache,
            "expired_cache_entries": expired_cache,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }
    
    def get_pricing_loader(self) -> PricingLoader:
        """Get the pricing loader instance.
        
        Returns:
            PricingLoader instance used by this registry
        """
        return self._pricing_loader
    
    def reload_pricing_config(self) -> None:
        """Reload pricing configuration from file.
        
        This method reloads the pricing configuration and updates
        all cached models with the new pricing information.
        """
        self._pricing_loader.reload_config()
        
        # Update pricing for all cached models
        for model_id, metadata in self._models.items():
            updated_metadata = self._pricing_loader.apply_pricing_to_metadata(metadata)
            self._models[model_id] = updated_metadata
