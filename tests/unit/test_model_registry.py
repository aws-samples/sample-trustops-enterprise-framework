"""Unit tests for ModelRegistry.

Tests the ModelRegistry implementation including model discovery,
registration, caching, filtering, and health checking.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.registry.model_registry import ModelRegistry
from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
    ModelPricing,
)
from src.adapters.base_adapter import BaseModelAdapter

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_model_metadata():
    """Create sample model metadata for testing."""
    return ModelMetadata(
        id="test-model-1",
        provider=ModelProvider.BEDROCK,
        name="Test Model 1",
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
        status=ModelStatus.ACTIVE,
        fine_tuning_support=True,
        input_modalities=["text"],
        output_modalities=["text"],
        max_tokens=4096,
        region="us-east-1",
        pricing=ModelPricing(
            input_price_per_1k_tokens=0.001,
            output_price_per_1k_tokens=0.002,
        ),
    )


@pytest.fixture
def mock_adapter():
    """Create a mock adapter."""
    adapter = Mock(spec=BaseModelAdapter)
    adapter.list_models = AsyncMock()
    adapter.get_model_info = AsyncMock()
    adapter.validate_connection = AsyncMock()
    adapter.supports_streaming = Mock(return_value=True)
    adapter.supports_fine_tuning = Mock(return_value=True)
    return adapter


@pytest.fixture
def registry():
    """Create a ModelRegistry instance."""
    return ModelRegistry(cache_ttl_seconds=3600)


class TestModelRegistryInit:
    """Test ModelRegistry initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        registry = ModelRegistry()
        assert registry.cache_ttl_seconds == 3600
        assert registry.dynamodb_table_name is None
        assert registry.enable_persistence is False
        assert len(registry._models) == 0
        assert len(registry._adapters) == 0

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        registry = ModelRegistry(
            cache_ttl_seconds=1800,
            dynamodb_table_name="test-table",
            enable_persistence=True
        )
        assert registry.cache_ttl_seconds == 1800
        assert registry.dynamodb_table_name == "test-table"
        assert registry.enable_persistence is True


class TestModelRegistration:
    """Test model registration functionality."""

    async def test_register_model_success(self, registry, sample_model_metadata):
        """Test successful model registration."""
        await registry.register_model(sample_model_metadata)
        
        # Verify model is in cache
        assert sample_model_metadata.id in registry._models
        assert registry._models[sample_model_metadata.id] == sample_model_metadata
        
        # Verify cache timestamp is set
        assert sample_model_metadata.id in registry._cache_timestamps

    async def test_register_model_empty_id(self, registry):
        """Test registration fails with empty model ID (Pydantic validation)."""
        # Pydantic validates this at model creation time
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            metadata = ModelMetadata(
                id="",
                provider=ModelProvider.BEDROCK,
                name="Test Model",
                capabilities=[],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=False,
                max_tokens=4096,
                region="us-east-1",
            )

    async def test_register_model_empty_name(self, registry):
        """Test registration fails with empty model name (Pydantic validation)."""
        # Pydantic validates this at model creation time
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            metadata = ModelMetadata(
                id="test-id",
                provider=ModelProvider.BEDROCK,
                name="",
                capabilities=[],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=False,
                max_tokens=4096,
                region="us-east-1",
            )

    async def test_register_model_updates_existing(
        self, registry, sample_model_metadata
    ):
        """Test registering a model with same ID updates existing entry."""
        # Register initial model
        await registry.register_model(sample_model_metadata)
        
        # Create updated metadata
        updated_metadata = sample_model_metadata.model_copy()
        updated_metadata.status = ModelStatus.INACTIVE
        
        # Register updated model
        await registry.register_model(updated_metadata)
        
        # Verify model is updated
        retrieved = await registry.get_model(sample_model_metadata.id)
        assert retrieved.status == ModelStatus.INACTIVE


class TestModelRetrieval:
    """Test model retrieval functionality."""

    async def test_get_model_success(self, registry, sample_model_metadata):
        """Test successful model retrieval."""
        await registry.register_model(sample_model_metadata)
        
        retrieved = await registry.get_model(sample_model_metadata.id)
        assert retrieved is not None
        assert retrieved.id == sample_model_metadata.id
        assert retrieved.name == sample_model_metadata.name

    async def test_get_model_not_found(self, registry):
        """Test retrieval of non-existent model returns None."""
        retrieved = await registry.get_model("non-existent-id")
        assert retrieved is None

    async def test_get_model_from_adapter(
        self, registry, mock_adapter, sample_model_metadata
    ):
        """Test model retrieval from adapter when not in cache."""
        # Register adapter
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)
        
        # Configure mock to return model
        mock_adapter.get_model_info.return_value = sample_model_metadata
        
        # Get model (not in cache)
        retrieved = await registry.get_model(sample_model_metadata.id)
        
        # Verify adapter was called
        mock_adapter.get_model_info.assert_called_once_with(sample_model_metadata.id)
        
        # Verify model is now cached
        assert sample_model_metadata.id in registry._models


class TestModelListing:
    """Test model listing and filtering functionality."""

    async def test_list_models_no_filters(self, registry):
        """Test listing all models without filters."""
        # Register multiple models
        models = [
            ModelMetadata(
                id=f"model-{i}",
                provider=ModelProvider.BEDROCK,
                name=f"Model {i}",
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=False,
                max_tokens=4096,
                region="us-east-1",
            )
            for i in range(3)
        ]
        
        for model in models:
            await registry.register_model(model)
        
        # List all models
        result = await registry.list_models()
        assert len(result) == 3

    async def test_list_models_filter_by_provider(self, registry):
        """Test filtering models by provider."""
        # Register models from different providers
        bedrock_model = ModelMetadata(
            id="bedrock-model",
            provider=ModelProvider.BEDROCK,
            name="Bedrock Model",
            capabilities=[],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1",
        )
        
        sagemaker_model = ModelMetadata(
            id="sagemaker-model",
            provider=ModelProvider.SAGEMAKER,
            name="SageMaker Model",
            capabilities=[],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1",
        )
        
        await registry.register_model(bedrock_model)
        await registry.register_model(sagemaker_model)
        
        # Filter by Bedrock
        result = await registry.list_models(provider=ModelProvider.BEDROCK)
        assert len(result) == 1
        assert result[0].id == "bedrock-model"

    async def test_list_models_filter_by_capability(self, registry):
        """Test filtering models by capability."""
        # Register models with different capabilities
        chat_model = ModelMetadata(
            id="chat-model",
            provider=ModelProvider.BEDROCK,
            name="Chat Model",
            capabilities=[ModelCapability.CHAT],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1",
        )
        
        embedding_model = ModelMetadata(
            id="embedding-model",
            provider=ModelProvider.BEDROCK,
            name="Embedding Model",
            capabilities=[ModelCapability.EMBEDDING],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1",
        )
        
        await registry.register_model(chat_model)
        await registry.register_model(embedding_model)
        
        # Filter by CHAT capability
        result = await registry.list_models(capability=ModelCapability.CHAT)
        assert len(result) == 1
        assert result[0].id == "chat-model"

    async def test_list_models_filter_by_status(self, registry):
        """Test filtering models by status."""
        # Register models with different statuses
        active_model = ModelMetadata(
            id="active-model",
            provider=ModelProvider.BEDROCK,
            name="Active Model",
            capabilities=[],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1",
        )
        
        inactive_model = ModelMetadata(
            id="inactive-model",
            provider=ModelProvider.BEDROCK,
            name="Inactive Model",
            capabilities=[],
            status=ModelStatus.INACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1",
        )
        
        await registry.register_model(active_model)
        await registry.register_model(inactive_model)
        
        # Filter by ACTIVE status
        result = await registry.list_models(status=ModelStatus.ACTIVE)
        assert len(result) == 1
        assert result[0].id == "active-model"

    async def test_list_models_filter_fine_tunable(self, registry):
        """Test filtering for fine-tunable models only."""
        # Register models with different fine-tuning support
        tunable_model = ModelMetadata(
            id="tunable-model",
            provider=ModelProvider.BEDROCK,
            name="Tunable Model",
            capabilities=[],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=True,
            max_tokens=4096,
            region="us-east-1",
        )
        
        non_tunable_model = ModelMetadata(
            id="non-tunable-model",
            provider=ModelProvider.BEDROCK,
            name="Non-Tunable Model",
            capabilities=[],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1",
        )
        
        await registry.register_model(tunable_model)
        await registry.register_model(non_tunable_model)
        
        # Filter for fine-tunable only
        result = await registry.list_models(fine_tunable_only=True)
        assert len(result) == 1
        assert result[0].id == "tunable-model"

    async def test_list_models_multiple_filters(self, registry):
        """Test filtering with multiple criteria."""
        # Register various models
        models = [
            ModelMetadata(
                id="model-1",
                provider=ModelProvider.BEDROCK,
                name="Model 1",
                capabilities=[ModelCapability.CHAT],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=True,
                max_tokens=4096,
                region="us-east-1",
            ),
            ModelMetadata(
                id="model-2",
                provider=ModelProvider.BEDROCK,
                name="Model 2",
                capabilities=[ModelCapability.CHAT],
                status=ModelStatus.INACTIVE,
                fine_tuning_support=True,
                max_tokens=4096,
                region="us-east-1",
            ),
            ModelMetadata(
                id="model-3",
                provider=ModelProvider.SAGEMAKER,
                name="Model 3",
                capabilities=[ModelCapability.CHAT],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=True,
                max_tokens=4096,
                region="us-east-1",
            ),
        ]
        
        for model in models:
            await registry.register_model(model)
        
        # Filter by provider, status, and fine-tunable
        result = await registry.list_models(
            provider=ModelProvider.BEDROCK,
            status=ModelStatus.ACTIVE,
            fine_tunable_only=True
        )
        assert len(result) == 1
        assert result[0].id == "model-1"


class TestModelStatusUpdate:
    """Test model status update functionality."""

    async def test_update_status_success(self, registry, sample_model_metadata):
        """Test successful status update."""
        await registry.register_model(sample_model_metadata)
        
        # Update status
        await registry.update_model_status(
            sample_model_metadata.id,
            ModelStatus.INACTIVE
        )
        
        # Verify status is updated
        model = await registry.get_model(sample_model_metadata.id)
        assert model.status == ModelStatus.INACTIVE

    async def test_update_status_not_found(self, registry):
        """Test status update fails for non-existent model."""
        with pytest.raises(ValueError, match="Model not found"):
            await registry.update_model_status(
                "non-existent-id",
                ModelStatus.INACTIVE
            )

    async def test_update_status_invalidates_cache(
        self, registry, sample_model_metadata
    ):
        """Test status update invalidates cache."""
        await registry.register_model(sample_model_metadata)
        
        # Verify cache is valid
        assert registry._is_cache_valid(sample_model_metadata.id)
        
        # Update status
        await registry.update_model_status(
            sample_model_metadata.id,
            ModelStatus.INACTIVE
        )
        
        # Verify cache is invalidated
        assert not registry._is_cache_valid(sample_model_metadata.id)


class TestModelDiscovery:
    """Test model discovery functionality."""

    async def test_discover_models_success(
        self, registry, mock_adapter, sample_model_metadata
    ):
        """Test successful model discovery."""
        # Register adapter
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)
        
        # Configure mock to return models
        mock_adapter.list_models.return_value = [sample_model_metadata]
        
        # Discover models
        discovered = await registry.discover_models()
        
        # Verify discovery
        assert len(discovered) == 1
        assert discovered[0].id == sample_model_metadata.id
        
        # Verify models are cached
        assert sample_model_metadata.id in registry._models

    async def test_discover_models_multiple_adapters(self, registry):
        """Test discovery from multiple adapters."""
        # Create multiple mock adapters
        adapter1 = Mock(spec=BaseModelAdapter)
        adapter1.list_models = AsyncMock()
        adapter2 = Mock(spec=BaseModelAdapter)
        adapter2.list_models = AsyncMock()
        
        # Configure mocks to return different models
        model1 = ModelMetadata(
            id="model-1",
            provider=ModelProvider.BEDROCK,
            name="Model 1",
            capabilities=[],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1",
        )
        
        model2 = ModelMetadata(
            id="model-2",
            provider=ModelProvider.BEDROCK,
            name="Model 2",
            capabilities=[],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1",
        )
        
        adapter1.list_models.return_value = [model1]
        adapter2.list_models.return_value = [model2]
        
        # Register adapters
        registry.register_adapter(ModelProvider.BEDROCK, adapter1)
        registry.register_adapter(ModelProvider.BEDROCK, adapter2)
        
        # Discover models
        discovered = await registry.discover_models()
        
        # Verify both models discovered
        assert len(discovered) == 2
        assert {m.id for m in discovered} == {"model-1", "model-2"}

    async def test_discover_models_adapter_failure(self, registry, mock_adapter):
        """Test discovery continues when one adapter fails."""
        # Create two adapters, one fails
        failing_adapter = Mock(spec=BaseModelAdapter)
        failing_adapter.list_models = AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )
        
        model = ModelMetadata(
            id="model-1",
            provider=ModelProvider.BEDROCK,
            name="Model 1",
            capabilities=[],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            max_tokens=4096,
            region="us-east-1",
        )
        mock_adapter.list_models.return_value = [model]
        
        # Register both adapters
        registry.register_adapter(ModelProvider.BEDROCK, failing_adapter)
        registry.register_adapter(ModelProvider.SAGEMAKER, mock_adapter)
        
        # Discover models - should succeed with one model
        discovered = await registry.discover_models()
        assert len(discovered) == 1
        assert discovered[0].id == "model-1"

    async def test_discover_models_all_fail(self, registry):
        """Test discovery raises error when all adapters fail."""
        # Create failing adapter
        failing_adapter = Mock(spec=BaseModelAdapter)
        failing_adapter.list_models = AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )
        
        # Register adapter
        registry.register_adapter(ModelProvider.BEDROCK, failing_adapter)
        
        # Discover models - should raise error
        with pytest.raises(RuntimeError, match="Failed to discover models"):
            await registry.discover_models()


class TestHealthCheck:
    """Test health check functionality."""

    async def test_health_check_success(
        self, registry, mock_adapter, sample_model_metadata
    ):
        """Test successful health check."""
        await registry.register_model(sample_model_metadata)
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)
        
        # Configure mock to return healthy
        mock_adapter.validate_connection.return_value = True
        
        # Perform health check
        is_healthy = await registry.health_check(sample_model_metadata.id)
        
        # Verify result
        assert is_healthy is True
        
        # Verify health check timestamp is updated
        model = await registry.get_model(sample_model_metadata.id)
        assert model.last_health_check is not None

    async def test_health_check_failure(
        self, registry, mock_adapter, sample_model_metadata
    ):
        """Test health check failure."""
        await registry.register_model(sample_model_metadata)
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)
        
        # Configure mock to return unhealthy
        mock_adapter.validate_connection.return_value = False
        
        # Perform health check
        is_healthy = await registry.health_check(sample_model_metadata.id)
        
        # Verify result
        assert is_healthy is False
        
        # Verify status is updated to INACTIVE (check directly in cache)
        assert registry._models[sample_model_metadata.id].status == ModelStatus.INACTIVE

    async def test_health_check_model_not_found(self, registry):
        """Test health check for non-existent model."""
        is_healthy = await registry.health_check("non-existent-id")
        assert is_healthy is False

    async def test_health_check_exception(
        self, registry, mock_adapter, sample_model_metadata
    ):
        """Test health check handles exceptions."""
        await registry.register_model(sample_model_metadata)
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)
        
        # Configure mock to raise exception
        mock_adapter.validate_connection.side_effect = RuntimeError("Connection error")
        
        # Perform health check
        is_healthy = await registry.health_check(sample_model_metadata.id)
        
        # Verify result
        assert is_healthy is False
        
        # Verify status is updated to INACTIVE (check directly in cache)
        assert registry._models[sample_model_metadata.id].status == ModelStatus.INACTIVE


class TestCaching:
    """Test caching functionality."""

    async def test_cache_validity(self, registry, sample_model_metadata):
        """Test cache validity checking."""
        await registry.register_model(sample_model_metadata)
        
        # Cache should be valid immediately
        assert registry._is_cache_valid(sample_model_metadata.id)

    async def test_cache_expiration(self, registry, sample_model_metadata):
        """Test cache expiration."""
        # Create registry with short TTL
        registry = ModelRegistry(cache_ttl_seconds=1)
        await registry.register_model(sample_model_metadata)
        
        # Cache should be valid initially
        assert registry._is_cache_valid(sample_model_metadata.id)
        
        # Wait for cache to expire
        import asyncio
        await asyncio.sleep(1.1)
        
        # Cache should be expired
        assert not registry._is_cache_valid(sample_model_metadata.id)

    async def test_cache_invalidation(self, registry, sample_model_metadata):
        """Test manual cache invalidation."""
        await registry.register_model(sample_model_metadata)
        
        # Cache should be valid
        assert registry._is_cache_valid(sample_model_metadata.id)
        
        # Invalidate cache
        registry._invalidate_cache(sample_model_metadata.id)
        
        # Cache should be invalid
        assert not registry._is_cache_valid(sample_model_metadata.id)

    async def test_clear_cache(self, registry, sample_model_metadata):
        """Test clearing all cache."""
        await registry.register_model(sample_model_metadata)
        
        # Verify model is cached
        assert len(registry._models) == 1
        
        # Clear cache
        await registry.clear_cache()
        
        # Verify cache is empty
        assert len(registry._models) == 0
        assert len(registry._cache_timestamps) == 0

    def test_cache_stats(self, registry):
        """Test cache statistics."""
        stats = registry.get_cache_stats()
        
        assert "total_models" in stats
        assert "valid_cache_entries" in stats
        assert "expired_cache_entries" in stats
        assert "cache_ttl_seconds" in stats
        assert stats["cache_ttl_seconds"] == 3600


class TestAdapterManagement:
    """Test adapter management functionality."""

    def test_register_adapter(self, registry, mock_adapter):
        """Test adapter registration."""
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)
        
        # Verify adapter is registered
        assert ModelProvider.BEDROCK in registry._provider_adapters
        assert mock_adapter in registry._provider_adapters[ModelProvider.BEDROCK]

    def test_register_multiple_adapters_same_provider(self, registry):
        """Test registering multiple adapters for same provider."""
        adapter1 = Mock(spec=BaseModelAdapter)
        adapter2 = Mock(spec=BaseModelAdapter)
        
        registry.register_adapter(ModelProvider.BEDROCK, adapter1)
        registry.register_adapter(ModelProvider.BEDROCK, adapter2)
        
        # Verify both adapters are registered
        assert len(registry._provider_adapters[ModelProvider.BEDROCK]) == 2

    async def test_get_adapter(self, registry, mock_adapter, sample_model_metadata):
        """Test getting adapter for a model."""
        await registry.register_model(sample_model_metadata)
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)
        
        # Get adapter
        adapter = registry.get_adapter(sample_model_metadata.id)
        
        # Verify correct adapter is returned
        assert adapter == mock_adapter

    def test_get_adapter_model_not_found(self, registry):
        """Test getting adapter for non-existent model."""
        adapter = registry.get_adapter("non-existent-id")
        assert adapter is None


    async def test_cache_hit_tracking(self, registry, sample_model_metadata):
        """Test cache hit tracking."""
        await registry.register_model(sample_model_metadata)

        # First access should be a cache hit (just registered)
        model = await registry.get_model(sample_model_metadata.id)
        assert model is not None

        # Check stats
        stats = registry.get_cache_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 0
        assert stats["hit_rate"] == 1.0

    async def test_cache_miss_tracking(
        self, registry, mock_adapter, sample_model_metadata
    ):
        """Test cache miss tracking."""
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)
        mock_adapter.get_model_info.return_value = sample_model_metadata

        # First access should be a cache miss (not in cache)
        model = await registry.get_model(sample_model_metadata.id)
        assert model is not None

        # Check stats
        stats = registry.get_cache_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 0.0

    async def test_cache_hit_rate_calculation(
        self, registry, mock_adapter, sample_model_metadata
    ):
        """Test cache hit rate calculation."""
        await registry.register_model(sample_model_metadata)
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)

        # Multiple accesses
        await registry.get_model(sample_model_metadata.id)  # Hit
        await registry.get_model(sample_model_metadata.id)  # Hit
        await registry.get_model("non-existent")  # Miss

        # Check stats
        stats = registry.get_cache_stats()
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert abs(stats["hit_rate"] - 0.666) < 0.01  # ~66.7%

    async def test_cache_stats_after_clear(
        self, registry, sample_model_metadata
    ):
        """Test cache stats are reset after clearing cache."""
        await registry.register_model(sample_model_metadata)
        await registry.get_model(sample_model_metadata.id)

        # Clear cache
        await registry.clear_cache()

        # Check stats are reset
        stats = registry.get_cache_stats()
        assert stats["total_models"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_rate"] == 0.0

    async def test_refresh_model_cache(
        self, registry, mock_adapter, sample_model_metadata
    ):
        """Test manual cache refresh for a specific model."""
        # Register initial model
        await registry.register_model(sample_model_metadata)
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)

        # Create updated metadata
        updated_metadata = sample_model_metadata.model_copy()
        updated_metadata.status = ModelStatus.INACTIVE
        mock_adapter.get_model_info.return_value = updated_metadata

        # Refresh cache
        refreshed = await registry.refresh_model_cache(
            sample_model_metadata.id
        )

        # Verify updated metadata is returned
        assert refreshed is not None
        assert refreshed.status == ModelStatus.INACTIVE

        # Verify cache is updated
        cached = await registry.get_model(sample_model_metadata.id)
        assert cached.status == ModelStatus.INACTIVE

    async def test_refresh_model_cache_not_found(
        self, registry, mock_adapter
    ):
        """Test cache refresh for non-existent model."""
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)
        mock_adapter.get_model_info.side_effect = ValueError("Not found")

        # Refresh non-existent model
        result = await registry.refresh_model_cache("non-existent")

        # Should return None
        assert result is None

    async def test_cache_reduces_api_calls(
        self, registry, mock_adapter, sample_model_metadata
    ):
        """Test that caching reduces API calls to adapters."""
        await registry.register_model(sample_model_metadata)
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)
        mock_adapter.get_model_info.return_value = sample_model_metadata

        # Multiple accesses to same model
        for _ in range(5):
            await registry.get_model(sample_model_metadata.id)

        # Adapter should not be called (all cache hits)
        mock_adapter.get_model_info.assert_not_called()

    async def test_expired_cache_triggers_refresh(
        self, registry, mock_adapter, sample_model_metadata
    ):
        """Test that expired cache triggers adapter call."""
        # Create registry with short TTL
        registry = ModelRegistry(cache_ttl_seconds=1)
        await registry.register_model(sample_model_metadata)
        registry.register_adapter(ModelProvider.BEDROCK, mock_adapter)

        # Create updated metadata
        updated_metadata = sample_model_metadata.model_copy()
        updated_metadata.status = ModelStatus.INACTIVE
        mock_adapter.get_model_info.return_value = updated_metadata

        # Wait for cache to expire
        import asyncio
        await asyncio.sleep(1.1)

        # Access model - should trigger refresh
        model = await registry.get_model(sample_model_metadata.id)

        # Verify adapter was called
        mock_adapter.get_model_info.assert_called_once()

        # Verify updated metadata is returned
        assert model.status == ModelStatus.INACTIVE
