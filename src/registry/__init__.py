"""Model Registry module for TrustOps Enterprise Framework.

This module provides centralized model discovery, registration, and metadata
management across multiple providers (Bedrock, SageMaker, External APIs).
"""

from src.registry.model_registry import ModelRegistry

__all__ = ["ModelRegistry"]
