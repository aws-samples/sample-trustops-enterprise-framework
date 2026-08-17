"""
Data models for the Model Registry.

This module defines the core data models for multi-provider model discovery,
registration, and inference in the TrustOps Enterprise Framework.

Requirements: 1.1, 1.2, 1.4, 1.8
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ModelProvider(str, Enum):
    """Enumeration of supported model providers.

    Requirement 1.2: Define ModelProvider enum
    """

    BEDROCK = "bedrock"
    SAGEMAKER = "sagemaker"
    EXTERNAL_API = "external_api"


class ModelCapability(str, Enum):
    """Enumeration of model capabilities.

    Requirement 1.8: Support model capability tags
    """

    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDING = "embedding"
    FINE_TUNABLE = "fine_tunable"
    MULTIMODAL = "multimodal"
    COMPLETION = "completion"


class ModelStatus(str, Enum):
    """Enumeration of model availability statuses.

    Requirement 1.5: Track model availability status
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    PROVISIONING = "provisioning"
    FAILED = "failed"


class ModelPricing(BaseModel):
    """Pricing information for a model.

    Requirement 1.4: Maintain pricing metadata for each model
    """

    input_price_per_1k_tokens: float = Field(
        ...,
        ge=0,
        description="Cost per 1000 input tokens"
    )
    output_price_per_1k_tokens: float = Field(
        ...,
        ge=0,
        description="Cost per 1000 output tokens"
    )
    fine_tuning_price_per_1k_tokens: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cost per 1000 tokens for fine-tuning"
    )
    currency: str = Field(
        default="USD",
        description="Currency for pricing"
    )


class ModelMetadata(BaseModel):
    """Metadata for a registered model.

    Requirement 1.1: Define ModelMetadata pydantic schema
    """

    id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the model"
    )
    provider: ModelProvider = Field(
        ...,
        description="The provider of this model"
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name of the model"
    )
    capabilities: list[ModelCapability] = Field(
        default_factory=list,
        description="List of capabilities this model supports"
    )
    status: ModelStatus = Field(
        default=ModelStatus.ACTIVE,
        description="Current availability status of the model"
    )
    fine_tuning_support: bool = Field(
        default=False,
        description="Whether this model supports fine-tuning"
    )
    input_modalities: list[str] = Field(
        default_factory=lambda: ["text"],
        description="Supported input modalities (e.g., text, image)"
    )
    output_modalities: list[str] = Field(
        default_factory=lambda: ["text"],
        description="Supported output modalities (e.g., text, image)"
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens supported by the model"
    )
    region: str = Field(
        default="us-east-1",
        description="AWS region where the model is available"
    )
    pricing: Optional[ModelPricing] = Field(
        default=None,
        description="Pricing information for the model"
    )
    last_health_check: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the last health check"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional provider-specific metadata"
    )


class InferenceRequest(BaseModel):
    """Request schema for model inference.

    Requirement 1.6: Unified inference interface
    """

    prompt: str = Field(
        ...,
        min_length=1,
        description="The input prompt for the model"
    )
    max_tokens: int = Field(
        default=1024,
        gt=0,
        le=100000,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    stop_sequences: Optional[list[str]] = Field(
        default=None,
        description="Sequences that stop generation when encountered"
    )

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is within valid range."""
        if v < 0.0 or v > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        """Ensure top_p is within valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        return v


class InferenceResponse(BaseModel):
    """Response schema for model inference.

    Requirement 1.6: Unified inference interface
    """

    text: str = Field(
        ...,
        description="The generated text response"
    )
    input_tokens: int = Field(
        ...,
        ge=0,
        description="Number of input tokens processed"
    )
    output_tokens: int = Field(
        ...,
        ge=0,
        description="Number of output tokens generated"
    )
    latency_ms: float = Field(
        ...,
        ge=0,
        description="Response latency in milliseconds"
    )
    model_id: str = Field(
        ...,
        description="ID of the model that generated the response"
    )
    finish_reason: str = Field(
        ...,
        description="Reason for completion (stop, length, content_filter)"
    )
