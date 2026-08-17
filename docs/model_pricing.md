# Model Pricing Configuration

This document describes the model pricing configuration system in the TrustOps Enterprise Framework.

## Overview

The pricing configuration system provides a centralized way to manage and apply pricing information for models across different providers (AWS Bedrock, SageMaker, External APIs). This enables accurate cost tracking and estimation for model inference and fine-tuning operations.

**Requirements:** 1.4, 1.14

## Configuration File

The pricing configuration is stored in `config/model_pricing.yaml`. This YAML file contains pricing information organized by provider.

### File Structure

```yaml
# Provider sections
bedrock:
  model-id-1:
    input_price_per_1k_tokens: 0.001
    output_price_per_1k_tokens: 0.002
    fine_tuning_price_per_1k_tokens: 0.008  # null if not supported
    currency: USD
  
  model-id-2:
    # ... pricing details

external_api:
  model-id-3:
    # ... pricing details

sagemaker:
  default:  # Default pricing for SageMaker models
    # ... pricing details

default:  # Fallback pricing for unknown models
  input_price_per_1k_tokens: 0.001
  output_price_per_1k_tokens: 0.002
  fine_tuning_price_per_1k_tokens: null
  currency: USD
```

### Pricing Fields

- **input_price_per_1k_tokens**: Cost per 1,000 input tokens (USD)
- **output_price_per_1k_tokens**: Cost per 1,000 output tokens (USD)
- **fine_tuning_price_per_1k_tokens**: Cost per 1,000 tokens for fine-tuning (null if not supported)
- **currency**: Currency code (default: USD)

## PricingLoader Class

The `PricingLoader` class provides methods to load and apply pricing information.

### Basic Usage

```python
from src.registry.pricing_loader import PricingLoader
from src.data_models.model import ModelProvider

# Initialize with default config path
loader = PricingLoader()

# Or specify custom config path
loader = PricingLoader("/path/to/custom/pricing.yaml")

# Load pricing configuration
pricing_data = loader.load_pricing_config()

# Get pricing for a specific model
pricing = loader.get_model_pricing(
    "anthropic.claude-3-haiku-20240307-v1:0",
    ModelProvider.BEDROCK
)

print(f"Input: ${pricing.input_price_per_1k_tokens} per 1K tokens")
print(f"Output: ${pricing.output_price_per_1k_tokens} per 1K tokens")
```

### Applying Pricing to Models

```python
from src.data_models.model import ModelMetadata, ModelProvider, ModelCapability, ModelStatus

# Create model metadata without pricing
metadata = ModelMetadata(
    id="anthropic.claude-3-haiku-20240307-v1:0",
    provider=ModelProvider.BEDROCK,
    name="Claude 3 Haiku",
    capabilities=[ModelCapability.TEXT_GENERATION],
    status=ModelStatus.ACTIVE,
    fine_tuning_support=False,
    max_tokens=200000,
    region="us-east-1"
)

# Apply pricing
updated_metadata = loader.apply_pricing_to_metadata(metadata)

# Now metadata.pricing contains the pricing information
print(updated_metadata.pricing)
```

### Batch Operations

```python
# Apply pricing to multiple models
models = [model1, model2, model3]
updated_models = loader.apply_pricing_to_models(models)

# Get all model IDs with pricing
all_bedrock_ids = loader.get_all_model_ids(ModelProvider.BEDROCK)
all_external_ids = loader.get_all_model_ids(ModelProvider.EXTERNAL_API)
```

## ModelRegistry Integration

The `ModelRegistry` automatically integrates with the `PricingLoader` to apply pricing information to discovered models.

### Usage

```python
from src.registry.model_registry import ModelRegistry

# Initialize registry (uses default pricing config)
registry = ModelRegistry()

# Or specify custom pricing config
registry = ModelRegistry(pricing_config_path="/path/to/pricing.yaml")

# Discover models - pricing is automatically applied
models = await registry.discover_models()

# All discovered models now have pricing information
for model in models:
    if model.pricing:
        print(f"{model.name}: ${model.pricing.input_price_per_1k_tokens} per 1K input tokens")
```

### Accessing the Pricing Loader

```python
# Get the pricing loader from registry
loader = registry.get_pricing_loader()

# Use loader methods
pricing = loader.get_model_pricing("model-id", ModelProvider.BEDROCK)
```

### Reloading Pricing Configuration

```python
# Reload pricing config and update all cached models
registry.reload_pricing_config()
```

## Cost Calculation

Use pricing information to calculate costs for model operations:

```python
from src.registry.pricing_loader import PricingLoader
from src.data_models.model import ModelProvider

loader = PricingLoader()
pricing = loader.get_model_pricing(
    "anthropic.claude-3-haiku-20240307-v1:0",
    ModelProvider.BEDROCK
)

# Example usage
input_tokens = 1500
output_tokens = 500

# Calculate costs
input_cost = (input_tokens / 1000) * pricing.input_price_per_1k_tokens
output_cost = (output_tokens / 1000) * pricing.output_price_per_1k_tokens
total_cost = input_cost + output_cost

print(f"Total cost: ${total_cost:.6f}")
```

## Pricing Fallback Logic

The pricing loader uses the following fallback logic:

1. **Exact match**: Look for exact model ID in provider section
2. **Provider default**: For SageMaker, use the "default" entry in sagemaker section
3. **Global default**: Use the "default" section for any unknown model

This ensures that all models have pricing information, even if not explicitly configured.

## Supported Models

The default configuration includes pricing for:

### AWS Bedrock Models
- **Anthropic Claude**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku, Claude 2.1, Claude 2, Claude Instant
- **Amazon Titan**: Titan Text Premier, Titan Text Express, Titan Text Lite, Titan Embed Text v2, Titan Embed Text v1
- **Meta Llama**: Llama 3.1 (405B, 70B, 8B), Llama 3 (70B, 8B), Llama 2 (70B, 13B)
- **Cohere**: Command R+, Command R, Command Text, Command Light, Embed English, Embed Multilingual
- **AI21 Labs**: Jamba 1.5 Large, Jamba 1.5 Mini, Jurassic-2 Ultra, Jurassic-2 Mid
- **Mistral AI**: Mistral Large, Mistral Small, Mixtral 8x7B, Mistral 7B

### External API Models
- **OpenAI**: GPT-4 Turbo, GPT-4, GPT-3.5 Turbo, Text Embedding 3 Large, Text Embedding 3 Small

### SageMaker Models
- Default pricing (actual costs vary by instance type)

## Updating Pricing

To update pricing information:

1. Edit `config/model_pricing.yaml`
2. Add or modify model entries following the existing format
3. Reload the configuration in your application:
   ```python
   registry.reload_pricing_config()
   ```

## Demo Script

Run the demo script to see pricing loader in action:

```bash
python demo/pricing_loader_demo.py
```

The demo shows:
- Loading pricing configuration
- Getting pricing for specific models
- Applying pricing to metadata
- ModelRegistry integration
- Cost calculation examples

## Testing

Unit tests are available in:
- `tests/unit/test_pricing_loader.py` - PricingLoader tests
- `tests/unit/test_model_registry_pricing.py` - Integration tests

Run tests:
```bash
pytest tests/unit/test_pricing_loader.py -v
pytest tests/unit/test_model_registry_pricing.py -v
```

## Notes

- Pricing information is cached after first load for performance
- All prices are in USD unless specified otherwise
- Pricing data is sourced from AWS Bedrock pricing documentation
- SageMaker pricing varies by instance type and deployment configuration
- External API pricing may change; verify with provider documentation
