"""
Model-specific format validator for AWS Bedrock fine-tuning.

This module validates that datasets meet the specific format requirements
for fine-tuning different model families on AWS Bedrock (Claude, Titan, Llama).

Requirement 2.10: Implement model-specific format validator
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from src.data_models.dataset import DatasetFormat, DatasetTaskType


class ModelFamily(str, Enum):
    """Supported AWS Bedrock model families for fine-tuning."""
    
    CLAUDE = "claude"
    TITAN = "titan"
    LLAMA = "llama"
    UNKNOWN = "unknown"


class ValidationResult:
    """Result of model-specific format validation."""
    
    def __init__(
        self,
        valid: bool,
        errors: Optional[list[str]] = None,
        warnings: Optional[list[str]] = None
    ):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.valid
    
    def __repr__(self) -> str:
        return (
            f"ValidationResult(valid={self.valid}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )


class ModelValidator:
    """Validates datasets against model-specific fine-tuning format requirements."""
    
    # Claude format requirements (JSONL with prompt/completion)
    CLAUDE_REQUIRED_FIELDS = ["prompt", "completion"]
    CLAUDE_OPTIONAL_FIELDS = ["system"]
    
    # Titan format requirements (JSONL with inputText/outputText)
    TITAN_REQUIRED_FIELDS = ["inputText", "outputText"]
    TITAN_OPTIONAL_FIELDS = []
    
    # Llama format requirements (JSONL with prompt/completion)
    LLAMA_REQUIRED_FIELDS = ["prompt", "completion"]
    LLAMA_OPTIONAL_FIELDS = []
    
    # Minimum dataset sizes per model family
    MIN_EXAMPLES = {
        ModelFamily.CLAUDE: 32,
        ModelFamily.TITAN: 32,
        ModelFamily.LLAMA: 32,
    }
    
    # Maximum token limits per model family
    MAX_TOKENS = {
        ModelFamily.CLAUDE: 32000,
        ModelFamily.TITAN: 8192,
        ModelFamily.LLAMA: 4096,
    }
    
    @staticmethod
    def detect_model_family(model_id: str) -> ModelFamily:
        """
        Detect model family from model ID.
        
        Args:
            model_id: AWS Bedrock model identifier
            
        Returns:
            ModelFamily enum value
        """
        model_id_lower = model_id.lower()
        
        if "claude" in model_id_lower or "anthropic" in model_id_lower:
            return ModelFamily.CLAUDE
        elif "titan" in model_id_lower or "amazon" in model_id_lower:
            return ModelFamily.TITAN
        elif "llama" in model_id_lower or "meta" in model_id_lower:
            return ModelFamily.LLAMA
        else:
            return ModelFamily.UNKNOWN
    
    @staticmethod
    def validate_for_model(
        file_path: Union[str, Path],
        model_id: str,
        dataset_format: Optional[DatasetFormat] = None
    ) -> ValidationResult:
        """
        Validate dataset format for a specific model.
        
        Args:
            file_path: Path to the dataset file
            model_id: AWS Bedrock model identifier
            dataset_format: Optional pre-detected format
            
        Returns:
            ValidationResult with validation status and messages
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ValidationResult(
                valid=False,
                errors=[f"File not found: {file_path}"]
            )
        
        # Detect model family
        model_family = ModelValidator.detect_model_family(model_id)
        
        if model_family == ModelFamily.UNKNOWN:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown model family for model ID: {model_id}"]
            )
        
        # Detect format if not provided
        if dataset_format is None:
            from src.datasets.format_detector import FormatDetector
            try:
                dataset_format = FormatDetector.detect_format(file_path)
            except ValueError as e:
                return ValidationResult(
                    valid=False,
                    errors=[f"Unable to detect dataset format: {e}"]
                )
        
        # All Bedrock fine-tuning requires JSONL format
        if dataset_format != DatasetFormat.JSONL:
            return ValidationResult(
                valid=False,
                errors=[
                    f"AWS Bedrock fine-tuning requires JSONL format, "
                    f"got {dataset_format.value}. "
                    f"Please convert your dataset to JSONL format."
                ]
            )
        
        # Validate based on model family
        if model_family == ModelFamily.CLAUDE:
            return ModelValidator._validate_claude_format(file_path)
        elif model_family == ModelFamily.TITAN:
            return ModelValidator._validate_titan_format(file_path)
        elif model_family == ModelFamily.LLAMA:
            return ModelValidator._validate_llama_format(file_path)
        
        return ValidationResult(
            valid=False,
            errors=[f"Validation not implemented for {model_family.value}"]
        )
    
    @staticmethod
    def _validate_claude_format(file_path: Path) -> ValidationResult:
        """
        Validate dataset for Claude fine-tuning.
        
        Claude format requirements:
        - JSONL format
        - Each line must have "prompt" and "completion" fields
        - Optional "system" field for system prompts
        - Minimum 32 examples
        - Maximum 32,000 tokens per example
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        examples = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(
                            f"Line {line_number}: Invalid JSON - {e.msg}"
                        )
                        continue
                    
                    if not isinstance(obj, dict):
                        errors.append(
                            f"Line {line_number}: Expected JSON object, "
                            f"got {type(obj).__name__}"
                        )
                        continue
                    
                    # Check required fields
                    missing_fields = []
                    for field in ModelValidator.CLAUDE_REQUIRED_FIELDS:
                        if field not in obj:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        errors.append(
                            f"Line {line_number}: Missing required fields: "
                            f"{', '.join(missing_fields)}"
                        )
                        continue
                    
                    # Validate field types
                    if not isinstance(obj.get("prompt"), str):
                        errors.append(
                            f"Line {line_number}: 'prompt' must be a string"
                        )
                    
                    if not isinstance(obj.get("completion"), str):
                        errors.append(
                            f"Line {line_number}: 'completion' must be a string"
                        )
                    
                    if "system" in obj and not isinstance(obj["system"], str):
                        errors.append(
                            f"Line {line_number}: 'system' must be a string"
                        )
                    
                    # Check for empty strings
                    if obj.get("prompt", "").strip() == "":
                        errors.append(
                            f"Line {line_number}: 'prompt' cannot be empty"
                        )
                    
                    if obj.get("completion", "").strip() == "":
                        errors.append(
                            f"Line {line_number}: 'completion' cannot be empty"
                        )
                    
                    examples.append(obj)
        
        except Exception as e:
            errors.append(f"Error reading file: {e}")
            return ValidationResult(valid=False, errors=errors)
        
        # Check minimum examples
        min_examples = ModelValidator.MIN_EXAMPLES[ModelFamily.CLAUDE]
        if len(examples) < min_examples:
            errors.append(
                f"Claude fine-tuning requires at least {min_examples} examples, "
                f"found {len(examples)}"
            )
        
        # Warn about token limits (approximate check)
        max_tokens = ModelValidator.MAX_TOKENS[ModelFamily.CLAUDE]
        for i, example in enumerate(examples, 1):
            # Rough token estimate: 1 token ≈ 4 characters
            prompt_tokens = len(example.get("prompt", "")) // 4
            completion_tokens = len(example.get("completion", "")) // 4
            total_tokens = prompt_tokens + completion_tokens
            
            if total_tokens > max_tokens:
                warnings.append(
                    f"Example {i}: Estimated {total_tokens} tokens exceeds "
                    f"recommended maximum of {max_tokens} tokens"
                )
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def _validate_titan_format(file_path: Path) -> ValidationResult:
        """
        Validate dataset for Titan fine-tuning.
        
        Titan format requirements:
        - JSONL format
        - Each line must have "inputText" and "outputText" fields
        - Minimum 32 examples
        - Maximum 8,192 tokens per example
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        examples = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(
                            f"Line {line_number}: Invalid JSON - {e.msg}"
                        )
                        continue
                    
                    if not isinstance(obj, dict):
                        errors.append(
                            f"Line {line_number}: Expected JSON object, "
                            f"got {type(obj).__name__}"
                        )
                        continue
                    
                    # Check required fields
                    missing_fields = []
                    for field in ModelValidator.TITAN_REQUIRED_FIELDS:
                        if field not in obj:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        errors.append(
                            f"Line {line_number}: Missing required fields: "
                            f"{', '.join(missing_fields)}"
                        )
                        continue
                    
                    # Validate field types
                    if not isinstance(obj.get("inputText"), str):
                        errors.append(
                            f"Line {line_number}: 'inputText' must be a string"
                        )
                    
                    if not isinstance(obj.get("outputText"), str):
                        errors.append(
                            f"Line {line_number}: 'outputText' must be a string"
                        )
                    
                    # Check for empty strings
                    if obj.get("inputText", "").strip() == "":
                        errors.append(
                            f"Line {line_number}: 'inputText' cannot be empty"
                        )
                    
                    if obj.get("outputText", "").strip() == "":
                        errors.append(
                            f"Line {line_number}: 'outputText' cannot be empty"
                        )
                    
                    examples.append(obj)
        
        except Exception as e:
            errors.append(f"Error reading file: {e}")
            return ValidationResult(valid=False, errors=errors)
        
        # Check minimum examples
        min_examples = ModelValidator.MIN_EXAMPLES[ModelFamily.TITAN]
        if len(examples) < min_examples:
            errors.append(
                f"Titan fine-tuning requires at least {min_examples} examples, "
                f"found {len(examples)}"
            )
        
        # Warn about token limits (approximate check)
        max_tokens = ModelValidator.MAX_TOKENS[ModelFamily.TITAN]
        for i, example in enumerate(examples, 1):
            # Rough token estimate: 1 token ≈ 4 characters
            input_tokens = len(example.get("inputText", "")) // 4
            output_tokens = len(example.get("outputText", "")) // 4
            total_tokens = input_tokens + output_tokens
            
            if total_tokens > max_tokens:
                warnings.append(
                    f"Example {i}: Estimated {total_tokens} tokens exceeds "
                    f"recommended maximum of {max_tokens} tokens"
                )
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def _validate_llama_format(file_path: Path) -> ValidationResult:
        """
        Validate dataset for Llama fine-tuning.
        
        Llama format requirements:
        - JSONL format
        - Each line must have "prompt" and "completion" fields
        - Minimum 32 examples
        - Maximum 4,096 tokens per example
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        examples = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(
                            f"Line {line_number}: Invalid JSON - {e.msg}"
                        )
                        continue
                    
                    if not isinstance(obj, dict):
                        errors.append(
                            f"Line {line_number}: Expected JSON object, "
                            f"got {type(obj).__name__}"
                        )
                        continue
                    
                    # Check required fields
                    missing_fields = []
                    for field in ModelValidator.LLAMA_REQUIRED_FIELDS:
                        if field not in obj:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        errors.append(
                            f"Line {line_number}: Missing required fields: "
                            f"{', '.join(missing_fields)}"
                        )
                        continue
                    
                    # Validate field types
                    if not isinstance(obj.get("prompt"), str):
                        errors.append(
                            f"Line {line_number}: 'prompt' must be a string"
                        )
                    
                    if not isinstance(obj.get("completion"), str):
                        errors.append(
                            f"Line {line_number}: 'completion' must be a string"
                        )
                    
                    # Check for empty strings
                    if obj.get("prompt", "").strip() == "":
                        errors.append(
                            f"Line {line_number}: 'prompt' cannot be empty"
                        )
                    
                    if obj.get("completion", "").strip() == "":
                        errors.append(
                            f"Line {line_number}: 'completion' cannot be empty"
                        )
                    
                    examples.append(obj)
        
        except Exception as e:
            errors.append(f"Error reading file: {e}")
            return ValidationResult(valid=False, errors=errors)
        
        # Check minimum examples
        min_examples = ModelValidator.MIN_EXAMPLES[ModelFamily.LLAMA]
        if len(examples) < min_examples:
            errors.append(
                f"Llama fine-tuning requires at least {min_examples} examples, "
                f"found {len(examples)}"
            )
        
        # Warn about token limits (approximate check)
        max_tokens = ModelValidator.MAX_TOKENS[ModelFamily.LLAMA]
        for i, example in enumerate(examples, 1):
            # Rough token estimate: 1 token ≈ 4 characters
            prompt_tokens = len(example.get("prompt", "")) // 4
            completion_tokens = len(example.get("completion", "")) // 4
            total_tokens = prompt_tokens + completion_tokens
            
            if total_tokens > max_tokens:
                warnings.append(
                    f"Example {i}: Estimated {total_tokens} tokens exceeds "
                    f"recommended maximum of {max_tokens} tokens"
                )
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def get_format_requirements(model_id: str) -> dict[str, Any]:
        """
        Get format requirements for a specific model.
        
        Args:
            model_id: AWS Bedrock model identifier
            
        Returns:
            Dictionary with format requirements
        """
        model_family = ModelValidator.detect_model_family(model_id)
        
        if model_family == ModelFamily.CLAUDE:
            return {
                "model_family": model_family.value,
                "format": "JSONL",
                "required_fields": ModelValidator.CLAUDE_REQUIRED_FIELDS,
                "optional_fields": ModelValidator.CLAUDE_OPTIONAL_FIELDS,
                "min_examples": ModelValidator.MIN_EXAMPLES[model_family],
                "max_tokens": ModelValidator.MAX_TOKENS[model_family],
                "description": (
                    "Claude fine-tuning requires JSONL format with 'prompt' and "
                    "'completion' fields. Optional 'system' field for system prompts."
                )
            }
        elif model_family == ModelFamily.TITAN:
            return {
                "model_family": model_family.value,
                "format": "JSONL",
                "required_fields": ModelValidator.TITAN_REQUIRED_FIELDS,
                "optional_fields": ModelValidator.TITAN_OPTIONAL_FIELDS,
                "min_examples": ModelValidator.MIN_EXAMPLES[model_family],
                "max_tokens": ModelValidator.MAX_TOKENS[model_family],
                "description": (
                    "Titan fine-tuning requires JSONL format with 'inputText' and "
                    "'outputText' fields."
                )
            }
        elif model_family == ModelFamily.LLAMA:
            return {
                "model_family": model_family.value,
                "format": "JSONL",
                "required_fields": ModelValidator.LLAMA_REQUIRED_FIELDS,
                "optional_fields": ModelValidator.LLAMA_OPTIONAL_FIELDS,
                "min_examples": ModelValidator.MIN_EXAMPLES[model_family],
                "max_tokens": ModelValidator.MAX_TOKENS[model_family],
                "description": (
                    "Llama fine-tuning requires JSONL format with 'prompt' and "
                    "'completion' fields."
                )
            }
        else:
            return {
                "model_family": "unknown",
                "error": f"Unknown model family for model ID: {model_id}"
            }
