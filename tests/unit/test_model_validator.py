"""
Unit tests for model-specific format validator.

Tests validation of datasets against AWS Bedrock fine-tuning format requirements
for different model families (Claude, Titan, Llama).

Requirement 2.10: Test model-specific format validator
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.datasets.model_validator import (
    ModelFamily,
    ModelValidator,
    ValidationResult
)
from src.data_models.dataset import DatasetFormat


class TestModelFamilyDetection:
    """Test model family detection from model IDs."""
    
    def test_detect_claude_family(self):
        """Test detection of Claude model family."""
        assert ModelValidator.detect_model_family(
            "anthropic.claude-3-sonnet-20240229-v1:0"
        ) == ModelFamily.CLAUDE
        
        assert ModelValidator.detect_model_family(
            "anthropic.claude-v2"
        ) == ModelFamily.CLAUDE
        
        assert ModelValidator.detect_model_family(
            "claude-instant-v1"
        ) == ModelFamily.CLAUDE
    
    def test_detect_titan_family(self):
        """Test detection of Titan model family."""
        assert ModelValidator.detect_model_family(
            "amazon.titan-text-express-v1"
        ) == ModelFamily.TITAN
        
        assert ModelValidator.detect_model_family(
            "amazon.titan-text-lite-v1"
        ) == ModelFamily.TITAN
        
        assert ModelValidator.detect_model_family(
            "titan-embed-text-v1"
        ) == ModelFamily.TITAN
    
    def test_detect_llama_family(self):
        """Test detection of Llama model family."""
        assert ModelValidator.detect_model_family(
            "meta.llama2-13b-chat-v1"
        ) == ModelFamily.LLAMA
        
        assert ModelValidator.detect_model_family(
            "meta.llama2-70b-v1"
        ) == ModelFamily.LLAMA
        
        assert ModelValidator.detect_model_family(
            "llama-2-7b"
        ) == ModelFamily.LLAMA
    
    def test_detect_unknown_family(self):
        """Test detection of unknown model family."""
        assert ModelValidator.detect_model_family(
            "unknown-model-v1"
        ) == ModelFamily.UNKNOWN
        
        assert ModelValidator.detect_model_family(
            "gpt-4"
        ) == ModelFamily.UNKNOWN


class TestClaudeFormatValidation:
    """Test validation for Claude fine-tuning format."""
    
    def test_valid_claude_format(self):
        """Test validation of valid Claude format dataset."""
        # Create valid Claude format dataset
        examples = [
            {"prompt": f"Question {i}?", "completion": f"Answer {i}"}
            for i in range(35)  # More than minimum 32
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "anthropic.claude-3-sonnet-20240229-v1:0",
                DatasetFormat.JSONL
            )
            
            assert result.valid
            assert len(result.errors) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_claude_with_system_prompt(self):
        """Test validation of Claude format with optional system field."""
        examples = [
            {
                "prompt": f"Question {i}?",
                "completion": f"Answer {i}",
                "system": "You are a helpful assistant."
            }
            for i in range(35)
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "anthropic.claude-v2",
                DatasetFormat.JSONL
            )
            
            assert result.valid
            assert len(result.errors) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_claude_missing_required_field(self):
        """Test validation fails when required field is missing."""
        examples = [
            {"prompt": f"Question {i}?"}  # Missing 'completion'
            for i in range(35)
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "anthropic.claude-3-sonnet-20240229-v1:0",
                DatasetFormat.JSONL
            )
            
            assert not result.valid
            assert len(result.errors) > 0
            assert any("Missing required fields" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()
    
    def test_claude_empty_field(self):
        """Test validation fails when field is empty."""
        examples = [
            {"prompt": "", "completion": f"Answer {i}"}  # Empty prompt
            for i in range(35)
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "anthropic.claude-v2",
                DatasetFormat.JSONL
            )
            
            assert not result.valid
            assert any("cannot be empty" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()
    
    def test_claude_insufficient_examples(self):
        """Test validation fails with insufficient examples."""
        examples = [
            {"prompt": f"Question {i}?", "completion": f"Answer {i}"}
            for i in range(20)  # Less than minimum 32
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "anthropic.claude-3-sonnet-20240229-v1:0",
                DatasetFormat.JSONL
            )
            
            assert not result.valid
            assert any("at least 32 examples" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()
    
    def test_claude_invalid_field_type(self):
        """Test validation fails when field has wrong type."""
        examples = [
            {"prompt": 123, "completion": f"Answer {i}"}  # prompt is int, not str
            for i in range(35)
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "anthropic.claude-v2",
                DatasetFormat.JSONL
            )
            
            assert not result.valid
            assert any("must be a string" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()


class TestTitanFormatValidation:
    """Test validation for Titan fine-tuning format."""
    
    def test_valid_titan_format(self):
        """Test validation of valid Titan format dataset."""
        examples = [
            {"inputText": f"Input {i}", "outputText": f"Output {i}"}
            for i in range(35)
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "amazon.titan-text-express-v1",
                DatasetFormat.JSONL
            )
            
            assert result.valid
            assert len(result.errors) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_titan_missing_required_field(self):
        """Test validation fails when required field is missing."""
        examples = [
            {"inputText": f"Input {i}"}  # Missing 'outputText'
            for i in range(35)
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "amazon.titan-text-lite-v1",
                DatasetFormat.JSONL
            )
            
            assert not result.valid
            assert any("Missing required fields" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()
    
    def test_titan_insufficient_examples(self):
        """Test validation fails with insufficient examples."""
        examples = [
            {"inputText": f"Input {i}", "outputText": f"Output {i}"}
            for i in range(25)  # Less than minimum 32
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "amazon.titan-text-express-v1",
                DatasetFormat.JSONL
            )
            
            assert not result.valid
            assert any("at least 32 examples" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()


class TestLlamaFormatValidation:
    """Test validation for Llama fine-tuning format."""
    
    def test_valid_llama_format(self):
        """Test validation of valid Llama format dataset."""
        examples = [
            {"prompt": f"Prompt {i}", "completion": f"Completion {i}"}
            for i in range(35)
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "meta.llama2-13b-chat-v1",
                DatasetFormat.JSONL
            )
            
            assert result.valid
            assert len(result.errors) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_llama_missing_required_field(self):
        """Test validation fails when required field is missing."""
        examples = [
            {"prompt": f"Prompt {i}"}  # Missing 'completion'
            for i in range(35)
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "meta.llama2-70b-v1",
                DatasetFormat.JSONL
            )
            
            assert not result.valid
            assert any("Missing required fields" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()
    
    def test_llama_insufficient_examples(self):
        """Test validation fails with insufficient examples."""
        examples = [
            {"prompt": f"Prompt {i}", "completion": f"Completion {i}"}
            for i in range(15)  # Less than minimum 32
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "meta.llama2-13b-chat-v1",
                DatasetFormat.JSONL
            )
            
            assert not result.valid
            assert any("at least 32 examples" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()


class TestFormatRequirements:
    """Test format requirements retrieval."""
    
    def test_get_claude_requirements(self):
        """Test getting format requirements for Claude."""
        requirements = ModelValidator.get_format_requirements(
            "anthropic.claude-3-sonnet-20240229-v1:0"
        )
        
        assert requirements["model_family"] == "claude"
        assert requirements["format"] == "JSONL"
        assert "prompt" in requirements["required_fields"]
        assert "completion" in requirements["required_fields"]
        assert requirements["min_examples"] == 32
        assert requirements["max_tokens"] == 32000
    
    def test_get_titan_requirements(self):
        """Test getting format requirements for Titan."""
        requirements = ModelValidator.get_format_requirements(
            "amazon.titan-text-express-v1"
        )
        
        assert requirements["model_family"] == "titan"
        assert requirements["format"] == "JSONL"
        assert "inputText" in requirements["required_fields"]
        assert "outputText" in requirements["required_fields"]
        assert requirements["min_examples"] == 32
        assert requirements["max_tokens"] == 8192
    
    def test_get_llama_requirements(self):
        """Test getting format requirements for Llama."""
        requirements = ModelValidator.get_format_requirements(
            "meta.llama2-13b-chat-v1"
        )
        
        assert requirements["model_family"] == "llama"
        assert requirements["format"] == "JSONL"
        assert "prompt" in requirements["required_fields"]
        assert "completion" in requirements["required_fields"]
        assert requirements["min_examples"] == 32
        assert requirements["max_tokens"] == 4096
    
    def test_get_unknown_requirements(self):
        """Test getting format requirements for unknown model."""
        requirements = ModelValidator.get_format_requirements(
            "unknown-model-v1"
        )
        
        assert requirements["model_family"] == "unknown"
        assert "error" in requirements


class TestValidationResult:
    """Test ValidationResult class."""
    
    def test_validation_result_valid(self):
        """Test ValidationResult with valid status."""
        result = ValidationResult(valid=True)
        
        assert result.valid
        assert bool(result) is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_validation_result_invalid(self):
        """Test ValidationResult with invalid status."""
        result = ValidationResult(
            valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        
        assert not result.valid
        assert bool(result) is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
    
    def test_validation_result_repr(self):
        """Test ValidationResult string representation."""
        result = ValidationResult(
            valid=False,
            errors=["Error 1"],
            warnings=["Warning 1", "Warning 2"]
        )
        
        repr_str = repr(result)
        assert "valid=False" in repr_str
        assert "errors=1" in repr_str
        assert "warnings=2" in repr_str


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nonexistent_file(self):
        """Test validation with nonexistent file."""
        result = ModelValidator.validate_for_model(
            "/nonexistent/path/file.jsonl",
            "anthropic.claude-v2",
            DatasetFormat.JSONL
        )
        
        assert not result.valid
        assert any("File not found" in err for err in result.errors)
    
    def test_unknown_model_family(self):
        """Test validation with unknown model family."""
        examples = [
            {"prompt": f"Question {i}?", "completion": f"Answer {i}"}
            for i in range(35)
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "unknown-model-v1",
                DatasetFormat.JSONL
            )
            
            assert not result.valid
            assert any("Unknown model family" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()
    
    def test_wrong_format(self):
        """Test validation with wrong format (not JSONL)."""
        # Create a CSV file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f:
            f.write("prompt,completion\n")
            f.write("Question 1,Answer 1\n")
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "anthropic.claude-v2",
                DatasetFormat.CSV
            )
            
            assert not result.valid
            assert any("requires JSONL format" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_json(self):
        """Test validation with invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            f.write('{"prompt": "Question 1", "completion": "Answer 1"}\n')
            f.write('invalid json line\n')  # Invalid JSON
            f.write('{"prompt": "Question 2", "completion": "Answer 2"}\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "anthropic.claude-v2",
                DatasetFormat.JSONL
            )
            
            assert not result.valid
            assert any("Invalid JSON" in err for err in result.errors)
        finally:
            Path(temp_path).unlink()
    
    def test_token_limit_warning(self):
        """Test warning for examples exceeding token limits."""
        # Create example with very long text (exceeds token limit)
        long_text = "word " * 10000  # ~40,000 characters = ~10,000 tokens
        examples = [
            {"prompt": long_text, "completion": long_text}
            for i in range(35)
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
            temp_path = f.name
        
        try:
            result = ModelValidator.validate_for_model(
                temp_path,
                "meta.llama2-13b-chat-v1",  # Llama has 4096 token limit
                DatasetFormat.JSONL
            )
            
            # Should be valid but with warnings
            assert result.valid
            assert len(result.warnings) > 0
            assert any("exceeds recommended maximum" in warn for warn in result.warnings)
        finally:
            Path(temp_path).unlink()
