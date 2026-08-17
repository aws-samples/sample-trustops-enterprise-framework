"""
Unit tests for DataValidator class.
"""
import pytest
import json
import tempfile
from pathlib import Path
from src.data_models import DataValidator, ValidationError


class TestDataValidator:
    """Test suite for DataValidator."""
    
    def test_valid_training_data(self):
        """Test validation passes for valid training data."""
        validator = DataValidator(min_sample_count=2)
        
        data = [
            {
                'id': 'ex1',
                'prompt': 'What is machine learning?',
                'completion': 'Machine learning is a subset of AI.'
            },
            {
                'id': 'ex2',
                'prompt': 'Explain neural networks.',
                'completion': 'Neural networks are computing systems inspired by biological neural networks.'
            }
        ]
        
        result = validator.validate_training_data(data)
        
        assert result.is_valid
        assert result.total_examples == 2
        assert result.valid_examples == 2
        assert len(result.errors) == 0
        assert result.statistics['total_examples'] == 2
    
    def test_valid_evaluation_data(self):
        """Test validation passes for valid evaluation data."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'eval1',
                'prompt': 'What is AI?',
                'source_documents': ['AI is artificial intelligence.'],
                'category': 'technical'
            }
        ]
        
        result = validator.validate_evaluation_data(data)
        
        assert result.is_valid
        assert result.total_examples == 1
        assert result.valid_examples == 1
        assert len(result.errors) == 0
    
    def test_missing_required_fields_training(self):
        """Test validation fails when required fields are missing in training data."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'ex1',
                'prompt': 'What is AI?'
                # Missing 'completion' field
            }
        ]
        
        result = validator.validate_training_data(data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any(e.error_type == 'missing_fields' for e in result.errors)
        assert any('completion' in e.message for e in result.errors)
    
    def test_missing_required_fields_evaluation(self):
        """Test validation fails when required fields are missing in evaluation data."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'eval1',
                'prompt': 'What is AI?'
                # Missing 'source_documents' and 'category'
            }
        ]
        
        result = validator.validate_evaluation_data(data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any(e.error_type == 'missing_fields' for e in result.errors)
    
    def test_invalid_field_types(self):
        """Test validation fails when field types are incorrect."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'ex1',
                'prompt': 123,  # Should be string
                'completion': 'Valid completion'
            }
        ]
        
        result = validator.validate_training_data(data)
        
        assert not result.is_valid
        assert any(e.error_type == 'invalid_field_type' for e in result.errors)
    
    def test_empty_dataset(self):
        """Test validation fails for empty dataset."""
        validator = DataValidator(min_sample_count=1)
        
        result = validator.validate_training_data([])
        
        assert not result.is_valid
        assert any(e.error_type == 'empty_dataset' for e in result.errors)
    
    def test_insufficient_samples(self):
        """Test validation fails when sample count is below minimum."""
        validator = DataValidator(min_sample_count=10)
        
        data = [
            {
                'id': 'ex1',
                'prompt': 'What is AI?',
                'completion': 'AI is artificial intelligence.'
            }
        ]
        
        result = validator.validate_training_data(data)
        
        assert not result.is_valid
        assert any(e.error_type == 'insufficient_samples' for e in result.errors)
    
    def test_prompt_too_short(self):
        """Test validation fails when prompt is too short."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'ex1',
                'prompt': '',  # Empty prompt
                'completion': 'Valid completion'
            }
        ]
        
        result = validator.validate_training_data(data)
        
        assert not result.is_valid
        assert any(e.error_type == 'invalid_prompt_length' for e in result.errors)
    
    def test_completion_too_short(self):
        """Test validation fails when completion is too short."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'ex1',
                'prompt': 'What is AI?',
                'completion': ''  # Empty completion
            }
        ]
        
        result = validator.validate_training_data(data)
        
        assert not result.is_valid
        assert any(e.error_type == 'invalid_completion_length' for e in result.errors)
    
    def test_long_prompt_warning(self):
        """Test warning is generated for very long prompts."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'ex1',
                'prompt': 'x' * 15000,  # Very long prompt
                'completion': 'Valid completion'
            }
        ]
        
        result = validator.validate_training_data(data)
        
        assert len(result.warnings) > 0
        assert any('very long' in w.lower() for w in result.warnings)
    
    def test_invalid_source_documents_type(self):
        """Test validation fails when source_documents is not a list."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'eval1',
                'prompt': 'What is AI?',
                'source_documents': 'Not a list',  # Should be list
                'category': 'technical'
            }
        ]
        
        result = validator.validate_evaluation_data(data)
        
        assert not result.is_valid
        assert any(e.error_type == 'invalid_field_type' for e in result.errors)
    
    def test_empty_source_documents_warning(self):
        """Test warning is generated for empty source documents."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'eval1',
                'prompt': 'What is AI?',
                'source_documents': [],  # Empty list
                'category': 'technical'
            }
        ]
        
        result = validator.validate_evaluation_data(data)
        
        assert len(result.warnings) > 0
        assert any('no source documents' in w.lower() for w in result.warnings)
    
    def test_invalid_source_document_element(self):
        """Test validation fails when source document element is not a string."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'eval1',
                'prompt': 'What is AI?',
                'source_documents': ['Valid doc', 123],  # Second element not string
                'category': 'technical'
            }
        ]
        
        result = validator.validate_evaluation_data(data)
        
        assert not result.is_valid
        assert any(e.error_type == 'invalid_source_document' for e in result.errors)
    
    def test_load_jsonl_file(self):
        """Test loading data from JSONL file."""
        validator = DataValidator(min_sample_count=1)
        
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({'prompt': 'Q1', 'completion': 'A1'}) + '\n')
            f.write(json.dumps({'prompt': 'Q2', 'completion': 'A2'}) + '\n')
            temp_path = f.name
        
        try:
            result = validator.validate_training_data(temp_path, data_format='jsonl')
            
            assert result.total_examples == 2
            assert result.valid_examples == 2
        finally:
            Path(temp_path).unlink()
    
    def test_load_json_file_list(self):
        """Test loading data from JSON file (list format)."""
        validator = DataValidator(min_sample_count=1)
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {'prompt': 'Q1', 'completion': 'A1'},
                {'prompt': 'Q2', 'completion': 'A2'}
            ], f)
            temp_path = f.name
        
        try:
            result = validator.validate_training_data(temp_path, data_format='json')
            
            assert result.total_examples == 2
            assert result.valid_examples == 2
        finally:
            Path(temp_path).unlink()
    
    def test_load_json_file_dict(self):
        """Test loading data from JSON file (dict with examples key)."""
        validator = DataValidator(min_sample_count=1)
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'examples': [
                    {'prompt': 'Q1', 'completion': 'A1'},
                    {'prompt': 'Q2', 'completion': 'A2'}
                ]
            }, f)
            temp_path = f.name
        
        try:
            result = validator.validate_training_data(temp_path, data_format='json')
            
            assert result.total_examples == 2
            assert result.valid_examples == 2
        finally:
            Path(temp_path).unlink()
    
    def test_file_not_found(self):
        """Test error handling when file doesn't exist."""
        validator = DataValidator(min_sample_count=1)
        
        result = validator.validate_training_data('/nonexistent/file.jsonl', data_format='jsonl')
        
        assert not result.is_valid
        assert any(e.error_type == 'file_load_error' for e in result.errors)
    
    def test_invalid_json(self):
        """Test error handling for malformed JSON."""
        validator = DataValidator(min_sample_count=1)
        
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"prompt": "Q1", invalid json}\n')
            temp_path = f.name
        
        try:
            result = validator.validate_training_data(temp_path, data_format='jsonl')
            
            assert not result.is_valid
            assert any(e.error_type == 'file_load_error' for e in result.errors)
        finally:
            Path(temp_path).unlink()
    
    def test_statistics_calculation(self):
        """Test that statistics are calculated correctly."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            {
                'id': 'ex1',
                'prompt': 'Short',
                'completion': 'Also short'
            },
            {
                'id': 'ex2',
                'prompt': 'This is a much longer prompt',
                'completion': 'This is a much longer completion text'
            }
        ]
        
        result = validator.validate_training_data(data)
        
        assert 'avg_prompt_length' in result.statistics
        assert 'max_prompt_length' in result.statistics
        assert 'min_prompt_length' in result.statistics
        assert 'avg_completion_length' in result.statistics
        assert result.statistics['total_examples'] == 2
        assert result.statistics['valid_examples'] == 2
    
    def test_strict_mode_warnings_as_errors(self):
        """Test that strict mode treats warnings as errors."""
        validator = DataValidator(min_sample_count=1, strict_mode=True)
        
        data = [
            {
                'id': 'eval1',
                'prompt': 'What is AI?',
                'source_documents': [],  # Empty - generates warning
                'category': 'technical'
            }
        ]
        
        result = validator.validate_evaluation_data(data)
        
        # In strict mode, warnings should cause validation to fail
        assert not result.is_valid
        assert len(result.warnings) > 0
    
    def test_non_strict_mode_allows_warnings(self):
        """Test that non-strict mode allows warnings."""
        validator = DataValidator(min_sample_count=1, strict_mode=False)
        
        data = [
            {
                'id': 'eval1',
                'prompt': 'What is AI?',
                'source_documents': [],  # Empty - generates warning
                'category': 'technical'
            }
        ]
        
        result = validator.validate_evaluation_data(data)
        
        # In non-strict mode, warnings should not cause validation to fail
        assert result.is_valid
        assert len(result.warnings) > 0
    
    def test_invalid_example_structure(self):
        """Test validation fails when example is not a dictionary."""
        validator = DataValidator(min_sample_count=1)
        
        data = [
            'not a dictionary',  # Invalid structure
            {'prompt': 'Valid', 'completion': 'Valid'}
        ]
        
        result = validator.validate_training_data(data)
        
        assert not result.is_valid
        assert any(e.error_type == 'invalid_structure' for e in result.errors)
    
    def test_encoding_validation_utf8(self):
        """Test that UTF-8 encoding is validated."""
        validator = DataValidator(min_sample_count=1)
        
        # Valid UTF-8 data
        data = [
            {
                'id': 'ex1',
                'prompt': 'What is AI? 你好',  # Contains Chinese characters
                'completion': 'AI is artificial intelligence. مرحبا'  # Contains Arabic
            }
        ]
        
        result = validator.validate_training_data(data)
        
        # Should pass - UTF-8 is valid
        assert result.is_valid
        assert result.valid_examples == 1
