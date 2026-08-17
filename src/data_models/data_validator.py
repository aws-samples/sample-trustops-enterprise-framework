"""
Dataset validation logic for TrustOps training and evaluation data.
"""
import json
from typing import List, Dict, Any, Union
from pathlib import Path
from .workflow import DataValidationResult, ValidationError


class DataValidator:
    """
    Validates training and evaluation datasets for TrustOps.
    
    Performs format validation, quality checks, and error reporting
    to ensure datasets meet AWS Bedrock requirements.
    """
    
    # Validation thresholds
    MIN_SAMPLE_COUNT = 10
    MIN_PROMPT_LENGTH = 1
    MIN_COMPLETION_LENGTH = 1
    MAX_PROMPT_LENGTH = 10000
    MAX_COMPLETION_LENGTH = 10000
    
    # Required fields for different dataset types
    TRAINING_REQUIRED_FIELDS = {'prompt', 'completion'}
    EVALUATION_REQUIRED_FIELDS = {'prompt', 'source_documents', 'category'}
    
    def __init__(
        self,
        min_sample_count: int = MIN_SAMPLE_COUNT,
        strict_mode: bool = False
    ):
        """
        Initialize DataValidator.
        
        Args:
            min_sample_count: Minimum number of valid samples required
            strict_mode: If True, warnings are treated as errors
        """
        self.min_sample_count = min_sample_count
        self.strict_mode = strict_mode
    
    def validate_training_data(
        self,
        data: Union[str, List[Dict[str, Any]]],
        data_format: str = 'jsonl'
    ) -> DataValidationResult:
        """
        Validate training data for fine-tuning.
        
        Args:
            data: Either a file path string or list of dictionaries
            data_format: Format of the data ('json' or 'jsonl')
            
        Returns:
            DataValidationResult with validation status and errors
        """
        errors: List[ValidationError] = []
        warnings: List[str] = []
        examples: List[Dict[str, Any]] = []
        
        # Load data if it's a file path
        if isinstance(data, str):
            try:
                examples = self._load_data_from_file(data, data_format)
            except Exception as e:
                errors.append(ValidationError(
                    error_type='file_load_error',
                    message=f'Failed to load data file: {str(e)}',
                    line_number=None,
                    example_id=None
                ))
                return self._create_validation_result(
                    examples=[], errors=errors, warnings=warnings
                )
        else:
            examples = data
        
        # Validate format and structure
        format_errors = self._validate_format(
            examples, self.TRAINING_REQUIRED_FIELDS, 'training'
        )
        errors.extend(format_errors)
        
        # Validate quality
        quality_errors, quality_warnings = self._validate_quality(
            examples, 'training'
        )
        errors.extend(quality_errors)
        warnings.extend(quality_warnings)
        
        # Validate character encoding
        encoding_errors = self._validate_encoding(examples)
        errors.extend(encoding_errors)
        
        # Check minimum sample count
        valid_examples = len(examples) - len([
            e for e in errors if e.example_id is not None
        ])
        
        if valid_examples < self.min_sample_count:
            errors.append(ValidationError(
                error_type='insufficient_samples',
                message=(
                    f'Dataset has {valid_examples} valid samples, '
                    f'minimum required is {self.min_sample_count}'
                ),
                line_number=None,
                example_id=None
            ))
        
        return self._create_validation_result(
            examples=examples, errors=errors, warnings=warnings
        )
    
    def validate_evaluation_data(
        self,
        data: Union[str, List[Dict[str, Any]]],
        data_format: str = 'json'
    ) -> DataValidationResult:
        """
        Validate evaluation dataset.
        
        Args:
            data: Either a file path string or list of dictionaries
            data_format: Format of the data ('json' or 'jsonl')
            
        Returns:
            DataValidationResult with validation status and errors
        """
        errors: List[ValidationError] = []
        warnings: List[str] = []
        examples: List[Dict[str, Any]] = []
        
        # Load data if it's a file path
        if isinstance(data, str):
            try:
                examples = self._load_data_from_file(data, data_format)
            except Exception as e:
                errors.append(ValidationError(
                    error_type='file_load_error',
                    message=f'Failed to load data file: {str(e)}',
                    line_number=None,
                    example_id=None
                ))
                return self._create_validation_result(
                    examples=[], errors=errors, warnings=warnings
                )
        else:
            examples = data
        
        # Validate format and structure
        format_errors = self._validate_format(
            examples, self.EVALUATION_REQUIRED_FIELDS, 'evaluation'
        )
        errors.extend(format_errors)
        
        # Validate quality
        quality_errors, quality_warnings = self._validate_quality(
            examples, 'evaluation'
        )
        errors.extend(quality_errors)
        warnings.extend(quality_warnings)
        
        # Validate character encoding
        encoding_errors = self._validate_encoding(examples)
        errors.extend(encoding_errors)
        
        return self._create_validation_result(
            examples=examples, errors=errors, warnings=warnings
        )
    
    def _load_data_from_file(
        self,
        file_path: str,
        data_format: str
    ) -> List[Dict[str, Any]]:
        """
        Load data from file.
        
        Args:
            file_path: Path to data file
            data_format: Format of the data ('json' or 'jsonl')
            
        Returns:
            List of example dictionaries
            
        Raises:
            ValueError: If format is invalid
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')
        
        if data_format == 'jsonl':
            examples = []
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                            f'Invalid JSON on line {line_num}: {e.msg}',
                            e.doc,
                            e.pos
                        )
            return examples
        
        elif data_format == 'json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle both list and dict with 'examples' key
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'examples' in data:
                return data['examples']
            else:
                raise ValueError(
                    'JSON must be a list or dict with "examples" key'
                )
        
        else:
            raise ValueError(f'Unsupported format: {data_format}')
    
    def _validate_format(
        self,
        examples: List[Dict[str, Any]],
        required_fields: set,
        dataset_type: str
    ) -> List[ValidationError]:
        """
        Validate format and required fields.
        
        Args:
            examples: List of example dictionaries
            required_fields: Set of required field names
            dataset_type: Type of dataset ('training' or 'evaluation')
            
        Returns:
            List of validation errors
        """
        errors: List[ValidationError] = []
        
        if not examples:
            errors.append(ValidationError(
                error_type='empty_dataset',
                message='Dataset is empty',
                line_number=None,
                example_id=None
            ))
            return errors
        
        for idx, example in enumerate(examples):
            # Check if example is a dictionary first
            if not isinstance(example, dict):
                errors.append(ValidationError(
                    error_type='invalid_structure',
                    message=f'Example must be a dictionary, got {type(example).__name__}',
                    line_number=idx + 1,
                    example_id=f'example_{idx}'
                ))
                continue
            
            example_id = example.get('id', f'example_{idx}')
            
            # Check required fields
            missing_fields = required_fields - set(example.keys())
            if missing_fields:
                errors.append(ValidationError(
                    error_type='missing_fields',
                    message=f'Missing required fields: {", ".join(sorted(missing_fields))}',
                    line_number=idx + 1,
                    example_id=example_id
                ))
            
            # Validate field types
            if 'prompt' in example and not isinstance(example['prompt'], str):
                errors.append(ValidationError(
                    error_type='invalid_field_type',
                    message=f'Field "prompt" must be a string, got {type(example["prompt"]).__name__}',
                    line_number=idx + 1,
                    example_id=example_id
                ))
            
            if dataset_type == 'training':
                if 'completion' in example and not isinstance(example['completion'], str):
                    errors.append(ValidationError(
                        error_type='invalid_field_type',
                        message=f'Field "completion" must be a string, got {type(example["completion"]).__name__}',
                        line_number=idx + 1,
                        example_id=example_id
                    ))
            
            if dataset_type == 'evaluation':
                if 'source_documents' in example and not isinstance(example['source_documents'], list):
                    errors.append(ValidationError(
                        error_type='invalid_field_type',
                        message=f'Field "source_documents" must be a list, got {type(example["source_documents"]).__name__}',
                        line_number=idx + 1,
                        example_id=example_id
                    ))
                
                if 'category' in example and not isinstance(example['category'], str):
                    errors.append(ValidationError(
                        error_type='invalid_field_type',
                        message=f'Field "category" must be a string, got {type(example["category"]).__name__}',
                        line_number=idx + 1,
                        example_id=example_id
                    ))
        
        return errors
    
    def _validate_quality(
        self,
        examples: List[Dict[str, Any]],
        dataset_type: str
    ) -> tuple[List[ValidationError], List[str]]:
        """
        Validate data quality.
        
        Args:
            examples: List of example dictionaries
            dataset_type: Type of dataset ('training' or 'evaluation')
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors: List[ValidationError] = []
        warnings: List[str] = []
        
        for idx, example in enumerate(examples):
            # Skip non-dict examples (already caught in format validation)
            if not isinstance(example, dict):
                continue
            
            example_id = example.get('id', f'example_{idx}')
            
            # Validate prompt length
            if 'prompt' in example and isinstance(example['prompt'], str):
                prompt_len = len(example['prompt'])
                
                if prompt_len < self.MIN_PROMPT_LENGTH:
                    errors.append(ValidationError(
                        error_type='invalid_prompt_length',
                        message=f'Prompt is too short ({prompt_len} chars), minimum is {self.MIN_PROMPT_LENGTH}',
                        line_number=idx + 1,
                        example_id=example_id
                    ))
                
                if prompt_len > self.MAX_PROMPT_LENGTH:
                    warnings.append(
                        f'Example {example_id}: Prompt is very long ({prompt_len} chars), '
                        f'may exceed model limits'
                    )
            
            # Validate completion/response length for training data
            if dataset_type == 'training' and 'completion' in example:
                if isinstance(example['completion'], str):
                    completion_len = len(example['completion'])
                    
                    if completion_len < self.MIN_COMPLETION_LENGTH:
                        errors.append(ValidationError(
                            error_type='invalid_completion_length',
                            message=f'Completion is too short ({completion_len} chars), minimum is {self.MIN_COMPLETION_LENGTH}',
                            line_number=idx + 1,
                            example_id=example_id
                        ))
                    
                    if completion_len > self.MAX_COMPLETION_LENGTH:
                        warnings.append(
                            f'Example {example_id}: Completion is very long ({completion_len} chars), '
                            f'may exceed model limits'
                        )
            
            # Validate source documents for evaluation data
            if dataset_type == 'evaluation' and 'source_documents' in example:
                if isinstance(example['source_documents'], list):
                    if len(example['source_documents']) == 0:
                        warnings.append(
                            f'Example {example_id}: No source documents provided, '
                            f'trust scoring may be limited'
                        )
                    
                    # Check that all source documents are strings
                    for doc_idx, doc in enumerate(example['source_documents']):
                        if not isinstance(doc, str):
                            errors.append(ValidationError(
                                error_type='invalid_source_document',
                                message=f'Source document {doc_idx} must be a string, got {type(doc).__name__}',
                                line_number=idx + 1,
                                example_id=example_id
                            ))
        
        return errors, warnings
    
    def _validate_encoding(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """
        Validate character encoding.
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            List of validation errors
        """
        errors: List[ValidationError] = []
        
        for idx, example in enumerate(examples):
            # Skip non-dict examples (already caught in format validation)
            if not isinstance(example, dict):
                continue
            
            example_id = example.get('id', f'example_{idx}')
            
            # Check prompt encoding
            if 'prompt' in example and isinstance(example['prompt'], str):
                try:
                    example['prompt'].encode('utf-8')
                except UnicodeEncodeError as e:
                    errors.append(ValidationError(
                        error_type='encoding_error',
                        message=f'Prompt contains invalid UTF-8 characters: {str(e)}',
                        line_number=idx + 1,
                        example_id=example_id
                    ))
            
            # Check completion encoding
            if 'completion' in example and isinstance(example['completion'], str):
                try:
                    example['completion'].encode('utf-8')
                except UnicodeEncodeError as e:
                    errors.append(ValidationError(
                        error_type='encoding_error',
                        message=f'Completion contains invalid UTF-8 characters: {str(e)}',
                        line_number=idx + 1,
                        example_id=example_id
                    ))
            
            # Check source documents encoding
            if 'source_documents' in example and isinstance(example['source_documents'], list):
                for doc_idx, doc in enumerate(example['source_documents']):
                    if isinstance(doc, str):
                        try:
                            doc.encode('utf-8')
                        except UnicodeEncodeError as e:
                            errors.append(ValidationError(
                                error_type='encoding_error',
                                message=f'Source document {doc_idx} contains invalid UTF-8 characters: {str(e)}',
                                line_number=idx + 1,
                                example_id=example_id
                            ))
        
        return errors
    
    def _create_validation_result(
        self,
        examples: List[Dict[str, Any]],
        errors: List[ValidationError],
        warnings: List[str]
    ) -> DataValidationResult:
        """
        Create validation result with statistics.
        
        Args:
            examples: List of example dictionaries
            errors: List of validation errors
            warnings: List of warning messages
            
        Returns:
            DataValidationResult
        """
        # Count valid examples (those without errors)
        example_ids_with_errors = {
            e.example_id for e in errors if e.example_id is not None
        }
        valid_examples = len(examples) - len(example_ids_with_errors)
        
        # Calculate statistics
        statistics = {
            'total_examples': len(examples),
            'valid_examples': valid_examples,
            'error_count': len(errors),
            'warning_count': len(warnings)
        }
        
        if examples:
            # Calculate average lengths (only for dict examples)
            prompt_lengths = [
                len(ex.get('prompt', ''))
                for ex in examples
                if isinstance(ex, dict) and isinstance(ex.get('prompt'), str)
            ]
            if prompt_lengths:
                statistics['avg_prompt_length'] = sum(prompt_lengths) / len(prompt_lengths)
                statistics['max_prompt_length'] = max(prompt_lengths)
                statistics['min_prompt_length'] = min(prompt_lengths)
            
            completion_lengths = [
                len(ex.get('completion', ''))
                for ex in examples
                if isinstance(ex, dict) and isinstance(ex.get('completion'), str)
            ]
            if completion_lengths:
                statistics['avg_completion_length'] = sum(completion_lengths) / len(completion_lengths)
                statistics['max_completion_length'] = max(completion_lengths)
                statistics['min_completion_length'] = min(completion_lengths)
        
        # Determine if validation passed
        is_valid = (
            len(errors) == 0 and
            valid_examples >= self.min_sample_count
        )
        
        # In strict mode, warnings are treated as errors
        if self.strict_mode and warnings:
            is_valid = False
        
        return DataValidationResult(
            is_valid=is_valid,
            total_examples=len(examples),
            valid_examples=valid_examples,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )
