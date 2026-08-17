"""
Format auto-detection for datasets.

This module provides automatic detection of dataset formats (JSONL, CSV, Parquet, HuggingFace)
and task types (QA, summarization, classification, text_generation, chat) based on file
headers, extensions, and content structure.

Requirement 2.2: Implement format auto-detection
"""

import csv
import json
import os
from pathlib import Path
from typing import Optional, Union

from src.data_models.dataset import DatasetFormat, DatasetTaskType
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FormatDetector:
    """Detects dataset format and task type from file content."""

    @staticmethod
    def detect_format(file_path: Union[str, Path]) -> DatasetFormat:
        """
        Auto-detect dataset format from file content and extension.

        Args:
            file_path: Path to the dataset file

        Returns:
            DatasetFormat enum value

        Raises:
            ValueError: If format cannot be detected or file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        # Check file extension first
        extension = file_path.suffix.lower()

        # Try Parquet detection
        if extension == ".parquet":
            return DatasetFormat.PARQUET

        # Try CSV detection
        if extension == ".csv":
            return DatasetFormat.CSV

        # Try JSONL detection (common extensions)
        if extension in [".jsonl", ".jsonlines", ".ndjson"]:
            return DatasetFormat.JSONL

        # For .json files, need to check content
        if extension == ".json":
            # Could be JSONL or HuggingFace format
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    try:
                        # Try parsing as JSON object
                        json.loads(first_line)
                        # Check if it's a single-line JSON (JSONL)
                        second_line = f.readline().strip()
                        if second_line:
                            # Multiple JSON objects on separate lines = JSONL
                            return DatasetFormat.JSONL
                        else:
                            # Single JSON object, could be HuggingFace
                            return DatasetFormat.HUGGINGFACE
                    except json.JSONDecodeError:
                        pass

        # Content-based detection for files without clear extension
        try:
            # Try Parquet
            try:
                import pyarrow.parquet as pq
                pq.read_table(str(file_path))
                return DatasetFormat.PARQUET
            except Exception as e:
                logger.debug("Not a Parquet file (%s): %s", file_path, e)

            # Try JSONL
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    try:
                        json.loads(first_line)
                        return DatasetFormat.JSONL
                    except json.JSONDecodeError:
                        pass

            # Try CSV
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    sample = f.read(8192)
                    f.seek(0)
                    sniffer = csv.Sniffer()
                    sniffer.sniff(sample)
                    return DatasetFormat.CSV
                except csv.Error:
                    pass

        except Exception as e:
            raise ValueError(f"Unable to detect format for {file_path}: {e}")

        raise ValueError(f"Unable to detect format for {file_path}")

    @staticmethod
    def detect_task_type(file_path: Union[str, Path]) -> DatasetTaskType:
        """
        Auto-detect task type from dataset field structure.

        Analyzes the fields/columns in the dataset to determine the task type:
        - QA: Has 'question' and 'answer' or 'prompt' and 'completion' with context
        - Summarization: Has 'document'/'text' and 'summary'
        - Classification: Has 'text' and 'label'/'category'
        - Chat: Has 'messages' or 'conversation' structure
        - Text Generation: Has 'prompt' and 'completion'

        Args:
            file_path: Path to the dataset file

        Returns:
            DatasetTaskType enum value

        Raises:
            ValueError: If task type cannot be detected
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        # First detect format
        format_type = FormatDetector.detect_format(file_path)

        # Extract field names based on format
        field_names = FormatDetector._extract_field_names(file_path, format_type)

        if not field_names:
            raise ValueError(f"Unable to extract field names from {file_path}")

        # Normalize field names to lowercase for comparison
        field_names_lower = {name.lower() for name in field_names}

        # Detect task type based on field patterns
        # QA detection
        if ("question" in field_names_lower and "answer" in field_names_lower) or \
           ("question" in field_names_lower and "response" in field_names_lower) or \
           (("prompt" in field_names_lower or "query" in field_names_lower) and 
            ("context" in field_names_lower or "passage" in field_names_lower)):
            return DatasetTaskType.QA

        # Chat detection
        if "messages" in field_names_lower or "conversation" in field_names_lower or \
           "conversations" in field_names_lower:
            return DatasetTaskType.CHAT

        # Summarization detection
        if (("document" in field_names_lower or "text" in field_names_lower or 
             "article" in field_names_lower) and 
            ("summary" in field_names_lower or "summarization" in field_names_lower)):
            return DatasetTaskType.SUMMARIZATION

        # Classification detection
        if (("text" in field_names_lower or "sentence" in field_names_lower or 
             "input" in field_names_lower) and 
            ("label" in field_names_lower or "category" in field_names_lower or 
             "class" in field_names_lower)):
            return DatasetTaskType.CLASSIFICATION

        # Text generation detection (most generic)
        if ("prompt" in field_names_lower and 
            ("completion" in field_names_lower or "response" in field_names_lower or 
             "output" in field_names_lower)):
            return DatasetTaskType.TEXT_GENERATION

        # Default to custom if no pattern matches
        return DatasetTaskType.CUSTOM

    @staticmethod
    def _extract_field_names(
        file_path: Path, 
        format_type: DatasetFormat
    ) -> set[str]:
        """
        Extract field names from a dataset file.

        Args:
            file_path: Path to the dataset file
            format_type: Detected format type

        Returns:
            Set of field names found in the dataset
        """
        field_names = set()

        try:
            if format_type == DatasetFormat.JSONL:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Read first valid JSON line
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                obj = json.loads(line)
                                if isinstance(obj, dict):
                                    field_names.update(obj.keys())
                                    break
                            except json.JSONDecodeError:
                                continue

            elif format_type == DatasetFormat.CSV:
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        field_names.update(reader.fieldnames)

            elif format_type == DatasetFormat.PARQUET:
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(str(file_path))
                    field_names.update(table.schema.names)
                except Exception as e:
                    logger.debug(
                        "Could not read Parquet schema for %s: %s", file_path, e
                    )

            elif format_type == DatasetFormat.HUGGINGFACE:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    try:
                        data = json.loads(content)
                        # HuggingFace datasets can have various structures
                        if isinstance(data, dict):
                            # Could be a dataset dict with splits
                            if "train" in data or "test" in data or "validation" in data:
                                # Get fields from first split
                                for split_name in ["train", "test", "validation"]:
                                    if split_name in data and data[split_name]:
                                        first_item = data[split_name][0] if isinstance(data[split_name], list) else data[split_name]
                                        if isinstance(first_item, dict):
                                            field_names.update(first_item.keys())
                                            break
                            else:
                                # Direct dict, use its keys
                                field_names.update(data.keys())
                        elif isinstance(data, list) and data:
                            # List of examples
                            if isinstance(data[0], dict):
                                field_names.update(data[0].keys())
                    except json.JSONDecodeError as e:
                        logger.debug(
                            "Invalid JSON in HuggingFace dataset %s: %s",
                            file_path, e
                        )

        except Exception as e:
            logger.debug("Could not extract fields from %s: %s", file_path, e)

        return field_names
