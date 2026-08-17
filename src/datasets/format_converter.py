"""
Format converter for bidirectional dataset conversion.

This module provides conversion between JSONL, CSV, and Parquet formats
while preserving all data during conversion.

Requirement 2.9: Implement format converter: JSONL ↔ CSV ↔ Parquet (bidirectional)
"""

import csv
import json
from pathlib import Path
from typing import Any, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_models.dataset import DatasetFormat, DatasetTaskType
from src.datasets.parsers.csv_parser import CSVParser
from src.datasets.parsers.jsonl_parser import JSONLParser
from src.datasets.parsers.parquet_parser import ParquetParser


class FormatConverter:
    """Converter for dataset formats with data preservation."""

    @staticmethod
    def convert(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        source_format: DatasetFormat,
        target_format: DatasetFormat,
        task_type: Optional[DatasetTaskType] = None,
        column_mapping: Optional[dict[str, str]] = None
    ) -> dict[str, Any]:
        """
        Convert a dataset from one format to another.

        Args:
            input_path: Path to input file
            output_path: Path to output file
            source_format: Source format (JSONL, CSV, or PARQUET)
            target_format: Target format (JSONL, CSV, or PARQUET)
            task_type: Optional task type for validation
            column_mapping: Optional column mapping for CSV parsing

        Returns:
            Dictionary with conversion results:
            - success: bool
            - rows_converted: int
            - source_format: str
            - target_format: str
            - output_path: str

        Raises:
            ValueError: If conversion fails or formats are invalid
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_path}")

        # If formats are the same, just copy
        if source_format == target_format:
            raise ValueError(
                f"Source and target formats are the same: {source_format}"
            )

        # Parse source data
        data = FormatConverter._parse_source(
            input_path, source_format, task_type, column_mapping
        )

        if not data:
            raise ValueError(f"No data found in {input_path}")

        # Write to target format
        FormatConverter._write_target(
            data, output_path, target_format
        )

        return {
            "success": True,
            "rows_converted": len(data),
            "source_format": source_format.value,
            "target_format": target_format.value,
            "output_path": str(output_path),
        }

    @staticmethod
    def _parse_source(
        file_path: Path,
        format: DatasetFormat,
        task_type: Optional[DatasetTaskType],
        column_mapping: Optional[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """
        Parse source file based on format.

        Args:
            file_path: Path to source file
            format: Source format
            task_type: Optional task type for validation
            column_mapping: Optional column mapping for CSV

        Returns:
            List of data rows as dictionaries

        Raises:
            ValueError: If format is unsupported or parsing fails
        """
        if format == DatasetFormat.JSONL:
            # Parse JSONL without validation if task_type not provided
            if task_type:
                return JSONLParser.parse(
                    file_path, task_type, validate=False
                )
            else:
                # Parse without validation
                data = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                return data

        elif format == DatasetFormat.CSV:
            # Parse CSV without validation if task_type not provided
            if task_type:
                return CSVParser.parse(
                    file_path, task_type, column_mapping, validate=False
                )
            else:
                # Parse without validation
                data = []
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        data.append(dict(row))
                return data

        elif format == DatasetFormat.PARQUET:
            # Parse Parquet without validation if task_type not provided
            if task_type:
                return ParquetParser.parse(
                    file_path, task_type, validate=False
                )
            else:
                # Parse without validation
                table = pq.read_table(str(file_path))
                data = []
                for batch in table.to_batches():
                    batch_dict = batch.to_pydict()
                    num_rows = len(batch_dict[table.schema.names[0]])
                    for i in range(num_rows):
                        row = {
                            col: batch_dict[col][i]
                            for col in table.schema.names
                        }
                        data.append(row)
                return data

        else:
            raise ValueError(f"Unsupported source format: {format}")

    @staticmethod
    def _write_target(
        data: list[dict[str, Any]],
        file_path: Path,
        format: DatasetFormat
    ) -> None:
        """
        Write data to target file based on format.

        Args:
            data: List of data rows as dictionaries
            file_path: Path to target file
            format: Target format

        Raises:
            ValueError: If format is unsupported or writing fails
        """
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == DatasetFormat.JSONL:
            FormatConverter._write_jsonl(data, file_path)

        elif format == DatasetFormat.CSV:
            FormatConverter._write_csv(data, file_path)

        elif format == DatasetFormat.PARQUET:
            FormatConverter._write_parquet(data, file_path)

        else:
            raise ValueError(f"Unsupported target format: {format}")

    @staticmethod
    def _write_jsonl(
        data: list[dict[str, Any]],
        file_path: Path
    ) -> None:
        """
        Write data to JSONL file.

        Args:
            data: List of data rows
            file_path: Output file path
        """
        with open(file_path, "w", encoding="utf-8") as f:
            for row in data:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")

    @staticmethod
    def _write_csv(
        data: list[dict[str, Any]],
        file_path: Path
    ) -> None:
        """
        Write data to CSV file.

        Args:
            data: List of data rows
            file_path: Output file path
        """
        if not data:
            raise ValueError("Cannot write empty data to CSV")

        # Get all unique field names from all rows
        fieldnames = set()
        for row in data:
            fieldnames.update(row.keys())

        # Sort fieldnames for consistent output
        fieldnames = sorted(fieldnames)

        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in data:
                # Handle complex types (lists, dicts) by converting to JSON
                processed_row = {}
                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        processed_row[key] = json.dumps(
                            value, ensure_ascii=False
                        )
                    else:
                        processed_row[key] = value

                writer.writerow(processed_row)

    @staticmethod
    def _write_parquet(
        data: list[dict[str, Any]],
        file_path: Path
    ) -> None:
        """
        Write data to Parquet file.

        Args:
            data: List of data rows
            file_path: Output file path
        """
        if not data:
            raise ValueError("Cannot write empty data to Parquet")

        # Convert list of dicts to PyArrow table
        # PyArrow will infer schema from the data
        table = pa.Table.from_pylist(data)

        # Write to Parquet file
        pq.write_table(table, str(file_path))

    @staticmethod
    def jsonl_to_csv(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        task_type: Optional[DatasetTaskType] = None
    ) -> dict[str, Any]:
        """
        Convert JSONL to CSV format.

        Args:
            input_path: Path to JSONL file
            output_path: Path to output CSV file
            task_type: Optional task type for validation

        Returns:
            Conversion results dictionary
        """
        return FormatConverter.convert(
            input_path,
            output_path,
            DatasetFormat.JSONL,
            DatasetFormat.CSV,
            task_type
        )

    @staticmethod
    def jsonl_to_parquet(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        task_type: Optional[DatasetTaskType] = None
    ) -> dict[str, Any]:
        """
        Convert JSONL to Parquet format.

        Args:
            input_path: Path to JSONL file
            output_path: Path to output Parquet file
            task_type: Optional task type for validation

        Returns:
            Conversion results dictionary
        """
        return FormatConverter.convert(
            input_path,
            output_path,
            DatasetFormat.JSONL,
            DatasetFormat.PARQUET,
            task_type
        )

    @staticmethod
    def csv_to_jsonl(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        task_type: Optional[DatasetTaskType] = None,
        column_mapping: Optional[dict[str, str]] = None
    ) -> dict[str, Any]:
        """
        Convert CSV to JSONL format.

        Args:
            input_path: Path to CSV file
            output_path: Path to output JSONL file
            task_type: Optional task type for validation
            column_mapping: Optional column mapping

        Returns:
            Conversion results dictionary
        """
        return FormatConverter.convert(
            input_path,
            output_path,
            DatasetFormat.CSV,
            DatasetFormat.JSONL,
            task_type,
            column_mapping
        )

    @staticmethod
    def csv_to_parquet(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        task_type: Optional[DatasetTaskType] = None,
        column_mapping: Optional[dict[str, str]] = None
    ) -> dict[str, Any]:
        """
        Convert CSV to Parquet format.

        Args:
            input_path: Path to CSV file
            output_path: Path to output Parquet file
            task_type: Optional task type for validation
            column_mapping: Optional column mapping

        Returns:
            Conversion results dictionary
        """
        return FormatConverter.convert(
            input_path,
            output_path,
            DatasetFormat.CSV,
            DatasetFormat.PARQUET,
            task_type,
            column_mapping
        )

    @staticmethod
    def parquet_to_jsonl(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        task_type: Optional[DatasetTaskType] = None
    ) -> dict[str, Any]:
        """
        Convert Parquet to JSONL format.

        Args:
            input_path: Path to Parquet file
            output_path: Path to output JSONL file
            task_type: Optional task type for validation

        Returns:
            Conversion results dictionary
        """
        return FormatConverter.convert(
            input_path,
            output_path,
            DatasetFormat.PARQUET,
            DatasetFormat.JSONL,
            task_type
        )

    @staticmethod
    def parquet_to_csv(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        task_type: Optional[DatasetTaskType] = None
    ) -> dict[str, Any]:
        """
        Convert Parquet to CSV format.

        Args:
            input_path: Path to Parquet file
            output_path: Path to output CSV file
            task_type: Optional task type for validation

        Returns:
            Conversion results dictionary
        """
        return FormatConverter.convert(
            input_path,
            output_path,
            DatasetFormat.PARQUET,
            DatasetFormat.CSV,
            task_type
        )
