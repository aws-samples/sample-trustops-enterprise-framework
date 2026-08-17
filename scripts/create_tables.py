"""
DynamoDB table creation script for TrustOps Enterprise Framework.

Creates all required DynamoDB tables with GSIs for the TrustOps
storage layer.

Requirements: 9.14

Usage:
    python scripts/create_tables.py [--region REGION] [--prefix PREFIX] [--endpoint-url URL]
"""

import argparse
import logging
import sys
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Table definitions matching infrastructure/dynamodb_tables.yaml.
#
# These are DynamoDB table names, not S3 bucket names: the namespace is scoped
# to the account and region, so a fixed default cannot be claimed by another
# account the way a globally unique bucket name can. See config/aws_config.py
# for why the S3 bucket names deliberately have no defaults.
TABLE_DEFINITIONS = [
    {
        "table_name": "trustops-evaluation-results",
        "key_schema": [
            {"AttributeName": "result_id", "KeyType": "HASH"},
        ],
        "attribute_definitions": [
            {"AttributeName": "result_id", "AttributeType": "S"},
            {"AttributeName": "result_type", "AttributeType": "S"},
            {"AttributeName": "model_id", "AttributeType": "S"},
            {"AttributeName": "workflow_id", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
        ],
        "global_secondary_indexes": [
            {
                "IndexName": "result-type-index",
                "KeySchema": [
                    {"AttributeName": "result_type", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "model-id-index",
                "KeySchema": [
                    {"AttributeName": "model_id", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "workflow-id-index",
                "KeySchema": [
                    {"AttributeName": "workflow_id", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    },
    {
        "table_name": "trustops-models",
        "key_schema": [
            {"AttributeName": "model_id", "KeyType": "HASH"},
        ],
        "attribute_definitions": [
            {"AttributeName": "model_id", "AttributeType": "S"},
            {"AttributeName": "provider", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"},
        ],
        "global_secondary_indexes": [
            {
                "IndexName": "provider-index",
                "KeySchema": [
                    {"AttributeName": "provider", "KeyType": "HASH"},
                    {"AttributeName": "model_id", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "status-index",
                "KeySchema": [
                    {"AttributeName": "status", "KeyType": "HASH"},
                    {"AttributeName": "model_id", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    },
    {
        "table_name": "trustops-dataset-metadata",
        "key_schema": [
            {"AttributeName": "dataset_id", "KeyType": "HASH"},
            {"AttributeName": "version", "KeyType": "RANGE"},
        ],
        "attribute_definitions": [
            {"AttributeName": "dataset_id", "AttributeType": "S"},
            {"AttributeName": "version", "AttributeType": "N"},
            {"AttributeName": "task_type", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
        ],
        "global_secondary_indexes": [
            {
                "IndexName": "task-type-index",
                "KeySchema": [
                    {"AttributeName": "task_type", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    },
    {
        "table_name": "trustops-evaluations",
        "key_schema": [
            {"AttributeName": "evaluation_id", "KeyType": "HASH"},
        ],
        "attribute_definitions": [
            {"AttributeName": "evaluation_id", "AttributeType": "S"},
            {"AttributeName": "model_id", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"},
        ],
        "global_secondary_indexes": [
            {
                "IndexName": "model-id-index",
                "KeySchema": [
                    {"AttributeName": "model_id", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "status-index",
                "KeySchema": [
                    {"AttributeName": "status", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    },
    {
        "table_name": "trustops-fine-tuning-jobs",
        "key_schema": [
            {"AttributeName": "job_id", "KeyType": "HASH"},
        ],
        "attribute_definitions": [
            {"AttributeName": "job_id", "AttributeType": "S"},
            {"AttributeName": "base_model_id", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
        ],
        "global_secondary_indexes": [
            {
                "IndexName": "base-model-index",
                "KeySchema": [
                    {"AttributeName": "base_model_id", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "status-index",
                "KeySchema": [
                    {"AttributeName": "status", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    },
    {
        "table_name": "trustops-workflows",
        "key_schema": [
            {"AttributeName": "workflow_id", "KeyType": "HASH"},
        ],
        "attribute_definitions": [
            {"AttributeName": "workflow_id", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"},
            {"AttributeName": "created_by", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
        ],
        "global_secondary_indexes": [
            {
                "IndexName": "status-index",
                "KeySchema": [
                    {"AttributeName": "status", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "created-by-index",
                "KeySchema": [
                    {"AttributeName": "created_by", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    },
    {
        "table_name": "trustops-access-logs",
        "key_schema": [
            {"AttributeName": "log_id", "KeyType": "HASH"},
        ],
        "attribute_definitions": [
            {"AttributeName": "log_id", "AttributeType": "S"},
            {"AttributeName": "result_id", "AttributeType": "S"},
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "S"},
        ],
        "global_secondary_indexes": [
            {
                "IndexName": "result-id-index",
                "KeySchema": [
                    {"AttributeName": "result_id", "KeyType": "HASH"},
                    {"AttributeName": "timestamp", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "user-id-index",
                "KeySchema": [
                    {"AttributeName": "user_id", "KeyType": "HASH"},
                    {"AttributeName": "timestamp", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    },
]


def create_table(dynamodb_client: Any, table_def: dict, prefix: str = "") -> str:
    """Create a single DynamoDB table.

    Args:
        dynamodb_client: boto3 DynamoDB client.
        table_def: Table definition dict.
        prefix: Optional prefix for table name.

    Returns:
        Created table name.
    """
    table_name = f"{prefix}{table_def['table_name']}" if prefix else table_def["table_name"]

    params: dict[str, Any] = {
        "TableName": table_name,
        "KeySchema": table_def["key_schema"],
        "AttributeDefinitions": table_def["attribute_definitions"],
        "BillingMode": "PAY_PER_REQUEST",
    }

    if table_def.get("global_secondary_indexes"):
        params["GlobalSecondaryIndexes"] = table_def["global_secondary_indexes"]

    dynamodb_client.create_table(**params)
    logger.info("Created table: %s", table_name)
    return table_name


def create_all_tables(
    dynamodb_client: Any,
    prefix: str = "",
) -> list[str]:
    """Create all required DynamoDB tables.

    Args:
        dynamodb_client: boto3 DynamoDB client.
        prefix: Optional prefix for table names.

    Returns:
        List of created table names.
    """
    created = []
    for table_def in TABLE_DEFINITIONS:
        table_name = f"{prefix}{table_def['table_name']}" if prefix else table_def["table_name"]
        try:
            # Check if table already exists
            dynamodb_client.describe_table(TableName=table_name)
            logger.info("Table %s already exists, skipping", table_name)
            created.append(table_name)
        except Exception:
            # Table doesn't exist, create it
            name = create_table(dynamodb_client, table_def, prefix)
            created.append(name)

    return created


def main():
    """Main entry point for table creation script."""
    parser = argparse.ArgumentParser(
        description="Create DynamoDB tables for TrustOps Enterprise Framework"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Table name prefix (e.g., 'dev-')",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="DynamoDB endpoint URL (for local development)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        import boto3

        kwargs: dict[str, Any] = {"region_name": args.region}
        if args.endpoint_url:
            kwargs["endpoint_url"] = args.endpoint_url

        client = boto3.client("dynamodb", **kwargs)
        created = create_all_tables(client, args.prefix)
        print(f"\nCreated {len(created)} tables:")
        for name in created:
            print(f"  - {name}")

    except ImportError:
        print("Error: boto3 is required. Install with: pip install boto3")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
