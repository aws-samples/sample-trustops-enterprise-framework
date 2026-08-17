"""Tests for DynamoDB table creation script."""

from unittest.mock import MagicMock, call

import pytest

from scripts.create_tables import (
    TABLE_DEFINITIONS,
    create_all_tables,
    create_table,
)


@pytest.fixture
def mock_client():
    return MagicMock()


class TestCreateTable:
    def test_creates_table(self, mock_client):
        table_def = TABLE_DEFINITIONS[0]  # trustops-evaluation-results
        name = create_table(mock_client, table_def)
        assert name == "trustops-evaluation-results"
        mock_client.create_table.assert_called_once()

    def test_creates_table_with_prefix(self, mock_client):
        table_def = TABLE_DEFINITIONS[0]
        name = create_table(mock_client, table_def, prefix="dev-")
        assert name == "dev-trustops-evaluation-results"

    def test_includes_gsis(self, mock_client):
        table_def = TABLE_DEFINITIONS[0]  # trustops-evaluation-results has 3 GSIs
        create_table(mock_client, table_def)
        call_args = mock_client.create_table.call_args[1]
        assert "GlobalSecondaryIndexes" in call_args
        assert len(call_args["GlobalSecondaryIndexes"]) == 3

    def test_uses_pay_per_request(self, mock_client):
        table_def = TABLE_DEFINITIONS[0]
        create_table(mock_client, table_def)
        call_args = mock_client.create_table.call_args[1]
        assert call_args["BillingMode"] == "PAY_PER_REQUEST"


class TestCreateAllTables:
    def test_creates_all_tables(self, mock_client):
        # All describe_table calls raise (tables don't exist)
        mock_client.describe_table.side_effect = Exception("not found")
        created = create_all_tables(mock_client)
        assert len(created) == len(TABLE_DEFINITIONS)
        assert mock_client.create_table.call_count == len(TABLE_DEFINITIONS)

    def test_skips_existing_tables(self, mock_client):
        # First table exists, rest don't
        def describe_side_effect(TableName):
            if TableName == "trustops-evaluation-results":
                return {"Table": {"TableName": TableName}}
            raise Exception("not found")

        mock_client.describe_table.side_effect = describe_side_effect
        created = create_all_tables(mock_client)
        assert len(created) == len(TABLE_DEFINITIONS)
        # One less create_table call since first table exists
        assert mock_client.create_table.call_count == len(TABLE_DEFINITIONS) - 1

    def test_with_prefix(self, mock_client):
        mock_client.describe_table.side_effect = Exception("not found")
        created = create_all_tables(mock_client, prefix="test-")
        assert all(name.startswith("test-") for name in created)

    def test_table_names(self, mock_client):
        mock_client.describe_table.side_effect = Exception("not found")
        created = create_all_tables(mock_client)
        expected_names = {
            "trustops-evaluation-results",
            "trustops-models",
            "trustops-dataset-metadata",
            "trustops-evaluations",
            "trustops-fine-tuning-jobs",
            "trustops-workflows",
            "trustops-access-logs",
        }
        assert set(created) == expected_names


class TestTableDefinitions:
    def test_all_tables_have_key_schema(self):
        for td in TABLE_DEFINITIONS:
            assert "key_schema" in td
            assert len(td["key_schema"]) >= 1

    def test_all_tables_have_attribute_definitions(self):
        for td in TABLE_DEFINITIONS:
            assert "attribute_definitions" in td
            assert len(td["attribute_definitions"]) >= 1

    def test_results_table_has_three_gsis(self):
        results_table = next(t for t in TABLE_DEFINITIONS if t["table_name"] == "trustops-evaluation-results")
        assert len(results_table["global_secondary_indexes"]) == 3

    def test_seven_tables_defined(self):
        assert len(TABLE_DEFINITIONS) == 7
