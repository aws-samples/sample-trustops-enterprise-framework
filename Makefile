.PHONY: help setup install test test-unit test-property test-all test-dashboard test-cli test-integration coverage clean deploy verify cli-install cli-dev run-dashboard create-tables lint format

help:
	@echo "TrustOps Enterprise Framework - Available Commands"
	@echo "=================================================="
	@echo "setup             - Create virtual environment and install dependencies"
	@echo "install           - Install dependencies only"
	@echo "cli-install       - Install TrustOps CLI"
	@echo "cli-dev           - Install TrustOps CLI in development mode"
	@echo "test              - Run all tests"
	@echo "test-unit         - Run unit tests only"
	@echo "test-property     - Run property-based tests only"
	@echo "test-dashboard    - Run dashboard tests"
	@echo "test-cli          - Run CLI tests"
	@echo "test-integration  - Run integration tests"
	@echo "coverage          - Run tests with coverage report"
	@echo "run-dashboard     - Run Streamlit dashboard"
	@echo "create-tables     - Create DynamoDB tables"
	@echo "verify            - Verify AWS infrastructure setup"
	@echo "deploy            - Deploy AWS infrastructure"
	@echo "clean             - Remove generated files and caches"
	@echo "lint              - Run code linting"
	@echo "format            - Format code with black"

setup:
	@echo "Setting up virtual environment..."
	./scripts/setup_venv.sh

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

cli-install:
	@echo "Installing TrustOps CLI..."
	pip install .
	@echo "CLI installed! Run 'trustops --help' to get started."

cli-dev:
	@echo "Installing TrustOps CLI in development mode..."
	pip install -e .
	@echo "CLI installed in development mode! Run 'trustops --help' to get started."

test:
	@echo "Running all tests..."
	pytest tests/

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -m unit

test-property:
	@echo "Running property-based tests..."
	pytest tests/property/ -m property

test-all:
	@echo "Running all tests with verbose output..."
	pytest tests/ -v

test-dashboard:
	@echo "Running dashboard tests..."
	pytest tests/dashboard/ -v

test-cli:
	@echo "Running CLI tests..."
	pytest tests/unit/test_cli_* -v

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

coverage:
	@echo "Running tests with coverage..."
	pytest --cov=src --cov-report=html --cov-report=term tests/

verify:
	@echo "Verifying AWS infrastructure..."
	python scripts/verify_aws_setup.py

deploy:
	@echo "Deploying AWS infrastructure..."
	cd infrastructure && ./deploy.sh

run-dashboard:
	@echo "Starting Streamlit dashboard..."
	python3 -m streamlit run dashboard/app.py --server.port 8501

create-tables:
	@echo "Creating DynamoDB tables..."
	python scripts/create_tables.py

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf htmlcov/
	rm -f .coverage
	@echo "Cleanup complete"

lint:
	@echo "Running linting..."
	@command -v pylint >/dev/null 2>&1 || { echo "pylint not installed. Install with: pip install pylint"; exit 1; }
	pylint src/

format:
	@echo "Formatting code..."
	@command -v black >/dev/null 2>&1 || { echo "black not installed. Install with: pip install black"; exit 1; }
	black src/ tests/
