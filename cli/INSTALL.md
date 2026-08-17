# TrustOps CLI Installation Guide

## Quick Install

### Option 1: Development Mode (Recommended for Development)

Install the CLI in development mode so changes are immediately reflected:

```bash
# From the project root
make cli-dev

# Or manually
pip install -e .
```

### Option 2: Standard Installation

Install the CLI as a regular package:

```bash
# From the project root
make cli-install

# Or manually
pip install .
```

### Option 3: Direct Execution (No Installation)

You can run the CLI directly without installation:

```bash
# From the project root
./trustops --help

# Or
python -m cli.main --help
```

## Verify Installation

After installation, verify the CLI is working:

```bash
trustops --version
trustops --help
```

You should see the TrustOps CLI help message with available commands.

## First-Time Setup

### 1. Configure AWS Credentials

Ensure your AWS credentials are configured:

```bash
aws configure
```

Or set environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

### 2. Configure TrustOps

Run the configuration wizard:

```bash
trustops configure
```

You'll be prompted for:
- AWS Region
- S3 bucket for datasets
- S3 bucket for results
- S3 bucket for artifacts
- DynamoDB table name for workflows

Configuration is saved to `~/.trustops/config.json`.

### 3. Verify AWS Setup

Verify your AWS infrastructure is set up correctly:

```bash
python scripts/verify_aws_setup.py
```

## Uninstall

To uninstall the TrustOps CLI:

```bash
pip uninstall trustops
```

## Troubleshooting

### Command Not Found

If you get "command not found" after installation:

1. Check if the installation directory is in your PATH:
   ```bash
   echo $PATH
   ```

2. Find where pip installed the CLI:
   ```bash
   pip show trustops
   ```

3. Add the installation directory to your PATH or use the full path to the executable.

### Import Errors

If you get import errors when running commands:

1. Ensure you're in the correct virtual environment:
   ```bash
   which python
   ```

2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Reinstall the CLI:
   ```bash
   pip install -e .
   ```

### AWS Credentials Issues

If you get AWS authentication errors:

1. Verify credentials are configured:
   ```bash
   aws sts get-caller-identity
   ```

2. Check the AWS region is set:
   ```bash
   echo $AWS_REGION
   ```

3. Verify IAM permissions for required services (Bedrock, S3, DynamoDB, etc.)

## Development

### Running Tests

Run CLI-related tests:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/unit/test_cli.py
```

### Code Style

Format CLI code:

```bash
make format
```

Lint CLI code:

```bash
make lint
```

## Support

For issues and questions:
- Check the [CLI README](README.md) for usage documentation
- Review the main [project README](../README.md)
- Check CloudWatch logs for detailed error information
