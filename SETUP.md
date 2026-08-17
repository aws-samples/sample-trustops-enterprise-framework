# TrustOps AWS Demo - Setup Guide

This guide walks you through setting up the TrustOps AWS Demo project from scratch.

## Prerequisites

### Required Software

- **Python 3.9+**: Check with `python3 --version`
- **AWS CLI**: Install from https://aws.amazon.com/cli/
- **Git**: For version control
- **Make**: For running common tasks (optional but recommended)

### AWS Requirements

- Active AWS account
- AWS credentials configured (IAM user or role with appropriate permissions)
- Access to AWS Bedrock (may require requesting access in some regions)
- Sufficient service limits for:
  - S3 buckets
  - DynamoDB tables
  - Lambda functions
  - CloudWatch log groups

### Required AWS Permissions

Your AWS user/role needs permissions for:
- S3 (create buckets, read/write objects, enable versioning)
- DynamoDB (create tables, read/write items)
- CloudWatch Logs (create log groups, write logs)
- AWS Bedrock (invoke models, create fine-tuning jobs)
- IAM (create roles for Lambda and Step Functions)
- Lambda (create and invoke functions)
- Step Functions (create and execute state machines)
- CloudFormation (create and update stacks)

## Step 1: Clone and Setup Python Environment

```bash
# Navigate to project directory
cd trustops-aws-demo

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or use the setup script
./scripts/setup_venv.sh
```

## Step 2: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or use Make
make install
```

## Step 3: Configure AWS Credentials

```bash
# Configure AWS CLI with your credentials
aws configure

# You'll be prompted for:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Default output format (json recommended)
```

Alternatively, set environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

## Step 4: Configure Environment Variables

```bash
# Copy the example environment file (it lives in infrastructure/)
cp infrastructure/env.example .env

# Edit .env with your preferred settings
# Most defaults are fine for initial setup
nano .env  # or use your preferred editor
```

Key variables to review:
- `AWS_REGION`: Your preferred AWS region
- `TRUSTOPS_*_BUCKET`: S3 bucket names (must be globally unique)
- `TRUST_SCORE_THRESHOLD`: Threshold for flagging low-confidence responses

## Step 5: Deploy AWS Infrastructure

```bash
# Deploy using the deployment script
cd infrastructure
./deploy.sh

# Or use Make from project root
make deploy
```

This will create:
- 4 S3 buckets (datasets, results, artifacts, access logs) with versioning enabled
- 7 DynamoDB tables (workflows, models, results metadata, and supporting tables)
- 4 CloudWatch log groups
- 8 Lambda functions (evaluation, fine-tuning, trust scoring, comparative
  evaluation, hallucination detection, metrics aggregation, recommendation)
- 3 Step Functions state machines (baseline evaluation, fine-tuning, comparative evaluation)
- IAM roles for Lambda, Step Functions, and Bedrock Knowledge Bases
- A KMS key, SNS topic, and SQS dead-letter queue
- 5 CloudWatch alarms

Each deployment generates a unique `DeploymentId` to avoid resource name collisions.

## Step 6: Update Environment Variables with Deployed Resources

After deployment completes, the script will display the created resource names. Update your `.env` file with these values:

```bash
# Example output from deployment:
# DatasetsBucket: trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
# ResultsBucket: trustops-results-<YOUR_ACCOUNT_ID>-<REGION>
# etc.

# Update .env accordingly
TRUSTOPS_DATASETS_BUCKET=trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_RESULTS_BUCKET=trustops-results-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_ARTIFACTS_BUCKET=trustops-artifacts-<YOUR_ACCOUNT_ID>-<REGION>
```

## Step 7: Verify Infrastructure Setup

```bash
# Run the verification script
python scripts/verify_aws_setup.py

# Or use Make
make verify
```

This script checks:
- AWS credentials are configured
- S3 buckets exist and are accessible
- DynamoDB tables exist
- CloudWatch log groups exist
- AWS Bedrock is accessible

All checks should pass before proceeding.

## Step 8: (Optional) Set Up a Bedrock Knowledge Base

Semantic similarity and evidence retrieval are backed by Amazon Bedrock
Knowledge Bases. Without one configured, `SemanticSimilarityAnalyzer` and
`KnowledgeBaseClient` run in mock mode, which is fine for local development
and for the full test suite.

To enable real retrieval, create a Knowledge Base with an S3 data source
pointing at the `knowledge-base/` prefix of your datasets bucket. The
Terraform stack in `infrastructure/terraform/` provisions this along with the
OpenSearch Serverless vector collection that Bedrock manages on its behalf:

```bash
cd infrastructure/terraform
terraform apply
```

Then record the resulting IDs in your `.env`:

```bash
KNOWLEDGE_BASE_ID=XXXXXXXXXX
KNOWLEDGE_BASE_DATA_SOURCE_ID=YYYYYYYYYY
```

Documents uploaded through `KnowledgeBaseClient.store_document()` land in S3
and are ingested on the next data source sync. Bedrock generates the
embeddings, so no embedding index needs to be managed directly.

## Step 9: Run Tests

```bash
# Run all tests
make test

# Or run specific test types
make test-unit        # Unit tests only
make test-property    # Property-based tests only

# Run with coverage
make coverage
```

## Step 10: Verify Installation

Create a simple test script to verify everything works:

```python
# test_installation.py
from config.aws_config import config
from src.utils.aws_utils import get_s3_client, get_dynamodb_client

print(f"Region: {config.region}")
print(f"Datasets bucket: {config.datasets_bucket}")

# Test S3 connection
s3 = get_s3_client()
print("✓ S3 client created")

# Test DynamoDB connection
dynamodb = get_dynamodb_client()
print("✓ DynamoDB client created")

print("\n✓ Installation verified!")
```

Run it:
```bash
python test_installation.py
```

## Troubleshooting

### Issue: AWS credentials not found

**Solution**: Run `aws configure` or set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Issue: Bedrock access denied

**Solution**: 
1. Ensure Bedrock is available in your region
2. Request access to Bedrock models in the AWS console
3. Wait for approval (can take a few hours)

### Issue: S3 bucket name already exists

**Solution**: S3 bucket names must be globally unique. Update the bucket names in your `.env` file:
```bash
TRUSTOPS_DATASETS_BUCKET=trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
```

### Issue: CloudFormation stack creation failed

**Solution**: 
1. Check the CloudFormation console for detailed error messages
2. Ensure you have sufficient permissions
3. Check service limits in your AWS account
4. Delete the failed stack and retry:
```bash
aws cloudformation delete-stack --stack-name trustops-infrastructure-dev
```

### Issue: Python version too old

**Solution**: Install Python 3.9 or higher:
```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install python3.9

# On macOS with Homebrew
brew install python@3.9
```

## Next Steps

After successful setup:

1. **Review the documentation**:
   - [Requirements](.kiro/specs/trustops-aws-demo/requirements.md)
   - [Design](.kiro/specs/trustops-aws-demo/design.md)
   - [Tasks](.kiro/specs/trustops-aws-demo/tasks.md)

2. **Start implementing features**: Follow the task list in `tasks.md`

3. **Run the demo**: Once implementation is complete, run the demo workflow

4. **Explore the dashboard**: Set up QuickSight for visualization

## Cleanup

To remove all AWS resources:

```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name trustops-infrastructure-dev

# Empty and delete S3 buckets (if needed)
aws s3 rm s3://your-bucket-name --recursive
aws s3 rb s3://your-bucket-name
```

If you provisioned the Bedrock Knowledge Base with Terraform, tear it down
from `infrastructure/terraform/` with `terraform destroy` so the associated
OpenSearch Serverless collection is removed too.

## Support

For issues or questions:
- Check the troubleshooting section above
- Review AWS service documentation
- Check CloudWatch logs for detailed error messages
