#!/usr/bin/env python3
"""Verify AWS infrastructure setup for TrustOps.

Discovers deployed resource names from CloudFormation stack outputs
so it works regardless of DeploymentId or environment.
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def get_region():
    """Get AWS region from env or default."""
    return os.getenv("AWS_REGION", "us-east-1")


def find_trustops_stack(cf_client):
    """Find the active TrustOps CloudFormation stack."""
    try:
        paginator = cf_client.get_paginator("describe_stacks")
        for page in paginator.paginate():
            for stack in page["Stacks"]:
                name = stack["StackName"]
                status = stack["StackStatus"]
                if name.startswith("trustops-") and "COMPLETE" in status:
                    return stack
    except ClientError:
        pass
    return None


def get_stack_outputs(stack):
    """Extract outputs from a CloudFormation stack as a dict."""
    outputs = {}
    for output in stack.get("Outputs", []):
        outputs[output["OutputKey"]] = output["OutputValue"]
    return outputs


def check_aws_credentials(region):
    """Check if AWS credentials are configured."""
    print("Checking AWS credentials...")
    try:
        sts = boto3.client("sts", region_name=region)
        identity = sts.get_caller_identity()
        print(f"  ✓ AWS credentials configured")
        print(f"    Account: {identity['Account']}")
        print(f"    User/Role: {identity['Arn']}")
        return True
    except NoCredentialsError:
        print("  ✗ AWS credentials not found")
        print("    Run 'aws configure' to set up credentials")
        return False
    except Exception as e:
        print(f"  ✗ Error checking credentials: {e}")
        return False


def check_s3_bucket(region, bucket_name):
    """Check if S3 bucket exists and is accessible."""
    try:
        s3 = boto3.client("s3", region_name=region)
        s3.head_bucket(Bucket=bucket_name)
        versioning = s3.get_bucket_versioning(Bucket=bucket_name)
        versioning_status = versioning.get("Status", "Disabled")
        print(f"  ✓ S3 bucket '{bucket_name}'  (versioning: {versioning_status})")
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            print(f"  ✗ S3 bucket '{bucket_name}' does not exist")
        elif code == "403":
            print(f"  ✗ Access denied to S3 bucket '{bucket_name}'")
        else:
            print(f"  ✗ Error accessing S3 bucket '{bucket_name}': {e}")
        return False


def check_dynamodb_table(region, table_name):
    """Check if DynamoDB table exists and is accessible."""
    try:
        dynamodb = boto3.client("dynamodb", region_name=region)
        resp = dynamodb.describe_table(TableName=table_name)
        status = resp["Table"]["TableStatus"]
        print(f"  ✓ DynamoDB table '{table_name}'  (status: {status})")
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ResourceNotFoundException":
            print(f"  ✗ DynamoDB table '{table_name}' does not exist")
        else:
            print(f"  ✗ Error accessing DynamoDB table '{table_name}': {e}")
        return False


def check_lambda_function(region, function_name):
    """Check if Lambda function exists."""
    try:
        lam = boto3.client("lambda", region_name=region)
        resp = lam.get_function(FunctionName=function_name)
        runtime = resp["Configuration"]["Runtime"]
        print(f"  ✓ Lambda '{function_name}'  (runtime: {runtime})")
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ResourceNotFoundException":
            print(f"  ✗ Lambda '{function_name}' does not exist")
        else:
            print(f"  ✗ Error accessing Lambda '{function_name}': {e}")
        return False


def check_knowledge_base(region, knowledge_base_id):
    """Check if the Bedrock Knowledge Base is reachable."""
    if not knowledge_base_id:
        print("  ⊘ No Knowledge Base configured (retrieval runs in mock mode)")
        return True  # Optional feature - don't fail the check
    try:
        client = boto3.client("bedrock-agent", region_name=region)
        response = client.get_knowledge_base(knowledgeBaseId=knowledge_base_id)
        status = response["knowledgeBase"]["status"]
        print(f"  ✓ Knowledge Base '{knowledge_base_id}' status: {status}")
        return status in ("ACTIVE", "UPDATING")
    except ClientError as e:
        print(f"  ✗ Error accessing Knowledge Base '{knowledge_base_id}': {e}")
        return False


def check_bedrock_access(region):
    """Check if Bedrock service is accessible."""
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        bedrock.list_foundation_models()
        print(f"  ✓ AWS Bedrock is accessible")
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "AccessDeniedException":
            print(f"  ✗ Access denied to AWS Bedrock")
            print(f"    Request model access in the AWS Console → Bedrock → Model access")
        else:
            print(f"  ✗ Error accessing AWS Bedrock: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error accessing AWS Bedrock: {e}")
        return False


def main():
    region = get_region()

    print("=" * 60)
    print("TrustOps AWS Infrastructure Verification")
    print("=" * 60)
    print(f"Region: {region}\n")

    checks = []

    # 1. Credentials
    checks.append(check_aws_credentials(region))
    print()

    # 2. Find CloudFormation stack
    print("Looking for TrustOps CloudFormation stack...")
    cf = boto3.client("cloudformation", region_name=region)
    stack = find_trustops_stack(cf)

    if stack:
        stack_name = stack["StackName"]
        outputs = get_stack_outputs(stack)
        print(f"  ✓ Found stack: {stack_name}  (status: {stack['StackStatus']})")
        print()

        # 3. S3 buckets
        print("Checking S3 buckets...")
        for key in ["DatasetsBucketName", "ResultsBucketName", "ArtifactsBucketName"]:
            bucket = outputs.get(key)
            if bucket:
                checks.append(check_s3_bucket(region, bucket))
            else:
                print(f"  ⊘ {key} not found in stack outputs")
                checks.append(False)
        print()

        # 4. DynamoDB tables
        print("Checking DynamoDB tables...")
        for key in ["WorkflowsTableName", "ModelsTableName"]:
            table = outputs.get(key)
            if table:
                checks.append(check_dynamodb_table(region, table))
            else:
                print(f"  ⊘ {key} not found in stack outputs")
                checks.append(False)
        print()

        # 5. Lambda functions
        print("Checking Lambda functions...")
        for key in [
            "EvaluationOrchestratorFunctionArn",
            "FineTuningOrchestratorFunctionArn",
            "TrustScoringFunctionArn",
            "ComparativeEvaluationFunctionArn",
        ]:
            arn = outputs.get(key)
            if arn:
                # Extract function name from ARN
                func_name = arn.split(":")[-1]
                checks.append(check_lambda_function(region, func_name))
            else:
                print(f"  ⊘ {key} not found in stack outputs")
                checks.append(False)
        print()

        # 6. Bedrock Knowledge Base (optional)
        print("Checking Bedrock Knowledge Base...")
        knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID") or outputs.get(
            "KnowledgeBaseId"
        )
        checks.append(check_knowledge_base(region, knowledge_base_id))
        print()

    else:
        print("  ✗ No active TrustOps CloudFormation stack found")
        print("    Deploy with: cd infrastructure && ./deploy.sh")
        checks.append(False)
        print()

    # 7. Bedrock
    print("Checking AWS Bedrock access...")
    checks.append(check_bedrock_access(region))
    print()

    # Summary
    passed = sum(checks)
    total = len(checks)
    print("=" * 60)
    print(f"Verification: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("✓ All checks passed. Infrastructure is ready.")
        return 0
    else:
        print("✗ Some checks failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
