#!/bin/bash

# TrustOps Infrastructure Deployment Script
# This script deploys the TrustOps infrastructure using either CloudFormation or Terraform

set -e

# Configuration
PROJECT_NAME="${PROJECT_NAME:-trustops}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
DEPLOYMENT_METHOD="${DEPLOYMENT_METHOD:-cloudformation}"  # cloudformation or terraform
DEPLOYMENT_ID="${DEPLOYMENT_ID:-$(date +%s | tail -c 7)}"
ALERT_EMAIL="${ALERT_EMAIL:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials are not configured. Please run 'aws configure'."
        exit 1
    fi
    
    # Check deployment method specific tools
    if [ "$DEPLOYMENT_METHOD" = "terraform" ]; then
        if ! command -v terraform &> /dev/null; then
            log_error "Terraform is not installed. Please install it first."
            exit 1
        fi
    fi
    
    log_info "Prerequisites check passed."
}

# Package Lambda functions
package_lambda_functions() {
    log_info "Packaging Lambda functions..."

    # Create temporary directory for packaging
    TEMP_DIR=$(mktemp -d)
    mkdir -p lambda

    # Package orchestrator handlers (from lambda_handlers/)
    for handler in evaluation_handler fine_tuning_handler trust_scoring_handler comparative_evaluation_handler; do
        log_info "Packaging ${handler}..."

        PACKAGE_DIR="${TEMP_DIR}/${handler}"
        mkdir -p "${PACKAGE_DIR}"

        cp "../lambda_handlers/${handler}.py" "${PACKAGE_DIR}/"
        cp -r ../src "${PACKAGE_DIR}/"

        if [ -f "requirements-lambda.txt" ]; then
            pip install -r requirements-lambda.txt -t "${PACKAGE_DIR}/" --quiet --no-cache-dir
        fi

        cd "${PACKAGE_DIR}"
        zip -r "${handler}.zip" . -q
        cd - > /dev/null

        mv "${PACKAGE_DIR}/${handler}.zip" "lambda/"
        log_info "Packaged ${handler}.zip"
    done

    # Package granular Step Functions handlers (from src/lambda_handlers/)
    for handler in evaluate_single_example aggregate_metrics generate_recommendation detect_hallucinations; do
        log_info "Packaging ${handler}..."

        PACKAGE_DIR="${TEMP_DIR}/${handler}"
        mkdir -p "${PACKAGE_DIR}"

        cp "../src/lambda_handlers/${handler}.py" "${PACKAGE_DIR}/"
        cp -r ../src "${PACKAGE_DIR}/"

        if [ -f "requirements-lambda.txt" ]; then
            pip install -r requirements-lambda.txt -t "${PACKAGE_DIR}/" --quiet --no-cache-dir
        fi

        cd "${PACKAGE_DIR}"
        zip -r "${handler}.zip" . -q
        cd - > /dev/null

        mv "${PACKAGE_DIR}/${handler}.zip" "lambda/"
        log_info "Packaged ${handler}.zip"
    done

    # Clean up
    rm -rf "${TEMP_DIR}"

    log_info "Lambda functions packaged successfully."
}

# Upload Lambda packages to S3
upload_lambda_packages() {
    log_info "Uploading Lambda packages to S3..."
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ARTIFACTS_BUCKET="${PROJECT_NAME}-artifacts-${ENVIRONMENT}-${DEPLOYMENT_ID}-${ACCOUNT_ID}"
    
    # Wait for the bucket to exist (created by CloudFormation)
    # If it already exists from a previous deploy, that's fine too
    log_info "Waiting for artifacts bucket: ${ARTIFACTS_BUCKET}"
    local retries=0
    while ! aws s3 ls "s3://${ARTIFACTS_BUCKET}" 2>/dev/null; do
        retries=$((retries + 1))
        if [ $retries -ge 30 ]; then
            log_error "Artifacts bucket ${ARTIFACTS_BUCKET} not found after waiting. Ensure CloudFormation stack deployed successfully."
            exit 1
        fi
        sleep 10
    done
    
    # Upload Lambda packages
    for handler_zip in lambda/*.zip; do
        log_info "Uploading $(basename ${handler_zip})..."
        aws s3 cp "${handler_zip}" "s3://${ARTIFACTS_BUCKET}/lambda/" --region "${AWS_REGION}"
    done
    
    log_info "Lambda packages uploaded successfully."
}

# Upload Step Functions definitions
upload_step_functions() {
    log_info "Uploading Step Functions definitions..."
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ARTIFACTS_BUCKET="${PROJECT_NAME}-artifacts-${ENVIRONMENT}-${DEPLOYMENT_ID}-${ACCOUNT_ID}"
    
    # Upload workflow definitions
    for workflow in ../step_functions/*.json; do
        if [ -f "${workflow}" ]; then
            log_info "Uploading $(basename ${workflow})..."
            aws s3 cp "${workflow}" "s3://${ARTIFACTS_BUCKET}/step_functions/" --region "${AWS_REGION}"
        fi
    done
    
    log_info "Step Functions definitions uploaded successfully."
}

# Update Lambda functions with real code from S3
update_lambda_code() {
    log_info "Updating Lambda functions with packaged code..."

    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ARTIFACTS_BUCKET="${PROJECT_NAME}-artifacts-${ENVIRONMENT}-${DEPLOYMENT_ID}-${ACCOUNT_ID}"

    declare -A HANDLER_MAP
    HANDLER_MAP[evaluation_handler]="${PROJECT_NAME}-evaluation-orchestrator-${ENVIRONMENT}-${DEPLOYMENT_ID}"
    HANDLER_MAP[fine_tuning_handler]="${PROJECT_NAME}-finetuning-orchestrator-${ENVIRONMENT}-${DEPLOYMENT_ID}"
    HANDLER_MAP[trust_scoring_handler]="${PROJECT_NAME}-trust-scoring-${ENVIRONMENT}-${DEPLOYMENT_ID}"
    HANDLER_MAP[comparative_evaluation_handler]="${PROJECT_NAME}-comparative-evaluation-${ENVIRONMENT}-${DEPLOYMENT_ID}"
    HANDLER_MAP[evaluate_single_example]="${PROJECT_NAME}-evaluate-single-example-${ENVIRONMENT}-${DEPLOYMENT_ID}"
    HANDLER_MAP[aggregate_metrics]="${PROJECT_NAME}-aggregate-metrics-${ENVIRONMENT}-${DEPLOYMENT_ID}"
    HANDLER_MAP[generate_recommendation]="${PROJECT_NAME}-generate-recommendation-${ENVIRONMENT}-${DEPLOYMENT_ID}"
    HANDLER_MAP[detect_hallucinations]="${PROJECT_NAME}-detect-hallucinations-${ENVIRONMENT}-${DEPLOYMENT_ID}"

    for handler in "${!HANDLER_MAP[@]}"; do
        FUNCTION_NAME="${HANDLER_MAP[$handler]}"
        log_info "Updating ${FUNCTION_NAME} with ${handler}.zip..."
        aws lambda update-function-code \
            --function-name "${FUNCTION_NAME}" \
            --s3-bucket "${ARTIFACTS_BUCKET}" \
            --s3-key "lambda/${handler}.zip" \
            --region "${AWS_REGION}" > /dev/null

        # Wait for update to complete before changing config
        aws lambda wait function-updated \
            --function-name "${FUNCTION_NAME}" \
            --region "${AWS_REGION}"

        aws lambda update-function-configuration \
            --function-name "${FUNCTION_NAME}" \
            --handler "${handler}.handler" \
            --region "${AWS_REGION}" > /dev/null
    done

    log_info "Lambda functions updated successfully."
}

# Deploy using CloudFormation
deploy_cloudformation() {
    log_info "Deploying infrastructure using CloudFormation..."
    
    STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}-${DEPLOYMENT_ID}"
    
    # Check if stack exists
    if aws cloudformation describe-stacks --stack-name "${STACK_NAME}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        log_info "Stack exists. Updating..."
        OPERATION="update-stack"
    else
        log_info "Stack does not exist. Creating..."
        OPERATION="create-stack"
    fi
    
    # Deploy stack
    aws cloudformation ${OPERATION} \
        --stack-name "${STACK_NAME}" \
        --template-body file://cloudformation-template.yaml \
        --parameters \
            ParameterKey=ProjectName,ParameterValue="${PROJECT_NAME}" \
            ParameterKey=Environment,ParameterValue="${ENVIRONMENT}" \
            ParameterKey=DeploymentId,ParameterValue="${DEPLOYMENT_ID}" \
            ParameterKey=AlertEmail,ParameterValue="${ALERT_EMAIL}" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "${AWS_REGION}"
    
    # Wait for stack operation to complete
    log_info "Waiting for stack operation to complete..."
    if [ "${OPERATION}" = "create-stack" ]; then
        aws cloudformation wait stack-create-complete \
            --stack-name "${STACK_NAME}" \
            --region "${AWS_REGION}"
    else
        aws cloudformation wait stack-update-complete \
            --stack-name "${STACK_NAME}" \
            --region "${AWS_REGION}" || true
    fi
    
    log_info "CloudFormation deployment completed."
}

# Deploy using Terraform
deploy_terraform() {
    log_info "Deploying infrastructure using Terraform..."
    
    cd terraform
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init
    
    # Create workspace if it doesn't exist
    terraform workspace select "${ENVIRONMENT}" 2>/dev/null || terraform workspace new "${ENVIRONMENT}"
    
    # Plan deployment
    log_info "Planning Terraform deployment..."
    terraform plan \
        -var="project_name=${PROJECT_NAME}" \
        -var="environment=${ENVIRONMENT}" \
        -var="aws_region=${AWS_REGION}" \
        -out=tfplan
    
    # Apply deployment
    log_info "Applying Terraform deployment..."
    terraform apply tfplan
    
    # Clean up plan file
    rm -f tfplan
    
    cd ..
    
    log_info "Terraform deployment completed."
}

# Display outputs
display_outputs() {
    log_info "Deployment outputs:"
    
    if [ "$DEPLOYMENT_METHOD" = "cloudformation" ]; then
        STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}-${DEPLOYMENT_ID}"
        aws cloudformation describe-stacks \
            --stack-name "${STACK_NAME}" \
            --region "${AWS_REGION}" \
            --query 'Stacks[0].Outputs' \
            --output table
    else
        cd terraform
        terraform output
        cd ..
    fi
}

# Main deployment flow
main() {
    log_info "Starting TrustOps infrastructure deployment..."
    log_info "Project: ${PROJECT_NAME}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Region: ${AWS_REGION}"
    log_info "Method: ${DEPLOYMENT_METHOD}"
    log_info "Deployment ID: ${DEPLOYMENT_ID}"
    
    # Check prerequisites
    check_prerequisites
    
    # Package Lambda functions (but don't upload yet — need buckets from CF)
    package_lambda_functions
    
    # Deploy infrastructure (creates S3 buckets, DynamoDB tables, etc.)
    if [ "$DEPLOYMENT_METHOD" = "terraform" ]; then
        deploy_terraform
    else
        deploy_cloudformation
    fi
    
    # Now upload Lambda packages and Step Functions to the CF-created buckets
    upload_lambda_packages
    
    # Update Lambda functions with real code from S3
    update_lambda_code
    
    upload_step_functions
    
    # Display outputs
    display_outputs
    
    log_info "Deployment completed successfully!"
}

# Run main function
main
