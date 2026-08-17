#!/bin/bash

# TrustOps Infrastructure Cleanup Script
# This script removes all TrustOps infrastructure resources
# WARNING: This will delete all data. Use with caution!

set -e

# Configuration
PROJECT_NAME="${PROJECT_NAME:-trustops}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
DEPLOYMENT_METHOD="${DEPLOYMENT_METHOD:-cloudformation}"

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

# Confirmation prompt
confirm_cleanup() {
    log_warn "WARNING: This will delete ALL TrustOps infrastructure resources!"
    log_warn "Project: ${PROJECT_NAME}"
    log_warn "Environment: ${ENVIRONMENT}"
    log_warn "Region: ${AWS_REGION}"
    log_warn "Method: ${DEPLOYMENT_METHOD}"
    echo ""
    log_warn "This action cannot be undone. All data will be permanently deleted."
    echo ""
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " CONFIRM
    
    if [ "$CONFIRM" != "yes" ]; then
        log_info "Cleanup cancelled."
        exit 0
    fi
}

# Empty S3 buckets before deletion
empty_s3_buckets() {
    log_info "Emptying S3 buckets..."
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    
    BUCKETS=(
        "${PROJECT_NAME}-datasets-${ENVIRONMENT}-${ACCOUNT_ID}"
        "${PROJECT_NAME}-results-${ENVIRONMENT}-${ACCOUNT_ID}"
        "${PROJECT_NAME}-artifacts-${ENVIRONMENT}-${ACCOUNT_ID}"
    )
    
    for BUCKET in "${BUCKETS[@]}"; do
        if aws s3 ls "s3://${BUCKET}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
            log_info "Emptying bucket: ${BUCKET}"
            
            # Delete all object versions
            aws s3api list-object-versions \
                --bucket "${BUCKET}" \
                --query 'Versions[].{Key:Key,VersionId:VersionId}' \
                --output json | \
            jq -r '.[] | "--key \"\(.Key)\" --version-id \"\(.VersionId)\""' | \
            xargs -I {} aws s3api delete-object --bucket "${BUCKET}" {}
            
            # Delete all delete markers
            aws s3api list-object-versions \
                --bucket "${BUCKET}" \
                --query 'DeleteMarkers[].{Key:Key,VersionId:VersionId}' \
                --output json | \
            jq -r '.[] | "--key \"\(.Key)\" --version-id \"\(.VersionId)\""' | \
            xargs -I {} aws s3api delete-object --bucket "${BUCKET}" {}
            
            log_info "Bucket emptied: ${BUCKET}"
        else
            log_warn "Bucket not found: ${BUCKET}"
        fi
    done
}

# Cleanup using CloudFormation
cleanup_cloudformation() {
    log_info "Cleaning up infrastructure using CloudFormation..."
    
    STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}"
    
    # Check if stack exists
    if aws cloudformation describe-stacks --stack-name "${STACK_NAME}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        log_info "Deleting CloudFormation stack: ${STACK_NAME}"
        
        aws cloudformation delete-stack \
            --stack-name "${STACK_NAME}" \
            --region "${AWS_REGION}"
        
        log_info "Waiting for stack deletion to complete..."
        aws cloudformation wait stack-delete-complete \
            --stack-name "${STACK_NAME}" \
            --region "${AWS_REGION}"
        
        log_info "CloudFormation stack deleted successfully."
    else
        log_warn "CloudFormation stack not found: ${STACK_NAME}"
    fi
}

# Cleanup using Terraform
cleanup_terraform() {
    log_info "Cleaning up infrastructure using Terraform..."
    
    cd terraform
    
    # Select workspace
    terraform workspace select "${ENVIRONMENT}" 2>/dev/null || {
        log_warn "Terraform workspace not found: ${ENVIRONMENT}"
        cd ..
        return
    }
    
    # Destroy infrastructure
    log_info "Destroying Terraform resources..."
    terraform destroy \
        -var="project_name=${PROJECT_NAME}" \
        -var="environment=${ENVIRONMENT}" \
        -var="aws_region=${AWS_REGION}" \
        -auto-approve
    
    # Delete workspace
    terraform workspace select default
    terraform workspace delete "${ENVIRONMENT}"
    
    cd ..
    
    log_info "Terraform resources destroyed successfully."
}

# Clean up local files
cleanup_local_files() {
    log_info "Cleaning up local files..."
    
    # Remove Lambda packages
    if [ -d "lambda" ]; then
        rm -rf lambda
        log_info "Removed Lambda packages"
    fi
    
    # Remove Terraform state files (if using local state)
    if [ -d "terraform/.terraform" ]; then
        cd terraform
        rm -rf .terraform .terraform.lock.hcl terraform.tfstate terraform.tfstate.backup *.tfplan
        cd ..
        log_info "Removed Terraform local state files"
    fi
}

# Main cleanup flow
main() {
    log_info "Starting TrustOps infrastructure cleanup..."
    
    # Confirm cleanup
    confirm_cleanup
    
    # Empty S3 buckets first
    empty_s3_buckets
    
    # Cleanup infrastructure
    if [ "$DEPLOYMENT_METHOD" = "terraform" ]; then
        cleanup_terraform
    else
        cleanup_cloudformation
    fi
    
    # Clean up local files
    cleanup_local_files
    
    log_info "Cleanup completed successfully!"
    log_info "All TrustOps infrastructure resources have been removed."
}

# Run main function
main
