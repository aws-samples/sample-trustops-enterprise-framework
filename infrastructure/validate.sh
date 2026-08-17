#!/bin/bash

# TrustOps Infrastructure Validation Script
# This script validates that all infrastructure components are deployed correctly

set -e

# Configuration
PROJECT_NAME="${PROJECT_NAME:-trustops}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

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

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

# Get account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Validation functions
validate_s3_buckets() {
    log_info "Validating S3 buckets..."
    
    # Check datasets bucket
    DATASETS_BUCKET="${PROJECT_NAME}-datasets-${ENVIRONMENT}-${ACCOUNT_ID}"
    if aws s3 ls "s3://${DATASETS_BUCKET}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Datasets bucket exists: ${DATASETS_BUCKET}"
        
        # Check versioning
        VERSIONING=$(aws s3api get-bucket-versioning --bucket "${DATASETS_BUCKET}" --query Status --output text)
        if [ "$VERSIONING" = "Enabled" ]; then
            check_pass "Datasets bucket versioning enabled"
        else
            check_fail "Datasets bucket versioning not enabled"
        fi
    else
        check_fail "Datasets bucket not found: ${DATASETS_BUCKET}"
    fi
    
    # Check results bucket
    RESULTS_BUCKET="${PROJECT_NAME}-results-${ENVIRONMENT}-${ACCOUNT_ID}"
    if aws s3 ls "s3://${RESULTS_BUCKET}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Results bucket exists: ${RESULTS_BUCKET}"
    else
        check_fail "Results bucket not found: ${RESULTS_BUCKET}"
    fi
    
    # Check artifacts bucket
    ARTIFACTS_BUCKET="${PROJECT_NAME}-artifacts-${ENVIRONMENT}-${ACCOUNT_ID}"
    if aws s3 ls "s3://${ARTIFACTS_BUCKET}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Artifacts bucket exists: ${ARTIFACTS_BUCKET}"
    else
        check_fail "Artifacts bucket not found: ${ARTIFACTS_BUCKET}"
    fi
}

validate_dynamodb_tables() {
    log_info "Validating DynamoDB tables..."
    
    # Check workflows table
    WORKFLOWS_TABLE="${PROJECT_NAME}-workflows-${ENVIRONMENT}"
    if aws dynamodb describe-table --table-name "${WORKFLOWS_TABLE}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Workflows table exists: ${WORKFLOWS_TABLE}"
        
        # Check indexes
        INDEXES=$(aws dynamodb describe-table --table-name "${WORKFLOWS_TABLE}" --region "${AWS_REGION}" --query 'Table.GlobalSecondaryIndexes[].IndexName' --output text)
        if echo "$INDEXES" | grep -q "workflow-type-index"; then
            check_pass "Workflows table has workflow-type-index"
        else
            check_fail "Workflows table missing workflow-type-index"
        fi
        if echo "$INDEXES" | grep -q "status-index"; then
            check_pass "Workflows table has status-index"
        else
            check_fail "Workflows table missing status-index"
        fi
    else
        check_fail "Workflows table not found: ${WORKFLOWS_TABLE}"
    fi
    
    # Check models table
    MODELS_TABLE="${PROJECT_NAME}-models-${ENVIRONMENT}"
    if aws dynamodb describe-table --table-name "${MODELS_TABLE}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Models table exists: ${MODELS_TABLE}"
    else
        check_fail "Models table not found: ${MODELS_TABLE}"
    fi
}

validate_lambda_functions() {
    log_info "Validating Lambda functions..."
    
    # Check evaluation orchestrator
    EVAL_FUNCTION="${PROJECT_NAME}-evaluation-orchestrator-${ENVIRONMENT}"
    if aws lambda get-function --function-name "${EVAL_FUNCTION}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Evaluation orchestrator function exists: ${EVAL_FUNCTION}"
    else
        check_fail "Evaluation orchestrator function not found: ${EVAL_FUNCTION}"
    fi
    
    # Check fine-tuning orchestrator
    FINETUNE_FUNCTION="${PROJECT_NAME}-finetuning-orchestrator-${ENVIRONMENT}"
    if aws lambda get-function --function-name "${FINETUNE_FUNCTION}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Fine-tuning orchestrator function exists: ${FINETUNE_FUNCTION}"
    else
        check_fail "Fine-tuning orchestrator function not found: ${FINETUNE_FUNCTION}"
    fi
    
    # Check trust scoring
    TRUST_FUNCTION="${PROJECT_NAME}-trust-scoring-${ENVIRONMENT}"
    if aws lambda get-function --function-name "${TRUST_FUNCTION}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Trust scoring function exists: ${TRUST_FUNCTION}"
    else
        check_fail "Trust scoring function not found: ${TRUST_FUNCTION}"
    fi
    
    # Check comparative evaluation
    COMPARE_FUNCTION="${PROJECT_NAME}-comparative-evaluation-${ENVIRONMENT}"
    if aws lambda get-function --function-name "${COMPARE_FUNCTION}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Comparative evaluation function exists: ${COMPARE_FUNCTION}"
    else
        check_fail "Comparative evaluation function not found: ${COMPARE_FUNCTION}"
    fi
}

validate_step_functions() {
    log_info "Validating Step Functions state machines..."
    
    # Check baseline evaluation
    BASELINE_SM="${PROJECT_NAME}-baseline-evaluation-${ENVIRONMENT}"
    BASELINE_ARN="arn:aws:states:${AWS_REGION}:${ACCOUNT_ID}:stateMachine:${BASELINE_SM}"
    if aws stepfunctions describe-state-machine --state-machine-arn "${BASELINE_ARN}" 2>&1 > /dev/null; then
        check_pass "Baseline evaluation state machine exists: ${BASELINE_SM}"
    else
        check_fail "Baseline evaluation state machine not found: ${BASELINE_SM}"
    fi
    
    # Check fine-tuning
    FINETUNE_SM="${PROJECT_NAME}-fine-tuning-${ENVIRONMENT}"
    FINETUNE_ARN="arn:aws:states:${AWS_REGION}:${ACCOUNT_ID}:stateMachine:${FINETUNE_SM}"
    if aws stepfunctions describe-state-machine --state-machine-arn "${FINETUNE_ARN}" 2>&1 > /dev/null; then
        check_pass "Fine-tuning state machine exists: ${FINETUNE_SM}"
    else
        check_fail "Fine-tuning state machine not found: ${FINETUNE_SM}"
    fi
    
    # Check comparative evaluation
    COMPARE_SM="${PROJECT_NAME}-comparative-evaluation-${ENVIRONMENT}"
    COMPARE_ARN="arn:aws:states:${AWS_REGION}:${ACCOUNT_ID}:stateMachine:${COMPARE_SM}"
    if aws stepfunctions describe-state-machine --state-machine-arn "${COMPARE_ARN}" 2>&1 > /dev/null; then
        check_pass "Comparative evaluation state machine exists: ${COMPARE_SM}"
    else
        check_fail "Comparative evaluation state machine not found: ${COMPARE_SM}"
    fi
}

validate_knowledge_base() {
    log_info "Validating Bedrock Knowledge Base..."

    # List knowledge bases and check for the one matching our project
    KB_NAME="${PROJECT_NAME}-kb-${ENVIRONMENT}"
    KB_ID=$(aws bedrock-agent list-knowledge-bases --region "${AWS_REGION}" \
        --query "knowledgeBaseSummaries[?name=='${KB_NAME}'].knowledgeBaseId" \
        --output text 2>/dev/null)

    if [ -n "$KB_ID" ] && [ "$KB_ID" != "None" ]; then
        check_pass "Knowledge Base exists: ${KB_NAME} (${KB_ID})"

        # Check KB status
        STATUS=$(aws bedrock-agent get-knowledge-base --knowledge-base-id "${KB_ID}" \
            --region "${AWS_REGION}" --query 'knowledgeBase.status' --output text 2>/dev/null)
        if [ "$STATUS" = "ACTIVE" ]; then
            check_pass "Knowledge Base is active"
        else
            check_fail "Knowledge Base status: ${STATUS}"
        fi
    else
        check_fail "Knowledge Base not found: ${KB_NAME}"
    fi
}

validate_cloudwatch() {
    log_info "Validating CloudWatch log groups..."
    
    # Check log groups
    LOG_GROUPS=(
        "/aws/${PROJECT_NAME}/${ENVIRONMENT}"
        "/aws/${PROJECT_NAME}/${ENVIRONMENT}/evaluation"
        "/aws/${PROJECT_NAME}/${ENVIRONMENT}/finetuning"
        "/aws/${PROJECT_NAME}/${ENVIRONMENT}/trustscoring"
    )
    
    for LOG_GROUP in "${LOG_GROUPS[@]}"; do
        if aws logs describe-log-groups --log-group-name-prefix "${LOG_GROUP}" --region "${AWS_REGION}" --query 'logGroups[0].logGroupName' --output text | grep -q "${LOG_GROUP}"; then
            check_pass "Log group exists: ${LOG_GROUP}"
        else
            check_fail "Log group not found: ${LOG_GROUP}"
        fi
    done
}

validate_iam_roles() {
    log_info "Validating IAM roles..."
    
    # Check Lambda execution role
    LAMBDA_ROLE="${PROJECT_NAME}-lambda-execution-${ENVIRONMENT}"
    if aws iam get-role --role-name "${LAMBDA_ROLE}" 2>&1 > /dev/null; then
        check_pass "Lambda execution role exists: ${LAMBDA_ROLE}"
    else
        check_fail "Lambda execution role not found: ${LAMBDA_ROLE}"
    fi
    
    # Check Step Functions execution role
    SF_ROLE="${PROJECT_NAME}-stepfunctions-${ENVIRONMENT}"
    if aws iam get-role --role-name "${SF_ROLE}" 2>&1 > /dev/null; then
        check_pass "Step Functions execution role exists: ${SF_ROLE}"
    else
        check_fail "Step Functions execution role not found: ${SF_ROLE}"
    fi
}

# Main validation flow
main() {
    log_info "Starting TrustOps infrastructure validation..."
    log_info "Project: ${PROJECT_NAME}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Region: ${AWS_REGION}"
    log_info "Account: ${ACCOUNT_ID}"
    echo ""
    
    # Run validations
    validate_s3_buckets
    echo ""
    validate_dynamodb_tables
    echo ""
    validate_lambda_functions
    echo ""
    validate_step_functions
    echo ""
    validate_knowledge_base
    echo ""
    validate_cloudwatch
    echo ""
    validate_iam_roles
    echo ""
    
    # Summary
    log_info "Validation Summary:"
    echo -e "${GREEN}Passed: ${PASSED}${NC}"
    echo -e "${RED}Failed: ${FAILED}${NC}"
    echo ""
    
    if [ $FAILED -eq 0 ]; then
        log_info "All validation checks passed! ✓"
        exit 0
    else
        log_error "Some validation checks failed. Please review the output above."
        exit 1
    fi
}

# Run main function
main
