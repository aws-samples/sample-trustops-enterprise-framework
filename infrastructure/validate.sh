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

# Auto-discover the active TrustOps CloudFormation stack and read outputs.
# This ensures resource names always match the deployed stack regardless of
# DEPLOYMENT_ID value.
DEPLOYMENT_ID="${DEPLOYMENT_ID:-}"
if [ -n "${DEPLOYMENT_ID}" ]; then
    STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}-${DEPLOYMENT_ID}"
else
    # Find the most recent active trustops stack
    STACK_NAME=$(aws cloudformation list-stacks --region "${AWS_REGION}" \
        --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE \
        --query "StackSummaries[?starts_with(StackName, '${PROJECT_NAME}-${ENVIRONMENT}-')].StackName | sort(@) | [-1]" \
        --output text 2>/dev/null)
fi

if [ -z "${STACK_NAME}" ] || [ "${STACK_NAME}" = "None" ]; then
    log_error "No active TrustOps CloudFormation stack found."
    log_error "Deploy first with: cd infrastructure && ./deploy.sh"
    exit 1
fi

log_info "Using stack: ${STACK_NAME}"

# Helper to read a stack output by key
get_output() {
    aws cloudformation describe-stacks --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text 2>/dev/null
}

# Validation functions
validate_s3_buckets() {
    log_info "Validating S3 buckets..."
    
    # Check datasets bucket
    DATASETS_BUCKET=$(get_output DatasetsBucketName)
    if [ -n "${DATASETS_BUCKET}" ] && aws s3 ls "s3://${DATASETS_BUCKET}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Datasets bucket exists: ${DATASETS_BUCKET}"
        
        # Check versioning
        VERSIONING=$(aws s3api get-bucket-versioning --bucket "${DATASETS_BUCKET}" --query Status --output text)
        if [ "$VERSIONING" = "Enabled" ]; then
            check_pass "Datasets bucket versioning enabled"
        else
            check_fail "Datasets bucket versioning not enabled"
        fi
    else
        check_fail "Datasets bucket not found: ${DATASETS_BUCKET:-<not in stack outputs>}"
    fi
    
    # Check results bucket
    RESULTS_BUCKET=$(get_output ResultsBucketName)
    if [ -n "${RESULTS_BUCKET}" ] && aws s3 ls "s3://${RESULTS_BUCKET}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Results bucket exists: ${RESULTS_BUCKET}"
    else
        check_fail "Results bucket not found: ${RESULTS_BUCKET:-<not in stack outputs>}"
    fi
    
    # Check artifacts bucket
    ARTIFACTS_BUCKET=$(get_output ArtifactsBucketName)
    if [ -n "${ARTIFACTS_BUCKET}" ] && aws s3 ls "s3://${ARTIFACTS_BUCKET}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Artifacts bucket exists: ${ARTIFACTS_BUCKET}"
    else
        check_fail "Artifacts bucket not found: ${ARTIFACTS_BUCKET:-<not in stack outputs>}"
    fi
}

validate_dynamodb_tables() {
    log_info "Validating DynamoDB tables..."
    
    # Check workflows table
    WORKFLOWS_TABLE=$(get_output WorkflowsTableName)
    if [ -n "${WORKFLOWS_TABLE}" ] && aws dynamodb describe-table --table-name "${WORKFLOWS_TABLE}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
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
        check_fail "Workflows table not found: ${WORKFLOWS_TABLE:-<not in stack outputs>}"
    fi
    
    # Check models table
    MODELS_TABLE=$(get_output ModelsTableName)
    if [ -n "${MODELS_TABLE}" ] && aws dynamodb describe-table --table-name "${MODELS_TABLE}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
        check_pass "Models table exists: ${MODELS_TABLE}"
    else
        check_fail "Models table not found: ${MODELS_TABLE:-<not in stack outputs>}"
    fi
}

validate_lambda_functions() {
    log_info "Validating Lambda functions..."
    
    for key in EvaluationOrchestratorFunctionArn FineTuningOrchestratorFunctionArn TrustScoringFunctionArn ComparativeEvaluationFunctionArn; do
        ARN=$(get_output "${key}")
        if [ -n "${ARN}" ]; then
            FUNC_NAME=$(echo "${ARN}" | awk -F: '{print $NF}')
            if aws lambda get-function --function-name "${FUNC_NAME}" --region "${AWS_REGION}" 2>&1 > /dev/null; then
                check_pass "Lambda function exists: ${FUNC_NAME}"
            else
                check_fail "Lambda function not found: ${FUNC_NAME}"
            fi
        else
            check_fail "Lambda output not found: ${key}"
        fi
    done
}

validate_step_functions() {
    log_info "Validating Step Functions state machines..."
    
    for key in BaselineEvaluationStateMachineArn FineTuningStateMachineArn ComparativeEvaluationStateMachineArn; do
        ARN=$(get_output "${key}")
        if [ -n "${ARN}" ]; then
            SM_NAME=$(echo "${ARN}" | awk -F: '{print $NF}')
            if aws stepfunctions describe-state-machine --state-machine-arn "${ARN}" 2>&1 > /dev/null; then
                check_pass "State machine exists: ${SM_NAME}"
            else
                check_fail "State machine not found: ${SM_NAME}"
            fi
        else
            check_fail "Step Functions output not found: ${key}"
        fi
    done
}

validate_knowledge_base() {
    log_info "Validating Bedrock Knowledge Base..."

    KB_ROLE_ARN=$(get_output KnowledgeBaseRoleArn)
    KB_ID=$(aws bedrock-agent list-knowledge-bases --region "${AWS_REGION}" \
        --query "knowledgeBaseSummaries[?contains(name, '${PROJECT_NAME}')].knowledgeBaseId | [0]" \
        --output text 2>/dev/null)

    if [ -n "$KB_ID" ] && [ "$KB_ID" != "None" ]; then
        check_pass "Knowledge Base exists: ${KB_ID}"

        STATUS=$(aws bedrock-agent get-knowledge-base --knowledge-base-id "${KB_ID}" \
            --region "${AWS_REGION}" --query 'knowledgeBase.status' --output text 2>/dev/null)
        if [ "$STATUS" = "ACTIVE" ]; then
            check_pass "Knowledge Base is active"
        else
            check_fail "Knowledge Base status: ${STATUS}"
        fi
    else
        log_info "  No Knowledge Base deployed (optional — retrieval runs in mock mode)"
        ((PASSED++))
    fi
}

validate_cloudwatch() {
    log_info "Validating CloudWatch log groups..."
    
    # Derive deployment ID from the stack name
    STACK_DEPLOY_ID=$(echo "${STACK_NAME}" | awk -F- '{print $NF}')
    
    # Check log groups (include DeploymentId in path)
    LOG_GROUPS=(
        "/aws/${PROJECT_NAME}/${ENVIRONMENT}/${STACK_DEPLOY_ID}"
        "/aws/${PROJECT_NAME}/${ENVIRONMENT}/${STACK_DEPLOY_ID}/evaluation"
        "/aws/${PROJECT_NAME}/${ENVIRONMENT}/${STACK_DEPLOY_ID}/finetuning"
        "/aws/${PROJECT_NAME}/${ENVIRONMENT}/${STACK_DEPLOY_ID}/trustscoring"
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
    
    # Check Lambda execution role (use stack output ARN)
    LAMBDA_ROLE_ARN=$(get_output LambdaExecutionRoleArn)
    if [ -n "${LAMBDA_ROLE_ARN}" ]; then
        LAMBDA_ROLE=$(echo "${LAMBDA_ROLE_ARN}" | awk -F/ '{print $NF}')
        if aws iam get-role --role-name "${LAMBDA_ROLE}" 2>&1 > /dev/null; then
            check_pass "Lambda execution role exists: ${LAMBDA_ROLE}"
        else
            check_fail "Lambda execution role not found: ${LAMBDA_ROLE}"
        fi
    else
        check_fail "Lambda role ARN not found in stack outputs"
    fi
    
    # Check Step Functions execution role
    SF_ROLE_ARN=$(get_output StepFunctionsExecutionRoleArn)
    if [ -n "${SF_ROLE_ARN}" ]; then
        SF_ROLE=$(echo "${SF_ROLE_ARN}" | awk -F/ '{print $NF}')
        if aws iam get-role --role-name "${SF_ROLE}" 2>&1 > /dev/null; then
            check_pass "Step Functions execution role exists: ${SF_ROLE}"
        else
            check_fail "Step Functions execution role not found: ${SF_ROLE}"
        fi
    else
        check_fail "Step Functions role ARN not found in stack outputs"
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
