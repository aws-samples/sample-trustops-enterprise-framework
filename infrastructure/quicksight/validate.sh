#!/bin/bash
# QuickSight Validation Script
# This script validates the QuickSight setup for TrustOps

set -e

# Configuration
PROJECT_NAME="${PROJECT_NAME:-trustops}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED++))
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
    ((WARNINGS++))
}

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo ""
log_info "Starting QuickSight validation for TrustOps..."
log_info "Project: ${PROJECT_NAME}, Environment: ${ENVIRONMENT}, Region: ${AWS_REGION}"
log_info "Account ID: ${ACCOUNT_ID}"
echo ""

# Test 1: Check QuickSight is enabled
log_info "Test 1: Checking if QuickSight is enabled..."
if aws quicksight describe-account-settings --aws-account-id "${ACCOUNT_ID}" --region "${AWS_REGION}" &> /dev/null; then
    log_success "QuickSight is enabled"
else
    log_fail "QuickSight is not enabled"
fi

# Test 2: Check QuickSight users exist
log_info "Test 2: Checking QuickSight users..."
USER_COUNT=$(aws quicksight list-users \
    --aws-account-id "${ACCOUNT_ID}" \
    --namespace default \
    --region "${AWS_REGION}" \
    --query 'length(UserList)' \
    --output text 2>/dev/null || echo "0")

if [ "${USER_COUNT}" -gt 0 ]; then
    log_success "Found ${USER_COUNT} QuickSight user(s)"
else
    log_fail "No QuickSight users found"
fi

# Test 3: Check S3 buckets exist
log_info "Test 3: Checking S3 buckets..."
RESULTS_BUCKET="${PROJECT_NAME}-results-${ENVIRONMENT}-${ACCOUNT_ID}"
DATASETS_BUCKET="${PROJECT_NAME}-datasets-${ENVIRONMENT}-${ACCOUNT_ID}"

if aws s3 ls "s3://${RESULTS_BUCKET}" &> /dev/null; then
    log_success "Results bucket exists: ${RESULTS_BUCKET}"
else
    log_fail "Results bucket not found: ${RESULTS_BUCKET}"
fi

if aws s3 ls "s3://${DATASETS_BUCKET}" &> /dev/null; then
    log_success "Datasets bucket exists: ${DATASETS_BUCKET}"
else
    log_fail "Datasets bucket not found: ${DATASETS_BUCKET}"
fi

# Test 4: Check S3 manifest file
log_info "Test 4: Checking S3 manifest file..."
if aws s3 ls "s3://${RESULTS_BUCKET}/quicksight/manifest.json" &> /dev/null; then
    log_success "S3 manifest file exists"
else
    log_warn "S3 manifest file not found (will be created during setup)"
fi

# Test 5: Check Glue database
log_info "Test 5: Checking Glue catalog database..."
GLUE_DB="${PROJECT_NAME}_${ENVIRONMENT}"
if aws glue get-database --name "${GLUE_DB}" --region "${AWS_REGION}" &> /dev/null; then
    log_success "Glue database exists: ${GLUE_DB}"
else
    log_fail "Glue database not found: ${GLUE_DB}"
fi

# Test 6: Check Glue tables
log_info "Test 6: Checking Glue catalog tables..."
if aws glue get-table --database-name "${GLUE_DB}" --name "evaluation_results" --region "${AWS_REGION}" &> /dev/null; then
    log_success "Glue table exists: evaluation_results"
else
    log_fail "Glue table not found: evaluation_results"
fi

if aws glue get-table --database-name "${GLUE_DB}" --name "comparative_results" --region "${AWS_REGION}" &> /dev/null; then
    log_success "Glue table exists: comparative_results"
else
    log_fail "Glue table not found: comparative_results"
fi

# Test 7: Check Athena workgroup
log_info "Test 7: Checking Athena workgroup..."
WORKGROUP="${PROJECT_NAME}-quicksight-${ENVIRONMENT}"
if aws athena get-work-group --work-group "${WORKGROUP}" --region "${AWS_REGION}" &> /dev/null; then
    log_success "Athena workgroup exists: ${WORKGROUP}"
else
    log_fail "Athena workgroup not found: ${WORKGROUP}"
fi

# Test 8: Check QuickSight data sources
log_info "Test 8: Checking QuickSight data sources..."
DATA_SOURCES=$(aws quicksight list-data-sources \
    --aws-account-id "${ACCOUNT_ID}" \
    --region "${AWS_REGION}" \
    --query "DataSources[?starts_with(Name, 'TrustOps')].Name" \
    --output text 2>/dev/null || echo "")

if [ -n "${DATA_SOURCES}" ]; then
    log_success "Found QuickSight data sources"
    echo "  ${DATA_SOURCES}"
else
    log_warn "No QuickSight data sources found (will be created during setup)"
fi

# Test 9: Check QuickSight datasets
log_info "Test 9: Checking QuickSight datasets..."
DATASETS=$(aws quicksight list-data-sets \
    --aws-account-id "${ACCOUNT_ID}" \
    --region "${AWS_REGION}" \
    --query "DataSetSummaries[?starts_with(Name, 'Baseline') || starts_with(Name, 'Comparative') || starts_with(Name, 'Cost') || starts_with(Name, 'Hallucination')].Name" \
    --output text 2>/dev/null || echo "")

if [ -n "${DATASETS}" ]; then
    log_success "Found QuickSight datasets"
    echo "  ${DATASETS}"
else
    log_warn "No QuickSight datasets found (will be created during setup)"
fi

# Test 10: Check QuickSight dashboards
log_info "Test 10: Checking QuickSight dashboards..."
DASHBOARDS=$(aws quicksight list-dashboards \
    --aws-account-id "${ACCOUNT_ID}" \
    --region "${AWS_REGION}" \
    --query "DashboardSummaryList[?starts_with(Name, '${PROJECT_NAME}')].Name" \
    --output text 2>/dev/null || echo "")

if [ -n "${DASHBOARDS}" ]; then
    log_success "Found QuickSight dashboards"
    echo "  ${DASHBOARDS}"
else
    log_warn "No QuickSight dashboards found (will be created manually)"
fi

# Test 11: Check IAM role for QuickSight
log_info "Test 11: Checking QuickSight IAM role..."
ROLE_NAME="${PROJECT_NAME}-quicksight-service-${ENVIRONMENT}"
if aws iam get-role --role-name "${ROLE_NAME}" &> /dev/null; then
    log_success "QuickSight IAM role exists: ${ROLE_NAME}"
else
    log_fail "QuickSight IAM role not found: ${ROLE_NAME}"
fi

# Test 12: Test Athena query
log_info "Test 12: Testing Athena query..."
QUERY_ID=$(aws athena start-query-execution \
    --query-string "SELECT COUNT(*) as count FROM ${GLUE_DB}.evaluation_results LIMIT 1" \
    --result-configuration "OutputLocation=s3://${RESULTS_BUCKET}/athena-results/" \
    --work-group "${WORKGROUP}" \
    --region "${AWS_REGION}" \
    --query 'QueryExecutionId' \
    --output text 2>/dev/null || echo "")

if [ -n "${QUERY_ID}" ]; then
    # Wait for query to complete
    sleep 3
    QUERY_STATE=$(aws athena get-query-execution \
        --query-execution-id "${QUERY_ID}" \
        --region "${AWS_REGION}" \
        --query 'QueryExecution.Status.State' \
        --output text 2>/dev/null || echo "FAILED")
    
    if [ "${QUERY_STATE}" == "SUCCEEDED" ]; then
        log_success "Athena query executed successfully"
    else
        log_warn "Athena query state: ${QUERY_STATE}"
    fi
else
    log_warn "Could not execute Athena query (may need data first)"
fi

# Test 13: Check for evaluation data
log_info "Test 13: Checking for evaluation data..."
DATA_COUNT=$(aws s3 ls "s3://${RESULTS_BUCKET}/evaluations/" --recursive 2>/dev/null | wc -l || echo "0")

if [ "${DATA_COUNT}" -gt 0 ]; then
    log_success "Found ${DATA_COUNT} evaluation result file(s)"
else
    log_warn "No evaluation data found (run demo to generate data)"
fi

# Summary
echo ""
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo -e "${GREEN}Passed:${NC}   ${PASSED}"
echo -e "${RED}Failed:${NC}   ${FAILED}"
echo -e "${YELLOW}Warnings:${NC} ${WARNINGS}"
echo "=========================================="
echo ""

if [ ${FAILED} -eq 0 ]; then
    log_success "All critical tests passed!"
    
    if [ ${WARNINGS} -gt 0 ]; then
        log_warn "Some optional components are missing. Run setup.sh to complete configuration."
    fi
    
    echo ""
    log_info "Next steps:"
    echo "  1. Run ./setup.sh to complete QuickSight setup"
    echo "  2. Generate evaluation data by running the demo"
    echo "  3. Create dashboards in QuickSight console"
    echo "  4. Share dashboards with team members"
    exit 0
else
    log_fail "Validation failed with ${FAILED} error(s)"
    echo ""
    log_info "Troubleshooting:"
    echo "  1. Ensure TrustOps infrastructure is deployed"
    echo "  2. Enable QuickSight in AWS Console"
    echo "  3. Run terraform apply in infrastructure/quicksight/"
    echo "  4. Check AWS permissions and service quotas"
    exit 1
fi
