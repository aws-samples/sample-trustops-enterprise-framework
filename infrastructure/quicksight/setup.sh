#!/bin/bash
# QuickSight Setup Script
# This script sets up QuickSight data sources, datasets, and dashboards for TrustOps

set -e

# Configuration
PROJECT_NAME="${PROJECT_NAME:-trustops}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
QUICKSIGHT_USER="${QUICKSIGHT_USER:-}"

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
    
    # Get AWS account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    log_info "AWS Account ID: ${ACCOUNT_ID}"
    
    # Check if QuickSight is enabled
    if ! aws quicksight describe-account-settings --aws-account-id "${ACCOUNT_ID}" --region "${AWS_REGION}" &> /dev/null; then
        log_error "QuickSight is not enabled for this account. Please enable it in the AWS Console."
        log_info "Visit: https://quicksight.aws.amazon.com/"
        exit 1
    fi
    
    log_info "Prerequisites check passed!"
}

# Get QuickSight user ARN
get_quicksight_user() {
    if [ -z "${QUICKSIGHT_USER}" ]; then
        log_info "Fetching QuickSight users..."
        QUICKSIGHT_USER=$(aws quicksight list-users \
            --aws-account-id "${ACCOUNT_ID}" \
            --namespace default \
            --region "${AWS_REGION}" \
            --query 'UserList[0].Arn' \
            --output text)
        
        if [ -z "${QUICKSIGHT_USER}" ] || [ "${QUICKSIGHT_USER}" == "None" ]; then
            log_error "No QuickSight users found. Please create a QuickSight user first."
            exit 1
        fi
    fi
    
    log_info "QuickSight User: ${QUICKSIGHT_USER}"
}

# Create S3 manifest file for QuickSight
create_s3_manifest() {
    log_info "Creating S3 manifest file..."
    
    RESULTS_BUCKET="${PROJECT_NAME}-results-${ENVIRONMENT}-${ACCOUNT_ID}"
    
    cat > /tmp/quicksight-manifest.json <<EOF
{
  "fileLocations": [
    {
      "URIPrefixes": [
        "s3://${RESULTS_BUCKET}/evaluations/",
        "s3://${RESULTS_BUCKET}/comparisons/"
      ]
    }
  ],
  "globalUploadSettings": {
    "format": "JSON"
  }
}
EOF
    
    # Upload manifest to S3
    aws s3 cp /tmp/quicksight-manifest.json "s3://${RESULTS_BUCKET}/quicksight/manifest.json"
    
    log_info "S3 manifest created and uploaded"
}

# Deploy Terraform resources
deploy_terraform() {
    log_info "Deploying QuickSight resources with Terraform..."
    
    cd "$(dirname "$0")"
    
    # Initialize Terraform if needed
    if [ ! -d ".terraform" ]; then
        terraform init
    fi
    
    # Apply Terraform configuration
    terraform apply \
        -var="project_name=${PROJECT_NAME}" \
        -var="environment=${ENVIRONMENT}" \
        -var="aws_region=${AWS_REGION}" \
        -auto-approve
    
    log_info "Terraform deployment completed"
}

# Create QuickSight dashboards using AWS CLI
create_dashboards() {
    log_info "Creating QuickSight dashboards..."
    
    # Read dashboard definitions
    DASHBOARDS=$(cat dashboards.json | jq -r '.dashboards[] | @base64')
    
    for dashboard in ${DASHBOARDS}; do
        _jq() {
            echo "${dashboard}" | base64 --decode | jq -r "${1}"
        }
        
        DASHBOARD_ID=$(_jq '.id')
        DASHBOARD_NAME=$(_jq '.name')
        
        log_info "Creating dashboard: ${DASHBOARD_NAME}"
        
        # Note: Dashboard creation requires QuickSight API calls with visual definitions
        # This is a placeholder - actual implementation would use QuickSight APIs
        log_warn "Dashboard ${DASHBOARD_NAME} definition created. Manual configuration required in QuickSight console."
    done
    
    log_info "Dashboard creation completed"
}

# Configure dashboard permissions
configure_permissions() {
    log_info "Configuring dashboard permissions..."
    
    # Get all dashboards
    DASHBOARD_IDS=$(aws quicksight list-dashboards \
        --aws-account-id "${ACCOUNT_ID}" \
        --region "${AWS_REGION}" \
        --query "DashboardSummaryList[?starts_with(Name, '${PROJECT_NAME}')].DashboardId" \
        --output text)
    
    for dashboard_id in ${DASHBOARD_IDS}; do
        log_info "Configuring permissions for dashboard: ${dashboard_id}"
        
        aws quicksight update-dashboard-permissions \
            --aws-account-id "${ACCOUNT_ID}" \
            --dashboard-id "${dashboard_id}" \
            --region "${AWS_REGION}" \
            --grant-permissions "Principal=${QUICKSIGHT_USER},Actions=quicksight:DescribeDashboard,quicksight:ListDashboardVersions,quicksight:UpdateDashboardPermissions,quicksight:QueryDashboard,quicksight:UpdateDashboard,quicksight:DeleteDashboard,quicksight:DescribeDashboardPermissions,quicksight:UpdateDashboardPublishedVersion" \
            || log_warn "Failed to update permissions for ${dashboard_id}"
    done
    
    log_info "Permissions configuration completed"
}

# Trigger initial data refresh
trigger_refresh() {
    log_info "Triggering initial data refresh..."
    
    DATASETS=(
        "${PROJECT_NAME}-baseline-evaluation-${ENVIRONMENT}"
        "${PROJECT_NAME}-comparative-evaluation-${ENVIRONMENT}"
        "${PROJECT_NAME}-cost-analysis-${ENVIRONMENT}"
        "${PROJECT_NAME}-hallucination-analysis-${ENVIRONMENT}"
    )
    
    for dataset_id in "${DATASETS[@]}"; do
        log_info "Refreshing dataset: ${dataset_id}"
        
        INGESTION_ID="initial-refresh-$(date +%s)"
        
        aws quicksight create-ingestion \
            --aws-account-id "${ACCOUNT_ID}" \
            --data-set-id "${dataset_id}" \
            --ingestion-id "${INGESTION_ID}" \
            --region "${AWS_REGION}" \
            || log_warn "Failed to trigger refresh for ${dataset_id}"
    done
    
    log_info "Data refresh triggered"
}

# Print dashboard URLs
print_dashboard_urls() {
    log_info "QuickSight Dashboard URLs:"
    echo ""
    
    DASHBOARDS=(
        "baseline-evaluation:Baseline Evaluation Dashboard"
        "comparative-evaluation:Comparative Evaluation Dashboard"
        "cost-analysis:Cost Analysis Dashboard"
        "hallucination-analysis:Hallucination Analysis Dashboard"
    )
    
    for dashboard in "${DASHBOARDS[@]}"; do
        IFS=':' read -r id name <<< "${dashboard}"
        DASHBOARD_ID="${PROJECT_NAME}-${id}-${ENVIRONMENT}"
        URL="https://${AWS_REGION}.quicksight.aws.amazon.com/sn/dashboards/${DASHBOARD_ID}"
        echo -e "${GREEN}${name}:${NC}"
        echo "  ${URL}"
        echo ""
    done
}

# Main execution
main() {
    log_info "Starting QuickSight setup for TrustOps..."
    log_info "Project: ${PROJECT_NAME}, Environment: ${ENVIRONMENT}, Region: ${AWS_REGION}"
    echo ""
    
    check_prerequisites
    get_quicksight_user
    create_s3_manifest
    deploy_terraform
    create_dashboards
    configure_permissions
    trigger_refresh
    
    echo ""
    log_info "QuickSight setup completed successfully!"
    echo ""
    print_dashboard_urls
    
    log_info "Next steps:"
    echo "  1. Visit the QuickSight console to customize dashboard layouts"
    echo "  2. Configure additional filters and parameters as needed"
    echo "  3. Share dashboards with team members"
    echo "  4. Set up email reports and alerts"
}

# Run main function
main "$@"
