# QuickSight Dashboard Setup Guide

This guide provides step-by-step instructions for setting up QuickSight dashboards for the TrustOps AWS Demo.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [QuickSight Account Setup](#quicksight-account-setup)
3. [Automated Setup](#automated-setup)
4. [Manual Setup](#manual-setup)
5. [Dashboard Configuration](#dashboard-configuration)
6. [Testing and Validation](#testing-and-validation)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### AWS Account Requirements

- AWS account with administrator access
- AWS CLI installed and configured
- Terraform 1.0+ installed
- Existing TrustOps infrastructure deployed (S3, DynamoDB, Lambda, etc.)

### QuickSight Requirements

- QuickSight subscription (Standard or Enterprise Edition)
- QuickSight user account created
- Permissions to create data sources, datasets, and dashboards

## QuickSight Account Setup

### Step 1: Enable QuickSight

1. Navigate to the QuickSight console:
   ```
   https://quicksight.aws.amazon.com/
   ```

2. If QuickSight is not enabled, click "Sign up for QuickSight"

3. Choose edition:
   - **Standard Edition**: $9/user/month (basic features)
   - **Enterprise Edition**: $18/user/month (recommended for advanced features)

4. Configure account:
   - Account name: `trustops-quicksight`
   - Notification email: Your email address
   - QuickSight region: Same as your TrustOps deployment region

5. Click "Finish" to complete setup

### Step 2: Grant AWS Service Access

1. In QuickSight console, click your username (top right)
2. Select "Manage QuickSight"
3. Click "Security & permissions" in left menu
4. Under "QuickSight access to AWS services", click "Add or remove"
5. Select the following services:
   - ✅ Amazon S3
   - ✅ Amazon Athena
   - ✅ AWS Glue
6. For S3, click "Select S3 buckets"
7. Select your TrustOps buckets:
   - `trustops-results-<YOUR_ACCOUNT_ID>-<REGION>`
   - `trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>`
8. Click "Finish"

### Step 3: Create QuickSight Users

1. In "Manage QuickSight", click "Manage users"
2. Click "Invite users"
3. Enter email addresses for team members
4. Select role:
   - **Author**: Can create and edit dashboards
   - **Reader**: Can only view dashboards
5. Click "Invite"

## Automated Setup

The automated setup script handles all configuration steps.

### Step 1: Set Environment Variables

```bash
export PROJECT_NAME=trustops
export ENVIRONMENT=dev
export AWS_REGION=us-east-1
export QUICKSIGHT_USER=arn:aws:quicksight:us-east-1:123456789012:user/default/admin
```

### Step 2: Run Setup Script

```bash
cd infrastructure/quicksight
./setup.sh
```

The script will:
1. ✅ Check prerequisites
2. ✅ Create S3 manifest file
3. ✅ Deploy Terraform resources
4. ✅ Configure data sources
5. ✅ Create datasets
6. ✅ Set up refresh schedules
7. ✅ Configure permissions
8. ✅ Trigger initial data refresh

### Step 3: Verify Setup

Check the output for dashboard URLs:

```
QuickSight Dashboard URLs:

Baseline Evaluation Dashboard:
  https://us-east-1.quicksight.aws.amazon.com/sn/dashboards/trustops-baseline-evaluation-dev

Comparative Evaluation Dashboard:
  https://us-east-1.quicksight.aws.amazon.com/sn/dashboards/trustops-comparative-evaluation-dev

Cost Analysis Dashboard:
  https://us-east-1.quicksight.aws.amazon.com/sn/dashboards/trustops-cost-analysis-dev

Hallucination Analysis Dashboard:
  https://us-east-1.quicksight.aws.amazon.com/sn/dashboards/trustops-hallucination-analysis-dev
```

## Manual Setup

If you prefer manual setup or need to customize the configuration:

### Step 1: Deploy Terraform Resources

```bash
cd infrastructure/quicksight

# Initialize Terraform
terraform init

# Review plan
terraform plan \
    -var="project_name=trustops" \
    -var="environment=dev" \
    -var="aws_region=us-east-1"

# Apply configuration
terraform apply \
    -var="project_name=trustops" \
    -var="environment=dev" \
    -var="aws_region=us-east-1"
```

This creates:
- IAM role for QuickSight
- Athena workgroup
- Glue catalog database and tables
- QuickSight data sources
- QuickSight datasets with refresh schedules

### Step 2: Create S3 Manifest

Create `manifest.json`:

```json
{
  "fileLocations": [
    {
      "URIPrefixes": [
        "s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>/evaluations/",
        "s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>/comparisons/"
      ]
    }
  ],
  "globalUploadSettings": {
    "format": "JSON"
  }
}
```

Upload to S3:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
RESULTS_BUCKET="trustops-results-${ACCOUNT_ID}-${AWS_REGION}"

aws s3 cp manifest.json "s3://${RESULTS_BUCKET}/quicksight/manifest.json"
```

### Step 3: Verify Data Sources

```bash
aws quicksight list-data-sources \
    --aws-account-id ${ACCOUNT_ID} \
    --region us-east-1
```

Expected output:
- `trustops-s3-results-dev`
- `trustops-dynamodb-workflows-dev`
- `trustops-athena-dev`

### Step 4: Verify Datasets

```bash
aws quicksight list-data-sets \
    --aws-account-id ${ACCOUNT_ID} \
    --region us-east-1
```

Expected output:
- `trustops-baseline-evaluation-dev`
- `trustops-comparative-evaluation-dev`
- `trustops-cost-analysis-dev`
- `trustops-hallucination-analysis-dev`

### Step 5: Trigger Initial Refresh

```bash
for dataset in baseline-evaluation comparative-evaluation cost-analysis hallucination-analysis; do
    aws quicksight create-ingestion \
        --aws-account-id ${ACCOUNT_ID} \
        --data-set-id "trustops-${dataset}-dev" \
        --ingestion-id "initial-$(date +%s)" \
        --region us-east-1
done
```

## Dashboard Configuration

### Creating Dashboards in QuickSight Console

#### 1. Baseline Evaluation Dashboard

1. Navigate to QuickSight console
2. Click "Analyses" → "New analysis"
3. Select dataset: "Baseline Evaluation Metrics"
4. Click "Create analysis"

**Add Visuals:**

a. **Trust Score Distribution (Histogram)**
   - Visual type: Histogram
   - Field: trust_score
   - Bins: 20
   - Title: "Trust Score Distribution"

b. **Trust Score by Category (Bar Chart)**
   - Visual type: Vertical bar chart
   - X-axis: category
   - Value: trust_score (Average)
   - Title: "Mean Trust Score by Category"

c. **Latency Histogram**
   - Visual type: Histogram
   - Field: latency_ms
   - Bins: 20
   - Title: "Response Latency Distribution"

d. **Trust Score Trend (Line Chart)**
   - Visual type: Line chart
   - X-axis: date
   - Value: trust_score (Average)
   - Title: "Trust Score Trend Over Time"

e. **Trust Score Categories (Pie Chart)**
   - Visual type: Pie chart
   - Group by: trust_score_category
   - Title: "Trust Score Categories"

f. **Sample Results (Table)**
   - Visual type: Table
   - Columns: example_id, trust_score, hallucination_rate, latency_ms, cost, passed
   - Sort by: trust_score (Descending)
   - Rows: 50

g. **KPIs**
   - Mean Trust Score: trust_score (Average)
   - P95 Latency: latency_ms (95th percentile)
   - Total Cost: cost (Sum)

**Add Filters:**
- model_id (Dropdown)
- date (Date range)
- category (Multi-select)

**Publish Dashboard:**
1. Click "Share" → "Publish dashboard"
2. Name: "Baseline Evaluation Dashboard"
3. Click "Publish dashboard"

#### 2. Comparative Evaluation Dashboard

Follow similar steps using "Comparative Evaluation Metrics" dataset.

**Key Visuals:**
- Trust Score Improvement (Bar chart)
- Hallucination Reduction (Bar chart)
- Cost Delta (Bar chart)
- Latency Delta (Bar chart)
- Improvement Trend (Line chart)
- Statistical Significance (Pie chart)
- Recommendations (Pie chart)
- Comparison Table

#### 3. Cost Analysis Dashboard

Use "Cost Analysis" dataset.

**Key Visuals:**
- Cost Trend (Line chart)
- Cost by Model (Bar chart)
- Cost-Performance Curve (Scatter plot)
- Cost Efficiency (Bar chart)
- Token Usage (Stacked bar chart)
- Cost per High-Trust Response (Bar chart)
- Daily Cost Breakdown (Stacked area chart)

#### 4. Hallucination Analysis Dashboard

Use "Hallucination Analysis" dataset.

**Key Visuals:**
- Hallucination Rate Trend (Line chart)
- Hallucination by Category (Bar chart)
- Hallucination by Model (Bar chart)
- Severity Distribution (Pie chart)
- Hallucination vs Trust Score (Scatter plot)
- Hallucination Heatmap
- Model Comparison (Grouped bar chart)

### Dashboard Customization Tips

1. **Color Schemes:**
   - High trust scores: Green
   - Medium trust scores: Yellow
   - Low trust scores: Red

2. **Formatting:**
   - Trust scores: 3 decimal places (0.000)
   - Percentages: 1 decimal place (0.0%)
   - Currency: 2-4 decimal places ($0.0000)
   - Latency: 1 decimal place with "ms" suffix

3. **Tooltips:**
   - Add relevant context to each visual
   - Include calculation methods for derived metrics

4. **Layout:**
   - Place KPIs at the top
   - Group related visuals together
   - Use consistent sizing and spacing

## Testing and Validation

### Step 1: Generate Test Data

Run a baseline evaluation to generate data:

```bash
cd demo
python run_demo.py --mode baseline
```

### Step 2: Verify Data in S3

```bash
aws s3 ls "s3://trustops-results-${ACCOUNT_ID}-${AWS_REGION}/evaluations/"
```

Expected: JSON files with evaluation results

### Step 3: Test Athena Query

```sql
SELECT 
    model_id,
    COUNT(*) as total_evaluations,
    AVG(trust_score) as avg_trust_score,
    AVG(hallucination_rate) as avg_hallucination_rate
FROM trustops_dev.evaluation_results
GROUP BY model_id;
```

### Step 4: Refresh Datasets

```bash
aws quicksight create-ingestion \
    --aws-account-id ${ACCOUNT_ID} \
    --data-set-id trustops-baseline-evaluation-dev \
    --ingestion-id test-refresh-$(date +%s) \
    --region us-east-1
```

Check status:

```bash
aws quicksight list-ingestions \
    --aws-account-id ${ACCOUNT_ID} \
    --data-set-id trustops-baseline-evaluation-dev \
    --region us-east-1
```

### Step 5: View Dashboards

1. Navigate to QuickSight console
2. Click "Dashboards"
3. Open "Baseline Evaluation Dashboard"
4. Verify data appears correctly
5. Test filters and interactions

## Troubleshooting

### Issue: No Data in Dashboards

**Possible Causes:**
1. No evaluation data generated yet
2. Data refresh not triggered
3. Athena query errors
4. Glue catalog misconfiguration

**Solutions:**

1. Check S3 for data:
   ```bash
   aws s3 ls "s3://trustops-results-${ACCOUNT_ID}-${AWS_REGION}/evaluations/" --recursive
   ```

2. Test Athena query:
   ```sql
   SELECT * FROM trustops_dev.evaluation_results LIMIT 10;
   ```

3. Check Glue table:
   ```bash
   aws glue get-table \
       --database-name trustops_dev \
       --name evaluation_results
   ```

4. Trigger manual refresh:
   ```bash
   aws quicksight create-ingestion \
       --aws-account-id ${ACCOUNT_ID} \
       --data-set-id trustops-baseline-evaluation-dev \
       --ingestion-id manual-$(date +%s) \
       --region us-east-1
   ```

### Issue: Permission Denied Errors

**Possible Causes:**
1. QuickSight service role lacks permissions
2. S3 bucket not in allowed list
3. IAM policy misconfiguration

**Solutions:**

1. Verify QuickSight service role:
   ```bash
   aws iam get-role --role-name trustops-quicksight-service-dev
   ```

2. Check S3 bucket permissions in QuickSight:
   - Manage QuickSight → Security & permissions
   - Verify buckets are selected

3. Update IAM policy if needed:
   ```bash
   cd infrastructure/quicksight
   terraform apply
   ```

### Issue: Slow Dashboard Performance

**Possible Causes:**
1. Large dataset size
2. Complex calculations
3. SPICE not used

**Solutions:**

1. Verify SPICE is enabled:
   - Dataset settings → Import mode: SPICE

2. Optimize Athena queries:
   - Add partitions to Glue tables
   - Use columnar formats (Parquet)

3. Reduce data volume:
   - Add date range filters
   - Aggregate data before import

### Issue: Refresh Failures

**Possible Causes:**
1. Athena query timeout
2. Data format errors
3. Insufficient SPICE capacity

**Solutions:**

1. Check ingestion status:
   ```bash
   aws quicksight describe-ingestion \
       --aws-account-id ${ACCOUNT_ID} \
       --data-set-id trustops-baseline-evaluation-dev \
       --ingestion-id {ingestion-id} \
       --region us-east-1
   ```

2. Review error messages in output

3. Increase SPICE capacity if needed:
   - Manage QuickSight → SPICE capacity
   - Purchase additional capacity

## Next Steps

After successful setup:

1. **Customize Dashboards:**
   - Adjust layouts and visuals
   - Add custom calculated fields
   - Configure parameters

2. **Share Dashboards:**
   - Share with team members
   - Set up email reports
   - Configure embedded dashboards

3. **Monitor Usage:**
   - Review dashboard access logs
   - Track SPICE usage
   - Monitor refresh schedules

4. **Optimize Performance:**
   - Review slow queries
   - Optimize data models
   - Adjust refresh schedules

5. **Set Up Alerts:**
   - Configure threshold alerts
   - Set up email notifications
   - Create custom metrics

## Support

For additional help:

- **AWS QuickSight Documentation**: https://docs.aws.amazon.com/quicksight/
- **TrustOps Documentation**: See main README.md
- **AWS Support**: Open a support case in AWS Console

## Validation Checklist

- [ ] QuickSight account enabled
- [ ] AWS service access configured
- [ ] Terraform resources deployed
- [ ] S3 manifest created
- [ ] Data sources created
- [ ] Datasets created
- [ ] Refresh schedules configured
- [ ] Test data generated
- [ ] Athena queries working
- [ ] Datasets refreshed successfully
- [ ] Dashboards created
- [ ] Visuals configured
- [ ] Filters working
- [ ] Permissions configured
- [ ] Dashboards shared with team
- [ ] Documentation reviewed

## Requirements Validation

This setup satisfies the following requirements:

- ✅ **Requirement 7.1**: Dashboard displays current evaluation status and recent runs
- ✅ **Requirement 7.2**: Baseline results show trust score distribution, latency, and sample outputs
- ✅ **Requirement 7.3**: Comparative results display side-by-side metrics with improvement percentages
- ✅ **Requirement 7.5**: Hallucination analysis visualizes rates over time and by category
- ✅ **Requirement 7.6**: Cost analysis displays trends, breakdowns, and cost-performance curves
- ✅ **Requirement 7.7**: Dashboard queries CloudWatch metrics and S3 results for visualizations
