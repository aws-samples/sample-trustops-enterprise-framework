# QuickSight Dashboard Configuration

This directory contains the QuickSight dashboard configuration for the TrustOps AWS Demo system.

## Overview

The QuickSight dashboards provide visualization and analysis capabilities for:

- **Baseline Evaluation**: Trust score distribution, latency histograms, and sample outputs
- **Comparative Evaluation**: Side-by-side metrics, improvement percentages, and statistical significance
- **Cost Analysis**: Cost trends, cost breakdowns by model, and cost-performance curves
- **Hallucination Analysis**: Hallucination rates over time and by category

## Architecture

### Data Flow

```
S3 Results → Glue Catalog → Athena → QuickSight Datasets → Dashboards
DynamoDB → QuickSight Data Source → Dashboards
CloudWatch Metrics → QuickSight → Dashboards
```

### Components

1. **Data Sources**:
   - S3 (via Athena): Evaluation results and comparison data
   - DynamoDB: Workflow and model metadata
   - CloudWatch: Metrics and logs

2. **Datasets**:
   - Baseline Evaluation Metrics
   - Comparative Evaluation Metrics
   - Cost Analysis
   - Hallucination Analysis

3. **Dashboards**:
   - Baseline Evaluation Dashboard
   - Comparative Evaluation Dashboard
   - Cost Analysis Dashboard
   - Hallucination Analysis Dashboard

## Prerequisites

### AWS Account Setup

1. **Enable QuickSight**:
   ```bash
   # Visit the QuickSight console
   https://quicksight.aws.amazon.com/
   
   # Sign up for QuickSight (if not already enabled)
   # Choose Enterprise Edition for advanced features
   ```

2. **Create QuickSight User**:
   - Go to QuickSight console
   - Navigate to "Manage QuickSight" → "Manage users"
   - Invite users or create service accounts

3. **Grant S3 Access**:
   - In QuickSight console, go to "Manage QuickSight" → "Security & permissions"
   - Under "QuickSight access to AWS services", click "Add or remove"
   - Select the S3 buckets: `{project}-results-{env}-{account-id}` and `{project}-datasets-{env}-{account-id}`
   - Click "Finish"

### Required Permissions

The QuickSight service role needs:
- S3: Read access to results and datasets buckets
- DynamoDB: Read access to workflows and models tables
- CloudWatch: Read access to metrics and logs
- Athena: Query execution permissions
- Glue: Read access to catalog

## Quick Start

### Automated Setup

Use the provided setup script for automated deployment:

```bash
# Set environment variables
export PROJECT_NAME=trustops
export ENVIRONMENT=dev
export AWS_REGION=us-east-1

# Run setup script
cd infrastructure/quicksight
./setup.sh
```

The script will:
1. Check prerequisites
2. Create S3 manifest file
3. Deploy Terraform resources (data sources, datasets, Glue catalog)
4. Configure dashboard permissions
5. Trigger initial data refresh
6. Print dashboard URLs

### Manual Setup

If you prefer manual setup or need to customize:

#### 1. Deploy Infrastructure

```bash
cd infrastructure/quicksight

# Initialize Terraform
terraform init

# Plan deployment
terraform plan \
    -var="project_name=trustops" \
    -var="environment=dev" \
    -var="aws_region=us-east-1"

# Apply deployment
terraform apply \
    -var="project_name=trustops" \
    -var="environment=dev" \
    -var="aws_region=us-east-1"
```

#### 2. Create S3 Manifest

Create a manifest file for QuickSight to discover S3 data:

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
aws s3 cp manifest.json s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>/quicksight/manifest.json
```

#### 3. Create Dashboards in QuickSight Console

1. **Navigate to QuickSight Console**:
   - Go to https://{region}.quicksight.aws.amazon.com/

2. **Create Analysis**:
   - Click "Analyses" → "New analysis"
   - Select dataset (e.g., "Baseline Evaluation Metrics")
   - Click "Create analysis"

3. **Add Visuals**:
   - Use the dashboard definitions in `dashboards.json` as a guide
   - Add visuals for each metric (histograms, bar charts, KPIs, etc.)
   - Configure filters and parameters

4. **Publish Dashboard**:
   - Click "Share" → "Publish dashboard"
   - Enter dashboard name
   - Click "Publish dashboard"

5. **Repeat for Other Dashboards**:
   - Comparative Evaluation Dashboard
   - Cost Analysis Dashboard
   - Hallucination Analysis Dashboard

## Dashboard Details

### 1. Baseline Evaluation Dashboard

**Purpose**: Monitor baseline model performance before fine-tuning

**Key Visuals**:
- Trust Score Distribution (Histogram)
- Trust Score by Category (Bar Chart)
- Latency Distribution (Histogram)
- Trust Score Trend Over Time (Line Chart)
- Trust Score Categories (Pie Chart)
- Sample Evaluation Results (Table)
- KPIs: Mean Trust Score, P95 Latency, Total Cost

**Filters**:
- Model ID
- Date Range
- Category

**Use Cases**:
- Establish baseline metrics before fine-tuning
- Identify categories with low trust scores
- Monitor latency and cost trends
- Review sample outputs for quality assessment

### 2. Comparative Evaluation Dashboard

**Purpose**: Compare baseline and fine-tuned model performance

**Key Visuals**:
- Trust Score Improvement (Bar Chart)
- Hallucination Rate Reduction (Bar Chart)
- Cost Delta per Query (Bar Chart)
- Latency Delta (Bar Chart)
- Improvement Trend Over Time (Line Chart)
- Statistical Significance (Pie Chart)
- Recommendations Distribution (Pie Chart)
- Detailed Comparison Table

**Filters**:
- Baseline Model ID
- Fine-tuned Model ID
- Date Range
- Recommendation

**Use Cases**:
- Quantify improvement from fine-tuning
- Make deployment decisions based on metrics
- Track improvement trends over multiple iterations
- Identify statistically significant improvements

### 3. Cost Analysis Dashboard

**Purpose**: Analyze cost-performance tradeoffs

**Key Visuals**:
- Cost Trend Over Time (Line Chart)
- Total Cost by Model (Bar Chart)
- Cost-Performance Curve (Scatter Plot)
- Cost Efficiency Score (Bar Chart)
- Token Usage by Model (Stacked Bar Chart)
- Cost per High-Trust Response (Bar Chart)
- Daily Cost Breakdown (Stacked Area Chart)
- KPIs: Total Cost, Avg Cost per Query, Avg Cost Efficiency

**Filters**:
- Model ID
- Date Range

**Use Cases**:
- Monitor cost trends and anomalies
- Compare cost efficiency across models
- Optimize cost-performance tradeoffs
- Project costs for production deployment

### 4. Hallucination Analysis Dashboard

**Purpose**: Track and analyze hallucination patterns

**Key Visuals**:
- Hallucination Rate Trend (Line Chart)
- Hallucination Rate by Category (Bar Chart)
- Hallucination Rate by Model (Bar Chart)
- Hallucination Severity Distribution (Pie Chart)
- Hallucination vs Trust Score (Scatter Plot)
- Hallucination Rate Heatmap
- Model Comparison by Category (Grouped Bar Chart)
- KPIs: Avg Hallucination Rate, High Severity Count

**Filters**:
- Model ID
- Category
- Date Range
- Hallucination Severity

**Use Cases**:
- Monitor hallucination rates over time
- Identify categories prone to hallucinations
- Compare hallucination rates across models
- Track improvement in hallucination reduction

## Data Refresh

### Automatic Refresh

Datasets are configured with daily refresh schedules at 6:00 AM UTC:

```hcl
schedule {
  refresh_type = "FULL_REFRESH"
  
  schedule_frequency {
    interval = "DAILY"
    time_of_the_day = "06:00"
  }
}
```

### Manual Refresh

To manually refresh a dataset:

```bash
# Using AWS CLI
aws quicksight create-ingestion \
    --aws-account-id {account-id} \
    --data-set-id trustops-baseline-evaluation-dev \
    --ingestion-id manual-refresh-$(date +%s) \
    --region us-east-1

# Or use the QuickSight console:
# 1. Go to Datasets
# 2. Select dataset
# 3. Click "Refresh now"
```

### Refresh Status

Check refresh status:

```bash
aws quicksight list-ingestions \
    --aws-account-id {account-id} \
    --data-set-id trustops-baseline-evaluation-dev \
    --region us-east-1
```

## Permissions and Sharing

### Dashboard Permissions

Configure who can view and edit dashboards:

```bash
# Grant view permissions
aws quicksight update-dashboard-permissions \
    --aws-account-id {account-id} \
    --dashboard-id trustops-baseline-evaluation-dev \
    --grant-permissions "Principal=arn:aws:quicksight:us-east-1:{account-id}:user/default/{username},Actions=quicksight:DescribeDashboard,quicksight:ListDashboardVersions,quicksight:QueryDashboard" \
    --region us-east-1
```

### Sharing Dashboards

1. **Via QuickSight Console**:
   - Open dashboard
   - Click "Share" → "Share dashboard"
   - Enter user emails
   - Set permissions (Viewer or Co-owner)
   - Click "Share"

2. **Via Email Reports**:
   - Open dashboard
   - Click "Share" → "Email report"
   - Configure schedule and recipients
   - Click "Send"

## Customization

### Adding Custom Visuals

1. Open analysis in QuickSight
2. Click "Add" → "Add visual"
3. Select visual type
4. Configure fields and formatting
5. Save and publish

### Creating Calculated Fields

Example: Cost per 1K tokens

```
({cost} / ({input_tokens} + {output_tokens})) * 1000
```

Example: Trust score category

```
ifelse({trust_score} >= 0.8, 'High', 
  ifelse({trust_score} >= 0.6, 'Medium', 'Low'))
```

### Adding Parameters

1. Click "Parameters" in left panel
2. Click "Create parameter"
3. Configure parameter (name, type, default value)
4. Use parameter in filters or calculated fields

## Troubleshooting

### Data Not Appearing

1. **Check S3 Data**:
   ```bash
   aws s3 ls s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>/evaluations/
   ```

2. **Verify Glue Catalog**:
   ```bash
   aws glue get-table \
       --database-name trustops_dev \
       --name evaluation_results
   ```

3. **Test Athena Query**:
   ```sql
   SELECT * FROM trustops_dev.evaluation_results LIMIT 10;
   ```

4. **Check Dataset Refresh Status**:
   ```bash
   aws quicksight list-ingestions \
       --aws-account-id {account-id} \
       --data-set-id trustops-baseline-evaluation-dev
   ```

### Permission Errors

1. **Verify QuickSight Service Role**:
   - Check IAM role has S3, DynamoDB, CloudWatch permissions
   - Verify trust relationship allows QuickSight service

2. **Check S3 Bucket Policy**:
   - Ensure QuickSight service role can read from buckets
   - Verify bucket is in allowed list in QuickSight settings

3. **Verify Athena Permissions**:
   - Check QuickSight can execute Athena queries
   - Verify Glue catalog access

### Slow Dashboard Performance

1. **Optimize SPICE Refresh**:
   - Use incremental refresh instead of full refresh
   - Schedule refreshes during off-peak hours

2. **Reduce Data Volume**:
   - Add date range filters
   - Aggregate data before importing to SPICE

3. **Optimize Athena Queries**:
   - Partition S3 data by date
   - Use columnar formats (Parquet) instead of JSON

## Cost Optimization

### QuickSight Pricing

- **Standard Edition**: $9/user/month
- **Enterprise Edition**: $18/user/month
- **SPICE Capacity**: $0.25/GB/month (first 10GB free per user)
- **Readers**: $0.30/session (max $5/month per reader)

### Cost Reduction Tips

1. **Use SPICE Efficiently**:
   - Import only necessary columns
   - Use appropriate data types
   - Compress data before import

2. **Optimize Refresh Schedules**:
   - Refresh only when data changes
   - Use incremental refresh for large datasets

3. **Share Dashboards**:
   - Use reader sessions for view-only access
   - Share dashboards instead of creating duplicates

## Monitoring and Alerts

### CloudWatch Metrics

Monitor QuickSight usage:

```bash
aws cloudwatch get-metric-statistics \
    --namespace AWS/QuickSight \
    --metric-name DataSetRefreshDuration \
    --dimensions Name=DataSetId,Value=trustops-baseline-evaluation-dev \
    --start-time 2024-01-01T00:00:00Z \
    --end-time 2024-01-02T00:00:00Z \
    --period 3600 \
    --statistics Average
```

### Email Alerts

Configure email alerts for:
- Trust score degradation
- High hallucination rates
- Cost anomalies
- Workflow failures

## Best Practices

1. **Data Quality**:
   - Validate data before importing to QuickSight
   - Use consistent data formats
   - Handle missing values appropriately

2. **Dashboard Design**:
   - Keep dashboards focused on specific use cases
   - Use consistent color schemes and formatting
   - Provide context with titles and descriptions

3. **Performance**:
   - Use SPICE for faster query performance
   - Optimize Athena queries with partitioning
   - Limit data volume with filters

4. **Security**:
   - Use row-level security for sensitive data
   - Regularly review and update permissions
   - Enable MFA for QuickSight users

5. **Maintenance**:
   - Monitor refresh failures
   - Update dashboards as requirements change
   - Archive unused dashboards

## Support and Resources

- **AWS QuickSight Documentation**: https://docs.aws.amazon.com/quicksight/
- **QuickSight Community**: https://repost.aws/tags/TA4ckwVwj6ShempRV_L95_Zg/amazon-quick-sight
- **TrustOps Documentation**: See main README.md

## Validation

This QuickSight configuration satisfies the following requirements:

- **Requirement 7.1**: Dashboard displays current evaluation status and recent runs
- **Requirement 7.2**: Baseline results show trust score distribution, latency, and sample outputs
- **Requirement 7.3**: Comparative results display side-by-side metrics with improvement percentages
- **Requirement 7.5**: Hallucination analysis visualizes rates over time and by category
- **Requirement 7.6**: Cost analysis displays trends, breakdowns, and cost-performance curves
- **Requirement 7.7**: Dashboard queries CloudWatch metrics and S3 results for visualizations

## Next Steps

After setting up QuickSight dashboards:

1. Run baseline and comparative evaluations to generate data
2. Verify data appears in dashboards
3. Customize dashboard layouts and visuals
4. Share dashboards with team members
5. Set up email reports and alerts
6. Monitor dashboard usage and performance
