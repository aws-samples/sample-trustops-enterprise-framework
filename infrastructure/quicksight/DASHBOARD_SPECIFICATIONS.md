# QuickSight Dashboard Specifications

This document provides detailed specifications for each QuickSight dashboard in the TrustOps system.

## Dashboard Overview

| Dashboard | Purpose | Primary Dataset | Key Metrics |
|-----------|---------|-----------------|-------------|
| Baseline Evaluation | Monitor baseline model performance | Baseline Evaluation Metrics | Trust Score, Latency, Cost |
| Comparative Evaluation | Compare baseline vs fine-tuned models | Comparative Evaluation Metrics | Improvement %, Statistical Significance |
| Cost Analysis | Analyze cost-performance tradeoffs | Cost Analysis | Cost Trends, Cost Efficiency |
| Hallucination Analysis | Track hallucination patterns | Hallucination Analysis | Hallucination Rate, Severity |

## 1. Baseline Evaluation Dashboard

### Purpose
Establish baseline metrics for foundation models before fine-tuning to enable comparison and track performance over time.

### Target Audience
- Data Scientists evaluating model performance
- ML Engineers monitoring model quality
- Product Managers reviewing model capabilities

### Key Questions Answered
1. What is the overall trust score distribution for the baseline model?
2. Which categories have the lowest trust scores?
3. What is the typical response latency?
4. How much does baseline evaluation cost?
5. What percentage of responses are high-trust?

### Visual Specifications

#### 1.1 Trust Score Distribution (Histogram)
- **Type**: Histogram
- **Data Field**: `trust_score`
- **Bins**: 20
- **X-Axis**: Trust Score (0.0 - 1.0)
- **Y-Axis**: Count
- **Color**: Single color (blue)
- **Tooltip**: Trust score range, count, percentage
- **Purpose**: Show distribution of trust scores across all evaluations

#### 1.2 Mean Trust Score by Category (Bar Chart)
- **Type**: Vertical bar chart
- **X-Axis**: `category`
- **Y-Axis**: `trust_score` (Average)
- **Sort**: Descending by trust score
- **Color**: Gradient based on trust score (red to green)
- **Tooltip**: Category, mean trust score, sample count
- **Purpose**: Identify categories with low trust scores

#### 1.3 Response Latency Distribution (Histogram)
- **Type**: Histogram
- **Data Field**: `latency_ms`
- **Bins**: 20
- **X-Axis**: Latency (ms)
- **Y-Axis**: Count
- **Color**: Single color (orange)
- **Tooltip**: Latency range, count, percentage
- **Purpose**: Understand latency characteristics

#### 1.4 Trust Score Trend Over Time (Line Chart)
- **Type**: Line chart
- **X-Axis**: `date`
- **Y-Axis**: `trust_score` (Average)
- **Line**: Solid, blue
- **Markers**: Show data points
- **Tooltip**: Date, mean trust score, sample count
- **Purpose**: Track trust score changes over time

#### 1.5 Trust Score Categories (Pie Chart)
- **Type**: Pie chart
- **Field**: `trust_score_category`
- **Categories**: High (≥0.8), Medium (0.6-0.8), Low (<0.6)
- **Colors**: Green (High), Yellow (Medium), Red (Low)
- **Labels**: Show percentage and count
- **Purpose**: Quick overview of trust score distribution

#### 1.6 Sample Evaluation Results (Table)
- **Type**: Table
- **Columns**:
  - `example_id`: Example identifier
  - `trust_score`: Trust score (3 decimals)
  - `hallucination_rate`: Hallucination rate (percentage)
  - `latency_ms`: Latency (1 decimal)
  - `cost`: Cost (4 decimals)
  - `passed`: Pass/Fail indicator
- **Sort**: By `trust_score` descending
- **Rows**: 50
- **Purpose**: Drill down into individual results

#### 1.7 KPI: Mean Trust Score
- **Type**: KPI
- **Field**: `trust_score`
- **Aggregation**: Average
- **Format**: 0.000
- **Comparison**: Previous period (if available)
- **Color**: Green if ≥0.8, Yellow if ≥0.6, Red if <0.6

#### 1.8 KPI: P95 Latency
- **Type**: KPI
- **Field**: `latency_ms`
- **Aggregation**: 95th percentile
- **Format**: 0.0 ms
- **Comparison**: Previous period
- **Color**: Green if <500ms, Yellow if <1000ms, Red if ≥1000ms

#### 1.9 KPI: Total Cost
- **Type**: KPI
- **Field**: `cost`
- **Aggregation**: Sum
- **Format**: $0.00
- **Comparison**: Previous period
- **Color**: Neutral

### Filters
1. **Model ID** (Dropdown): Filter by specific model
2. **Date Range** (Date range picker): Filter by evaluation date
3. **Category** (Multi-select): Filter by evaluation category

### Layout
```
+------------------+------------------+------------------+
|  Mean Trust Score|   P95 Latency   |   Total Cost     |
|      (KPI)       |      (KPI)      |      (KPI)       |
+------------------+------------------+------------------+
|                                                        |
|         Trust Score Distribution (Histogram)          |
|                                                        |
+---------------------------+---------------------------+
|                           |                           |
|  Mean Trust Score by      |  Trust Score Categories   |
|  Category (Bar Chart)     |  (Pie Chart)              |
|                           |                           |
+---------------------------+---------------------------+
|                                                        |
|    Response Latency Distribution (Histogram)          |
|                                                        |
+-------------------------------------------------------+
|                                                        |
|      Trust Score Trend Over Time (Line Chart)         |
|                                                        |
+-------------------------------------------------------+
|                                                        |
|       Sample Evaluation Results (Table)               |
|                                                        |
+-------------------------------------------------------+
```

## 2. Comparative Evaluation Dashboard

### Purpose
Compare baseline and fine-tuned model performance to make deployment decisions based on quantifiable improvements.

### Target Audience
- Data Scientists evaluating fine-tuning effectiveness
- ML Engineers making deployment decisions
- Stakeholders reviewing ROI of fine-tuning

### Key Questions Answered
1. How much did trust scores improve after fine-tuning?
2. Was the hallucination rate reduced?
3. What is the cost impact of fine-tuning?
4. Are improvements statistically significant?
5. Should we deploy the fine-tuned model?

### Visual Specifications

#### 2.1 Trust Score Improvement (Bar Chart)
- **Type**: Vertical bar chart
- **X-Axis**: `finetuned_model_id`
- **Y-Axis**: `trust_score_improvement` (percentage points)
- **Color**: By `improvement_category` (Significant, Moderate, Minimal)
- **Tooltip**: Model ID, improvement, category
- **Reference Line**: 0% (no improvement)

#### 2.2 Hallucination Rate Reduction (Bar Chart)
- **Type**: Vertical bar chart
- **X-Axis**: `finetuned_model_id`
- **Y-Axis**: `hallucination_reduction` (percentage points)
- **Color**: Green if positive, red if negative
- **Format**: 0.0%
- **Reference Line**: 0% (no change)

#### 2.3 Cost Delta per Query (Bar Chart)
- **Type**: Vertical bar chart
- **X-Axis**: `finetuned_model_id`
- **Y-Axis**: `cost_delta_per_query`
- **Color**: Red if increased, green if decreased
- **Format**: $0.0000
- **Reference Line**: $0 (no change)

#### 2.4 Latency Delta (Bar Chart)
- **Type**: Vertical bar chart
- **X-Axis**: `finetuned_model_id`
- **Y-Axis**: `latency_delta_ms`
- **Color**: Red if increased, green if decreased
- **Format**: 0.0 ms
- **Reference Line**: 0ms (no change)

#### 2.5 Improvement Trend Over Time (Line Chart)
- **Type**: Line chart
- **X-Axis**: `date`
- **Y-Axis**: `trust_score_improvement` (Average)
- **Lines**: One per model
- **Markers**: Show data points
- **Reference Line**: 0% (no improvement)

#### 2.6 Statistical Significance (Pie Chart)
- **Type**: Pie chart
- **Field**: `statistical_significance`
- **Categories**: Significant, Not Significant
- **Colors**: Green (Significant), Gray (Not Significant)
- **Labels**: Show percentage and count

#### 2.7 Recommendations Distribution (Pie Chart)
- **Type**: Pie chart
- **Field**: `recommendation`
- **Categories**: Deploy, Iterate, Reject
- **Colors**: Green (Deploy), Yellow (Iterate), Red (Reject)
- **Labels**: Show percentage and count

#### 2.8 Detailed Comparison Table
- **Type**: Table
- **Columns**:
  - `baseline_model_id`
  - `finetuned_model_id`
  - `trust_score_improvement` (%)
  - `hallucination_reduction` (%)
  - `cost_delta_percentage` (%)
  - `statistical_significance` (Yes/No)
  - `recommendation`
- **Sort**: By `trust_score_improvement` descending
- **Conditional Formatting**: Color code recommendations

#### 2.9 KPI: Avg Trust Score Improvement
- **Type**: KPI
- **Field**: `trust_score_improvement`
- **Aggregation**: Average
- **Format**: +0.0%
- **Color**: Green if positive, red if negative

#### 2.10 KPI: Avg Hallucination Reduction
- **Type**: KPI
- **Field**: `hallucination_reduction`
- **Aggregation**: Average
- **Format**: +0.0%
- **Color**: Green if positive, red if negative

### Filters
1. **Baseline Model ID** (Dropdown)
2. **Fine-tuned Model ID** (Dropdown)
3. **Date Range** (Date range picker)
4. **Recommendation** (Multi-select): Deploy, Iterate, Reject

### Layout
```
+---------------------------+---------------------------+
| Avg Trust Score Improvement| Avg Hallucination Reduction|
|         (KPI)             |         (KPI)             |
+---------------------------+---------------------------+
|                           |                           |
| Trust Score Improvement   | Hallucination Reduction   |
| (Bar Chart)               | (Bar Chart)               |
|                           |                           |
+---------------------------+---------------------------+
|                           |                           |
| Cost Delta per Query      | Latency Delta             |
| (Bar Chart)               | (Bar Chart)               |
|                           |                           |
+---------------------------+---------------------------+
|                                                        |
|    Improvement Trend Over Time (Line Chart)           |
|                                                        |
+---------------------------+---------------------------+
|                           |                           |
| Statistical Significance  | Recommendations           |
| (Pie Chart)               | (Pie Chart)               |
|                           |                           |
+-------------------------------------------------------+
|                                                        |
|       Detailed Comparison Table                       |
|                                                        |
+-------------------------------------------------------+
```

## 3. Cost Analysis Dashboard

### Purpose
Analyze cost-performance tradeoffs to optimize model selection and deployment decisions.

### Target Audience
- Finance teams tracking AI costs
- ML Engineers optimizing cost efficiency
- Product Managers balancing cost and quality

### Key Questions Answered
1. What are the cost trends over time?
2. Which models are most cost-efficient?
3. What is the cost per high-trust response?
4. How do costs scale with usage?
5. What are the projected costs for production?

### Visual Specifications

#### 3.1 Cost Trend Over Time (Line Chart)
- **Type**: Line chart
- **X-Axis**: `date`
- **Y-Axis**: `cost` (Sum)
- **Format**: $0.00
- **Trend Line**: Show moving average
- **Tooltip**: Date, total cost, query count

#### 3.2 Total Cost by Model (Bar Chart)
- **Type**: Vertical bar chart
- **X-Axis**: `model_id`
- **Y-Axis**: `cost` (Sum)
- **Sort**: Descending by cost
- **Format**: $0.00
- **Color**: Gradient by cost

#### 3.3 Cost-Performance Curve (Scatter Plot)
- **Type**: Scatter plot
- **X-Axis**: `cost`
- **Y-Axis**: `trust_score`
- **Size**: `total_tokens`
- **Color**: `model_id`
- **Tooltip**: Model, cost, trust score, tokens
- **Purpose**: Identify optimal cost-performance balance

#### 3.4 Cost Efficiency Score (Bar Chart)
- **Type**: Vertical bar chart
- **X-Axis**: `model_id`
- **Y-Axis**: `cost_efficiency_score` (Average)
- **Sort**: Descending by efficiency
- **Format**: 0.00
- **Color**: Green for high efficiency

#### 3.5 Token Usage by Model (Stacked Bar Chart)
- **Type**: Stacked bar chart
- **X-Axis**: `model_id`
- **Y-Axis**: `input_tokens` and `output_tokens` (Sum)
- **Colors**: Blue (input), Orange (output)
- **Tooltip**: Model, input tokens, output tokens, total

#### 3.6 Cost per High-Trust Response (Bar Chart)
- **Type**: Vertical bar chart
- **X-Axis**: `model_id`
- **Y-Axis**: `cost_per_high_trust_response` (Average)
- **Format**: $0.0000
- **Sort**: Ascending by cost
- **Purpose**: Identify most cost-effective models for quality

#### 3.7 Daily Cost Breakdown (Stacked Area Chart)
- **Type**: Stacked area chart
- **X-Axis**: `date`
- **Y-Axis**: `cost` (Sum)
- **Stack By**: `model_id`
- **Format**: $0.00
- **Purpose**: Track cost distribution over time

#### 3.8 KPI: Total Cost
- **Type**: KPI
- **Field**: `cost`
- **Aggregation**: Sum
- **Format**: $0.00
- **Comparison**: Previous period

#### 3.9 KPI: Avg Cost per Query
- **Type**: KPI
- **Field**: `cost`
- **Aggregation**: Average
- **Format**: $0.0000
- **Comparison**: Previous period

#### 3.10 KPI: Avg Cost Efficiency
- **Type**: KPI
- **Field**: `cost_efficiency_score`
- **Aggregation**: Average
- **Format**: 0.00
- **Comparison**: Previous period

### Filters
1. **Model ID** (Multi-select)
2. **Date Range** (Date range picker)

### Layout
```
+------------------+------------------+------------------+
|   Total Cost     | Avg Cost/Query  | Avg Cost Efficiency|
|      (KPI)       |      (KPI)      |      (KPI)       |
+------------------+------------------+------------------+
|                           |                           |
| Cost Trend Over Time      | Total Cost by Model       |
| (Line Chart)              | (Bar Chart)               |
|                           |                           |
+---------------------------+---------------------------+
|                                                        |
|       Cost-Performance Curve (Scatter Plot)           |
|                                                        |
+---------------------------+---------------------------+
|                           |                           |
| Cost Efficiency Score     | Token Usage by Model      |
| (Bar Chart)               | (Stacked Bar Chart)       |
|                           |                           |
+---------------------------+---------------------------+
|                           |                           |
| Cost per High-Trust       | Daily Cost Breakdown      |
| Response (Bar Chart)      | (Stacked Area Chart)      |
|                           |                           |
+-------------------------------------------------------+
```

## 4. Hallucination Analysis Dashboard

### Purpose
Track and analyze hallucination patterns to improve model reliability and reduce factual errors.

### Target Audience
- Data Scientists monitoring model reliability
- ML Engineers improving model grounding
- Compliance teams tracking accuracy

### Key Questions Answered
1. What is the overall hallucination rate?
2. Which categories have the highest hallucination rates?
3. How do hallucination rates vary by model?
4. Are hallucination rates improving over time?
5. What is the relationship between hallucinations and trust scores?

### Visual Specifications

#### 4.1 Hallucination Rate Trend (Line Chart)
- **Type**: Line chart
- **X-Axis**: `date`
- **Y-Axis**: `hallucination_rate` (Average)
- **Format**: 0.0%
- **Trend Line**: Show moving average
- **Reference Line**: Target rate (e.g., 5%)

#### 4.2 Hallucination Rate by Category (Bar Chart)
- **Type**: Vertical bar chart
- **X-Axis**: `category`
- **Y-Axis**: `hallucination_rate` (Average)
- **Sort**: Descending by rate
- **Format**: 0.0%
- **Color**: Red gradient by severity

#### 4.3 Hallucination Rate by Model (Bar Chart)
- **Type**: Vertical bar chart
- **X-Axis**: `model_id`
- **Y-Axis**: `hallucination_rate` (Average)
- **Sort**: Descending by rate
- **Format**: 0.0%
- **Color**: By model

#### 4.4 Hallucination Severity Distribution (Pie Chart)
- **Type**: Pie chart
- **Field**: `hallucination_severity`
- **Categories**: High (≥30%), Medium (10-30%), Low (<10%)
- **Colors**: Red (High), Yellow (Medium), Green (Low)
- **Labels**: Show percentage and count

#### 4.5 Hallucination vs Trust Score (Scatter Plot)
- **Type**: Scatter plot
- **X-Axis**: `hallucination_rate`
- **Y-Axis**: `trust_score`
- **Color**: `model_id`
- **Tooltip**: Model, hallucination rate, trust score
- **Purpose**: Show inverse relationship

#### 4.6 Hallucination Rate Heatmap
- **Type**: Heatmap
- **X-Axis**: `date`
- **Y-Axis**: `category`
- **Value**: `hallucination_rate` (Average)
- **Color Scale**: Green (low) to Red (high)
- **Purpose**: Identify patterns over time and categories

#### 4.7 Model Comparison by Category (Grouped Bar Chart)
- **Type**: Grouped bar chart
- **X-Axis**: `category`
- **Y-Axis**: `hallucination_rate` (Average)
- **Group By**: `model_id`
- **Format**: 0.0%
- **Purpose**: Compare models across categories

#### 4.8 KPI: Avg Hallucination Rate
- **Type**: KPI
- **Field**: `hallucination_rate`
- **Aggregation**: Average
- **Format**: 0.0%
- **Comparison**: Previous period
- **Color**: Green if <5%, Yellow if <10%, Red if ≥10%

#### 4.9 KPI: High Severity Count
- **Type**: KPI
- **Field**: `hallucination_severity`
- **Filter**: High
- **Aggregation**: Count
- **Comparison**: Previous period
- **Color**: Red if increasing, green if decreasing

### Filters
1. **Model ID** (Multi-select)
2. **Category** (Multi-select)
3. **Date Range** (Date range picker)
4. **Hallucination Severity** (Multi-select): High, Medium, Low

### Layout
```
+---------------------------+---------------------------+
| Avg Hallucination Rate    | High Severity Count       |
|         (KPI)             |         (KPI)             |
+---------------------------+---------------------------+
|                                                        |
|    Hallucination Rate Trend Over Time (Line Chart)    |
|                                                        |
+---------------------------+---------------------------+
|                           |                           |
| Hallucination by Category | Hallucination by Model    |
| (Bar Chart)               | (Bar Chart)               |
|                           |                           |
+---------------------------+---------------------------+
|                           |                           |
| Severity Distribution     | Hallucination vs Trust    |
| (Pie Chart)               | Score (Scatter Plot)      |
|                           |                           |
+-------------------------------------------------------+
|                                                        |
|       Hallucination Rate Heatmap                      |
|                                                        |
+-------------------------------------------------------+
|                                                        |
|    Model Comparison by Category (Grouped Bar Chart)   |
|                                                        |
+-------------------------------------------------------+
```

## Dashboard Refresh Schedule

All dashboards are configured with daily refresh at 6:00 AM UTC:

- **Baseline Evaluation Dashboard**: Daily at 6:00 AM UTC
- **Comparative Evaluation Dashboard**: Daily at 6:00 AM UTC
- **Cost Analysis Dashboard**: Daily at 6:00 AM UTC
- **Hallucination Analysis Dashboard**: Daily at 6:00 AM UTC

Manual refresh can be triggered at any time via the QuickSight console or AWS CLI.

## Color Schemes

### Trust Score Colors
- **High (≥0.8)**: #2ECC71 (Green)
- **Medium (0.6-0.8)**: #F39C12 (Yellow)
- **Low (<0.6)**: #E74C3C (Red)

### Improvement Colors
- **Positive**: #2ECC71 (Green)
- **Neutral**: #95A5A6 (Gray)
- **Negative**: #E74C3C (Red)

### Severity Colors
- **High**: #E74C3C (Red)
- **Medium**: #F39C12 (Yellow)
- **Low**: #2ECC71 (Green)

### Model Colors
Use distinct colors for different models:
- Model 1: #3498DB (Blue)
- Model 2: #9B59B6 (Purple)
- Model 3: #E67E22 (Orange)
- Model 4: #1ABC9C (Teal)

## Accessibility

All dashboards follow accessibility best practices:

- **Color Contrast**: Minimum 4.5:1 ratio for text
- **Alternative Text**: All visuals have descriptive alt text
- **Keyboard Navigation**: Full keyboard support
- **Screen Reader Support**: Proper ARIA labels
- **Color Independence**: Information not conveyed by color alone

## Performance Optimization

- **SPICE**: All datasets use SPICE for fast query performance
- **Aggregation**: Pre-aggregate data where possible
- **Partitioning**: S3 data partitioned by date
- **Incremental Refresh**: Use incremental refresh for large datasets
- **Query Optimization**: Optimize Athena queries with appropriate filters

## Validation

These dashboard specifications satisfy the following requirements:

- ✅ **Requirement 7.1**: Dashboard displays current evaluation status and recent runs
- ✅ **Requirement 7.2**: Baseline results show trust score distribution, latency, and sample outputs
- ✅ **Requirement 7.3**: Comparative results display side-by-side metrics with improvement percentages
- ✅ **Requirement 7.5**: Hallucination analysis visualizes rates over time and by category
- ✅ **Requirement 7.6**: Cost analysis displays trends, breakdowns, and cost-performance curves
- ✅ **Requirement 7.7**: Dashboard queries CloudWatch metrics and S3 results for visualizations
