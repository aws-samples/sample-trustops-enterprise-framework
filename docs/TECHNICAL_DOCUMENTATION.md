# TrustOps AWS Demo - Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Interactions](#component-interactions)
3. [Data Models and Interfaces](#data-models-and-interfaces)
4. [Trust Scoring Algorithm](#trust-scoring-algorithm)
5. [Hallucination Detection](#hallucination-detection)
6. [Cost Calculation](#cost-calculation)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)

## Architecture Overview

TrustOps is a trust-first fine-tuning and evaluation framework for enterprise GenAI that leverages AWS managed services to provide quantifiable trust scoring, automated model comparison, hallucination detection, and cost-performance optimization.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│                    (CLI / Dashboard / API)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                    Orchestration Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Step         │  │ Evaluation   │  │ Fine-Tuning  │         │
│  │ Functions    │  │ Orchestrator │  │ Orchestrator │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                      Processing Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Trust        │  │ Metrics      │  │ Workflow     │         │
│  │ Scoring      │  │ Aggregator   │  │ Manager      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                      AWS Services Layer                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Bedrock  │ │ S3       │ │Bedrock KB│ │ DynamoDB │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│  │CloudWatch│ │ Lambda   │ │QuickSight│                       │
│  └──────────┘ └──────────┘ └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### Core Design Principles


- **Trust-First**: Every model output includes a quantifiable trust score (0-1 scale)
- **Comparison-Driven**: Always measure against baseline before deployment decisions
- **Cost-Aware**: Track and optimize cost alongside quality metrics
- **Automated**: Minimize manual evaluation through systematic testing
- **Auditable**: Maintain complete lineage from data to deployment decision
- **Serverless**: Leverage AWS managed services for scalability and operational efficiency

### AWS Service Responsibilities

| Service | Purpose | Key Responsibilities |
|---------|---------|---------------------|
| **AWS Bedrock** | Model hosting and fine-tuning | - Host foundation models (Claude, Llama, Titan)<br>- Execute fine-tuning jobs<br>- Provide inference API<br>- Manage model versions |
| **Amazon S3** | Data storage | - Store training/evaluation datasets with versioning<br>- Store evaluation results and workflow manifests<br>- Archive historical data with lifecycle policies |
| **AWS Lambda** | Serverless compute | - Orchestrate evaluation workflows<br>- Calculate trust scores in real-time<br>- Validate data quality<br>- Aggregate results |
| **Bedrock Knowledge Bases** | Semantic retrieval | - Ingest and index source documents<br>- Calculate semantic similarity<br>- Detect hallucinations via grounding analysis |
| **Amazon DynamoDB** | Metadata storage | - Store workflow metadata and status<br>- Track model versions<br>- Maintain trust score history |
| **Amazon CloudWatch** | Monitoring and logging | - Collect trust score metrics<br>- Monitor performance (latency, cost, errors)<br>- Log workflow events for audit trail |
| **AWS Step Functions** | Workflow orchestration | - Orchestrate multi-step evaluation workflows<br>- Handle error recovery and retries<br>- Coordinate parallel model evaluations |
| **Amazon QuickSight** | Visualization | - Visualize trust metrics and trends<br>- Display baseline vs. fine-tuned comparisons<br>- Show cost-performance tradeoffs |


## Component Interactions

### Baseline Evaluation Flow

```
User → CLI/API
  ↓
Step Functions (baseline_evaluation_workflow)
  ↓
Evaluation Orchestrator Lambda
  ↓
1. Load dataset from S3
2. Validate dataset format
3. For each prompt:
   ├─→ Invoke Foundation Model (Bedrock)
   ├─→ Calculate Trust Score (Trust Scoring Lambda)
   └─→ Store individual result (S3)
4. Aggregate metrics (MetricsAggregator)
5. Store aggregate results (S3 + DynamoDB)
6. Log metrics (CloudWatch)
  ↓
Results available in Dashboard (QuickSight)
```

### Fine-Tuning Flow

```
User → CLI/API
  ↓
Step Functions (fine_tuning_workflow)
  ↓
Fine-Tuning Orchestrator Lambda
  ↓
1. Load training data from S3
2. Validate data quality (DataValidator)
   ├─→ Format validation
   ├─→ Quality checks
   └─→ Return errors if invalid
3. Create fine-tuning job (Bedrock)
4. Poll job status (exponential backoff)
5. Store fine-tuned model metadata (DynamoDB)
6. Calculate training costs
7. Log progress (CloudWatch)
  ↓
Fine-tuned model ready for evaluation
```

### Comparative Evaluation Flow

```
User → CLI/API
  ↓
Step Functions (comparative_evaluation_workflow)
  ↓
Evaluation Orchestrator Lambda
  ↓
1. Load dataset from S3
2. For each prompt (parallel):
   ├─→ Invoke Baseline Model (Bedrock)
   ├─→ Invoke Fine-Tuned Model (Bedrock)
   ├─→ Calculate Trust Scores (both)
   └─→ Calculate Semantic Similarity (Bedrock Knowledge Bases)
3. Compute improvement metrics
4. Calculate hallucination rate reduction
5. Generate comparison report
6. Store results (S3 + DynamoDB)
  ↓
Comparison report available in Dashboard
```

### Foundation Model Comparison Flow

```
User → CLI/API
  ↓
Step Functions (foundation_model_comparison_workflow)
  ↓
Evaluation Orchestrator Lambda
  ↓
1. Validate model IDs
   ├─→ Check model IDs are valid Bedrock identifiers
   ├─→ Verify models are different
   └─→ Retrieve model metadata (pricing, family)
2. Load dataset from S3
3. Calculate dataset checksum for reproducibility
4. For each prompt (parallel):
   ├─→ Invoke Model 1 (Bedrock) with identical inference params
   ├─→ Invoke Model 2 (Bedrock) with identical inference params
   ├─→ Calculate Trust Scores (both) using TrustScoringEngine
   ├─→ Detect Hallucinations (both) using TrustScoringEngine
   └─→ Record Metrics (latency, tokens, cost)
5. Aggregate metrics for both models
   ├─→ Calculate means, medians, std devs
   ├─→ Calculate hallucination rates
   └─→ Calculate total costs
6. Compute comparison metrics
   ├─→ Trust score improvement (Model 2 - Model 1)
   ├─→ Hallucination reduction (Model 1 rate - Model 2 rate)
   ├─→ Cost delta (Model 2 - Model 1)
   ├─→ Latency delta (Model 2 - Model 1)
   └─→ Statistical significance testing
7. Generate recommendation
   ├─→ DEPLOY_MODEL_1: Model 1 significantly better
   ├─→ DEPLOY_MODEL_2: Model 2 significantly better
   └─→ ITERATE: No significant difference
8. Store results (S3 + DynamoDB)
  ↓
Comparison report available in Dashboard
```

### Foundation Model Comparison Architecture

The Foundation Model Comparison feature extends TrustOps to enable comparison of any two AWS Bedrock foundation models. It reuses all existing TrustOps components and follows the established orchestrator pattern.

#### Component Integration

```
┌─────────────────────────────────────────────────────────────────┐
│              Foundation Model Comparison Layer                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Foundation Model Comparison Orchestrator                 │  │
│  │  - Model ID validation                                    │  │
│  │  - Parallel model evaluation                              │  │
│  │  - Comparison metrics calculation                         │  │
│  │  - Recommendation generation                              │  │
│  └────┬──────────────┬──────────────┬─────────────────┬──────┘  │
└───────┼──────────────┼──────────────┼─────────────────┼─────────┘
        │              │              │                 │
        ▼              ▼              ▼                 ▼
┌──────────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────────┐
│ REUSED:      │ │ REUSED:  │ │ REUSED:     │ │ REUSED:      │
│ Bedrock      │ │ Trust    │ │ Workflow    │ │ S3 Storage   │
│ Client       │ │ Scoring  │ │ Manager     │ │ Manager      │
│              │ │ Engine   │ │             │ │              │
└──────────────┘ └──────────┘ └─────────────┘ └──────────────┘
```

#### Data Flow: Input to Recommendation

```
1. INPUT VALIDATION
   ┌─────────────────────────────────────────────────────────┐
   │ User Input:                                             │
   │ - model_id_1: "anthropic.claude-3-haiku-20240307-v1:0" │
   │ - model_id_2: "amazon.titan-text-express-v1"           │
   │ - dataset_s3_uri: "s3://bucket/eval-dataset.json"      │
   │ - workflow_id: "compare-20240115-abc123"               │
   │ - inference_params: {temperature: 0.7, max_tokens: 2048}│
   └─────────────────────────────────────────────────────────┘
                            ↓
   ┌─────────────────────────────────────────────────────────┐
   │ Model Validator (NEW):                                  │
   │ ✓ Validate model IDs are valid Bedrock identifiers     │
   │ ✓ Verify models are different                          │
   │ ✓ Retrieve model metadata (pricing, family)            │
   └─────────────────────────────────────────────────────────┘
                            ↓
2. WORKFLOW INITIALIZATION
   ┌─────────────────────────────────────────────────────────┐
   │ WorkflowManager (REUSED):                               │
   │ - Create workflow record in DynamoDB                    │
   │ - Set status: "running"                                 │
   │ - Store configuration and metadata                      │
   └─────────────────────────────────────────────────────────┘
                            ↓
3. DATASET LOADING
   ┌─────────────────────────────────────────────────────────┐
   │ S3StorageManager (REUSED):                              │
   │ - Load evaluation dataset from S3                       │
   │ - Validate dataset format (EvaluationDataset)           │
   │ - Calculate SHA-256 checksum for reproducibility        │
   └─────────────────────────────────────────────────────────┘
                            ↓
4. PARALLEL MODEL EVALUATION
   ┌─────────────────────────────────────────────────────────┐
   │ For each example in dataset:                            │
   │                                                         │
   │ Model 1 Evaluation          Model 2 Evaluation         │
   │ ┌─────────────────┐         ┌─────────────────┐       │
   │ │ BedrockClient   │         │ BedrockClient   │       │
   │ │ (REUSED)        │         │ (REUSED)        │       │
   │ │ - Invoke model  │         │ - Invoke model  │       │
   │ │ - Record tokens │         │ - Record tokens │       │
   │ │ - Record latency│         │ - Record latency│       │
   │ └────────┬────────┘         └────────┬────────┘       │
   │          ↓                           ↓                 │
   │ ┌─────────────────┐         ┌─────────────────┐       │
   │ │ TrustScoring    │         │ TrustScoring    │       │
   │ │ Engine (REUSED) │         │ Engine (REUSED) │       │
   │ │ - 5 components  │         │ - 5 components  │       │
   │ │ - Hallucination │         │ - Hallucination │       │
   │ │   detection     │         │   detection     │       │
   │ └────────┬────────┘         └────────┬────────┘       │
   │          ↓                           ↓                 │
   │ ┌─────────────────┐         ┌─────────────────┐       │
   │ │ EvaluationResult│         │ EvaluationResult│       │
   │ │ (REUSED)        │         │ (REUSED)        │       │
   │ └─────────────────┘         └─────────────────┘       │
   └─────────────────────────────────────────────────────────┘
                            ↓
5. METRICS AGGREGATION
   ┌─────────────────────────────────────────────────────────┐
   │ MetricsAggregator (REUSED):                             │
   │                                                         │
   │ Model 1 Metrics:            Model 2 Metrics:           │
   │ - Mean trust score: 0.72    - Mean trust score: 0.85   │
   │ - Hallucination rate: 0.25  - Hallucination rate: 0.12 │
   │ - Total cost: $12.50        - Total cost: $15.20       │
   │ - Mean latency: 850ms       - Mean latency: 920ms      │
   │ - Std dev: 0.15             - Std dev: 0.10            │
   └─────────────────────────────────────────────────────────┘
                            ↓
6. COMPARISON METRICS + STATISTICAL SIGNIFICANCE
   ┌─────────────────────────────────────────────────────────┐
   │ MetricsAggregator + StatisticalTestEngine:              │
   │                                                         │
   │ Improvement Metrics:                                    │
   │ - Trust score improvement: +0.13 (18% improvement)      │
   │ - Hallucination reduction: 0.13 (52% reduction)         │
   │ - Cost delta: +$2.70 (22% increase)                     │
   │ - Latency delta: +70ms (8% increase)                    │
   │                                                         │
   │ Statistical Testing (paired, per-example vectors):      │
   │ - Test selection: n=250 >= 20, primary = paired t-test  │
   │ - p_value_ttest: 0.0003 (significant at p < 0.05)      │
   │ - p_value_wilcoxon: 0.0005 (also computed)             │
   │ - 95% CI: [-0.17, -0.09] (does not contain 0)          │
   │ - test_type_used: "ttest"                              │
   │                                                         │
   │ Break-Even Analysis (BreakEvenCalculator):              │
   │ - Cost premium/query: $0.0108                           │
   │ - Hallucination reduction: 0.13                         │
   │ - Remediation cost: $50/hallucination                   │
   │ - Break-even volume: 2 queries                          │
   │ - Monthly savings at 10K queries: $64,892               │
   └─────────────────────────────────────────────────────────┘
                            ↓
7. DEPLOY/ITERATE/REJECT RECOMMENDATION
   ┌─────────────────────────────────────────────────────────┐
   │ Recommendation Logic:                                   │
   │                                                         │
   │ IF trust_improvement >= 10% AND hallucination_reduction │
   │    >= 5% AND cost_increase <= 20% AND p_value < 0.05:  │
   │   recommendation = "DEPLOY"                             │
   │   justification = "Significant trust improvement (13%)  │
   │                    and hallucination reduction (13%)     │
   │                    with acceptable cost (22%). p=0.0003" │
   │                                                         │
   │ ELSE IF trust_improvement >= 5% OR (improvement > 0    │
   │    AND NOT statistically_significant):                  │
   │   recommendation = "ITERATE"                            │
   │   justification = "Moderate improvement or results not  │
   │                    statistically significant. Consider   │
   │                    additional fine-tuning iterations."   │
   │                                                         │
   │ ELSE:                                                   │
   │   recommendation = "REJECT"                             │
   │   justification = "Insufficient improvement in trust or │
   │                    hallucination metrics. Fine-tuning    │
   │                    did not provide meaningful benefits." │
   └─────────────────────────────────────────────────────────┘
                            ↓
8. RESULTS STORAGE
   ┌─────────────────────────────────────────────────────────┐
   │ S3StorageManager (REUSED):                              │
   │ - Store comparison report (JSON)                        │
   │ - Store individual evaluation results                   │
   │ - Organize by workflow ID hierarchy                     │
   │                                                         │
   │ WorkflowManager (REUSED):                               │
   │ - Update workflow status: "completed"                   │
   │ - Store results S3 URI                                  │
   │ - Log completion event to CloudWatch                    │
   └─────────────────────────────────────────────────────────┘
                            ↓
9. OUTPUT
   ┌─────────────────────────────────────────────────────────┐
   │ ComparativeEvaluationResult:                            │
   │ {                                                       │
   │   "workflow_id": "compare-20240115-abc123",             │
   │   "baseline_model_id": "anthropic.claude-3-haiku...",   │
   │   "finetuned_model_id": "amazon.titan-text-express-v1", │
   │   "improvement_metrics": {                              │
   │     "trust_score_improvement": 0.13,                    │
   │     "hallucination_reduction": 0.13,                    │
   │     "p_value_ttest": 0.0003,                            │
   │     "p_value_wilcoxon": 0.0005,                         │
   │     "confidence_interval_lower": -0.17,                 │
   │     "confidence_interval_upper": -0.09,                 │
   │     "test_type_used": "ttest",                          │
   │     "statistical_significance": true,                   │
   │     "recommendation": "deploy",                         │
   │     "justification": "Significant trust..."             │
   │   },                                                    │
   │   "results_s3_uri": "s3://bucket/results/..."           │
   │ }                                                       │
   └─────────────────────────────────────────────────────────┘
```

#### Integration with Existing TrustOps Components

The Foundation Model Comparison feature is designed as an **extension** of TrustOps, not a separate system. It achieves seamless integration by:

**1. Reusing Existing Components (No Modifications)**

| Component | Usage | Integration Point |
|-----------|-------|-------------------|
| **TrustScoringEngine** | Calculate all 5 trust score components and detect hallucinations | Called for each model response with identical parameters |
| **SemanticSimilarityAnalyzer** | Calculate grounding scores for hallucination detection | Used by TrustScoringEngine (no direct calls) |
| **WorkflowManager** | Track workflow state in DynamoDB | Create workflow, update status, store metadata |
| **S3StorageManager** | Load datasets and store results | Load EvaluationDataset, store comparison results |
| **BedrockClient** | Invoke foundation models | Invoke both models with identical inference params |
| **MetricsAggregator** | Aggregate metrics and calculate improvements | Aggregate results for both models, compute deltas |

**2. Following Existing Patterns**

- **Orchestrator Pattern**: `run_foundation_model_comparison()` method added to existing `EvaluationOrchestrator` class
- **Data Models**: Reuses `EvaluationResult`, `BaselineMetrics`, `ImprovementMetrics`, `ComparativeEvaluationResult`
- **Error Handling**: Uses existing retry logic with exponential backoff for S3/DynamoDB operations
- **Logging**: Uses existing CloudWatch logging infrastructure
- **Storage Structure**: Follows existing S3 hierarchy pattern: `results/{workflow_id}/`

**3. New Components (Minimal Additions)**

| Component | Purpose | Location |
|-----------|---------|----------|
| **Model Validator** | Validate Bedrock model IDs and retrieve metadata | `src/utils/model_validator.py` |
| **Foundation Compare CLI** | CLI command for foundation model comparison | `cli/commands/foundation_compare.py` |
| **Foundation Compare Lambda** | Lambda handler for serverless execution | `lambda_handlers/foundation_model_comparison_handler.py` |

**4. Coexistence with Existing Features**

The feature coexists with existing TrustOps capabilities:

- **Baseline Evaluation**: Evaluate a single foundation model
- **Comparative Evaluation**: Compare baseline vs fine-tuned model
- **Foundation Model Comparison**: Compare any two foundation models (NEW)

All three features share the same infrastructure, data models, and evaluation logic.

#### Key Design Decisions

**1. Identical Evaluation Conditions**
- Both models receive identical prompts from the same dataset
- Both models use identical inference parameters (temperature, max_tokens)
- Both models are evaluated using the same TrustScoringEngine configuration
- This ensures fair, unbiased comparison

**2. Statistical Significance Testing**
- Trust score differences are tested for statistical significance
- Uses appropriate statistical methods (t-test or Mann-Whitney U test)
- Recommendations only suggest deployment when improvements are statistically significant
- Prevents false positives from random variation

**3. Cost-Benefit Analysis**
- Tracks per-query costs for both models based on token usage and model pricing
- Factors cost differences into recommendation justification
- Enables informed decisions balancing quality and cost

**4. Reproducibility**
- Dataset checksums ensure identical data across runs
- Workflow records store complete configuration
- Results include all metadata needed to reproduce evaluation
- Supports audit and compliance requirements


## Data Models and Interfaces

### Core Data Models

#### EvaluationExample

Represents a single evaluation test case.

```python
@dataclass
class EvaluationExample:
    prompt: str                          # Input prompt for the model
    expected_response: Optional[str]     # Expected output (optional)
    source_documents: List[str]          # Context documents for grounding
    category: str                        # Category (e.g., "technical", "business")
    metadata: Dict[str, Any]             # Additional metadata
```

**Example:**
```json
{
  "prompt": "What is the capital of France?",
  "expected_response": "Paris",
  "source_documents": ["France is a country in Europe. Its capital is Paris."],
  "category": "general",
  "metadata": {"difficulty": "easy"}
}
```

#### EvaluationDataset

Collection of evaluation examples with metadata.

```python
@dataclass
class EvaluationDataset:
    dataset_id: str                      # Unique identifier
    name: str                            # Human-readable name
    description: str                     # Dataset description
    examples: List[EvaluationExample]    # List of examples
    created_at: datetime                 # Creation timestamp
    version: str                         # Version identifier
```

#### ModelResponse

Response from a model inference with metadata.

```python
@dataclass
class ModelResponse:
    response_id: str                     # Unique response identifier
    model_id: str                        # Model identifier (Bedrock ARN)
    prompt: str                          # Input prompt
    response_text: str                   # Generated response
    input_tokens: int                    # Number of input tokens
    output_tokens: int                   # Number of output tokens
    latency_ms: float                    # Response latency in milliseconds
    timestamp: datetime                  # Response timestamp
    metadata: Dict[str, Any]             # Additional metadata
```


#### TrustScore

Trust score with component breakdown.

```python
@dataclass
class TrustScoreComponents:
    context_grounding: float             # Semantic similarity with sources (0-1)
    output_structure: float              # Schema compliance (0-1)
    uncertainty_indicators: float        # Certainty level (0-1)
    factual_consistency: float           # Internal consistency (0-1)
    response_completeness: float         # Completeness (0-1)

@dataclass
class TrustScore:
    overall_score: float                 # Weighted composite score (0-1)
    components: TrustScoreComponents     # Component breakdown
    confidence_level: str                # "high", "medium", or "low"
    flagged_for_review: bool             # True if below threshold
    explanation: str                     # Human-readable explanation
```

**Example:**
```json
{
  "overall_score": 0.82,
  "components": {
    "context_grounding": 0.85,
    "output_structure": 0.90,
    "uncertainty_indicators": 0.75,
    "factual_consistency": 0.80,
    "response_completeness": 0.80
  },
  "confidence_level": "high",
  "flagged_for_review": false,
  "explanation": "Trust score 0.82: all components within acceptable range"
}
```

#### HallucinationAnalysis

Analysis of potential hallucinations in a response.

```python
@dataclass
class HallucinationSpan:
    text: str                            # Flagged text span
    start_idx: int                       # Start position in response
    end_idx: int                         # End position in response
    grounding_score: float               # Grounding score (0-1)
    evidence_documents: List[Tuple[str, float]]  # (doc_snippet, similarity)

@dataclass
class HallucinationAnalysis:
    has_hallucinations: bool             # True if hallucinations detected
    hallucination_rate: float            # Ratio of unsupported claims (0-1)
    flagged_spans: List[HallucinationSpan]  # Flagged text spans
    overall_grounding_score: float       # Average grounding score (0-1)
```


## Trust Scoring Algorithm

### Overview

The Trust Scoring Engine calculates a quantifiable trust score (0-1) for model outputs using five weighted components. The algorithm provides real-time confidence assessment to identify low-confidence responses that may require human review.

### Component Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| Context Grounding | 30% | Semantic similarity with source documents |
| Output Structure | 20% | Schema compliance and formatting quality |
| Uncertainty Indicators | 15% | Detection of hedging language patterns |
| Factual Consistency | 15% | Internal contradiction detection |
| Response Completeness | 20% | Adequacy in addressing the prompt |

### Calculation Formula

```
overall_score = (
    context_grounding × 0.30 +
    output_structure × 0.20 +
    uncertainty_indicators × 0.15 +
    factual_consistency × 0.15 +
    response_completeness × 0.20
)
```

The overall score is clamped to the range [0, 1].

### Component Algorithms

#### 1. Context Grounding (30%)

Measures how well the response is grounded in source documents using semantic similarity.

**Algorithm:**
1. For each source document:
   - Calculate semantic similarity between response and document using embeddings
   - Use cosine similarity on Titan embedding vectors
2. Return maximum similarity score (best grounding)

**Scoring:**
- 1.0: Perfect semantic match with source documents
- 0.7-0.9: Strong grounding
- 0.5-0.7: Moderate grounding
- <0.5: Weak or no grounding

**Implementation:**
```python
def _calculate_context_grounding(response: str, source_documents: List[str]) -> float:
    similarities = []
    for doc in source_documents:
        similarity = semantic_similarity_analyzer.calculate_similarity(response, doc)
        similarities.append(similarity)
    return max(similarities) if similarities else 0.5
```


#### 2. Output Structure (20%)

Validates response structure and formatting quality.

**Checks:**
- Response is not empty
- Reasonable length (10-10,000 characters)
- Proper sentence structure
- Appropriate special character ratio (<30%)
- Proper capitalization

**Scoring:**
- 1.0: All structure checks pass
- 0.75: Most checks pass
- 0.5: Some checks pass
- 0.0: Response is empty or severely malformed

#### 3. Uncertainty Indicators (15%)

Detects hedging language that indicates uncertainty.

**Hedging Patterns Detected:**
- Modal verbs: might, may, could, possibly, perhaps, maybe, probably, likely
- Appearance verbs: seems, appears, suggests, indicates
- Opinion markers: I think, I believe, I guess, I suppose
- Uncertainty markers: not sure, uncertain, unclear, ambiguous
- Approximations: approximately, roughly, about, around
- Quantifiers: some, several, many, few, most

**Algorithm:**
1. Count occurrences of hedging patterns (case-insensitive)
2. Calculate hedging density: (hedges / word_count) × 100
3. Convert to certainty score: max(0, 1.0 - hedging_density / 10)

**Scoring:**
- 1.0: No hedging language (highly certain)
- 0.7-0.9: Minimal hedging
- 0.5-0.7: Moderate hedging
- <0.5: Excessive hedging (low certainty)

#### 4. Factual Consistency (15%)

Detects internal contradictions in the response.

**Contradiction Patterns:**
- yes/true/correct ↔ no/false/incorrect
- always/never ↔ sometimes/occasionally
- all/every/none ↔ some/few/many
- increase/rise/grow ↔ decrease/fall/decline

**Algorithm:**
1. Check for presence of contradictory pattern pairs
2. Count contradictions
3. Calculate consistency score: max(0, 1.0 - contradiction_count / 3)

**Scoring:**
- 1.0: No contradictions detected
- 0.7-0.9: One minor contradiction
- 0.3-0.7: Multiple contradictions
- 0.0: Severe contradictions (3+)


#### 5. Response Completeness (20%)

Assesses whether the response adequately addresses the prompt.

**Checks:**
- Response length ≥ 20 characters
- Response is not just repeating the prompt
- Substantive content (≥ 10 words)
- Complete thought (ends with proper punctuation)

**Scoring:**
- 1.0: All completeness checks pass
- 0.75: Most checks pass
- 0.5: Some checks pass
- 0.0: Response is empty or trivial

### Confidence Level Classification

Based on the overall trust score:

| Overall Score | Confidence Level | Action |
|---------------|------------------|--------|
| 0.80 - 1.00 | High | Accept response |
| 0.60 - 0.79 | Medium | Review recommended |
| 0.00 - 0.59 | Low | Flag for review |

### Threshold-Based Flagging

Responses are flagged for review if:
```
overall_score < review_threshold
```

Default threshold: **0.6** (configurable via `TRUST_SCORE_THRESHOLD` environment variable)

### Performance Characteristics

- **Target Latency**: <500ms per response
- **Actual Latency**: Typically 100-300ms depending on:
  - Number of source documents
  - Response length
  - Knowledge Base retrieval time for semantic similarity

### Example Trust Score Calculation

**Input:**
- Response: "Paris is the capital of France. It is located in northern France."
- Source Documents: ["France is a country in Europe. Its capital is Paris."]
- Prompt: "What is the capital of France?"

**Component Scores:**
- Context Grounding: 0.92 (high semantic similarity)
- Output Structure: 0.95 (well-formed response)
- Uncertainty Indicators: 1.0 (no hedging)
- Factual Consistency: 1.0 (no contradictions)
- Response Completeness: 1.0 (complete answer)

**Overall Score:**
```
0.92 × 0.30 + 0.95 × 0.20 + 1.0 × 0.15 + 1.0 × 0.15 + 1.0 × 0.20
= 0.276 + 0.19 + 0.15 + 0.15 + 0.20
= 0.966
```

**Result:** Trust Score = 0.97 (High confidence, not flagged)


## Hallucination Detection

### Overview

The hallucination detection system identifies claims in model responses that lack supporting evidence in source documents. It uses semantic similarity analysis to verify grounding and flags unsupported claims as potential hallucinations.

### Methodology

#### 1. Claim Extraction

**Algorithm:**
1. Split response into sentences using punctuation delimiters (`.`, `!`, `?`)
2. Filter out very short sentences (<10 characters)
3. Track position of each claim (start_idx, end_idx)

**Example:**
```
Response: "Paris is the capital of France. It has a population of 50 million."
Claims:
  1. "Paris is the capital of France." (0-33)
  2. "It has a population of 50 million." (34-69)
```

#### 2. Evidence Search

For each claim:
1. Generate embedding vector using Bedrock Titan embedding model
2. Search source documents using semantic similarity
3. Calculate cosine similarity between claim and each document
4. Identify best matching document and similarity score

**Implementation:**
```python
def _find_grounding_evidence(claim: str, source_documents: List[str]) -> tuple:
    evidence_docs = []
    for doc in source_documents:
        similarity = semantic_similarity_analyzer.calculate_similarity(claim, doc)
        evidence_docs.append((doc[:100], similarity))
    
    evidence_docs.sort(key=lambda x: x[1], reverse=True)
    best_score = evidence_docs[0][1] if evidence_docs else 0.0
    return (best_score, evidence_docs[:3])
```

#### 3. Hallucination Classification

**Thresholds:**
- Default similarity threshold: **0.7** (configurable via `HALLUCINATION_SIMILARITY_THRESHOLD`)

**Classification Logic:**
```
IF grounding_score < similarity_threshold THEN
    claim is classified as hallucination
ELSE
    claim is grounded
END IF
```


#### 4. Hallucination Rate Calculation

**Formula:**
```
hallucination_rate = number_of_unsupported_claims / total_number_of_claims
```

**Range:** [0, 1]
- 0.0: No hallucinations detected
- 0.5: Half of claims are unsupported
- 1.0: All claims are unsupported

#### 5. Overall Grounding Score

**Formula:**
```
overall_grounding_score = average(grounding_scores_for_all_claims)
```

This provides a continuous measure of how well the entire response is grounded.

### Hallucination Analysis Output

```python
HallucinationAnalysis(
    has_hallucinations=True,
    hallucination_rate=0.5,
    flagged_spans=[
        HallucinationSpan(
            text="It has a population of 50 million.",
            start_idx=34,
            end_idx=69,
            grounding_score=0.42,
            evidence_documents=[
                ("France is a country in Europe...", 0.42),
                ("Paris is the capital city...", 0.38)
            ]
        )
    ],
    overall_grounding_score=0.71
)
```

### Hallucination Rate Reduction

When comparing baseline and fine-tuned models:

**Formula:**
```
hallucination_reduction = baseline_hallucination_rate - finetuned_hallucination_rate
```

**Example:**
- Baseline hallucination rate: 0.35 (35% of claims unsupported)
- Fine-tuned hallucination rate: 0.12 (12% of claims unsupported)
- Reduction: 0.23 (23 percentage points improvement)

### Threshold Configuration

Adjust the similarity threshold based on your use case:

| Threshold | Use Case | Characteristics |
|-----------|----------|-----------------|
| 0.9 | High precision | Strict grounding, fewer false positives |
| 0.7 | Balanced (default) | Good balance of precision and recall |
| 0.5 | High recall | Catch more hallucinations, more false positives |


## Cost Calculation

### Overview

TrustOps tracks and calculates costs for all AWS service usage, enabling cost-performance optimization decisions. Costs are calculated based on token usage, API calls, and AWS service consumption.

### Pricing Assumptions

#### Bedrock Model Pricing (USD per 1,000 tokens)

| Model | Input Tokens | Output Tokens |
|-------|--------------|---------------|
| Claude v2 | $0.008 | $0.024 |
| Claude v2.1 | $0.008 | $0.024 |
| Claude Instant | $0.0008 | $0.0024 |
| Llama 2 70B | $0.00195 | $0.00256 |
| Titan Text Express | $0.0008 | $0.0016 |

**Configuration:** Set via environment variables:
```bash
CLAUDE_V2_INPUT_COST=0.008
CLAUDE_V2_OUTPUT_COST=0.024
```

#### Embedding Model Pricing

| Model | Cost per 1,000 tokens |
|-------|----------------------|
| Titan Embeddings | $0.0001 |

#### Other AWS Services

| Service | Pricing Model | Typical Cost |
|---------|---------------|--------------|
| S3 Storage | $0.023/GB/month | ~$1-5/month |
| DynamoDB | On-demand: $1.25/million writes | ~$2-10/month |
| Lambda | $0.20/million requests + compute | ~$5-20/month |
| Bedrock Knowledge Base | OpenSearch Serverless OCU-hours (only if provisioned) | $0 when unused |
| CloudWatch | $0.50/GB ingested | ~$1-5/month |

### Cost Calculation Formulas

#### 1. Per-Query Cost

```python
def calculate_query_cost(input_tokens: int, output_tokens: int, 
                        input_cost_per_1k: float, output_cost_per_1k: float) -> float:
    """
    Calculate cost for a single query.
    
    Returns: Cost in USD
    """
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    return input_cost + output_cost
```

**Example:**
- Input tokens: 500
- Output tokens: 200
- Model: Claude v2
- Cost = (500/1000 × $0.008) + (200/1000 × $0.024) = $0.004 + $0.0048 = $0.0088


#### 2. Evaluation Run Cost

```python
def calculate_evaluation_cost(results: List[EvaluationResult], 
                              pricing_config: PricingConfig) -> float:
    """
    Calculate total cost for an evaluation run.
    
    Returns: Total cost in USD
    """
    total_cost = 0.0
    for result in results:
        total_cost += calculate_query_cost(
            result.model_response.input_tokens,
            result.model_response.output_tokens,
            pricing_config.input_cost_per_1k,
            pricing_config.output_cost_per_1k
        )
    return total_cost
```

#### 3. Cost Per High-Trust Response

```python
def calculate_cost_per_high_trust_response(
    total_cost: float,
    results: List[EvaluationResult],
    high_trust_threshold: float = 0.8
) -> float:
    """
    Calculate cost per high-trust response.
    
    Returns: Cost per high-trust response in USD
    """
    high_trust_count = sum(
        1 for r in results 
        if r.trust_score.overall_score >= high_trust_threshold
    )
    
    if high_trust_count == 0:
        return float('inf')
    
    return total_cost / high_trust_count
```

#### 4. Fine-Tuning Cost

```python
def calculate_fine_tuning_cost(
    training_tokens: int,
    storage_gb: float,
    training_hours: float
) -> float:
    """
    Calculate fine-tuning cost.
    
    Components:
    - Training compute: Based on model size and training time
    - Storage: Training data and model artifacts
    
    Returns: Total fine-tuning cost in USD
    """
    # Bedrock fine-tuning pricing (example for Claude)
    training_cost_per_hour = 12.00  # Varies by model
    storage_cost_per_gb_month = 0.023
    
    training_cost = training_hours * training_cost_per_hour
    storage_cost = storage_gb * storage_cost_per_gb_month
    
    return training_cost + storage_cost
```


#### 5. Cost Projections

```python
def calculate_cost_projections(
    cost_per_query: float,
    query_volumes: List[int]
) -> Dict[int, float]:
    """
    Calculate cost projections for various query volumes.
    
    Args:
        cost_per_query: Average cost per query
        query_volumes: List of monthly query volumes
    
    Returns: Dictionary mapping volume to projected monthly cost
    """
    return {
        volume: cost_per_query * volume
        for volume in query_volumes
    }
```

**Example:**
```python
cost_per_query = 0.0088  # $0.0088 per query
volumes = [1000, 10000, 100000, 1000000]

projections = calculate_cost_projections(cost_per_query, volumes)
# {
#     1000: 8.80,
#     10000: 88.00,
#     100000: 880.00,
#     1000000: 8800.00
# }
```

### Cost-Performance Metrics

#### Cost Efficiency Score

```python
def calculate_cost_efficiency(
    mean_trust_score: float,
    total_cost: float
) -> float:
    """
    Calculate cost efficiency (trust score per dollar).
    
    Higher is better.
    
    Returns: Trust score per dollar
    """
    if total_cost == 0:
        return 0.0
    return mean_trust_score / total_cost
```

#### Cost-Performance Comparison

```python
@dataclass
class CostPerformanceMetrics:
    total_cost: float                           # Total evaluation cost
    cost_per_query: float                       # Average cost per query
    cost_per_high_trust_response: float         # Cost per high-trust response
    cost_per_token: float                       # Average cost per token
    mean_trust_score: float                     # Average trust score
    cost_efficiency_score: float                # Trust score per dollar
    projected_monthly_cost: Dict[int, float]    # Volume → cost projections
```


## API Reference

### EvaluationOrchestrator

Main orchestrator for running evaluations.

#### run_baseline_evaluation

```python
def run_baseline_evaluation(
    self,
    model_id: str,
    dataset_s3_uri: str,
    workflow_id: str
) -> BaselineEvaluationResult:
    """
    Execute baseline evaluation for a foundation model.
    
    Args:
        model_id: AWS Bedrock model identifier (e.g., "anthropic.claude-v2")
        dataset_s3_uri: S3 URI of evaluation dataset (e.g., "s3://bucket/dataset.json")
        workflow_id: Unique workflow identifier
        
    Returns:
        BaselineEvaluationResult with metrics and trust scores
        
    Raises:
        DataValidationError: If dataset format is invalid
        BedrockError: If model invocation fails
        S3Error: If dataset cannot be loaded
    """
```

**Example Usage:**
```python
from src.orchestration.evaluation_orchestrator import EvaluationOrchestrator

orchestrator = EvaluationOrchestrator()
result = orchestrator.run_baseline_evaluation(
    model_id="anthropic.claude-v2",
    dataset_s3_uri="s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval-data.json",
    workflow_id="baseline-20240115-abc123"
)

print(f"Mean trust score: {result.metrics.mean_trust_score}")
print(f"Total cost: ${result.metrics.total_cost:.2f}")
```

#### run_comparative_evaluation

```python
def run_comparative_evaluation(
    self,
    baseline_model_id: str,
    finetuned_model_id: str,
    dataset_s3_uri: str,
    workflow_id: str
) -> ComparativeEvaluationResult:
    """
    Execute comparative evaluation between two models.
    
    Args:
        baseline_model_id: Foundation model identifier
        finetuned_model_id: Fine-tuned model identifier
        dataset_s3_uri: S3 URI of evaluation dataset
        workflow_id: Unique workflow identifier
        
    Returns:
        ComparativeEvaluationResult with side-by-side metrics
        
    Raises:
        DataValidationError: If dataset format is invalid
        BedrockError: If model invocation fails
        S3Error: If dataset cannot be loaded
    """
```


### TrustScoringEngine

Calculate trust scores for model outputs.

#### calculate_trust_score

```python
def calculate_trust_score(
    self,
    response: str,
    prompt: str,
    source_documents: List[str],
    expected_schema: Optional[Dict] = None
) -> TrustScore:
    """
    Calculate composite trust score for a model response.
    
    Args:
        response: Model-generated response text
        prompt: Original prompt/query
        source_documents: Context documents for grounding check
        expected_schema: Optional schema for structure validation
        
    Returns:
        TrustScore object with overall score and component breakdown
        
    Performance:
        Target latency: <500ms
        Typical latency: 100-300ms
    """
```

**Example Usage:**
```python
from src.trust_scoring.trust_scoring_engine import TrustScoringEngine

engine = TrustScoringEngine(review_threshold=0.7)
trust_score = engine.calculate_trust_score(
    response="Paris is the capital of France.",
    prompt="What is the capital of France?",
    source_documents=["France is a country in Europe. Its capital is Paris."]
)

print(f"Overall score: {trust_score.overall_score}")
print(f"Confidence: {trust_score.confidence_level}")
print(f"Flagged: {trust_score.flagged_for_review}")
```

#### detect_hallucinations

```python
def detect_hallucinations(
    self,
    response: str,
    source_documents: List[str],
    similarity_threshold: float = 0.7
) -> HallucinationAnalysis:
    """
    Detect potential hallucinations in response.
    
    Args:
        response: Model-generated response text
        source_documents: Ground truth documents
        similarity_threshold: Minimum similarity for grounding (default: 0.7)
        
    Returns:
        HallucinationAnalysis with flagged spans and evidence scores
    """
```


### FineTuningOrchestrator

Manage fine-tuning pipeline.

#### validate_training_data

```python
def validate_training_data(
    self,
    training_data_s3_uri: str
) -> DataValidationResult:
    """
    Validate training data meets quality standards.
    
    Args:
        training_data_s3_uri: S3 URI of training data
        
    Returns:
        DataValidationResult with validation status and errors
        
    Validation Checks:
        - Format compliance (JSONL with required fields)
        - Minimum sample count (≥10 examples)
        - Prompt-completion pairs present
        - Character encoding (UTF-8)
        - No duplicate examples
    """
```

#### start_fine_tuning_job

```python
def start_fine_tuning_job(
    self,
    base_model_id: str,
    training_data_s3_uri: str,
    job_name: str,
    hyperparameters: Dict[str, Any]
) -> FineTuningJob:
    """
    Initiate fine-tuning job via AWS Bedrock.
    
    Args:
        base_model_id: Foundation model to fine-tune
        training_data_s3_uri: S3 URI of validated training data
        job_name: Unique job identifier
        hyperparameters: Training configuration
            - epochs: Number of training epochs (default: 3)
            - learning_rate: Learning rate (default: 0.0001)
            - batch_size: Batch size (default: 8)
        
    Returns:
        FineTuningJob object with job ID and status
        
    Raises:
        DataValidationError: If training data is invalid
        BedrockError: If job creation fails
    """
```

**Example Usage:**
```python
from src.orchestration.fine_tuning_orchestrator import FineTuningOrchestrator

orchestrator = FineTuningOrchestrator()

# Validate training data
validation = orchestrator.validate_training_data(
    "s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/training-data.jsonl"
)

if validation.is_valid:
    # Start fine-tuning
    job = orchestrator.start_fine_tuning_job(
        base_model_id="anthropic.claude-v2",
        training_data_s3_uri="s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/training-data.jsonl",
        job_name="finetune-20240115-xyz",
        hyperparameters={
            "epochs": 5,
            "learning_rate": 0.0001,
            "batch_size": 8
        }
    )
    print(f"Job ID: {job.job_id}")
```

#### run_foundation_model_comparison

```python
def run_foundation_model_comparison(
    self,
    model_id_1: str,
    model_id_2: str,
    dataset_s3_uri: str,
    workflow_id: str,
    inference_params: Optional[Dict[str, Any]] = None
) -> ComparativeEvaluationResult:
    """
    Execute foundation model comparison workflow.
    
    Compares any two AWS Bedrock foundation models on identical evaluation
    dataset with identical inference parameters. Calculates trust scores,
    detects hallucinations, aggregates metrics, performs statistical
    significance testing, and generates deployment recommendation.
    
    Args:
        model_id_1: First model identifier (e.g., "anthropic.claude-3-haiku-20240307-v1:0")
        model_id_2: Second model identifier (e.g., "amazon.titan-text-express-v1")
        dataset_s3_uri: S3 URI of evaluation dataset (e.g., "s3://bucket/dataset.json")
        workflow_id: Unique workflow identifier
        inference_params: Optional inference parameters
            - temperature: Sampling temperature (default: 0.7)
            - max_tokens: Maximum output tokens (default: 2048)
        
    Returns:
        ComparativeEvaluationResult with:
            - model_1_metrics: Aggregated metrics for first model
            - model_2_metrics: Aggregated metrics for second model
            - comparison_metrics: Improvement metrics and statistical significance
            - recommendation: "DEPLOY_MODEL_1", "DEPLOY_MODEL_2", or "ITERATE"
            - justification: Explanation of recommendation
            - results_s3_uri: S3 location of detailed results
        
    Raises:
        ValueError: If model IDs are invalid or identical
        DataValidationError: If dataset format is invalid
        BedrockError: If model invocation fails
        S3Error: If dataset cannot be loaded or results cannot be stored
        
    Example:
        >>> orchestrator = EvaluationOrchestrator()
        >>> result = orchestrator.run_foundation_model_comparison(
        ...     model_id_1="anthropic.claude-3-haiku-20240307-v1:0",
        ...     model_id_2="amazon.titan-text-express-v1",
        ...     dataset_s3_uri="s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval-data.json",
        ...     workflow_id="compare-20240115-abc123",
        ...     inference_params={"temperature": 0.7, "max_tokens": 2048}
        ... )
        >>> print(f"Recommendation: {result.recommendation}")
        >>> print(f"Trust score improvement: {result.comparison_metrics.trust_score_improvement:.2%}")
        >>> print(f"Cost delta: ${result.comparison_metrics.cost_delta_per_query:.4f}")
    
    Performance:
        - Evaluation time: ~2-5 minutes for 100 examples (parallel execution)
        - Cost: Depends on model pricing and dataset size
        - Typical cost: $5-20 for 100 examples with Claude/Titan models
    
    Workflow Steps:
        1. Validate model IDs (different, valid Bedrock identifiers)
        2. Create workflow record in DynamoDB
        3. Load and validate evaluation dataset from S3
        4. Calculate dataset checksum for reproducibility
        5. Evaluate Model 1 on all examples (parallel)
        6. Evaluate Model 2 on identical examples (parallel)
        7. Aggregate metrics for both models
        8. Calculate comparison metrics and statistical significance
        9. Generate recommendation based on trust score improvement
        10. Store results in S3 and update workflow status
    
    Recommendation Logic:
        - DEPLOY_MODEL_2: Model 2 has statistically significant trust score improvement
        - DEPLOY_MODEL_1: Model 1 has statistically significant trust score improvement
        - ITERATE: No statistically significant difference detected
    
    Statistical Significance:
        - Uses t-test or Mann-Whitney U test depending on distribution
        - Significance threshold: p < 0.05
        - Accounts for sample size and variance
    """
```

**Example Usage:**
```python
from src.orchestration.evaluation_orchestrator import EvaluationOrchestrator

orchestrator = EvaluationOrchestrator()

# Compare Claude Haiku vs Titan Express
result = orchestrator.run_foundation_model_comparison(
    model_id_1="anthropic.claude-3-haiku-20240307-v1:0",
    model_id_2="amazon.titan-text-express-v1",
    dataset_s3_uri="s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval-data.json",
    workflow_id="compare-claude-titan-20240115"
)

# Display results
print(f"Recommendation: {result.recommendation}")
print(f"Justification: {result.justification}")
print(f"\nModel 1 ({result.model_1_id}):")
print(f"  Mean trust score: {result.model_1_metrics.mean_trust_score:.3f}")
print(f"  Hallucination rate: {result.model_1_metrics.hallucination_rate:.2%}")
print(f"  Total cost: ${result.model_1_metrics.total_cost:.2f}")
print(f"\nModel 2 ({result.model_2_id}):")
print(f"  Mean trust score: {result.model_2_metrics.mean_trust_score:.3f}")
print(f"  Hallucination rate: {result.model_2_metrics.hallucination_rate:.2%}")
print(f"  Total cost: ${result.model_2_metrics.total_cost:.2f}")
print(f"\nImprovement:")
print(f"  Trust score: {result.comparison_metrics.trust_score_improvement:+.3f} "
      f"({result.comparison_metrics.trust_score_improvement_percentage:+.1%})")
print(f"  Hallucination reduction: {result.comparison_metrics.hallucination_reduction:+.3f} "
      f"({result.comparison_metrics.hallucination_reduction_percentage:+.1%})")
print(f"  Cost delta: ${result.comparison_metrics.cost_delta_per_query:+.4f} "
      f"({result.comparison_metrics.cost_delta_percentage:+.1%})")
print(f"  Statistical significance: p={result.comparison_metrics.trust_score_p_value:.4f}")
```


### WorkflowManager

Manage workflow state and audit logging.

#### create_workflow

```python
def create_workflow(
    self,
    workflow_type: str,
    configuration: Dict[str, Any]
) -> str:
    """
    Create new workflow instance.
    
    Args:
        workflow_type: Type of workflow ("baseline", "comparative", "fine-tuning")
        configuration: Workflow configuration parameters
        
    Returns:
        Unique workflow identifier (UUID format)
        
    Side Effects:
        - Creates workflow record in DynamoDB
        - Logs workflow creation event to CloudWatch
        - Stores workflow manifest in S3
    """
```

#### reproduce_workflow

```python
def reproduce_workflow(
    self,
    workflow_id: str
) -> str:
    """
    Reproduce a previous workflow with identical configuration.
    
    Args:
        workflow_id: Original workflow identifier
        
    Returns:
        New workflow identifier for reproduction
        
    Raises:
        WorkflowNotFoundError: If original workflow doesn't exist
        
    Behavior:
        - Loads original workflow manifest from S3
        - Creates new workflow with same configuration
        - Links new workflow to original for audit trail
    """
```

**Example Usage:**
```python
from src.orchestration.workflow_manager import WorkflowManager

manager = WorkflowManager()

# Create workflow
workflow_id = manager.create_workflow(
    workflow_type="baseline",
    configuration={
        "model_id": "anthropic.claude-v2",
        "dataset_s3_uri": "s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval-data.json"
    }
)

# Later, reproduce the workflow
new_workflow_id = manager.reproduce_workflow(workflow_id)
```


### CLI Commands

#### Foundation Model Comparison

Compare any two AWS Bedrock foundation models to determine which better suits your use case.

**Command:**
```bash
trustops foundation-compare \
  --model-id-1 <MODEL_ID_1> \
  --model-id-2 <MODEL_ID_2> \
  --dataset <S3_URI> \
  [--workflow-id <WORKFLOW_ID>] \
  [--temperature <FLOAT>] \
  [--max-tokens <INT>]
```

**Options:**

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--model-id-1` | Yes | - | First model identifier (AWS Bedrock model ID) |
| `--model-id-2` | Yes | - | Second model identifier (AWS Bedrock model ID) |
| `--dataset` | Yes | - | S3 URI of evaluation dataset |
| `--workflow-id` | No | Auto-generated | Unique workflow identifier |
| `--temperature` | No | 0.7 | Inference temperature (0.0-1.0) |
| `--max-tokens` | No | 2048 | Maximum output tokens |

**Examples:**

1. **Compare Claude Haiku vs Titan Express:**
```bash
trustops foundation-compare \
  --model-id-1 anthropic.claude-3-haiku-20240307-v1:0 \
  --model-id-2 amazon.titan-text-express-v1 \
  --dataset s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval-data.json
```

2. **Compare Claude Sonnet vs Llama 2 with custom parameters:**
```bash
trustops foundation-compare \
  --model-id-1 anthropic.claude-3-sonnet-20240229-v1:0 \
  --model-id-2 meta.llama2-70b-chat-v1 \
  --dataset s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/technical-eval.json \
  --temperature 0.5 \
  --max-tokens 1024 \
  --workflow-id compare-claude-llama-20240115
```

3. **Compare different Claude versions:**
```bash
trustops foundation-compare \
  --model-id-1 anthropic.claude-3-haiku-20240307-v1:0 \
  --model-id-2 anthropic.claude-3-sonnet-20240229-v1:0 \
  --dataset s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval-data.json
```

**Output:**

The command displays a comprehensive comparison report including:

```
Foundation Model Comparison Results
====================================

Workflow ID: compare-20240115-abc123
Dataset: s3://trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>/eval-data.json
Examples Evaluated: 100

Model 1: anthropic.claude-3-haiku-20240307-v1:0
------------------------------------------------
Mean Trust Score:        0.782
Hallucination Rate:      18.5%
Total Cost:              $12.45
Mean Latency:            850ms
High Confidence:         72/100 responses

Model 2: amazon.titan-text-express-v1
--------------------------------------
Mean Trust Score:        0.695
Hallucination Rate:      28.3%
Total Cost:              $8.20
Mean Latency:            620ms
High Confidence:         58/100 responses

Comparison Metrics
------------------
Trust Score Improvement:     -0.087 (-11.1%)
Hallucination Reduction:     -0.098 (-9.8 pp)
Cost Delta:                  -$4.25 (-34.1%)
Latency Delta:               -230ms (-27.1%)
Statistical Significance:    p=0.002 (significant)

Recommendation: DEPLOY_MODEL_1
-------------------------------
Model 1 (Claude Haiku) shows significantly better trust scores
(p=0.002) with 11% higher trust and 10 percentage points lower
hallucination rate. While Model 1 costs 34% more per query, the
quality improvement justifies the additional cost for high-trust
applications.

Results stored at: s3://trustops-results-<YOUR_ACCOUNT_ID>-<REGION>/compare-20240115-abc123/
```

**Error Handling:**

The CLI provides clear error messages for common issues:

```bash
# Identical model IDs
Error: Cannot compare identical models. Please provide two different model IDs.

# Invalid model ID
Error: Invalid model ID: invalid-model-123. Must be a valid AWS Bedrock foundation model identifier.

# Dataset not found
Error: Failed to load dataset from s3://bucket/missing.json: NoSuchKey

# Invalid dataset format
Error: Invalid dataset format: Missing required field 'prompt' in example 5
```

**Exit Codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid arguments or validation error |
| 2 | Dataset loading error |
| 3 | Model invocation error |
| 4 | Results storage error |


## Configuration

### Environment Variables

TrustOps uses environment variables for configuration. Create a `.env` file in the project root:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_PROFILE=default

# S3 Buckets
TRUSTOPS_DATASETS_BUCKET=trustops-datasets-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_RESULTS_BUCKET=trustops-results-<YOUR_ACCOUNT_ID>-<REGION>
TRUSTOPS_ARTIFACTS_BUCKET=trustops-artifacts-<YOUR_ACCOUNT_ID>-<REGION>

# DynamoDB Tables
TRUSTOPS_WORKFLOWS_TABLE=trustops-workflows
TRUSTOPS_MODELS_TABLE=trustops-models

# CloudWatch
TRUSTOPS_LOG_GROUP=/aws/trustops

# Bedrock Knowledge Bases (optional - retrieval runs in mock mode if unset)
KNOWLEDGE_BASE_ID=
KNOWLEDGE_BASE_DATA_SOURCE_ID=

# Trust Scoring
TRUST_SCORE_THRESHOLD=0.7
HALLUCINATION_SIMILARITY_THRESHOLD=0.7

# Lambda Configuration
LAMBDA_TIMEOUT=900
LAMBDA_MEMORY=3008

# Model Configuration
DEFAULT_EMBEDDING_MODEL=amazon.titan-embed-text-v1
DEFAULT_FOUNDATION_MODEL=anthropic.claude-v2

# Bedrock Fine-Tuning
BEDROCK_EXECUTION_ROLE_ARN=arn:aws:iam::123456789012:role/BedrockExecutionRole

# Cost Configuration (USD per 1000 tokens)
CLAUDE_V2_INPUT_COST=0.008
CLAUDE_V2_OUTPUT_COST=0.024
```


### Configuration Object

Access configuration in code:

```python
from config.aws_config import config

# AWS settings
print(config.region)
print(config.datasets_bucket)

# Trust scoring settings
print(config.trust_score_threshold)
print(config.hallucination_similarity_threshold)

# Model settings
print(config.default_foundation_model)
print(config.default_embedding_model)

# Cost settings
print(config.claude_v2_input_cost)
print(config.claude_v2_output_cost)
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRUST_SCORE_THRESHOLD` | 0.7 | Threshold for flagging low-confidence responses |
| `HALLUCINATION_SIMILARITY_THRESHOLD` | 0.7 | Minimum similarity for grounding claims |
| `LAMBDA_TIMEOUT` | 900 | Lambda function timeout in seconds |
| `LAMBDA_MEMORY` | 3008 | Lambda function memory in MB |
| `DEFAULT_FOUNDATION_MODEL` | anthropic.claude-v2 | Default model for evaluations |

### Adjusting Thresholds

**Trust Score Threshold:**
- **Higher (0.8-0.9)**: Stricter quality control, more responses flagged
- **Lower (0.5-0.6)**: More permissive, fewer responses flagged

**Hallucination Threshold:**
- **Higher (0.8-0.9)**: Stricter grounding requirements, more hallucinations detected
- **Lower (0.5-0.6)**: More lenient, fewer hallucinations detected

---

**Document Version:** 1.0  
**Last Updated:** 2024-01-15  
**Maintained By:** TrustOps Team
