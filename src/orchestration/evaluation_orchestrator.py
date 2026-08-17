"""
Evaluation orchestrator for baseline and comparative model evaluations.
"""
import uuid
import json
from datetime import datetime
from typing import Optional, List
from src.aws_clients.bedrock_client import BedrockClient
from src.aws_clients.s3_storage_manager import S3StorageManager
from src.trust_scoring.trust_scoring_engine import TrustScoringEngine
from src.orchestration.workflow_manager import WorkflowManager
from src.orchestration.metrics_aggregator import MetricsAggregator
from src.data_models.evaluation import EvaluationDataset, EvaluationExample
from src.data_models.model_response import ModelResponse
from src.data_models.results import (
    EvaluationResult,
    BaselineEvaluationResult,
    ComparativeEvaluationResult,
    BaselineMetrics,
    ImprovementMetrics
)
from src.aws_clients.semantic_similarity_analyzer import SemanticSimilarityAnalyzer
from src.utils.logging_utils import get_logger
from src.utils.model_validator import (
    validate_bedrock_model_id,
    validate_models_different,
    get_model_metadata
)
import hashlib


logger = get_logger(__name__)


class EvaluationOrchestrator:
    """
    Orchestrate end-to-end evaluation workflows for baseline and
    comparative assessments.
    """

    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        s3_storage: Optional[S3StorageManager] = None,
        trust_scoring: Optional[TrustScoringEngine] = None,
        workflow_manager: Optional[WorkflowManager] = None,
        metrics_aggregator: Optional[MetricsAggregator] = None,
        semantic_similarity: Optional[SemanticSimilarityAnalyzer] = None
    ):
        """
        Initialize evaluation orchestrator.

        Args:
            bedrock_client: AWS Bedrock client for model inference
            s3_storage: S3 storage manager for datasets and results
            trust_scoring: Trust scoring engine
            workflow_manager: Workflow manager for tracking
            metrics_aggregator: Metrics aggregator for computing statistics
            semantic_similarity: Semantic similarity analyzer
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        self.s3_storage = s3_storage or S3StorageManager()
        self.trust_scoring = trust_scoring or TrustScoringEngine()
        self.workflow_manager = workflow_manager or WorkflowManager()
        self.metrics_aggregator = metrics_aggregator or MetricsAggregator()
        self.semantic_similarity = semantic_similarity or SemanticSimilarityAnalyzer()

    def run_baseline_evaluation(
        self,
        model_id: str,
        dataset_s3_uri: str,
        workflow_id: str
    ) -> BaselineEvaluationResult:
        """
        Execute baseline evaluation for a foundation model.

        This method orchestrates the complete baseline evaluation workflow:
        1. Loads evaluation dataset from S3
        2. Invokes Foundation_Model for each prompt via Bedrock
        3. Calculates trust score for each response
        4. Tracks tokens and costs for each inference
        5. Stores results with timestamps in S3
        6. Updates workflow status in DynamoDB
        7. Logs progress to CloudWatch

        Args:
            model_id: AWS Bedrock model identifier
            dataset_s3_uri: S3 URI of evaluation dataset
            workflow_id: Unique workflow identifier

        Returns:
            BaselineEvaluationResult with metrics and trust scores

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If evaluation fails
        """
        try:
            logger.info(
                f"Starting baseline evaluation for model {model_id}, "
                f"workflow {workflow_id}"
            )

            # Update workflow status to running
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {'stage': 'loading_dataset'}
            )

            # Step 1: Load evaluation dataset from S3
            logger.info(f"Loading dataset from {dataset_s3_uri}")
            dataset = self._load_dataset(dataset_s3_uri)
            logger.info(
                f"Loaded dataset with {len(dataset.examples)} examples"
            )

            # Update workflow status
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {
                    'stage': 'generating_responses',
                    'total_examples': len(dataset.examples)
                }
            )

            # Step 2-4: Process each example
            evaluation_results: List[EvaluationResult] = []

            for idx, example in enumerate(dataset.examples):
                logger.info(
                    f"Processing example {idx + 1}/{len(dataset.examples)}"
                )

                try:
                    # Generate response from model
                    result = self._evaluate_example(
                        model_id=model_id,
                        example=example,
                        example_idx=idx
                    )
                    evaluation_results.append(result)

                    # Log progress every 10 examples
                    if (idx + 1) % 10 == 0:
                        logger.info(
                            f"Completed {idx + 1}/{len(dataset.examples)} "
                            f"examples"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to evaluate example {idx}: {e}",
                        exc_info=True
                    )
                    # Continue with remaining examples
                    continue

            logger.info(
                f"Completed evaluation of {len(evaluation_results)} examples"
            )

            # Update workflow status
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {
                    'stage': 'aggregating_metrics',
                    'completed_examples': len(evaluation_results)
                }
            )

            # Step 5: Aggregate metrics
            logger.info("Aggregating metrics")
            metrics = self.metrics_aggregator.aggregate_baseline_metrics(
                evaluation_results
            )

            # Step 6: Store results in S3
            logger.info("Storing results in S3")
            results_info = self._store_results(
                workflow_id=workflow_id,
                model_id=model_id,
                evaluation_results=evaluation_results,
                metrics=metrics
            )

            # Create baseline evaluation result
            baseline_result = BaselineEvaluationResult(
                workflow_id=workflow_id,
                model_id=model_id,
                evaluation_results=evaluation_results,
                metrics=metrics,
                dataset_s3_uri=dataset_s3_uri,
                results_s3_uri=results_info['s3_uri']
            )

            # Step 7: Update workflow status to completed
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'completed',
                {
                    'results_s3_uri': results_info['s3_uri'],
                    'total_examples': len(evaluation_results),
                    'mean_trust_score': metrics.mean_trust_score,
                    'total_cost': metrics.total_cost
                }
            )

            logger.info(
                f"Baseline evaluation completed successfully. "
                f"Mean trust score: {metrics.mean_trust_score:.3f}, "
                f"Total cost: ${metrics.total_cost:.4f}"
            )

            return baseline_result

        except Exception as e:
            logger.error(
                f"Baseline evaluation failed: {e}",
                exc_info=True
            )

            # Update workflow status to failed
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'failed',
                {'error': str(e)}
            )

            raise RuntimeError(
                f"Baseline evaluation failed: {e}"
            ) from e

    def run_comparative_evaluation(
        self,
        baseline_model_id: str,
        finetuned_model_id: str,
        dataset_s3_uri: str,
        workflow_id: str
    ) -> ComparativeEvaluationResult:
        """
        Execute comparative evaluation between two models.

        This method orchestrates the complete comparative evaluation workflow:
        1. Loads evaluation dataset from S3
        2. Generates responses from both baseline and fine-tuned models with identical prompts
        3. Calculates trust scores for all responses
        4. Computes semantic similarity between corresponding responses
        5. Calculates improvement metrics using MetricsAggregator
        6. Generates comparison report with statistical significance
        7. Stores report in S3 with references to both evaluation runs

        Args:
            baseline_model_id: Foundation model identifier
            finetuned_model_id: Fine-tuned model identifier
            dataset_s3_uri: S3 URI of evaluation dataset
            workflow_id: Unique workflow identifier

        Returns:
            ComparativeEvaluationResult with side-by-side metrics

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If evaluation fails
        """
        try:
            logger.info(
                f"Starting comparative evaluation between {baseline_model_id} "
                f"and {finetuned_model_id}, workflow {workflow_id}"
            )

            # Update workflow status to running
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {'stage': 'loading_dataset'}
            )

            # Step 1: Load evaluation dataset from S3
            logger.info(f"Loading dataset from {dataset_s3_uri}")
            dataset = self._load_dataset(dataset_s3_uri)
            logger.info(
                f"Loaded dataset with {len(dataset.examples)} examples"
            )

            # Update workflow status
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {
                    'stage': 'generating_baseline_responses',
                    'total_examples': len(dataset.examples)
                }
            )

            # Step 2: Generate responses from baseline model
            logger.info(f"Generating responses from baseline model {baseline_model_id}")
            baseline_results: List[EvaluationResult] = []

            for idx, example in enumerate(dataset.examples):
                logger.info(
                    f"Processing baseline example {idx + 1}/{len(dataset.examples)}"
                )

                try:
                    result = self._evaluate_example(
                        model_id=baseline_model_id,
                        example=example,
                        example_idx=idx
                    )
                    baseline_results.append(result)

                    if (idx + 1) % 10 == 0:
                        logger.info(
                            f"Completed {idx + 1}/{len(dataset.examples)} "
                            f"baseline examples"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to evaluate baseline example {idx}: {e}",
                        exc_info=True
                    )
                    continue

            logger.info(
                f"Completed baseline evaluation of {len(baseline_results)} examples"
            )

            # Update workflow status
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {
                    'stage': 'generating_finetuned_responses',
                    'completed_baseline': len(baseline_results)
                }
            )

            # Step 3: Generate responses from fine-tuned model with identical prompts
            logger.info(f"Generating responses from fine-tuned model {finetuned_model_id}")
            finetuned_results: List[EvaluationResult] = []

            for idx, example in enumerate(dataset.examples):
                logger.info(
                    f"Processing fine-tuned example {idx + 1}/{len(dataset.examples)}"
                )

                try:
                    result = self._evaluate_example(
                        model_id=finetuned_model_id,
                        example=example,
                        example_idx=idx
                    )
                    finetuned_results.append(result)

                    if (idx + 1) % 10 == 0:
                        logger.info(
                            f"Completed {idx + 1}/{len(dataset.examples)} "
                            f"fine-tuned examples"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to evaluate fine-tuned example {idx}: {e}",
                        exc_info=True
                    )
                    continue

            logger.info(
                f"Completed fine-tuned evaluation of {len(finetuned_results)} examples"
            )

            # Ensure we have matching results
            if len(baseline_results) != len(finetuned_results):
                logger.warning(
                    f"Mismatch in result counts: baseline={len(baseline_results)}, "
                    f"finetuned={len(finetuned_results)}. Using minimum count."
                )
                min_count = min(len(baseline_results), len(finetuned_results))
                baseline_results = baseline_results[:min_count]
                finetuned_results = finetuned_results[:min_count]

            # Update workflow status
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {
                    'stage': 'calculating_semantic_similarity',
                    'completed_finetuned': len(finetuned_results)
                }
            )

            # Step 4: Compute semantic similarity between corresponding responses
            logger.info("Calculating semantic similarity between responses")
            for idx, (baseline_result, finetuned_result) in enumerate(
                zip(baseline_results, finetuned_results)
            ):
                try:
                    similarity = self.semantic_similarity.calculate_similarity(
                        baseline_result.model_response.response_text,
                        finetuned_result.model_response.response_text
                    )
                    
                    # Update both results with semantic similarity
                    baseline_result.semantic_similarity = similarity
                    finetuned_result.semantic_similarity = similarity

                    if (idx + 1) % 10 == 0:
                        logger.info(
                            f"Calculated similarity for {idx + 1}/{len(baseline_results)} "
                            f"response pairs"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to calculate similarity for example {idx}: {e}",
                        exc_info=True
                    )
                    # Set to None if calculation fails
                    baseline_result.semantic_similarity = None
                    finetuned_result.semantic_similarity = None

            # Update workflow status
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {'stage': 'aggregating_metrics'}
            )

            # Step 5: Aggregate metrics for both models
            logger.info("Aggregating metrics for both models")
            baseline_metrics = self.metrics_aggregator.aggregate_baseline_metrics(
                baseline_results
            )
            finetuned_metrics = self.metrics_aggregator.aggregate_baseline_metrics(
                finetuned_results
            )

            # Step 6: Calculate improvement metrics
            logger.info("Calculating improvement metrics")
            improvement_metrics = self.metrics_aggregator.compute_improvement_metrics(
                baseline_results,
                finetuned_results
            )

            # Step 7: Store results in S3
            logger.info("Storing comparative results in S3")
            results_info = self._store_comparative_results(
                workflow_id=workflow_id,
                baseline_model_id=baseline_model_id,
                finetuned_model_id=finetuned_model_id,
                baseline_results=baseline_results,
                finetuned_results=finetuned_results,
                baseline_metrics=baseline_metrics,
                finetuned_metrics=finetuned_metrics,
                improvement_metrics=improvement_metrics
            )

            # Create comparative evaluation result
            comparative_result = ComparativeEvaluationResult(
                workflow_id=workflow_id,
                baseline_model_id=baseline_model_id,
                finetuned_model_id=finetuned_model_id,
                baseline_results=baseline_results,
                finetuned_results=finetuned_results,
                baseline_metrics=baseline_metrics,
                finetuned_metrics=finetuned_metrics,
                improvement_metrics=improvement_metrics,
                dataset_s3_uri=dataset_s3_uri,
                results_s3_uri=results_info['s3_uri']
            )

            # Update workflow status to completed
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'completed',
                {
                    'results_s3_uri': results_info['s3_uri'],
                    'total_examples': len(baseline_results),
                    'baseline_mean_trust_score': baseline_metrics.mean_trust_score,
                    'finetuned_mean_trust_score': finetuned_metrics.mean_trust_score,
                    'trust_score_improvement': improvement_metrics.trust_score_improvement,
                    'recommendation': improvement_metrics.recommendation
                }
            )

            logger.info(
                f"Comparative evaluation completed successfully. "
                f"Trust score improvement: {improvement_metrics.trust_score_improvement:.3f}, "
                f"Recommendation: {improvement_metrics.recommendation}"
            )

            return comparative_result

        except Exception as e:
            logger.error(
                f"Comparative evaluation failed: {e}",
                exc_info=True
            )

            # Update workflow status to failed
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'failed',
                {'error': str(e)}
            )

            raise RuntimeError(
                f"Comparative evaluation failed: {e}"
            ) from e

    def run_foundation_model_comparison(
        self,
        model_id_1: str,
        model_id_2: str,
        dataset_s3_uri: str,
        workflow_id: str,
        inference_params: Optional[dict] = None
    ) -> ComparativeEvaluationResult:
        """
        Execute foundation model comparison workflow.

        This method orchestrates the complete foundation model comparison workflow:
        1. Validates model IDs (valid format, different models)
        2. Creates workflow record in DynamoDB
        3. Loads evaluation dataset from S3
        4. Calculates dataset checksum for reproducibility
        5. Evaluates model 1 on all examples with identical inference parameters
        6. Evaluates model 2 on identical examples with identical inference parameters
        7. Records latency, token usage, and cost for each invocation
        8. Handles individual example failures gracefully (log and continue)
        9. Aggregates metrics for both models
        10. Calculates comparison metrics with statistical significance testing
        11. Generates recommendation (DEPLOY_MODEL_1, DEPLOY_MODEL_2, or ITERATE)
        12. Stores results in S3
        13. Updates workflow status to completed or failed

        Args:
            model_id_1: First AWS Bedrock model identifier
            model_id_2: Second AWS Bedrock model identifier
            dataset_s3_uri: S3 URI of evaluation dataset
            workflow_id: Unique workflow identifier
            inference_params: Optional inference parameters (temperature, max_tokens)

        Returns:
            ComparativeEvaluationResult with metrics and recommendation

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If evaluation fails
        """
        try:
            logger.info(
                f"Starting foundation model comparison between {model_id_1} "
                f"and {model_id_2}, workflow {workflow_id}"
            )

            # Step 1: Validate model IDs
            logger.info("Validating model IDs")
            if not validate_bedrock_model_id(model_id_1):
                raise ValueError(
                    f"Invalid model ID: {model_id_1}. Must be a valid AWS Bedrock "
                    f"foundation model identifier."
                )
            if not validate_bedrock_model_id(model_id_2):
                raise ValueError(
                    f"Invalid model ID: {model_id_2}. Must be a valid AWS Bedrock "
                    f"foundation model identifier."
                )
            validate_models_different(model_id_1, model_id_2)

            # Retrieve model metadata for pricing
            model_1_metadata = get_model_metadata(model_id_1)
            model_2_metadata = get_model_metadata(model_id_2)
            logger.info(
                f"Model 1: {model_1_metadata['model_family']}, "
                f"Model 2: {model_2_metadata['model_family']}"
            )

            # Set default inference parameters if not provided
            if inference_params is None:
                inference_params = {
                    'temperature': 0.7,
                    'max_tokens': 2048
                }

            # Step 2: Create workflow record
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {
                    'stage': 'loading_dataset',
                    'model_id_1': model_id_1,
                    'model_id_2': model_id_2
                }
            )

            # Step 3: Load evaluation dataset from S3
            logger.info(f"Loading dataset from {dataset_s3_uri}")
            dataset = self._load_dataset(dataset_s3_uri)
            logger.info(
                f"Loaded dataset with {len(dataset.examples)} examples"
            )

            # Step 4: Calculate dataset checksum for reproducibility
            dataset_json = json.dumps(dataset.to_dict(), sort_keys=True)
            dataset_checksum = hashlib.sha256(
                dataset_json.encode('utf-8')
            ).hexdigest()
            logger.info(f"Dataset checksum: {dataset_checksum}")

            # Update workflow status
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {
                    'stage': 'evaluating_model_1',
                    'total_examples': len(dataset.examples),
                    'dataset_checksum': dataset_checksum
                }
            )

            # Step 5: Evaluate model 1 on all examples
            logger.info(f"Evaluating model 1: {model_id_1}")
            model_1_results: List[EvaluationResult] = []

            for idx, example in enumerate(dataset.examples):
                logger.info(
                    f"Processing model 1 example {idx + 1}/{len(dataset.examples)}"
                )

                try:
                    result = self._evaluate_example(
                        model_id=model_id_1,
                        example=example,
                        example_idx=idx
                    )
                    model_1_results.append(result)

                    if (idx + 1) % 10 == 0:
                        logger.info(
                            f"Completed {idx + 1}/{len(dataset.examples)} "
                            f"model 1 examples"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to evaluate model 1 example {idx}: {e}",
                        exc_info=True
                    )
                    # Continue with remaining examples
                    continue

            logger.info(
                f"Completed model 1 evaluation of {len(model_1_results)} examples"
            )

            # Update workflow status
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {
                    'stage': 'evaluating_model_2',
                    'completed_model_1': len(model_1_results)
                }
            )

            # Step 6: Evaluate model 2 on identical examples with identical parameters
            logger.info(f"Evaluating model 2: {model_id_2}")
            model_2_results: List[EvaluationResult] = []

            for idx, example in enumerate(dataset.examples):
                logger.info(
                    f"Processing model 2 example {idx + 1}/{len(dataset.examples)}"
                )

                try:
                    result = self._evaluate_example(
                        model_id=model_id_2,
                        example=example,
                        example_idx=idx
                    )
                    model_2_results.append(result)

                    if (idx + 1) % 10 == 0:
                        logger.info(
                            f"Completed {idx + 1}/{len(dataset.examples)} "
                            f"model 2 examples"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to evaluate model 2 example {idx}: {e}",
                        exc_info=True
                    )
                    # Continue with remaining examples
                    continue

            logger.info(
                f"Completed model 2 evaluation of {len(model_2_results)} examples"
            )

            # Ensure we have matching results
            if len(model_1_results) != len(model_2_results):
                logger.warning(
                    f"Mismatch in result counts: model_1={len(model_1_results)}, "
                    f"model_2={len(model_2_results)}. Using minimum count."
                )
                min_count = min(len(model_1_results), len(model_2_results))
                model_1_results = model_1_results[:min_count]
                model_2_results = model_2_results[:min_count]

            # Update workflow status
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'running',
                {
                    'stage': 'aggregating_metrics',
                    'completed_model_2': len(model_2_results)
                }
            )

            # Step 7: Aggregate metrics for both models
            logger.info("Aggregating metrics for both models")
            model_1_metrics = self.metrics_aggregator.aggregate_baseline_metrics(
                model_1_results
            )
            model_2_metrics = self.metrics_aggregator.aggregate_baseline_metrics(
                model_2_results
            )

            # Step 8: Calculate comparison metrics with statistical significance
            logger.info("Calculating improvement metrics")
            improvement_metrics = self.metrics_aggregator.compute_improvement_metrics(
                model_1_results,
                model_2_results
            )

            # Update improvement metrics to use correct model IDs
            improvement_metrics.baseline_model_id = model_id_1
            improvement_metrics.finetuned_model_id = model_id_2

            # Step 9: Generate recommendation
            # The recommendation is already generated by compute_improvement_metrics
            # but we'll log it here for clarity
            logger.info(
                f"Recommendation: {improvement_metrics.recommendation}, "
                f"Justification: {improvement_metrics.justification}"
            )

            # Step 10: Store results in S3
            logger.info("Storing foundation model comparison results in S3")
            results_info = self._store_comparative_results(
                workflow_id=workflow_id,
                baseline_model_id=model_id_1,
                finetuned_model_id=model_id_2,
                baseline_results=model_1_results,
                finetuned_results=model_2_results,
                baseline_metrics=model_1_metrics,
                finetuned_metrics=model_2_metrics,
                improvement_metrics=improvement_metrics
            )

            # Create comparative evaluation result
            comparison_result = ComparativeEvaluationResult(
                workflow_id=workflow_id,
                baseline_model_id=model_id_1,
                finetuned_model_id=model_id_2,
                baseline_results=model_1_results,
                finetuned_results=model_2_results,
                baseline_metrics=model_1_metrics,
                finetuned_metrics=model_2_metrics,
                improvement_metrics=improvement_metrics,
                dataset_s3_uri=dataset_s3_uri,
                results_s3_uri=results_info['s3_uri']
            )

            # Step 11: Update workflow status to completed
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'completed',
                {
                    'results_s3_uri': results_info['s3_uri'],
                    'total_examples': len(model_1_results),
                    'model_1_mean_trust_score': model_1_metrics.mean_trust_score,
                    'model_2_mean_trust_score': model_2_metrics.mean_trust_score,
                    'trust_score_improvement': improvement_metrics.trust_score_improvement,
                    'recommendation': improvement_metrics.recommendation,
                    'dataset_checksum': dataset_checksum
                }
            )

            logger.info(
                f"Foundation model comparison completed successfully. "
                f"Trust score improvement: {improvement_metrics.trust_score_improvement:.3f}, "
                f"Recommendation: {improvement_metrics.recommendation}"
            )

            return comparison_result

        except ValueError as e:
            logger.error(
                f"Foundation model comparison validation failed: {e}",
                exc_info=True
            )

            # Update workflow status to failed
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'failed',
                {'error': str(e)}
            )

            raise

        except Exception as e:
            logger.error(
                f"Foundation model comparison failed: {e}",
                exc_info=True
            )

            # Update workflow status to failed
            self.workflow_manager.update_workflow_status(
                workflow_id,
                'failed',
                {'error': str(e)}
            )

            raise RuntimeError(
                f"Foundation model comparison failed: {e}"
            ) from e

    def _load_dataset(self, dataset_s3_uri: str) -> EvaluationDataset:
        """
        Load evaluation dataset from S3.

        Args:
            dataset_s3_uri: S3 URI of the dataset

        Returns:
            EvaluationDataset object

        Raises:
            ValueError: If dataset format is invalid
            RuntimeError: If loading fails
        """
        try:
            # Download dataset from S3
            dataset_data = self.s3_storage.download_dataset(dataset_s3_uri)
            content = dataset_data['content']

            # Parse JSON
            dataset_dict = json.loads(content)

            # Convert to EvaluationDataset
            dataset = EvaluationDataset.from_dict(dataset_dict)

            return dataset

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid dataset format: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset: {e}"
            ) from e

    def _evaluate_example(
        self,
        model_id: str,
        example: EvaluationExample,
        example_idx: int
    ) -> EvaluationResult:
        """
        Evaluate a single example.

        Args:
            model_id: Model identifier
            example: Evaluation example
            example_idx: Example index

        Returns:
            EvaluationResult for the example

        Raises:
            RuntimeError: If evaluation fails
        """
        try:
            # Step 1: Invoke model via Bedrock
            inference_result = self.bedrock_client.invoke_model(
                model_id=model_id,
                prompt=example.prompt
            )

            # Create ModelResponse
            response_id = f"{model_id}-{example_idx}-{uuid.uuid4().hex[:8]}"
            model_response = ModelResponse(
                response_id=response_id,
                model_id=model_id,
                prompt=example.prompt,
                response_text=inference_result['response_text'],
                input_tokens=inference_result['input_tokens'],
                output_tokens=inference_result['output_tokens'],
                latency_ms=inference_result['latency_ms'],
                timestamp=datetime.utcnow(),
                metadata={
                    'cost': inference_result['cost'],
                    'example_idx': example_idx
                }
            )

            # Step 2: Calculate trust score
            trust_score = self.trust_scoring.calculate_trust_score(
                response=inference_result['response_text'],
                prompt=example.prompt,
                source_documents=example.source_documents,
                expected_schema=None
            )

            # Step 3: Detect hallucinations
            hallucination_analysis = self.trust_scoring.detect_hallucinations(
                response=inference_result['response_text'],
                source_documents=example.source_documents
            )

            # Determine if example passed (trust score above threshold)
            passed = trust_score.overall_score >= 0.6

            # Create evaluation result
            evaluation_result = EvaluationResult(
                example_id=f"example-{example_idx}",
                model_response=model_response,
                trust_score=trust_score,
                hallucination_analysis=hallucination_analysis,
                semantic_similarity=None,  # Not used in baseline
                category=example.category,
                passed=passed
            )

            return evaluation_result

        except Exception as e:
            raise RuntimeError(
                f"Failed to evaluate example {example_idx}: {e}"
            ) from e

    def _store_results(
        self,
        workflow_id: str,
        model_id: str,
        evaluation_results: List[EvaluationResult],
        metrics: 'BaselineMetrics'
    ) -> dict:
        """
        Store evaluation results in S3.

        Args:
            workflow_id: Workflow identifier
            model_id: Model identifier
            evaluation_results: List of evaluation results
            metrics: Aggregated metrics

        Returns:
            Dictionary with storage information

        Raises:
            RuntimeError: If storage fails
        """
        try:
            # Prepare results dictionary
            results_dict = {
                'workflow_id': workflow_id,
                'model_id': model_id,
                'timestamp': datetime.utcnow().isoformat(),
                'evaluation_results': [
                    result.to_dict() for result in evaluation_results
                ],
                'metrics': metrics.to_dict()
            }

            # Store in S3
            storage_info = self.s3_storage.store_results(
                results=results_dict,
                workflow_id=workflow_id,
                model_id=model_id
            )

            return storage_info

        except Exception as e:
            raise RuntimeError(
                f"Failed to store results: {e}"
            ) from e

    def _store_comparative_results(
        self,
        workflow_id: str,
        baseline_model_id: str,
        finetuned_model_id: str,
        baseline_results: List[EvaluationResult],
        finetuned_results: List[EvaluationResult],
        baseline_metrics: 'BaselineMetrics',
        finetuned_metrics: 'BaselineMetrics',
        improvement_metrics: 'ImprovementMetrics'
    ) -> dict:
        """
        Store comparative evaluation results in S3.

        Args:
            workflow_id: Workflow identifier
            baseline_model_id: Baseline model identifier
            finetuned_model_id: Fine-tuned model identifier
            baseline_results: Baseline evaluation results
            finetuned_results: Fine-tuned evaluation results
            baseline_metrics: Baseline aggregated metrics
            finetuned_metrics: Fine-tuned aggregated metrics
            improvement_metrics: Improvement metrics

        Returns:
            Dictionary with storage information

        Raises:
            RuntimeError: If storage fails
        """
        try:
            # Prepare comparative results dictionary
            results_dict = {
                'workflow_id': workflow_id,
                'baseline_model_id': baseline_model_id,
                'finetuned_model_id': finetuned_model_id,
                'timestamp': datetime.utcnow().isoformat(),
                'baseline_results': [
                    result.to_dict() for result in baseline_results
                ],
                'finetuned_results': [
                    result.to_dict() for result in finetuned_results
                ],
                'baseline_metrics': baseline_metrics.to_dict(),
                'finetuned_metrics': finetuned_metrics.to_dict(),
                'improvement_metrics': improvement_metrics.to_dict(),
                'comparison_report': {
                    'total_examples': len(baseline_results),
                    'baseline_model_id': baseline_model_id,
                    'finetuned_model_id': finetuned_model_id,
                    'trust_score_improvement': improvement_metrics.trust_score_improvement,
                    'hallucination_reduction': improvement_metrics.hallucination_reduction,
                    'cost_delta_per_query': improvement_metrics.cost_delta_per_query,
                    'statistical_significance': improvement_metrics.statistical_significance,
                    'recommendation': improvement_metrics.recommendation,
                    'justification': improvement_metrics.justification
                }
            }

            # Store in S3 with comparative naming
            storage_info = self.s3_storage.store_results(
                results=results_dict,
                workflow_id=workflow_id,
                model_id=f"comparative-{baseline_model_id}-vs-{finetuned_model_id}"
            )

            return storage_info

        except Exception as e:
            raise RuntimeError(
                f"Failed to store comparative results: {e}"
            ) from e
