"""
AWS Lambda handler for model evaluation.

Receives evaluation requests via API Gateway or Step Functions,
delegates to the EvaluationEngine, and returns results.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point for evaluation requests.

    Expects event with:
        - model_id (str): Model to evaluate.
        - dataset_id (str): Dataset to use.
        - evaluation_type (str): 'baseline' or 'comparative'.
        - inference_params (dict, optional): max_tokens, temperature.
        - concurrency (int, optional): Parallel invocations.

    Returns:
        dict with statusCode and JSON body containing evaluation results.
    """
    try:
        model_id = event.get("model_id")
        dataset_id = event.get("dataset_id")
        evaluation_type = event.get("evaluation_type", "baseline")

        # Log only operation metadata. The raw event can carry prompts and
        # dataset content, which may be sensitive in a production adaptation.
        logger.info(
            "Evaluation request received",
            extra={
                "model_id": model_id,
                "dataset_id": dataset_id,
                "evaluation_type": evaluation_type,
            },
        )

        if not model_id or not dataset_id:
            return _response(400, {"error": "model_id and dataset_id are required"})

        from src.clients.inference_client import InferenceClient
        from src.data_models.evaluation import EvaluationConfig, InferenceParams
        from src.evaluation.evaluation_engine import EvaluationEngine
        from src.trust_scoring.trust_scoring_engine import TrustScoringEngine

        inference_params = event.get("inference_params", {})
        config = EvaluationConfig(
            model_id=model_id,
            dataset_id=dataset_id,
            inference_params=InferenceParams(
                max_tokens=inference_params.get("max_tokens", 1024),
                temperature=inference_params.get("temperature", 0.0),
            ),
            concurrency=event.get("concurrency", 5),
        )

        # Engine setup is deferred to actual invocation via Step Functions
        return _response(200, {
            "evaluation_type": evaluation_type,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "status": "accepted",
            "config": {
                "concurrency": config.concurrency,
                "max_tokens": config.inference_params.max_tokens,
                "temperature": config.inference_params.temperature,
            },
        })

    except Exception as e:
        logger.exception("Evaluation handler error")
        return _response(500, {"error": str(e)})


def _response(status_code: int, body: dict) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
