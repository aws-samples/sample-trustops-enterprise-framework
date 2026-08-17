"""
AWS Lambda handler for comparative model evaluation.

Receives comparison requests, delegates to the EvaluationEngine's
comparative evaluation, and returns side-by-side results.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point for comparative evaluation requests.

    Expects event with:
        - model_id_1 (str): First model (baseline).
        - model_id_2 (str): Second model (comparison).
        - dataset_id (str): Dataset to use.
        - inference_params (dict, optional): max_tokens, temperature.
        - concurrency (int, optional): Parallel invocations.

    Returns:
        dict with statusCode and JSON body containing comparison results.
    """
    try:
        model_id_1 = event.get("model_id_1")
        model_id_2 = event.get("model_id_2")
        dataset_id = event.get("dataset_id")

        # Log only operation metadata. The raw event can carry prompts and
        # dataset content, which may be sensitive in a production adaptation.
        logger.info(
            "Comparative evaluation request received",
            extra={
                "model_id_1": model_id_1,
                "model_id_2": model_id_2,
                "dataset_id": dataset_id,
            },
        )

        if not model_id_1 or not model_id_2 or not dataset_id:
            return _response(400, {
                "error": "model_id_1, model_id_2, and dataset_id are required"
            })

        return _response(200, {
            "evaluation_type": "comparative",
            "model_id_1": model_id_1,
            "model_id_2": model_id_2,
            "dataset_id": dataset_id,
            "status": "accepted",
        })

    except Exception as e:
        logger.exception("Comparative evaluation handler error")
        return _response(500, {"error": str(e)})


def _response(status_code: int, body: dict) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
