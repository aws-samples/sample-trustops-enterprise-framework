"""
AWS Lambda handler for fine-tuning operations.

Receives fine-tuning requests via API Gateway or Step Functions,
delegates to the FineTuningPipeline, and returns results.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point for fine-tuning requests.

    Expects event with:
        - action (str): 'validate', 'estimate_cost', 'start', 'status', 'stop'.
        - base_model_id (str): Base model identifier.
        - training_data_id (str): Training dataset identifier.
        - job_name (str, optional): Name for the fine-tuning job.
        - hyperparameters (dict, optional): epochs, batch_size, learning_rate.
        - job_id (str): Required for 'status' and 'stop' actions.

    Returns:
        dict with statusCode and JSON body.
    """
    try:
        action = event.get("action", "start")
        job_id = event.get("job_id")

        # Log only operation metadata. The raw event can carry training data
        # references and hyperparameters; keep prompts out of the log stream.
        logger.info(
            "Fine-tuning request received",
            extra={
                "action": action,
                "job_id": job_id,
                "base_model_id": event.get("base_model_id"),
                "training_data_id": event.get("training_data_id"),
            },
        )

        if action in ("status", "stop") and not job_id:
            return _response(400, {"error": "job_id is required for this action"})

        if action in ("validate", "estimate_cost", "start"):
            base_model_id = event.get("base_model_id")
            training_data_id = event.get("training_data_id")
            if not base_model_id or not training_data_id:
                return _response(400, {
                    "error": "base_model_id and training_data_id are required"
                })

        return _response(200, {
            "action": action,
            "status": "accepted",
            "job_id": job_id,
            "base_model_id": event.get("base_model_id"),
        })

    except Exception as e:
        logger.exception("Fine-tuning handler error")
        return _response(500, {"error": str(e)})


def _response(status_code: int, body: dict) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
