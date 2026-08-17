"""
AWS Lambda handler for trust score calculation.

Receives scoring requests, delegates to the TrustScoringEngine,
and returns trust score results.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point for trust scoring requests.

    Expects event with:
        - response (str): Model-generated response text.
        - prompt (str): Original prompt.
        - source_documents (list[str], optional): Context documents.
        - expected_schema (dict, optional): Expected output schema.

    Returns:
        dict with statusCode and JSON body containing trust score.
    """
    try:
        logger.info("Trust scoring request received")

        response_text = event.get("response")
        prompt = event.get("prompt")

        if not response_text or not prompt:
            return _response(400, {"error": "response and prompt are required"})

        from src.trust_scoring.trust_scoring_engine import TrustScoringEngine

        engine = TrustScoringEngine()
        source_documents = event.get("source_documents", [])
        expected_schema = event.get("expected_schema")

        result = engine.calculate_trust_score(
            response=response_text,
            prompt=prompt,
            source_documents=source_documents,
            expected_schema=expected_schema,
        )

        return _response(200, {
            "overall_score": result.overall_score,
            "confidence_level": result.confidence_level,
            "flagged_for_review": result.flagged_for_review,
            "explanation": result.explanation,
            "components": {
                "context_grounding": result.components.context_grounding,
                "output_structure": result.components.output_structure,
                "uncertainty_indicators": result.components.uncertainty_indicators,
                "factual_consistency": result.components.factual_consistency,
                "response_completeness": result.components.response_completeness,
            },
        })

    except Exception as e:
        logger.exception("Trust scoring handler error")
        return _response(500, {"error": str(e)})


def _response(status_code: int, body: dict) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
