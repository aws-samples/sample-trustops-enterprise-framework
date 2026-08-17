"""Lambda handler for evaluating a single example against a Bedrock model."""

import json
import os
import time

import boto3
from botocore.exceptions import ClientError

TRANSIENT_ERROR_CODES = (
    "ThrottlingException",
    "ServiceUnavailableException",
    "TooManyRequestsException",
)


def handler(event: dict, context) -> dict:
    for field in ("prompt", "model_id", "example_id"):
        if field not in event:
            return {
                "statusCode": 400,
                "error": "ValidationError",
                "message": f"Missing required field: {field}",
            }

    prompt = event["prompt"]
    model_id = event["model_id"]
    example_id = event["example_id"]

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

    max_tokens = int(event.get("max_tokens", 2048))

    try:
        start = time.time()
        # Use the unified Converse API so this handler works across model
        # families (Amazon Nova, Claude 3+, Llama 3, etc.) without per-provider
        # request/response schemas.
        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens},
        )
        latency_ms = (time.time() - start) * 1000

        content = response.get("output", {}).get("message", {}).get("content", [])
        model_response = content[0].get("text", "") if content else ""
        usage = response.get("usage", {})

        return {
            "model_response": model_response,
            "latency_ms": latency_ms,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "example_id": example_id,
        }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in TRANSIENT_ERROR_CODES:
            raise
        return {
            "statusCode": 500,
            "error": "ProcessingError",
            "message": str(e),
        }
