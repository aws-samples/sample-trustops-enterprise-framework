"""Lambda handler for aggregating evaluation metrics from S3 results."""

import json
import os
import statistics

import boto3
from botocore.exceptions import ClientError

TRANSIENT_ERROR_CODES = (
    "ThrottlingException",
    "ServiceUnavailableException",
    "TooManyRequestsException",
)


def handler(event: dict, context) -> dict:
    for field in ("result_s3_uris", "model_id", "output_s3_uri"):
        if field not in event:
            return {
                "statusCode": 400,
                "error": "ValidationError",
                "message": f"Missing required field: {field}",
            }

    result_s3_uris = event["result_s3_uris"]
    model_id = event["model_id"]
    output_s3_uri = event["output_s3_uri"]

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    try:
        results = []
        for uri in result_s3_uris:
            bucket, key = _parse_s3_uri(uri)
            response = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(response["Body"].read())
            results.append(data)

        trust_scores = [r.get("trust_score", {}).get("overall_score", 0.0) for r in results]
        hallucination_rates = [
            r.get("hallucination_analysis", {}).get("hallucination_rate", 0.0)
            for r in results
        ]
        latencies = [r.get("latency_ms", 0.0) for r in results]
        input_tokens = sum(r.get("input_tokens", 0) for r in results)
        output_tokens = sum(r.get("output_tokens", 0) for r in results)

        total_cost = (input_tokens / 1000) * 0.008 + (output_tokens / 1000) * 0.024

        metrics = {
            "model_id": model_id,
            "total_examples": len(results),
            "mean_trust_score": statistics.mean(trust_scores) if trust_scores else 0.0,
            "median_trust_score": statistics.median(trust_scores) if trust_scores else 0.0,
            "hallucination_rate": statistics.mean(hallucination_rates) if hallucination_rates else 0.0,
            "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "total_cost": total_cost,
        }

        # Store output
        out_bucket, out_key = _parse_s3_uri(output_s3_uri)
        s3.put_object(
            Bucket=out_bucket,
            Key=out_key,
            Body=json.dumps(metrics),
            ContentType="application/json",
        )

        return {"metrics": metrics, "output_s3_uri": output_s3_uri}

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in TRANSIENT_ERROR_CODES:
            raise
        return {
            "statusCode": 500,
            "error": "ProcessingError",
            "message": str(e),
        }


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    path = uri.replace("s3://", "")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""
