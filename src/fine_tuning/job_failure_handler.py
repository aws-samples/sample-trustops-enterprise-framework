"""
Job failure handler for fine-tuning jobs.

Parses error messages from provider responses and provides
user-friendly recovery suggestions based on the error type.

Requirements: 4.12
"""

from dataclasses import dataclass, field
from typing import Any

from src.data_models.fine_tuning import FineTuningJob, FineTuningStatus


@dataclass
class FailureAnalysis:
    """Analysis of a failed fine-tuning job.

    Attributes:
        job_id: The ID of the failed job.
        error_category: Categorized error type.
        error_message: The original error message from the provider.
        user_friendly_message: A clear explanation of what went wrong.
        recovery_suggestions: Actionable steps the user can take.
        is_retryable: Whether the error is likely transient and retryable.
    """

    job_id: str
    error_category: str
    error_message: str
    user_friendly_message: str
    recovery_suggestions: list[str] = field(default_factory=list)
    is_retryable: bool = False


# Error patterns and their categorizations
_ERROR_PATTERNS: list[tuple[list[str], str, str, list[str], bool]] = [
    # (keywords, category, user_message, suggestions, retryable)
    (
        ["access denied", "not authorized", "permission", "forbidden"],
        "permission_error",
        "The fine-tuning job failed due to insufficient permissions.",
        [
            "Verify the IAM role has the required permissions for fine-tuning.",
            "Check that the role can access the training data S3 bucket.",
            "Ensure the role has bedrock:CreateModelCustomizationJob permission.",
        ],
        False,
    ),
    (
        ["throttl", "rate limit", "too many requests", "rate exceeded"],
        "throttling_error",
        "The fine-tuning job was throttled due to rate limits.",
        [
            "Wait a few minutes and retry the job.",
            "Check your AWS account service quotas for fine-tuning.",
            "Consider requesting a quota increase if this persists.",
        ],
        True,
    ),
    (
        ["resource limit", "quota", "limit exceeded", "capacity"],
        "quota_error",
        "The fine-tuning job exceeded a resource quota or capacity limit.",
        [
            "Check your AWS account fine-tuning quotas.",
            "Request a service quota increase via the AWS console.",
            "Try using a smaller dataset or fewer epochs.",
        ],
        False,
    ),
    (
        ["invalid", "validation", "format", "malformed", "schema"],
        "validation_error",
        "The fine-tuning job failed due to data validation issues.",
        [
            "Verify your training data format matches the model requirements.",
            "Run the format validator before submitting the job.",
            "Check that all required fields are present in every record.",
        ],
        False,
    ),
    (
        ["timeout", "timed out", "deadline"],
        "timeout_error",
        "The fine-tuning job timed out.",
        [
            "The training may have taken longer than expected.",
            "Try reducing the dataset size or number of epochs.",
            "Retry the job — transient infrastructure issues may have caused this.",
        ],
        True,
    ),
    (
        ["not found", "does not exist", "no such"],
        "not_found_error",
        "A required resource was not found.",
        [
            "Verify the base model ID is correct and available in your region.",
            "Check that the training data S3 URI is accessible.",
            "Ensure the output S3 bucket exists.",
        ],
        False,
    ),
    (
        ["internal", "server error", "service error", "500"],
        "internal_error",
        "An internal service error occurred during fine-tuning.",
        [
            "This is typically a transient issue. Retry the job.",
            "If the error persists, check the AWS Health Dashboard.",
            "Contact AWS Support if the issue continues.",
        ],
        True,
    ),
    (
        ["out of memory", "oom", "memory"],
        "memory_error",
        "The fine-tuning job ran out of memory.",
        [
            "Try reducing the batch size in your hyperparameters.",
            "Reduce the max_seq_length parameter.",
            "Use a smaller model variant if available.",
        ],
        False,
    ),
    (
        ["convergence", "nan", "diverge", "explod"],
        "training_error",
        "The training process encountered numerical issues.",
        [
            "Try reducing the learning rate.",
            "Increase warmup steps to stabilize early training.",
            "Check your training data for outliers or corrupted examples.",
        ],
        False,
    ),
]


def _categorize_error(error_message: str) -> tuple[str, str, list[str], bool]:
    """Categorize an error message and return recovery info.

    Args:
        error_message: The raw error message from the provider.

    Returns:
        Tuple of (category, user_message, suggestions, is_retryable).
    """
    error_lower = error_message.lower()

    for keywords, category, user_msg, suggestions, retryable in _ERROR_PATTERNS:
        if any(kw in error_lower for kw in keywords):
            return category, user_msg, suggestions, retryable

    # Default unknown error
    return (
        "unknown_error",
        "The fine-tuning job failed with an unexpected error.",
        [
            "Review the full error message for details.",
            "Check AWS CloudWatch logs for additional context.",
            "Retry the job if the error seems transient.",
            "Contact support if the issue persists.",
        ],
        False,
    )


def handle_job_failure(job: FineTuningJob) -> FailureAnalysis:
    """Analyze a failed fine-tuning job and provide recovery suggestions.

    Parses the error message from the job, categorizes the failure,
    and returns user-friendly guidance for resolution.

    Args:
        job: The failed fine-tuning job.

    Returns:
        FailureAnalysis with categorized error and recovery suggestions.
    """
    if job.status != FineTuningStatus.FAILED:
        return FailureAnalysis(
            job_id=job.job_id,
            error_category="not_failed",
            error_message="",
            user_friendly_message=(
                f"Job '{job.job_id}' has not failed "
                f"(status: {job.status.value})."
            ),
            recovery_suggestions=[],
            is_retryable=False,
        )

    error_msg = job.error_message or "No error message provided"
    category, user_msg, suggestions, retryable = _categorize_error(error_msg)

    return FailureAnalysis(
        job_id=job.job_id,
        error_category=category,
        error_message=error_msg,
        user_friendly_message=user_msg,
        recovery_suggestions=suggestions,
        is_retryable=retryable,
    )


def format_failure_report(analysis: FailureAnalysis) -> str:
    """Format a failure analysis into a human-readable report.

    Args:
        analysis: The failure analysis to format.

    Returns:
        Formatted string report.
    """
    lines = [
        f"Fine-Tuning Job Failure Report",
        "=" * 45,
        f"Job ID:    {analysis.job_id}",
        f"Category:  {analysis.error_category}",
        f"Retryable: {'Yes' if analysis.is_retryable else 'No'}",
        "",
        f"What happened:",
        f"  {analysis.user_friendly_message}",
        "",
        f"Original error:",
        f"  {analysis.error_message}",
        "",
        "Recovery suggestions:",
    ]
    for i, suggestion in enumerate(analysis.recovery_suggestions, 1):
        lines.append(f"  {i}. {suggestion}")
    lines.append("=" * 45)
    return "\n".join(lines)
