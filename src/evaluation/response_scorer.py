"""
Per-response scoring for the Evaluation Engine.

Combines trust scoring and hallucination detection for each evaluation
response, bridging batch runner results with the scoring engines.

Requirements: 3.4
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.data_models.hallucination import HallucinationResult
from src.data_models.trust_score import TrustScoreResult
from src.evaluation.batch_runner import BatchItemResult
from src.hallucination.hallucination_detector import HallucinationDetector
from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine

logger = logging.getLogger(__name__)


@dataclass
class ScoredResponse:
    """Result of scoring a single response.

    Attributes:
        batch_item: The original batch item result.
        trust_score_result: Trust score from the TrustScoringEngine.
        hallucination_result: Hallucination detection result (None if
            detector not configured or no source documents).
        scoring_error: Error message if scoring failed.
    """

    batch_item: BatchItemResult
    trust_score_result: Optional[TrustScoreResult] = None
    hallucination_result: Optional[HallucinationResult] = None
    scoring_error: Optional[str] = None


class ResponseScorer:
    """Score individual responses using trust scoring and hallucination detection.

    This class integrates TrustScoringEngine and HallucinationDetector to
    produce a combined ScoredResponse for each batch item.

    Requirements:
    - 3.4: Calculate Trust_Score for each response
    """

    def __init__(
        self,
        trust_engine: TrustScoringEngine,
        hallucination_detector: Optional[HallucinationDetector] = None,
    ):
        """
        Args:
            trust_engine: Engine for computing trust scores.
            hallucination_detector: Optional detector for hallucination analysis.
        """
        self.trust_engine = trust_engine
        self.hallucination_detector = hallucination_detector

    async def score_response(
        self,
        item: BatchItemResult,
        source_documents: Optional[list[str]] = None,
        model_id: Optional[str] = None,
    ) -> ScoredResponse:
        """Score a single batch item response.

        Runs hallucination detection first (if detector configured and
        source_documents available), then passes the result to the trust
        scoring engine for enhanced grounding.

        Args:
            item: A BatchItemResult from the batch runner.
            source_documents: Optional context documents for grounding.
            model_id: Optional model ID for consistency scoring.

        Returns:
            ScoredResponse with trust and hallucination results.
        """
        if not item.success or not item.response_text:
            return ScoredResponse(
                batch_item=item,
                scoring_error=item.error or "No response to score",
            )

        try:
            # Step 1: Hallucination detection (sync call, if configured)
            hallucination_result: Optional[HallucinationResult] = None
            if self.hallucination_detector is not None and source_documents:
                hallucination_result = self.hallucination_detector.detect(
                    response=item.response_text,
                    source_documents=source_documents,
                )

            # Step 2: Trust scoring (async), passing hallucination result
            trust_score_result = await self.trust_engine.score_response(
                prompt=item.prompt,
                response=item.response_text,
                expected_response=item.expected_response,
                source_documents=source_documents,
                model_id=model_id,
                hallucination_result=hallucination_result,
            )

            return ScoredResponse(
                batch_item=item,
                trust_score_result=trust_score_result,
                hallucination_result=hallucination_result,
            )
        except Exception as exc:
            logger.warning(
                "Scoring failed for item %d: %s", item.index, exc
            )
            return ScoredResponse(
                batch_item=item,
                scoring_error=str(exc),
            )

    async def score_batch(
        self,
        items: list[BatchItemResult],
        source_documents: Optional[list[str]] = None,
        model_id: Optional[str] = None,
    ) -> list[ScoredResponse]:
        """Score multiple batch item responses.

        Processes each item individually. If scoring fails for one response,
        the error is captured and processing continues with the remaining items.

        Args:
            items: List of BatchItemResult objects.
            source_documents: Optional context documents shared across items.
            model_id: Optional model ID for consistency scoring.

        Returns:
            List of ScoredResponse objects, one per input item.
        """
        results: list[ScoredResponse] = []
        for item in items:
            scored = await self.score_response(
                item=item,
                source_documents=source_documents,
                model_id=model_id,
            )
            results.append(scored)
        return results
