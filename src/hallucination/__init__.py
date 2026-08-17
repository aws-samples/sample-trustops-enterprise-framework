"""
Hallucination detection and measurement components.

This module provides tools for detecting and measuring hallucinations in
model outputs by comparing claims against source documents.

Requirements: 6.1-6.13
"""

from src.hallucination.claim_classifier import ClaimClassifier
from src.hallucination.claim_extractor import ClaimExtractor
from src.hallucination.evidence_searcher import EvidenceSearcher
from src.hallucination.hallucination_detector import HallucinationDetector

__all__ = [
    "ClaimClassifier",
    "ClaimExtractor",
    "EvidenceSearcher",
    "HallucinationDetector",
]
