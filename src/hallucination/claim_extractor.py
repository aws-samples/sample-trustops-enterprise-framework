"""
Claim Extractor for Hallucination Detection.

This module implements claim extraction from model responses, splitting text
into individual claims using sentence segmentation and identifying claim
boundaries for hallucination analysis.

Requirements: 6.1
"""

import re
from typing import Optional

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from src.data_models.hallucination import Claim, HallucinationConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ClaimExtractor:
    """
    Extract individual claims from model responses.
    
    The claim extractor splits response text into individual claims using
    sentence segmentation (NLTK or fallback regex-based approach). Each claim
    is identified with its text content and character position boundaries
    (start_idx, end_idx) in the original response.
    
    Claims are classified as factual or opinion based on hedging language
    detection. This classification is used downstream to filter which claims
    should be verified against source documents.
    
    Requirement 6.1: Extract individual claims from model responses
    """
    
    def __init__(self, config: Optional[HallucinationConfig] = None):
        """
        Initialize the claim extractor.
        
        Args:
            config: Hallucination detection configuration (optional)
        """
        self.config = config or HallucinationConfig()
        
        # Ensure NLTK punkt tokenizer is available
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                # Download punkt tokenizer if not available
                try:
                    nltk.download('punkt', quiet=True)
                except Exception as e:
                    # If download fails, we'll fall back to regex
                    logger.debug(
                        "NLTK punkt download failed, using regex fallback: %s", e
                    )
    
    def extract_claims(self, response: str) -> list[Claim]:
        """
        Extract individual claims from a model response.
        
        This method splits the response into sentences using NLTK's sentence
        tokenizer (or a fallback regex-based approach if NLTK is unavailable).
        Each sentence is treated as a claim and classified as factual or opinion
        based on hedging language detection.
        
        Args:
            response: The model response text to extract claims from
            
        Returns:
            List of Claim objects with text, start_idx, end_idx, and classification
            
        Requirement 6.1: Split response into individual claims using sentence
        segmentation and identify claim boundaries (start_idx, end_idx)
        """
        if not response or not response.strip():
            return []
        
        # Split response into sentences
        sentences = self._segment_sentences(response)
        
        # Extract claims with position tracking
        claims = []
        current_pos = 0
        
        for sentence in sentences:
            # Find the sentence in the original text starting from current position
            # This preserves exact character positions including whitespace
            start_idx = response.find(sentence, current_pos)
            
            if start_idx == -1:
                # Sentence not found (shouldn't happen with proper segmentation)
                # Skip this sentence
                continue
            
            end_idx = start_idx + len(sentence)
            
            # Classify the claim
            is_factual, is_opinion, is_hedged = self._classify_claim(sentence)
            
            # Create claim object
            claim = Claim(
                text=sentence,
                start_idx=start_idx,
                end_idx=end_idx,
                is_factual=is_factual,
                is_opinion=is_opinion,
                is_hedged=is_hedged
            )
            
            claims.append(claim)
            
            # Update position for next search
            current_pos = end_idx
            
            # Respect max claims limit
            if len(claims) >= self.config.max_claims_per_response:
                break
        
        return claims
    
    def _segment_sentences(self, text: str) -> list[str]:
        """
        Segment text into sentences using NLTK or fallback regex.
        
        Prefers NLTK's sent_tokenize for accurate sentence boundary detection.
        Falls back to regex-based segmentation if NLTK is unavailable.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of sentence strings
        """
        if NLTK_AVAILABLE:
            try:
                # Use NLTK's sentence tokenizer for accurate segmentation
                sentences = sent_tokenize(text)
                return [s.strip() for s in sentences if s.strip()]
            except Exception as e:
                # Fall back to regex if NLTK fails
                logger.debug(
                    "NLTK sentence tokenization failed, using regex fallback: %s", e
                )
        
        # Fallback: regex-based sentence segmentation
        # Split on sentence-ending punctuation followed by whitespace and capital letter
        # This is a simple heuristic and less accurate than NLTK
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _classify_claim(self, claim_text: str) -> tuple[bool, bool, bool]:
        """
        Classify a claim as factual/opinion and hedged/unhedged.
        
        Classification logic:
        - Hedged: Contains hedging keywords (might, possibly, I think, etc.)
        - Opinion: Contains opinion indicators or is hedged
        - Factual: Not an opinion (can still be hedged)
        
        Args:
            claim_text: The claim text to classify
            
        Returns:
            Tuple of (is_factual, is_opinion, is_hedged)
        """
        claim_lower = claim_text.lower()
        
        # Check for hedging language
        is_hedged = any(
            keyword in claim_lower
            for keyword in self.config.hedging_keywords
        )
        
        # Opinion indicators (in addition to hedging)
        opinion_indicators = [
            "i think", "i believe", "in my opinion", "i feel",
            "it seems", "it appears", "arguably", "presumably",
            "i would say", "i suspect", "i assume"
        ]
        
        has_opinion_indicator = any(
            indicator in claim_lower
            for indicator in opinion_indicators
        )
        
        # Classify as opinion if it has opinion indicators or is heavily hedged
        is_opinion = has_opinion_indicator or is_hedged
        
        # Factual claims are those that are not opinions
        # Note: A factual claim can still be hedged (e.g., "The data might indicate...")
        is_factual = not is_opinion
        
        return is_factual, is_opinion, is_hedged
