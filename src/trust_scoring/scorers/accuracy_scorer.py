"""
Accuracy Scorer for Trust Scoring Engine.

This module implements the accuracy dimension of trust scoring, comparing
model responses against expected answers using multiple methods:
- Exact match: Case-insensitive, normalized string comparison
- Fuzzy match: Levenshtein distance-based similarity
- Semantic similarity: Embedding-based cosine similarity

Requirements: 5.1, 5.3
"""

import re
from typing import Optional

from src.aws_clients.semantic_similarity_analyzer import SemanticSimilarityAnalyzer
from src.data_models.trust_score import DimensionScore, TrustDimension


class AccuracyScorer:
    """
    Calculate accuracy scores by comparing responses to expected answers.
    
    The accuracy scorer supports three scoring methods:
    1. Exact Match: Normalized string comparison (case-insensitive, whitespace normalized)
    2. Fuzzy Match: Levenshtein distance-based similarity ratio
    3. Semantic Similarity: Embedding-based cosine similarity
    
    The final accuracy score is the maximum of all three methods, ensuring
    that semantically correct answers score well even if phrasing differs.
    
    Requirement 5.3: Implement accuracy scorer with exact, fuzzy, and semantic matching
    """
    
    def __init__(
        self,
        similarity_analyzer: Optional[SemanticSimilarityAnalyzer] = None,
        fuzzy_threshold: float = 0.8,
        semantic_threshold: float = 0.7
    ):
        """
        Initialize the accuracy scorer.
        
        Args:
            similarity_analyzer: Semantic similarity analyzer for embedding-based comparison
            fuzzy_threshold: Threshold for fuzzy match to be considered accurate (default: 0.8)
            semantic_threshold: Threshold for semantic similarity to be considered accurate (default: 0.7)
        """
        self.similarity_analyzer = similarity_analyzer or SemanticSimilarityAnalyzer()
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
    
    async def calculate_accuracy(
        self,
        response: str,
        expected_response: Optional[str] = None
    ) -> DimensionScore:
        """
        Calculate accuracy score for a response against expected answer.
        
        This method combines three scoring approaches:
        1. Exact match (normalized)
        2. Fuzzy match (Levenshtein distance)
        3. Semantic similarity (embeddings)
        
        The final score is the maximum of all three methods to ensure
        semantically correct answers are recognized.
        
        Args:
            response: The model's response text
            expected_response: The expected/reference answer (optional)
            
        Returns:
            DimensionScore with accuracy score in [0, 1] range
            
        Requirement 5.3: Compare response against expected answer using
        exact match, fuzzy match, and semantic similarity
        """
        # If no expected response provided, return neutral score
        if not expected_response or not expected_response.strip():
            return DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.5,
                confidence=0.3,
                details={
                    "method": "no_reference",
                    "message": "No expected response provided for comparison"
                },
                checks_passed=[],
                checks_failed=["no_reference_answer"]
            )
        
        # If response is empty, return zero score
        if not response or not response.strip():
            return DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.0,
                confidence=1.0,
                details={
                    "method": "empty_response",
                    "message": "Response is empty"
                },
                checks_passed=[],
                checks_failed=["empty_response"]
            )
        
        # Calculate all three scoring methods
        exact_score = self._exact_match_score(response, expected_response)
        fuzzy_score = self._fuzzy_match_score(response, expected_response)
        semantic_score = await self._semantic_similarity_score(response, expected_response)
        
        # Use maximum score across all methods
        # This ensures semantically correct answers score well
        final_score = max(exact_score, fuzzy_score, semantic_score)
        
        # Determine which checks passed
        checks_passed = []
        checks_failed = []
        
        if exact_score == 1.0:
            checks_passed.append("exact_match")
        else:
            checks_failed.append("exact_match")
        
        if fuzzy_score >= self.fuzzy_threshold:
            checks_passed.append("fuzzy_match")
        else:
            checks_failed.append("fuzzy_match")
        
        if semantic_score >= self.semantic_threshold:
            checks_passed.append("semantic_similarity")
        else:
            checks_failed.append("semantic_similarity")
        
        # Calculate confidence based on agreement between methods
        # High confidence when multiple methods agree
        scores = [exact_score, fuzzy_score, semantic_score]
        score_variance = sum((s - final_score) ** 2 for s in scores) / len(scores)
        confidence = max(0.5, 1.0 - score_variance)
        
        return DimensionScore(
            dimension=TrustDimension.ACCURACY,
            score=final_score,
            confidence=confidence,
            details={
                "exact_match_score": exact_score,
                "fuzzy_match_score": fuzzy_score,
                "semantic_similarity_score": semantic_score,
                "method_used": "maximum",
                "response_length": len(response),
                "expected_length": len(expected_response)
            },
            checks_passed=checks_passed,
            checks_failed=checks_failed
        )
    
    def _exact_match_score(self, response: str, expected: str) -> float:
        """
        Calculate exact match score with normalization.
        
        Normalization includes:
        - Convert to lowercase
        - Remove extra whitespace
        - Remove punctuation
        - Trim leading/trailing whitespace
        
        Args:
            response: Model response
            expected: Expected answer
            
        Returns:
            1.0 if exact match after normalization, 0.0 otherwise
        """
        normalized_response = self._normalize_text(response)
        normalized_expected = self._normalize_text(expected)
        
        return 1.0 if normalized_response == normalized_expected else 0.0
    
    def _fuzzy_match_score(self, response: str, expected: str) -> float:
        """
        Calculate fuzzy match score using Levenshtein distance.
        
        The Levenshtein distance measures the minimum number of single-character
        edits (insertions, deletions, substitutions) needed to transform one
        string into another. This is converted to a similarity ratio in [0, 1].
        
        Args:
            response: Model response
            expected: Expected answer
            
        Returns:
            Similarity ratio in [0, 1] where 1.0 is identical
        """
        # Normalize both strings for comparison
        normalized_response = self._normalize_text(response)
        normalized_expected = self._normalize_text(expected)
        
        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(normalized_response, normalized_expected)
        
        # Convert to similarity ratio
        max_len = max(len(normalized_response), len(normalized_expected))
        if max_len == 0:
            return 1.0  # Both empty strings
        
        similarity = 1.0 - (distance / max_len)
        return max(0.0, min(1.0, similarity))
    
    async def _semantic_similarity_score(self, response: str, expected: str) -> float:
        """
        Calculate semantic similarity using embeddings.
        
        This method uses the Bedrock Titan Embed v2 model (via SemanticSimilarityAnalyzer)
        to compute embeddings for both texts and calculate cosine similarity.
        
        Args:
            response: Model response
            expected: Expected answer
            
        Returns:
            Cosine similarity in [0, 1] range
        """
        try:
            # Use semantic similarity analyzer to compute embedding-based similarity
            similarity = self.similarity_analyzer.calculate_similarity(response, expected)
            
            # Ensure score is in valid range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            # If semantic similarity fails, return neutral score
            # This ensures the scorer is resilient to API failures
            return 0.5
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Normalization steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Normalize whitespace (collapse multiple spaces)
        4. Trim leading/trailing whitespace
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation (keep alphanumeric and spaces)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        The Levenshtein distance is the minimum number of single-character
        edits (insertions, deletions, or substitutions) required to change
        one string into the other.
        
        This implementation uses dynamic programming with O(m*n) time complexity
        and O(min(m,n)) space complexity.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Levenshtein distance (number of edits)
        """
        # Ensure s1 is the shorter string for space optimization
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        # Handle edge cases
        if len(s1) == 0:
            return len(s2)
        if len(s2) == 0:
            return len(s1)
        
        # Initialize previous row of distances
        # This is the edit distance from empty string to s2[:j]
        previous_row = range(len(s2) + 1)
        
        # Calculate edit distances
        for i, c1 in enumerate(s1):
            # Current row starts with edit distance from s1[:i+1] to empty string
            current_row = [i + 1]
            
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (0 if c1 == c2 else 1)
                
                # Take minimum cost
                current_row.append(min(insertions, deletions, substitutions))
            
            # Move to next row
            previous_row = current_row
        
        return previous_row[-1]
