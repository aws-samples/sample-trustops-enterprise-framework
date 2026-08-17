"""
Trust Scoring Engine for calculating quantifiable trust scores for model outputs.
"""
import re
from typing import List, Optional, Dict, Any
from src.data_models.model_response import TrustScore, TrustScoreComponents, HallucinationAnalysis, HallucinationSpan
from src.aws_clients.semantic_similarity_analyzer import SemanticSimilarityAnalyzer


class TrustScoringEngine:
    """
    Calculate quantifiable trust scores for model outputs using multiple signals.
    
    Trust Score Components:
    - Context Grounding (30%): Semantic similarity with source documents
    - Output Structure (20%): Schema compliance and completeness
    - Uncertainty Indicators (15%): Detection of hedging language
    - Factual Consistency (15%): Internal contradiction detection
    - Response Completeness (20%): Presence of required information
    """
    
    # Component weights
    WEIGHTS = {
        'context_grounding': 0.30,
        'output_structure': 0.20,
        'uncertainty_indicators': 0.15,
        'factual_consistency': 0.15,
        'response_completeness': 0.20
    }
    
    # Hedging language patterns
    HEDGING_PATTERNS = [
        r'\b(might|may|could|possibly|perhaps|maybe|probably|likely|unlikely)\b',
        r'\b(seems?|appears?|suggests?|indicates?)\b',
        r'\b(I think|I believe|I guess|I suppose|in my opinion)\b',
        r'\b(not sure|uncertain|unclear|ambiguous)\b',
        r'\b(approximately|roughly|about|around)\b',
        r'\b(some|several|many|few|most)\b',
    ]
    
    # Contradiction patterns
    CONTRADICTION_PATTERNS = [
        (r'\b(yes|true|correct)\b', r'\b(no|false|incorrect)\b'),
        (r'\b(always|never)\b', r'\b(sometimes|occasionally)\b'),
        (r'\b(all|every|none)\b', r'\b(some|few|many)\b'),
        (r'\b(increase|rise|grow)\b', r'\b(decrease|fall|decline)\b'),
    ]
    
    def __init__(
        self,
        similarity_analyzer: Optional[SemanticSimilarityAnalyzer] = None,
        review_threshold: float = 0.6
    ):
        """
        Initialize trust scoring engine.
        
        Args:
            similarity_analyzer: Semantic similarity analyzer instance
            review_threshold: Trust score threshold for flagging reviews (default: 0.6)
        """
        self.similarity_analyzer = similarity_analyzer or SemanticSimilarityAnalyzer()
        self.review_threshold = review_threshold
    
    def calculate_trust_score(
        self,
        response: str,
        prompt: str,
        source_documents: List[str],
        expected_schema: Optional[Dict] = None
    ) -> TrustScore:
        """
        Calculate composite trust score for a model response.
        
        Args:
            response: Model-generated response text
            prompt: Original prompt/query
            source_documents: Context documents for grounding check
            expected_schema: Optional schema for structure validation
            
        Returns:
            TrustScore object with overall score and component breakdown
        """
        # Calculate individual components
        context_grounding = self._calculate_context_grounding(
            response, source_documents
        )
        output_structure = self._validate_output_structure(
            response, expected_schema
        )
        uncertainty_indicators = self._detect_uncertainty_indicators(response)
        factual_consistency = self._check_factual_consistency(response)
        response_completeness = self._check_response_completeness(
            response, prompt
        )
        
        # Create components object
        components = TrustScoreComponents(
            context_grounding=context_grounding,
            output_structure=output_structure,
            uncertainty_indicators=uncertainty_indicators,
            factual_consistency=factual_consistency,
            response_completeness=response_completeness
        )
        
        # Calculate weighted overall score
        overall_score = (
            context_grounding * self.WEIGHTS['context_grounding'] +
            output_structure * self.WEIGHTS['output_structure'] +
            uncertainty_indicators * self.WEIGHTS['uncertainty_indicators'] +
            factual_consistency * self.WEIGHTS['factual_consistency'] +
            response_completeness * self.WEIGHTS['response_completeness']
        )
        
        # Ensure score is in [0, 1] range
        overall_score = max(0.0, min(1.0, overall_score))
        
        # Determine confidence level
        if overall_score >= 0.8:
            confidence_level = "high"
        elif overall_score >= 0.6:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Flag for review if below threshold
        flagged_for_review = overall_score < self.review_threshold
        
        # Generate explanation
        explanation = self._generate_explanation(components, overall_score)
        
        return TrustScore(
            overall_score=overall_score,
            components=components,
            confidence_level=confidence_level,
            flagged_for_review=flagged_for_review,
            explanation=explanation
        )
    
    def detect_hallucinations(
        self,
        response: str,
        source_documents: List[str],
        similarity_threshold: float = 0.7
    ) -> HallucinationAnalysis:
        """
        Detect potential hallucinations in response.
        
        Args:
            response: Model-generated response text
            source_documents: Ground truth documents
            similarity_threshold: Minimum similarity for grounding
            
        Returns:
            HallucinationAnalysis with flagged spans and evidence scores
        """
        # Extract claims from response (sentences)
        claims = self._extract_claims(response)
        
        if not claims:
            return HallucinationAnalysis(
                has_hallucinations=False,
                hallucination_rate=0.0,
                flagged_spans=[],
                overall_grounding_score=1.0
            )
        
        flagged_spans = []
        grounding_scores = []
        
        # Check each claim against source documents
        for claim_text, start_idx, end_idx in claims:
            # Find best matching document
            grounding_score, evidence_docs = self._find_grounding_evidence(
                claim_text, source_documents
            )
            
            grounding_scores.append(grounding_score)
            
            # Flag if below threshold
            if grounding_score < similarity_threshold:
                flagged_spans.append(
                    HallucinationSpan(
                        text=claim_text,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        grounding_score=grounding_score,
                        evidence_documents=evidence_docs
                    )
                )
        
        # Calculate metrics
        hallucination_rate = len(flagged_spans) / len(claims) if claims else 0.0
        has_hallucinations = len(flagged_spans) > 0
        overall_grounding_score = (
            sum(grounding_scores) / len(grounding_scores)
            if grounding_scores else 0.0
        )
        
        return HallucinationAnalysis(
            has_hallucinations=has_hallucinations,
            hallucination_rate=hallucination_rate,
            flagged_spans=flagged_spans,
            overall_grounding_score=overall_grounding_score
        )
    
    def _calculate_context_grounding(
        self,
        response: str,
        source_documents: List[str]
    ) -> float:
        """
        Calculate context grounding score using semantic similarity.
        
        Args:
            response: Model response text
            source_documents: Source documents for grounding
            
        Returns:
            Grounding score (0-1)
        """
        if not source_documents:
            # No source documents to ground against
            return 0.5  # Neutral score
        
        try:
            # Calculate similarity with each source document
            similarities = []
            for doc in source_documents:
                if doc.strip():  # Skip empty documents
                    similarity = self.similarity_analyzer.calculate_similarity(
                        response, doc
                    )
                    similarities.append(similarity)
            
            if not similarities:
                return 0.5
            
            # Return maximum similarity (best grounding)
            return max(similarities)
            
        except Exception:
            # If similarity calculation fails, return neutral score
            return 0.5
    
    def _validate_output_structure(
        self,
        response: str,
        expected_schema: Optional[Dict]
    ) -> float:
        """
        Validate output structure compliance.
        
        Args:
            response: Model response text
            expected_schema: Expected schema (optional)
            
        Returns:
            Structure validation score (0-1)
        """
        if expected_schema is None:
            # No schema to validate against, check basic structure
            return self._check_basic_structure(response)
        
        # Schema validation logic would go here
        # For now, implement basic checks
        score = 0.0
        checks = 0
        
        # Check if response is not empty
        if response and response.strip():
            score += 1.0
            checks += 1
        
        # Check for reasonable length
        if 10 <= len(response) <= 10000:
            score += 1.0
            checks += 1
        
        # Check for proper formatting (no excessive special characters)
        special_char_ratio = sum(
            1 for c in response if not c.isalnum() and not c.isspace()
        ) / max(len(response), 1)
        
        if special_char_ratio < 0.3:
            score += 1.0
            checks += 1
        
        return score / max(checks, 1)
    
    def _check_basic_structure(self, response: str) -> float:
        """
        Check basic response structure quality.
        
        Args:
            response: Model response text
            
        Returns:
            Structure score (0-1)
        """
        if not response or not response.strip():
            return 0.0
        
        score = 0.0
        
        # Check 1: Has reasonable length
        if 10 <= len(response) <= 10000:
            score += 0.25
        
        # Check 2: Has proper sentence structure
        sentences = re.split(r'[.!?]+', response)
        if len(sentences) >= 1:
            score += 0.25
        
        # Check 3: Not too many special characters
        special_char_ratio = sum(
            1 for c in response if not c.isalnum() and not c.isspace()
        ) / max(len(response), 1)
        
        if special_char_ratio < 0.3:
            score += 0.25
        
        # Check 4: Has proper capitalization
        if response[0].isupper():
            score += 0.25
        
        return score
    
    def _detect_uncertainty_indicators(self, response: str) -> float:
        """
        Detect uncertainty indicators (hedging language).
        
        Args:
            response: Model response text
            
        Returns:
            Certainty score (0-1), higher means more certain
        """
        if not response:
            return 0.5
        
        # Count hedging patterns
        hedging_count = 0
        response_lower = response.lower()
        
        for pattern in self.HEDGING_PATTERNS:
            matches = re.findall(pattern, response_lower, re.IGNORECASE)
            hedging_count += len(matches)
        
        # Calculate hedging density (hedges per 100 words)
        word_count = len(response.split())
        hedging_density = (hedging_count / max(word_count, 1)) * 100
        
        # Convert to certainty score (inverse of hedging)
        # 0 hedges = 1.0 score, 10+ hedges per 100 words = 0.0 score
        certainty_score = max(0.0, 1.0 - (hedging_density / 10.0))
        
        return certainty_score
    
    def _check_factual_consistency(self, response: str) -> float:
        """
        Check for internal contradictions in response.
        
        Args:
            response: Model response text
            
        Returns:
            Consistency score (0-1)
        """
        if not response:
            return 1.0  # Empty response has no contradictions
        
        response_lower = response.lower()
        contradiction_count = 0
        
        # Check for contradictory patterns
        for pattern1, pattern2 in self.CONTRADICTION_PATTERNS:
            has_pattern1 = bool(re.search(pattern1, response_lower, re.IGNORECASE))
            has_pattern2 = bool(re.search(pattern2, response_lower, re.IGNORECASE))
            
            # If both patterns present, potential contradiction
            if has_pattern1 and has_pattern2:
                contradiction_count += 1
        
        # Calculate consistency score
        # 0 contradictions = 1.0, 3+ contradictions = 0.0
        consistency_score = max(0.0, 1.0 - (contradiction_count / 3.0))
        
        return consistency_score
    
    def _check_response_completeness(self, response: str, prompt: str) -> float:
        """
        Check if response adequately addresses the prompt.
        
        Args:
            response: Model response text
            prompt: Original prompt
            
        Returns:
            Completeness score (0-1)
        """
        if not response or not response.strip():
            return 0.0
        
        score = 0.0
        
        # Check 1: Response is not too short
        if len(response) >= 20:
            score += 0.25
        
        # Check 2: Response is not just repeating the prompt
        if response.lower() != prompt.lower():
            score += 0.25
        
        # Check 3: Response has substantive content (not just filler)
        words = response.split()
        if len(words) >= 10:
            score += 0.25
        
        # Check 4: Response appears to be a complete thought
        # (ends with proper punctuation)
        if response.rstrip()[-1:] in '.!?':
            score += 0.25
        
        return score
    
    def _generate_explanation(
        self,
        components: TrustScoreComponents,
        overall_score: float
    ) -> str:
        """
        Generate human-readable explanation of trust score.
        
        Args:
            components: Trust score components
            overall_score: Overall trust score
            
        Returns:
            Explanation string
        """
        explanations = []
        
        # Identify weak components
        if components.context_grounding < 0.6:
            explanations.append("low grounding in source documents")
        
        if components.output_structure < 0.6:
            explanations.append("structural issues in response")
        
        if components.uncertainty_indicators < 0.6:
            explanations.append("high uncertainty in language")
        
        if components.factual_consistency < 0.6:
            explanations.append("potential internal contradictions")
        
        if components.response_completeness < 0.6:
            explanations.append("incomplete response")
        
        if explanations:
            return f"Trust score {overall_score:.2f}: " + ", ".join(explanations)
        else:
            return f"Trust score {overall_score:.2f}: all components within acceptable range"
    
    def _extract_claims(self, response: str) -> List[tuple]:
        """
        Extract claims (sentences) from response.
        
        Args:
            response: Model response text
            
        Returns:
            List of (claim_text, start_idx, end_idx) tuples
        """
        if not response:
            return []
        
        claims = []
        
        # Split into sentences
        sentences = re.split(r'([.!?]+)', response)
        
        current_pos = 0
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
            
            claim_text = (sentence + punctuation).strip()
            
            if claim_text and len(claim_text) > 10:  # Skip very short sentences
                start_idx = current_pos
                end_idx = current_pos + len(sentence + punctuation)
                claims.append((claim_text, start_idx, end_idx))
            
            current_pos += len(sentence + punctuation)
        
        return claims
    
    def _find_grounding_evidence(
        self,
        claim: str,
        source_documents: List[str]
    ) -> tuple:
        """
        Find grounding evidence for a claim in source documents.
        
        Args:
            claim: Claim text to verify
            source_documents: Source documents to search
            
        Returns:
            Tuple of (best_score, evidence_documents)
            where evidence_documents is List[Tuple[str, float]]
        """
        if not source_documents:
            return (0.0, [])
        
        try:
            evidence_docs = []
            
            for doc in source_documents:
                if doc.strip():
                    similarity = self.similarity_analyzer.calculate_similarity(
                        claim, doc
                    )
                    evidence_docs.append((doc[:100], similarity))  # Store first 100 chars
            
            # Sort by similarity score
            evidence_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return best score and top 3 evidence documents
            best_score = evidence_docs[0][1] if evidence_docs else 0.0
            top_evidence = evidence_docs[:3]
            
            return (best_score, top_evidence)
            
        except Exception:
            return (0.0, [])
