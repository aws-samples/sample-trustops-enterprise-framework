"""
Unit tests for TrustScoringEngine.
"""
import pytest
from unittest.mock import Mock, MagicMock
from src.trust_scoring.trust_scoring_engine import TrustScoringEngine
from src.data_models.model_response import TrustScore, TrustScoreComponents, HallucinationAnalysis


class TestTrustScoringEngine:
    """Test suite for TrustScoringEngine."""
    
    @pytest.fixture
    def mock_similarity_analyzer(self):
        """Create mock similarity analyzer."""
        analyzer = Mock()
        analyzer.calculate_similarity = Mock(return_value=0.8)
        return analyzer
    
    @pytest.fixture
    def engine(self, mock_similarity_analyzer):
        """Create TrustScoringEngine instance with mock analyzer."""
        return TrustScoringEngine(
            similarity_analyzer=mock_similarity_analyzer,
            review_threshold=0.6
        )
    
    def test_calculate_trust_score_basic(self, engine):
        """Test basic trust score calculation."""
        response = "The capital of France is Paris. This is a well-known fact."
        prompt = "What is the capital of France?"
        source_docs = ["Paris is the capital and largest city of France."]
        
        trust_score = engine.calculate_trust_score(
            response=response,
            prompt=prompt,
            source_documents=source_docs
        )
        
        assert isinstance(trust_score, TrustScore)
        assert 0.0 <= trust_score.overall_score <= 1.0
        assert trust_score.confidence_level in ["high", "medium", "low"]
        assert isinstance(trust_score.flagged_for_review, bool)
        assert isinstance(trust_score.explanation, str)
    
    def test_trust_score_components_present(self, engine):
        """Test that all trust score components are calculated."""
        response = "Test response"
        prompt = "Test prompt"
        source_docs = ["Test document"]
        
        trust_score = engine.calculate_trust_score(
            response=response,
            prompt=prompt,
            source_documents=source_docs
        )
        
        components = trust_score.components
        assert isinstance(components, TrustScoreComponents)
        assert 0.0 <= components.context_grounding <= 1.0
        assert 0.0 <= components.output_structure <= 1.0
        assert 0.0 <= components.uncertainty_indicators <= 1.0
        assert 0.0 <= components.factual_consistency <= 1.0
        assert 0.0 <= components.response_completeness <= 1.0
    
    def test_trust_score_in_valid_range(self, engine):
        """Test that overall trust score is always in [0, 1] range."""
        test_cases = [
            ("Short", "prompt", []),
            ("A very long response with lots of detail and information.", "prompt", ["doc"]),
            ("", "prompt", []),
            ("Maybe this could be correct.", "prompt", ["doc"]),
        ]
        
        for response, prompt, docs in test_cases:
            trust_score = engine.calculate_trust_score(response, prompt, docs)
            assert 0.0 <= trust_score.overall_score <= 1.0
    
    def test_confidence_level_high(self, engine, mock_similarity_analyzer):
        """Test high confidence level assignment."""
        # Mock high similarity
        mock_similarity_analyzer.calculate_similarity.return_value = 0.95
        
        response = "The capital of France is Paris."
        prompt = "What is the capital of France?"
        source_docs = ["Paris is the capital of France."]
        
        trust_score = engine.calculate_trust_score(response, prompt, source_docs)
        
        # With high similarity and good structure, should be high confidence
        assert trust_score.overall_score >= 0.7
    
    def test_confidence_level_low(self, engine, mock_similarity_analyzer):
        """Test low confidence level assignment."""
        # Mock low similarity
        mock_similarity_analyzer.calculate_similarity.return_value = 0.2
        
        response = "Maybe it could be something."
        prompt = "What is the answer?"
        source_docs = ["Unrelated document"]
        
        trust_score = engine.calculate_trust_score(response, prompt, source_docs)
        
        # With low similarity and hedging language, should be lower confidence
        assert trust_score.confidence_level in ["low", "medium"]
    
    def test_flagged_for_review_below_threshold(self, engine):
        """Test that responses below threshold are flagged."""
        engine.review_threshold = 0.8
        
        # Create a response that will score low
        response = "I'm not sure."
        prompt = "What is the answer?"
        source_docs = []
        
        trust_score = engine.calculate_trust_score(response, prompt, source_docs)
        
        if trust_score.overall_score < 0.8:
            assert trust_score.flagged_for_review is True
    
    def test_not_flagged_above_threshold(self, engine, mock_similarity_analyzer):
        """Test that responses above threshold are not flagged."""
        # Mock high similarity
        mock_similarity_analyzer.calculate_similarity.return_value = 0.95
        
        engine.review_threshold = 0.5
        
        response = "The capital of France is Paris."
        prompt = "What is the capital of France?"
        source_docs = ["Paris is the capital of France."]
        
        trust_score = engine.calculate_trust_score(response, prompt, source_docs)
        
        if trust_score.overall_score >= 0.5:
            assert trust_score.flagged_for_review is False
    
    def test_detect_hallucinations_no_hallucinations(self, engine, mock_similarity_analyzer):
        """Test hallucination detection with well-grounded response."""
        # Mock high similarity
        mock_similarity_analyzer.calculate_similarity.return_value = 0.9
        
        response = "Paris is the capital of France."
        source_docs = ["Paris is the capital and largest city of France."]
        
        analysis = engine.detect_hallucinations(response, source_docs)
        
        assert isinstance(analysis, HallucinationAnalysis)
        assert analysis.has_hallucinations is False
        assert analysis.hallucination_rate == 0.0
        assert len(analysis.flagged_spans) == 0
        assert analysis.overall_grounding_score > 0.7
    
    def test_detect_hallucinations_with_hallucinations(self, engine, mock_similarity_analyzer):
        """Test hallucination detection with ungrounded claims."""
        # Mock low similarity
        mock_similarity_analyzer.calculate_similarity.return_value = 0.3
        
        response = "The moon is made of cheese. It tastes like cheddar."
        source_docs = ["The moon is Earth's natural satellite."]
        
        analysis = engine.detect_hallucinations(
            response, source_docs, similarity_threshold=0.7
        )
        
        assert isinstance(analysis, HallucinationAnalysis)
        assert analysis.has_hallucinations is True
        assert analysis.hallucination_rate > 0.0
        assert len(analysis.flagged_spans) > 0
    
    def test_detect_hallucinations_empty_response(self, engine):
        """Test hallucination detection with empty response."""
        response = ""
        source_docs = ["Some document"]
        
        analysis = engine.detect_hallucinations(response, source_docs)
        
        assert analysis.has_hallucinations is False
        assert analysis.hallucination_rate == 0.0
        assert len(analysis.flagged_spans) == 0
    
    def test_detect_hallucinations_no_source_docs(self, engine):
        """Test hallucination detection with no source documents."""
        response = "Some claim about something."
        source_docs = []
        
        analysis = engine.detect_hallucinations(response, source_docs)
        
        # Without source docs, grounding score should be 0
        assert analysis.overall_grounding_score == 0.0
    
    def test_context_grounding_with_no_sources(self, engine):
        """Test context grounding returns neutral score with no sources."""
        score = engine._calculate_context_grounding("Test response", [])
        assert score == 0.5
    
    def test_context_grounding_with_empty_sources(self, engine):
        """Test context grounding with empty source documents."""
        score = engine._calculate_context_grounding("Test response", ["", "  "])
        assert score == 0.5
    
    def test_uncertainty_detection_certain_language(self, engine):
        """Test uncertainty detection with certain language."""
        response = "The answer is definitely Paris. This is correct."
        score = engine._detect_uncertainty_indicators(response)
        
        # Should have high certainty (low hedging)
        assert score > 0.8
    
    def test_uncertainty_detection_uncertain_language(self, engine):
        """Test uncertainty detection with hedging language."""
        response = "Maybe it could possibly be Paris. I think it might be correct."
        score = engine._detect_uncertainty_indicators(response)
        
        # Should have low certainty (high hedging)
        assert score < 0.7
    
    def test_factual_consistency_no_contradictions(self, engine):
        """Test factual consistency with no contradictions."""
        response = "The sky is blue. Water is wet."
        score = engine._check_factual_consistency(response)
        
        assert score >= 0.8
    
    def test_factual_consistency_with_contradictions(self, engine):
        """Test factual consistency with contradictions."""
        response = "The answer is yes. No, the answer is false. It's always true but never correct."
        score = engine._check_factual_consistency(response)
        
        # Should detect contradictions
        assert score < 1.0
    
    def test_response_completeness_complete(self, engine):
        """Test response completeness with complete response."""
        response = "The capital of France is Paris, which is located in the north-central part of the country."
        prompt = "What is the capital of France?"
        score = engine._check_response_completeness(response, prompt)
        
        assert score >= 0.75
    
    def test_response_completeness_incomplete(self, engine):
        """Test response completeness with incomplete response."""
        response = "Um"
        prompt = "What is the capital of France?"
        score = engine._check_response_completeness(response, prompt)
        
        assert score < 0.5
    
    def test_response_completeness_empty(self, engine):
        """Test response completeness with empty response."""
        score = engine._check_response_completeness("", "prompt")
        assert score == 0.0
    
    def test_output_structure_validation_basic(self, engine):
        """Test basic output structure validation."""
        response = "This is a well-formed response with proper structure."
        score = engine._validate_output_structure(response, None)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should pass basic checks
    
    def test_output_structure_validation_empty(self, engine):
        """Test output structure validation with empty response."""
        score = engine._validate_output_structure("", None)
        assert score < 0.5
    
    def test_extract_claims_basic(self, engine):
        """Test claim extraction from response."""
        response = "Paris is the capital. France is in Europe. The Eiffel Tower is famous."
        claims = engine._extract_claims(response)
        
        assert len(claims) > 0
        assert all(isinstance(claim, tuple) for claim in claims)
        assert all(len(claim) == 3 for claim in claims)
    
    def test_extract_claims_empty(self, engine):
        """Test claim extraction from empty response."""
        claims = engine._extract_claims("")
        assert claims == []
    
    def test_find_grounding_evidence_with_sources(self, engine, mock_similarity_analyzer):
        """Test finding grounding evidence with source documents."""
        mock_similarity_analyzer.calculate_similarity.return_value = 0.85
        
        claim = "Paris is the capital of France."
        sources = ["Paris is the capital city of France.", "France is in Europe."]
        
        score, evidence = engine._find_grounding_evidence(claim, sources)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(evidence, list)
        assert len(evidence) <= 3
    
    def test_find_grounding_evidence_no_sources(self, engine):
        """Test finding grounding evidence with no sources."""
        claim = "Some claim"
        sources = []
        
        score, evidence = engine._find_grounding_evidence(claim, sources)
        
        assert score == 0.0
        assert evidence == []
    
    def test_generate_explanation(self, engine):
        """Test explanation generation."""
        components = TrustScoreComponents(
            context_grounding=0.9,
            output_structure=0.8,
            uncertainty_indicators=0.7,
            factual_consistency=0.85,
            response_completeness=0.9
        )
        
        explanation = engine._generate_explanation(components, 0.85)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "0.85" in explanation
    
    def test_generate_explanation_with_issues(self, engine):
        """Test explanation generation with low scores."""
        components = TrustScoreComponents(
            context_grounding=0.3,
            output_structure=0.4,
            uncertainty_indicators=0.5,
            factual_consistency=0.2,
            response_completeness=0.3
        )
        
        explanation = engine._generate_explanation(components, 0.35)
        
        assert isinstance(explanation, str)
        assert "low grounding" in explanation or "grounding" in explanation.lower()
    
    def test_weighted_score_calculation(self, engine, mock_similarity_analyzer):
        """Test that weighted score calculation uses correct weights."""
        # Mock specific similarity
        mock_similarity_analyzer.calculate_similarity.return_value = 0.8
        
        response = "Test response with good structure."
        prompt = "Test prompt"
        source_docs = ["Test document"]
        
        trust_score = engine.calculate_trust_score(response, prompt, source_docs)
        
        # Verify weights sum to 1.0
        weights = engine.WEIGHTS
        assert abs(sum(weights.values()) - 1.0) < 0.01
        
        # Verify overall score is weighted combination
        components = trust_score.components
        expected_score = (
            components.context_grounding * weights['context_grounding'] +
            components.output_structure * weights['output_structure'] +
            components.uncertainty_indicators * weights['uncertainty_indicators'] +
            components.factual_consistency * weights['factual_consistency'] +
            components.response_completeness * weights['response_completeness']
        )
        
        assert abs(trust_score.overall_score - expected_score) < 0.01
