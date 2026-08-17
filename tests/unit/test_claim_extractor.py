"""
Unit tests for ClaimExtractor.

Tests claim extraction, sentence segmentation, and claim classification
functionality for hallucination detection.

Requirements: 6.1
"""

import pytest

from src.data_models.hallucination import Claim, HallucinationConfig
from src.hallucination.claim_extractor import ClaimExtractor


class TestClaimExtractor:
    """Test suite for ClaimExtractor class."""
    
    def test_extract_claims_simple_response(self):
        """Test extracting claims from a simple multi-sentence response."""
        extractor = ClaimExtractor()
        response = "The sky is blue. Water is wet. Fire is hot."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 3
        assert claims[0].text == "The sky is blue."
        assert claims[1].text == "Water is wet."
        assert claims[2].text == "Fire is hot."
        
        # Check positions
        assert claims[0].start_idx == 0
        assert claims[0].end_idx == 16
        assert claims[1].start_idx == 17
        assert claims[2].start_idx == 31
    
    def test_extract_claims_empty_response(self):
        """Test extracting claims from empty response."""
        extractor = ClaimExtractor()
        
        claims = extractor.extract_claims("")
        assert len(claims) == 0
        
        claims = extractor.extract_claims("   ")
        assert len(claims) == 0
    
    def test_extract_claims_single_sentence(self):
        """Test extracting claims from single sentence response."""
        extractor = ClaimExtractor()
        response = "Python is a programming language."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 1
        assert claims[0].text == "Python is a programming language."
        assert claims[0].start_idx == 0
        assert claims[0].end_idx == len(response)
    
    def test_extract_claims_with_complex_punctuation(self):
        """Test extracting claims with various punctuation marks."""
        extractor = ClaimExtractor()
        response = "Is this a question? Yes, it is! This is an exclamation."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 3
        assert "question?" in claims[0].text
        assert "Yes, it is!" in claims[1].text
        assert "exclamation" in claims[2].text
    
    def test_extract_claims_respects_max_limit(self):
        """Test that max_claims_per_response limit is respected."""
        config = HallucinationConfig(max_claims_per_response=2)
        extractor = ClaimExtractor(config)
        
        response = "First sentence. Second sentence. Third sentence. Fourth sentence."
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 2
        assert claims[0].text == "First sentence."
        assert claims[1].text == "Second sentence."
    
    def test_extract_claims_with_newlines(self):
        """Test extracting claims from response with newlines."""
        extractor = ClaimExtractor()
        response = "First line.\nSecond line.\n\nThird line after blank."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) >= 3
        assert any("First line" in claim.text for claim in claims)
        assert any("Second line" in claim.text for claim in claims)
        assert any("Third line" in claim.text for claim in claims)
    
    def test_classify_factual_claim(self):
        """Test classification of factual claims."""
        extractor = ClaimExtractor()
        response = "The Earth orbits the Sun."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 1
        assert claims[0].is_factual is True
        assert claims[0].is_opinion is False
        assert claims[0].is_hedged is False
    
    def test_classify_hedged_claim(self):
        """Test classification of hedged claims."""
        extractor = ClaimExtractor()
        response = "The data might indicate a trend."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 1
        assert claims[0].is_hedged is True
        assert claims[0].is_opinion is True  # Hedged claims are treated as opinions
    
    def test_classify_opinion_claim(self):
        """Test classification of opinion claims."""
        extractor = ClaimExtractor()
        response = "I think this is the best approach."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 1
        assert claims[0].is_opinion is True
        assert claims[0].is_factual is False
    
    def test_classify_multiple_hedging_keywords(self):
        """Test detection of various hedging keywords."""
        extractor = ClaimExtractor()
        
        hedged_responses = [
            "This possibly could work.",
            "Perhaps we should consider this.",
            "It seems like a good idea.",
            "This probably will succeed.",
            "It appears to be correct.",
            "This may be the solution.",
            "This likely will help.",
            "I believe this is true."
        ]
        
        for response in hedged_responses:
            claims = extractor.extract_claims(response)
            assert len(claims) >= 1
            assert claims[0].is_hedged is True or claims[0].is_opinion is True, \
                f"Failed to detect hedging/opinion in: {response}"
    
    def test_claim_boundaries_accuracy(self):
        """Test that claim boundaries accurately reflect positions in original text."""
        extractor = ClaimExtractor()
        response = "First claim here. Second claim follows. Third and final claim."
        
        claims = extractor.extract_claims(response)
        
        # Verify each claim can be extracted using its boundaries
        for claim in claims:
            extracted_text = response[claim.start_idx:claim.end_idx]
            assert extracted_text == claim.text
    
    def test_claim_boundaries_with_whitespace(self):
        """Test claim boundaries with various whitespace patterns."""
        extractor = ClaimExtractor()
        response = "First.  Second with double space.   Third with triple."
        
        claims = extractor.extract_claims(response)
        
        # Verify boundaries are correct
        for claim in claims:
            extracted_text = response[claim.start_idx:claim.end_idx]
            assert extracted_text == claim.text
    
    def test_extract_claims_with_abbreviations(self):
        """Test handling of abbreviations that contain periods."""
        extractor = ClaimExtractor()
        response = "Dr. Smith works at NASA. He has a Ph.D. in physics."
        
        claims = extractor.extract_claims(response)
        
        # Should handle abbreviations reasonably
        # NLTK should handle this well, regex fallback may split differently
        assert len(claims) >= 1
        assert all(claim.start_idx < claim.end_idx for claim in claims)
    
    def test_extract_claims_preserves_order(self):
        """Test that claims are extracted in order of appearance."""
        extractor = ClaimExtractor()
        response = "Alpha. Beta. Gamma. Delta. Epsilon."
        
        claims = extractor.extract_claims(response)
        
        # Verify claims are in order
        for i in range(len(claims) - 1):
            assert claims[i].start_idx < claims[i + 1].start_idx
    
    def test_custom_hedging_keywords(self):
        """Test using custom hedging keywords in config."""
        config = HallucinationConfig(
            hedging_keywords=["uncertain", "unclear", "ambiguous"]
        )
        extractor = ClaimExtractor(config)
        
        response = "This is uncertain territory."
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 1
        assert claims[0].is_hedged is True
    
    def test_extract_claims_long_response(self):
        """Test extracting claims from a longer, more realistic response."""
        extractor = ClaimExtractor()
        response = """
        The TrustOps Framework is a comprehensive platform for LLM evaluation.
        It supports multiple model providers including AWS Bedrock and SageMaker.
        The framework includes trust scoring across five dimensions.
        Users can fine-tune models with their own datasets.
        Hallucination detection helps identify unsupported claims.
        """
        
        claims = extractor.extract_claims(response)
        
        # Should extract multiple claims
        assert len(claims) >= 4
        
        # All claims should have valid boundaries
        for claim in claims:
            assert claim.start_idx >= 0
            assert claim.end_idx > claim.start_idx
            assert claim.end_idx <= len(response)
            
            # Verify extraction
            extracted = response[claim.start_idx:claim.end_idx]
            assert extracted == claim.text
    
    def test_claim_object_validation(self):
        """Test that extracted claims are valid Claim objects."""
        extractor = ClaimExtractor()
        response = "This is a test claim."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 1
        claim = claims[0]
        
        # Verify all required fields are present
        assert isinstance(claim, Claim)
        assert isinstance(claim.text, str)
        assert isinstance(claim.start_idx, int)
        assert isinstance(claim.end_idx, int)
        assert isinstance(claim.is_factual, bool)
        assert isinstance(claim.is_opinion, bool)
        assert isinstance(claim.is_hedged, bool)
    
    def test_extract_claims_with_quotes(self):
        """Test extracting claims containing quoted text."""
        extractor = ClaimExtractor()
        response = 'The author said "this is important." The quote was noted.'
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) >= 1
        # Verify boundaries work with quotes
        for claim in claims:
            extracted = response[claim.start_idx:claim.end_idx]
            assert extracted == claim.text
    
    def test_extract_claims_with_numbers(self):
        """Test extracting claims containing numbers and decimals."""
        extractor = ClaimExtractor()
        response = "The value is 3.14. The count is 42. The ratio is 1.618."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 3
        assert all(claim.start_idx < claim.end_idx for claim in claims)
    
    def test_factual_vs_opinion_distinction(self):
        """Test clear distinction between factual and opinion claims."""
        extractor = ClaimExtractor()
        
        factual_response = "Water boils at 100 degrees Celsius."
        opinion_response = "I believe water is the best beverage."
        
        factual_claims = extractor.extract_claims(factual_response)
        opinion_claims = extractor.extract_claims(opinion_response)
        
        assert factual_claims[0].is_factual is True
        assert factual_claims[0].is_opinion is False
        
        assert opinion_claims[0].is_opinion is True
        assert opinion_claims[0].is_factual is False


class TestClaimExtractorEdgeCases:
    """Test edge cases and error handling for ClaimExtractor."""
    
    def test_extract_claims_none_input(self):
        """Test handling of None input."""
        extractor = ClaimExtractor()
        
        # Should handle None gracefully
        claims = extractor.extract_claims(None)
        assert claims == []
    
    def test_extract_claims_only_punctuation(self):
        """Test response with only punctuation."""
        extractor = ClaimExtractor()
        response = "... !!! ???"
        
        claims = extractor.extract_claims(response)
        
        # May extract as single claim or none depending on tokenizer
        # Should not crash
        assert isinstance(claims, list)
    
    def test_extract_claims_very_long_sentence(self):
        """Test handling of very long single sentence."""
        extractor = ClaimExtractor()
        
        # Create a very long sentence
        long_sentence = "This is a very long sentence " * 100 + "with an ending."
        
        claims = extractor.extract_claims(long_sentence)
        
        assert len(claims) >= 1
        assert claims[0].end_idx <= len(long_sentence)
    
    def test_extract_claims_unicode_characters(self):
        """Test handling of unicode characters."""
        extractor = ClaimExtractor()
        response = "The café is open. The price is €10. The temperature is 20°C."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) >= 2
        # Verify boundaries work with unicode
        for claim in claims:
            extracted = response[claim.start_idx:claim.end_idx]
            assert extracted == claim.text
    
    def test_extract_claims_mixed_languages(self):
        """Test handling of mixed language content (if applicable)."""
        extractor = ClaimExtractor()
        response = "Hello world. Bonjour le monde. Hola mundo."
        
        claims = extractor.extract_claims(response)
        
        assert len(claims) == 3
        assert all(claim.start_idx < claim.end_idx for claim in claims)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
