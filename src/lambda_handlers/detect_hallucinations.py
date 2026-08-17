"""Lambda handler for claim-level hallucination detection."""

import re


def handler(event: dict, context) -> dict:
    for field in ("model_response", "source_documents", "example_id"):
        if field not in event:
            return {
                "statusCode": 400,
                "error": "ValidationError",
                "message": f"Missing required field: {field}",
            }

    model_response = event["model_response"]
    source_documents = event["source_documents"]
    example_id = event["example_id"]

    claims = _extract_claims(model_response)

    if not claims:
        return {
            "hallucination_analysis": {
                "example_id": example_id,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "hallucination_rate": 0.0,
                "claims": [],
            }
        }

    source_text = " ".join(source_documents).lower()
    source_words = set(source_text.split())

    classified_claims = []
    supported_count = 0

    for claim in claims:
        confidence = _compute_support_score(claim, source_words, source_text)
        classification = "supported" if confidence >= 0.3 else "unsupported"
        if classification == "supported":
            supported_count += 1
        classified_claims.append({
            "text": claim,
            "classification": classification,
            "confidence": round(confidence, 3),
        })

    unsupported_count = len(claims) - supported_count
    hallucination_rate = unsupported_count / len(claims)

    return {
        "hallucination_analysis": {
            "example_id": example_id,
            "total_claims": len(claims),
            "supported_claims": supported_count,
            "unsupported_claims": unsupported_count,
            "hallucination_rate": round(hallucination_rate, 4),
            "claims": classified_claims,
        }
    }


def _extract_claims(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) > 10 and not _is_hedged(s):
            claims.append(s)
    return claims


def _is_hedged(sentence: str) -> bool:
    hedges = [
        "i think", "in my opinion", "it seems", "perhaps",
        "maybe", "might be", "could be", "i believe",
    ]
    lower = sentence.lower()
    return any(h in lower for h in hedges)


def _compute_support_score(
    claim: str, source_words: set[str], source_text: str
) -> float:
    claim_words = set(claim.lower().split())
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "it", "this", "that", "these", "those", "and", "or", "but",
        "not", "no", "if", "then", "so", "as",
    }
    meaningful_words = claim_words - stop_words

    if not meaningful_words:
        return 0.5

    overlap = meaningful_words & source_words
    word_score = len(overlap) / len(meaningful_words)

    # Bigram overlap for phrase-level matching
    claim_bigrams = _get_bigrams(claim.lower())
    source_bigrams = _get_bigrams(source_text)
    if claim_bigrams:
        bigram_overlap = len(claim_bigrams & source_bigrams) / len(claim_bigrams)
    else:
        bigram_overlap = 0.0

    return 0.6 * word_score + 0.4 * bigram_overlap


def _get_bigrams(text: str) -> set[str]:
    words = text.split()
    return {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}
