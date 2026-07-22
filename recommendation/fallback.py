"""Generic, safe fallback recommendation — used when no KB entry clears the similarity threshold."""


def build_fallback(disease, matched_entries=None, reason="below_similarity_threshold"):
    matched_entries = matched_entries or []
    return {
        "recommendation": (
            "No sufficiently similar clinical pattern was found in the knowledge base for this "
            "specific combination of lab values. Please consult a qualified physician for "
            "interpretation of these results; this system's automated recommendation is not "
            "reliable for this specific pattern."
        ),
        "diet_advice": "Maintain a balanced, generally healthy diet pending clinical review.",
        "lifestyle_advice": "Maintain regular activity and follow up with a healthcare provider.",
        "suggested_follow_up_tests": [],
        "urgency_level": "Monitor",
        "evidence_tags": [],
        "matched_entries": matched_entries,
        "primary_entry_id": None,
        "similarity_score": matched_entries[0]["similarity_score"] if matched_entries else 0.0,
        "match_confidence": "low_fallback",
        "is_fallback": True,
        "fallback_reason": reason,
        "disease": disease,
    }
