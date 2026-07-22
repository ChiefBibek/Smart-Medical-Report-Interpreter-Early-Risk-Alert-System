"""
Confidence Adjustment — Module 1, "Confidence Adjustment".

Combines the model's own optional-field-based confidence with completeness,
optional per-field OCR confidence, and clinical-consistency severity, using a
multiplicative combination (so it can never go negative) and a floor (so a
valid mandatory-only prediction is never reported as having ~0% confidence).
"""

CONFIDENCE_FLOOR = 10.0

COMPLETENESS_FACTORS = {"Excellent": 1.00, "Good": 0.95, "Fair": 0.85, "Poor": 0.65}
SEVERITY_FACTORS = {"info": 1.00, "warning": 0.90, "critical": 0.70}


def adjust_confidence(base_confidence, completeness_label, ocr_confidences, consistency_severities, blocked=False):
    """
    base_confidence: the model's own confidence (0-100), from disease_models._compute_confidence.
    ocr_confidences: list[float 0-100] for fields where the caller supplied "<field>_ocr_confidence".
                      Empty/absent -> ignored entirely (no penalty).
    consistency_severities: list[str] of severities ("info"/"warning"/"critical") from consistency.py.
                      The single worst severity determines the penalty (not multiplicative stacking).
    Returns {base_confidence, completeness_factor, ocr_factor, consistency_factor,
             adjusted_confidence, penalty_points}.
    """
    if blocked:
        return {
            "base_confidence": base_confidence, "completeness_factor": 0.0,
            "ocr_factor": 0.0, "consistency_factor": 0.0,
            "adjusted_confidence": 0.0, "penalty_points": round(base_confidence, 1),
        }

    completeness_factor = COMPLETENESS_FACTORS.get(completeness_label, 1.00)

    if not ocr_confidences:
        ocr_factor = 1.0
    else:
        avg = sum(ocr_confidences) / len(ocr_confidences)
        ocr_factor = 0.6 + 0.004 * avg  # avg=100 -> 1.0 ; avg=0 -> 0.6 (OCR is a supplementary signal, not the sole driver)

    consistency_factor = min((SEVERITY_FACTORS.get(s, 1.0) for s in consistency_severities), default=1.0)

    adjusted = base_confidence * completeness_factor * ocr_factor * consistency_factor
    adjusted = max(CONFIDENCE_FLOOR, min(100.0, adjusted))

    return {
        "base_confidence": round(base_confidence, 1),
        "completeness_factor": round(completeness_factor, 3),
        "ocr_factor": round(ocr_factor, 3),
        "consistency_factor": round(consistency_factor, 3),
        "adjusted_confidence": round(adjusted, 1),
        "penalty_points": round(base_confidence - adjusted, 1),
    }
