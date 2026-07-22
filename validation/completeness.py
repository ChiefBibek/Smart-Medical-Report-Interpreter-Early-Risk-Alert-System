"""
Report Completeness Score — Module 1, "Report Completeness Score".

Mandatory fields are weighted more heavily than optional fields so a report
missing its mandatory field(s) can never score as "complete" just because it
happens to include a lot of optional extras.
"""

MANDATORY_WEIGHT = 50.0  # percentage points reserved for the mandatory block


def compute_completeness(mandatory_total, mandatory_provided, optional_total, optional_provided):
    """
    Returns {score: float 0-100, label: str, mandatory_provided, mandatory_total,
              optional_provided, optional_total}.
    """
    if mandatory_total == 0:
        mandatory_component = 100.0
    else:
        mandatory_component = MANDATORY_WEIGHT * (mandatory_provided / mandatory_total)

    if optional_total == 0:
        score = 100.0 * (mandatory_provided / mandatory_total) if mandatory_total else 100.0
    else:
        optional_component = (100.0 - MANDATORY_WEIGHT) * (optional_provided / optional_total)
        score = mandatory_component + optional_component

    score = round(min(100.0, max(0.0, score)), 1)
    return {
        "score": score,
        "label": label_for_score(score),
        "mandatory_provided": mandatory_provided,
        "mandatory_total": mandatory_total,
        "optional_provided": optional_provided,
        "optional_total": optional_total,
    }


def label_for_score(score):
    if score >= 90:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"
