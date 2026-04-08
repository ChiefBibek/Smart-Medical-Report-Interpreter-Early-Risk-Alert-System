"""
Explainability Layer — SHAP-like without any library
======================================================
Computes feature contributions as:
    contribution_i = normalized_value_i × weight_i

Then normalizes to percentage importance.

This follows the same principle as SHAP's linear explanation:
"each feature's contribution = its value × its learned weight"
"""

import numpy as np
from preprocessing import REFERENCE_RANGES, FEATURE_NAMES


def compute_feature_contributions(normalized_values, weights, feature_names):
    """
    Calculate per-feature contribution to the logistic regression output.

    Parameters
    ----------
    normalized_values : ndarray, shape (n_features,)
        Min-max normalized feature values in [0, 1].
    weights : ndarray, shape (n_features,)
        Learned weights from logistic regression.
    feature_names : list[str]

    Returns
    -------
    contributions : dict  {feature_name: signed_contribution}
    pct_importance : dict {feature_name: percentage_of_total_magnitude}
    """
    # Raw signed contributions
    contributions = {}
    for i, name in enumerate(feature_names):
        # Center at 0.5 so contribution is relative to "average"
        contributions[name] = float((normalized_values[i] - 0.5) * weights[i])

    # Absolute magnitude for percentage
    total_abs = sum(abs(v) for v in contributions.values()) + 1e-9
    pct_importance = {
        name: round(abs(v) / total_abs * 100, 1)
        for name, v in contributions.items()
    }

    return contributions, pct_importance


def rank_features(pct_importance):
    """Return features sorted by importance, descending."""
    return sorted(pct_importance.items(), key=lambda x: x[1], reverse=True)


def generate_explanation_text(contributions, flags, top_n=3):
    """
    Generate a human-readable template-based explanation.

    Parameters
    ----------
    contributions : dict {feature: signed_contribution}
    flags : dict {feature: status}  from preprocessing.flag_abnormal
    top_n : int  — number of top contributors to mention
    """
    # Identify top contributors that are also abnormal
    abnormal_contribs = [
        (name, abs(v)) for name, v in contributions.items()
        if flags.get(name, "normal") != "normal"
    ]
    abnormal_contribs.sort(key=lambda x: x[1], reverse=True)
    top_abnormal = [name for name, _ in abnormal_contribs[:top_n]]

    # Build readable feature phrases
    phrase_map = {
        "hemoglobin": "low hemoglobin",
        "glucose":    "elevated glucose",
        "rbc":        "abnormal RBC count",
        "wbc":        "abnormal WBC count",
        "platelets":  "abnormal platelet count",
        "creatinine": "elevated creatinine",
    }

    if top_abnormal:
        phrases = [phrase_map.get(f, f) for f in top_abnormal]
        if len(phrases) == 1:
            feature_str = phrases[0]
        elif len(phrases) == 2:
            feature_str = f"{phrases[0]} and {phrases[1]}"
        else:
            feature_str = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
        return (
            f"The risk assessment is primarily influenced by {feature_str}. "
            f"These parameters fall outside the normal reference range and "
            f"contribute most significantly to the elevated risk score."
        )
    else:
        return (
            "All major parameters are within or near the normal range. "
            "The risk score reflects the combined pattern of values rather than "
            "any single critical deviation."
        )


def format_explanation_report(contributions, pct_importance, flags, probability, category):
    """
    Full formatted explanation report for display or printing.
    """
    lines = [
        "=" * 55,
        "  EXPLAINABILITY REPORT",
        "=" * 55,
        f"  Final risk probability : {probability:.3f}",
        f"  Risk category          : {category.upper()}",
        "",
        "  Feature contributions (% importance):",
        "-" * 55,
    ]

    ranked = rank_features(pct_importance)
    for name, pct in ranked:
        contrib = contributions[name]
        direction = "↑ increases risk" if contrib > 0 else "↓ decreases risk"
        flag_note = f"  [{flags.get(name, 'normal')}]" if flags.get(name) != "normal" else ""
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"  {name:<12} {bar} {pct:5.1f}% {direction}{flag_note}")

    lines.append("")
    lines.append("  Narrative:")
    lines.append(f"  {generate_explanation_text(contributions, flags)}")
    lines.append("=" * 55)
    return "\n".join(lines)
