"""
Risk Categorization & Safety Layer
=====================================
Implements:
 - Risk probability → category mapping
 - Rule-based overrides (critical flags)
 - Confidence handling
 - Borderline uncertainty highlighting

Medical principle: prefer false positives over false negatives.
"""


# ─────────────────────────────────────────────
# Risk thresholds
# ─────────────────────────────────────────────
THRESHOLDS = {
    "low":        (0.00, 0.20),
    "borderline": (0.20, 0.50),
    "moderate":   (0.50, 0.75),
    "high":       (0.75, 1.00),
}

RISK_ICONS = {
    "low":        "🟢",
    "borderline": "⚠️",
    "moderate":   "🟡",
    "high":       "🔴",
}

RISK_MESSAGES = {
    "low":        "No immediate concern. Maintain healthy habits.",
    "borderline": "Uncertain result. Consider re-testing and consulting a professional.",
    "moderate":   "Some parameters need attention. Monitor closely and consult a doctor.",
    "high":       "Immediate professional attention is recommended.",
}

# ─────────────────────────────────────────────
# Rule-based override triggers
# Prefer false positives — these force High Risk
# regardless of the ML score
# ─────────────────────────────────────────────
CRITICAL_RULES = [
    {
        "feature":   "hemoglobin",
        "condition": lambda v: v < 9.0,
        "reason":    "Hemoglobin critically low (< 9 g/dL) — severe anemia indicator",
        "force":     "high",
    },
    {
        "feature":   "glucose",
        "condition": lambda v: v > 200,
        "reason":    "Glucose significantly elevated (> 200 mg/dL) — diabetes/hyperglycemia indicator",
        "force":     "high",
    },
    {
        "feature":   "wbc",
        "condition": lambda v: v > 20,
        "reason":    "WBC severely elevated (> 20 K/uL) — possible serious infection or hematologic issue",
        "force":     "high",
    },
    {
        "feature":   "platelets",
        "condition": lambda v: v < 50,
        "reason":    "Platelets critically low (< 50 K/uL) — bleeding risk",
        "force":     "high",
    },
    {
        "feature":   "creatinine",
        "condition": lambda v: v > 3.0,
        "reason":    "Creatinine severely elevated (> 3.0 mg/dL) — possible renal impairment",
        "force":     "high",
    },
]


def categorize_risk(probability):
    """Map a probability to a risk category string."""
    for category, (lo, hi) in THRESHOLDS.items():
        if lo <= probability < hi:
            return category
    return "high"  # fallback for p == 1.0


def apply_safety_overrides(probability, values_dict):
    """
    Check critical rules and apply overrides.

    Returns
    -------
    adjusted_probability : float
        The probability, possibly increased if a critical rule fires.
    overrides : list[str]
        Human-readable descriptions of applied rules.
    """
    adjusted_probability = probability
    overrides = []

    for rule in CRITICAL_RULES:
        feature = rule["feature"]
        if feature in values_dict and rule["condition"](values_dict[feature]):
            overrides.append(rule["reason"])
            if rule["force"] == "high":
                adjusted_probability = max(adjusted_probability, 0.76)

    return adjusted_probability, overrides


def get_confidence(probability):
    """
    Confidence is lower near decision boundaries (0.2, 0.5, 0.75).
    A prediction near a boundary is labelled 'uncertain'.
    """
    boundaries = [0.2, 0.5, 0.75]
    min_dist = min(abs(probability - b) for b in boundaries)

    if min_dist < 0.05:
        return "low confidence — near decision boundary"
    elif min_dist < 0.12:
        return "moderate confidence"
    else:
        return "high confidence"


class RiskResult:
    """Holds the complete risk assessment for one patient."""

    def __init__(self, raw_probability, values_dict):
        adjusted_prob, overrides = apply_safety_overrides(raw_probability, values_dict)

        self.raw_probability      = raw_probability
        self.adjusted_probability = adjusted_prob
        self.category             = categorize_risk(adjusted_prob)
        self.icon                 = RISK_ICONS[self.category]
        self.message              = RISK_MESSAGES[self.category]
        self.overrides            = overrides
        self.confidence           = get_confidence(adjusted_prob)
        self.override_applied     = len(overrides) > 0

    def summary(self):
        lines = [
            f"{self.icon}  Risk category : {self.category.upper()}",
            f"   Probability  : {self.adjusted_probability:.3f}",
            f"   Confidence   : {self.confidence}",
        ]
        if self.overrides:
            lines.append("   ⚠️  Safety overrides applied:")
            for o in self.overrides:
                lines.append(f"      • {o}")
        lines.append(f"   {self.message}")
        lines.append("")
        lines.append("DISCLAIMER: This system is for awareness only and is not a")
        lines.append("replacement for professional medical advice.")
        return "\n".join(lines)
