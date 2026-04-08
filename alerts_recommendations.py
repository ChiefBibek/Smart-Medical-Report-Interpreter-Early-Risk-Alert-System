"""
Alert System & Preventive Recommendation Engine
=================================================
Rule-based — no prescriptions, only general advice.
No diagnoses are made. All recommendations are preventive.
"""

from preprocessing import REFERENCE_RANGES


# ─────────────────────────────────────────────
# Alert system
# ─────────────────────────────────────────────
def generate_alerts(flags, overrides, risk_category):
    """
    Produce a prioritized list of alerts.

    Returns list of dicts: {level, icon, message, priority}
    """
    alerts = []
    priority_counter = 0

    # Category-level alert
    level_alert = {
        "high":       ("🔴", "HIGH",       "Immediate professional attention recommended."),
        "moderate":   ("🟡", "MODERATE",   "Several parameters need monitoring. Schedule a check-up."),
        "borderline": ("⚠️",  "BORDERLINE", "Result is uncertain. Re-testing and consultation are advised."),
        "low":        ("🟢", "LOW",        "No immediate concern detected. Maintain healthy habits."),
    }
    icon, level, msg = level_alert.get(risk_category, ("⚠️", "UNKNOWN", "Result inconclusive."))
    alerts.append({"level": level, "icon": icon, "message": msg, "priority": 0})

    # Safety override alerts
    for o in overrides:
        priority_counter += 1
        alerts.append({
            "level":    "HIGH",
            "icon":     "🔴",
            "message":  f"Rule override: {o}",
            "priority": priority_counter,
        })

    # Per-parameter alerts
    flag_messages = {
        "critical_low":  ("🔴", "HIGH",     "critically below normal — {feature} = {value}"),
        "critical_high": ("🔴", "HIGH",     "critically above normal — {feature} = {value}"),
        "low":           ("🟡", "MODERATE", "below normal range — {feature} = {value}"),
        "high":          ("🟡", "MODERATE", "above normal range — {feature} = {value}"),
    }

    for feature, status in flags.items():
        if status == "normal":
            continue
        icon_s, level_s, tmpl = flag_messages[status]
        r = REFERENCE_RANGES.get(feature, {})
        msg_s = tmpl.format(
            feature=feature.capitalize(),
            value=f"({r.get('unit', '')})",
        )
        priority_counter += 1
        alerts.append({
            "level":    level_s,
            "icon":     icon_s,
            "message":  msg_s,
            "priority": priority_counter,
        })

    # Sort: HIGH first, then by priority
    level_order = {"HIGH": 0, "MODERATE": 1, "BORDERLINE": 2, "LOW": 3}
    alerts.sort(key=lambda a: (level_order.get(a["level"], 9), a["priority"]))
    return alerts


# ─────────────────────────────────────────────
# Recommendation engine (rule-based)
# ─────────────────────────────────────────────
RECOMMENDATIONS_DB = {
    "hemoglobin_low": [
        "Consider increasing intake of iron-rich foods such as spinach, lentils, red meat, and fortified cereals.",
        "Vitamin C-rich foods (citrus, bell peppers) can help with iron absorption.",
        "Consult a healthcare professional — low hemoglobin may indicate anemia requiring specific treatment.",
    ],
    "glucose_high": [
        "Reduce refined sugar and high-glycemic foods (white bread, sugary drinks, processed snacks).",
        "Increase fiber intake — vegetables, legumes, and whole grains slow glucose absorption.",
        "Regular moderate exercise (30 min/day walking) can improve insulin sensitivity.",
        "Monitor blood sugar regularly and consult an endocrinologist if persistently elevated.",
    ],
    "rbc_low": [
        "Low RBC may be related to nutritional deficiencies — ensure adequate iron, B12, and folate intake.",
        "Discuss with your doctor if you experience fatigue, shortness of breath, or pallor.",
    ],
    "rbc_high": [
        "Elevated RBC can be associated with dehydration — ensure adequate daily water intake.",
        "Consult a doctor to rule out underlying conditions such as polycythemia.",
    ],
    "wbc_high": [
        "Elevated WBC may indicate infection or inflammation. Rest and consult a doctor.",
        "Avoid tobacco and excessive alcohol, which can affect white blood cell levels.",
    ],
    "platelets_low": [
        "Low platelets increase bleeding risk — avoid contact sports until reviewed by a doctor.",
        "Seek prompt medical attention if you notice unusual bruising or prolonged bleeding.",
    ],
    "creatinine_high": [
        "Stay well hydrated — adequate water intake supports kidney function.",
        "Reduce high-protein dietary intake temporarily and consult a nephrologist if elevated.",
        "Avoid NSAIDs (ibuprofen, naproxen) without medical guidance — they can stress kidneys.",
    ],
    "general_high_risk": [
        "Schedule an appointment with your general physician within the next few days.",
        "Bring this report and any previous medical records to your consultation.",
    ],
    "general_low_risk": [
        "Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean protein.",
        "Aim for at least 150 minutes of moderate physical activity per week.",
        "Stay hydrated and ensure 7–9 hours of sleep per night.",
        "Schedule a routine health check-up annually.",
    ],
}


def generate_recommendations(flags, risk_category):
    """
    Map abnormal flags and risk category to preventive recommendations.

    Returns list of {text, category} dicts.
    No prescriptions. General advice only.
    """
    recs = []
    seen = set()

    def add(key, category="general"):
        for text in RECOMMENDATIONS_DB.get(key, []):
            if text not in seen:
                seen.add(text)
                recs.append({"text": text, "category": category})

    # Parameter-specific
    for feature, status in flags.items():
        if "low" in status:
            add(f"{feature}_low", feature)
        elif "high" in status:
            add(f"{feature}_high", feature)

    # Risk level
    if risk_category in ("high", "moderate"):
        add("general_high_risk", "priority")
    else:
        add("general_low_risk", "lifestyle")

    return recs


def format_alert_report(alerts, recommendations):
    """Print-friendly alert and recommendation report."""
    lines = ["=" * 55, "  ALERT REPORT", "=" * 55]
    for a in alerts:
        lines.append(f"  {a['icon']}  [{a['level']}] {a['message']}")
    lines += ["", "=" * 55, "  PREVENTIVE RECOMMENDATIONS", "=" * 55]
    for i, r in enumerate(recommendations, 1):
        lines.append(f"  {i}. {r['text']}")
    lines += ["", "  ─" * 27,
              "  DISCLAIMER: These are general wellness suggestions only.",
              "  This system does not provide medical diagnoses.",
              "  Always consult a qualified healthcare professional.", "=" * 55]
    return "\n".join(lines)
