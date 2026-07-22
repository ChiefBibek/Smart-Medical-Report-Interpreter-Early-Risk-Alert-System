"""
Missing-Test Context Builder — Module 1, "Suggested Additional Tests".

Builds a natural-language context string describing which fields are
abnormal and which optional fields are missing, WITHOUT deciding which tests
to suggest — that decision is delegated to the shared semantic retrieval
engine (recommendation/retrieval_engine.py :: suggest_tests()), which ranks
candidate tests by embedding similarity against the knowledge base. This
module only assembles descriptive text; it contains no disease-specific
if/elif branching on what to recommend.
"""

from validation.ranges import Tier, get_bounds


def _direction(disease, field, value):
    bounds = get_bounds(disease, field)
    if bounds is None:
        return "abnormal"
    _, _, normal_min, normal_max = bounds
    midpoint = (normal_min + normal_max) / 2.0
    return "low" if value < midpoint else "high"


def build_missing_test_context(disease, normalized_values, tiers, missing_optional_fields,
                                field_labels, consistency_warnings, completeness_label):
    """
    normalized_values: {field: float_or_None} canonical-unit values actually present.
    tiers: {field: Tier} classification for each present field (from ranges.classify_value).
    missing_optional_fields: list[str] of optional field names not provided.
    field_labels: {field: "Human readable label (unit)"}.
    consistency_warnings: list[{message: str, ...}] from consistency.run_consistency_checks().
    completeness_label: str, from completeness.compute_completeness().

    Returns (context_dict, context_string) — the string is what gets embedded/searched.
    """
    provided = {f: v for f, v in normalized_values.items() if v is not None}
    abnormal_fields = [
        {"field": f, "value": v, "tier": tiers[f].value, "direction": _direction(disease, f, v)}
        for f, v in provided.items()
        if tiers.get(f) is Tier.EXTREME
    ]

    missing_labels = [field_labels.get(f, f) for f in missing_optional_fields]
    warning_messages = [w["message"] for w in consistency_warnings]

    context = {
        "disease": disease,
        "provided_fields": provided,
        "abnormal_fields": abnormal_fields,
        "missing_optional_fields": list(missing_optional_fields),
        "missing_optional_field_labels": missing_labels,
        "consistency_warnings": warning_messages,
        "completeness_label": completeness_label,
    }

    sentences = [f"Patient lab profile assessed for {disease}."]
    if abnormal_fields:
        parts = [f"{a['field']} is {a['direction']} at {a['value']}" for a in abnormal_fields]
        sentences.append("Abnormal findings: " + "; ".join(parts) + ".")
    if missing_labels:
        sentences.append("The following optional fields were not provided: " + ", ".join(missing_labels) + ".")
    if warning_messages:
        sentences.append("Data quality notes: " + "; ".join(warning_messages) + ".")
    sentences.append(f"Overall data completeness: {completeness_label}.")

    return context, " ".join(sentences)
