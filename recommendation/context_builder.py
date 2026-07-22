"""
Patient Clinical Context Generator — Module 2.

Turns disease + prediction + SHAP-style contributions + lab values +
validation results + (optional) patient history into ONE natural-language
paragraph, which gets Sentence-BERT embedded and semantically matched
against the knowledge base (retrieval_engine.py). This module only ASSEMBLES
descriptive text — it makes no decision about which recommendation to
return; that decision happens entirely in the embedding-similarity retrieval
step, never here.

Every field in `all_factors` is described, not just the top-3 `top_factors`,
so a clinically decisive but numerically low-SHAP-weight field (e.g.
ferritin) is never dropped from the embedded text — dropping it would break
the iron-deficiency vs. further-workup discrimination the spec requires.

Qualitative low/high/normal labeling reuses validation/ranges.py's real
clinical reference ranges (the same bounds table the Validation Engine uses),
rather than a training-mean-deviation heuristic — more medically accurate,
and immune to the bimodal-training-distribution mislabeling risk a
mean-deviation approach would have near a feature's decision boundary.
"""

import re

from validation.ranges import get_bounds

_EMOJI_PREFIX = re.compile(r"^[^\w]+")
_UNIT_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def _clean_label(display_label):
    """Strip a trailing "(unit)" suffix, e.g. "Body Temperature (°C)" -> "Body Temperature".
    KB scenario text never includes units/raw numbers (it's written in flowing clinical
    prose, e.g. "generally below approximately 7 g/dL"), so per-field sentences that keep
    repeating units and exact decimal values are register-mismatched against the corpus
    they're being matched to — that mismatch measurably dilutes retrieval quality more
    than dropping the number costs, since severity/polarity is already carried by the
    low/elevated/normal wording and by the safety-override alert text."""
    return _UNIT_SUFFIX.sub("", display_label).strip()


def _qualitative_label(disease, field, value):
    bounds = get_bounds(disease, field)
    if bounds is None or value is None:
        return None
    _, _, normal_min, normal_max = bounds
    if value < normal_min:
        return "low"
    if value > normal_max:
        return "high"
    return "within the normal range"


def build_clinical_context(disease, risk_level, risk_probability, top_factors, all_factors,
                            raw_values, fields_imputed, fields_missing_optional, field_labels,
                            prediction_confidence, validation_warnings=None,
                            completeness_label=None, patient_history=None, alerts=None):
    """
    raw_values: {field: float_or_None} in canonical units (validation_result.normalized_input).
    all_factors / top_factors: the model's existing SHAP-style output — list of
                {feature, importance_pct/contribution, imputed}, sorted descending by importance.
    alerts: the model's existing rule-based safety-override alert strings (e.g. "CRITICAL:
                Hemoglobin < 9 g/dL"). These carry strong, already-clinically-validated
                language ("severe", "critically low", "depleted") that a small embedding
                model latches onto far more reliably than a templated "Low X: value" list —
                without them, generic per-field labels can be dominated by shared lab-test
                vocabulary (e.g. a severely abnormal panel can score deceptively close to a
                "all normal" scenario purely because both mention the same six test names).
    Returns a single natural-language paragraph (str) ready for sentence embedding.
    """
    fields_imputed = set(fields_imputed or [])
    validation_warnings = validation_warnings or []
    sentences = [f"Patient lab profile assessed for {disease} risk."]

    if risk_level is not None and risk_probability is not None:
        sentences.append(
            f"Model-predicted risk level is {risk_level} with an estimated probability of "
            f"{round(risk_probability * 100)}%."
        )

    for factor in (all_factors or []):
        field = factor["feature"]
        value = raw_values.get(field)
        if value is None:
            continue
        label = _qualitative_label(disease, field, value)
        clean_label = _clean_label(field_labels.get(field, field))
        imputed_note = " (estimated, not directly measured)" if field in fields_imputed else ""
        if label == "within the normal range":
            sentences.append(f"{clean_label} is within the normal range{imputed_note}.")
        elif label == "high":
            sentences.append(f"{clean_label} is elevated{imputed_note}.")
        elif label == "low":
            sentences.append(f"{clean_label} is low{imputed_note}.")
        else:
            sentences.append(f"{clean_label}: {value}{imputed_note}.")

    if top_factors:
        primary = [f["feature"] for f in top_factors[:2]]
        if primary:
            sentences.append("The model identifies " + " and ".join(primary) + " as the primary contributing factors.")

    if alerts:
        # Strip only the leading emoji glyph; keep words like "CRITICAL"/"depleted"/
        # "severe" intact — those carry the semantic weight the embedding needs.
        clean_alerts = [_EMOJI_PREFIX.sub("", a).strip() for a in alerts]
        sentences.append("Clinical alerts: " + "; ".join(clean_alerts) + ".")

    if fields_missing_optional:
        sentences.append("The following optional fields were not provided: " + ", ".join(fields_missing_optional) + ".")

    if validation_warnings:
        sentences.append("Data quality notes: " + "; ".join(validation_warnings) + ".")

    if completeness_label is not None:
        sentences.append(
            f"Overall data completeness: {completeness_label} (prediction confidence {prediction_confidence}%)."
        )

    if patient_history:
        conditions = patient_history.get("conditions") if isinstance(patient_history, dict) else None
        if conditions:
            sentences.append("Patient history: " + ", ".join(conditions) + ".")

    return " ".join(sentences)
