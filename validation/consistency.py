"""
Clinical Consistency Check — Module 1, "Clinical Consistency Check".

Cross-parameter contradiction detection, expressed as a declarative rule
table rather than nested if/else, so rules stay independently inspectable
and extensible. This is deterministic/rule-based by design (safety-critical
medical logic) — a different constraint than the recommendation engine,
which must avoid hardcoded decision logic.

A rule is skipped (not evaluated) if any required field is missing, or if
its `applicability_fn` returns False (e.g. Friedewald's formula is only
valid when triglycerides < 400 mg/dL).
"""

from dataclasses import dataclass, field
from typing import Callable, Tuple


@dataclass(frozen=True)
class ConsistencyRule:
    rule_id: str
    disease: str
    fields_required: Tuple[str, ...]
    check_fn: Callable[[dict], str]  # returns a message string, or None if consistent
    severity: str  # "info" | "warning" | "critical"
    applicability_fn: Callable[[dict], bool] = field(default=lambda values: True)


def _anemia_rule_of_three(values):
    hgb, hct = values["hemoglobin"], values["hematocrit"]
    expected_hct = hgb * 3.0
    if abs(hct - expected_hct) > 3.0:
        return (f"Hematocrit ({hct}%) is inconsistent with hemoglobin ({hgb} g/dL) under the "
                f"hematology 'rule of three' (expected ~{round(expected_hct, 1)}%).")
    return None


def _anemia_mch_definition(values):
    hgb, rbc, mch = values["hemoglobin"], values["rbc"], values["mch"]
    if rbc <= 0:
        return None
    expected_mch = hgb * 10.0 / rbc
    if abs(mch - expected_mch) > max(2.0, 0.10 * expected_mch):
        return (f"MCH ({mch} pg) is inconsistent with hemoglobin and RBC "
                f"(expected ~{round(expected_mch, 1)} pg from MCH = Hgb x 10 / RBC).")
    return None


def _anemia_rbc_mcv_hct(values):
    rbc, mcv, hct = values["rbc"], values["mcv"], values["hematocrit"]
    expected_hct2 = rbc * mcv / 10.0
    if abs(hct - expected_hct2) > 4.0:
        return (f"Hematocrit ({hct}%) is inconsistent with RBC and MCV "
                f"(expected ~{round(expected_hct2, 1)}% from RBC x MCV / 10).")
    return None


def _diabetes_glucose_hba1c_mismatch(values):
    glucose, hba1c = values["glucose"], values["hba1c"]
    if glucose > 200 and hba1c < 5.7:
        return (f"Glucose ({glucose} mg/dL) is diabetic-range but HbA1c ({hba1c}%) is normal — "
                f"suggests acute stress hyperglycemia or a data-entry error rather than sustained diabetes.")
    return None


def _diabetes_glucose_hba1c_mismatch_severe(values):
    glucose, hba1c = values["glucose"], values["hba1c"]
    if glucose > 300 and hba1c < 6.5:
        return (f"Glucose ({glucose} mg/dL) is severely elevated but HbA1c ({hba1c}%) is below the "
                f"diabetic threshold — a large, clinically significant mismatch. Verify fasting status and re-test.")
    return None


def _diabetes_hba1c_glucose_reverse_mismatch(values):
    glucose, hba1c = values["glucose"], values["hba1c"]
    if hba1c > 8 and glucose < 100:
        return (f"HbA1c ({hba1c}%) indicates poor long-term control but the current glucose ({glucose} mg/dL) "
                f"is normal — verify fasting status and recent medication changes.")
    return None


def _infection_differential_sum_high(values):
    total = values["neutrophils"] + values["lymphocytes"]
    if total > 102:
        return (f"Neutrophils + lymphocytes ({total}%) exceeds 100% (with rounding tolerance) — "
                f"check for a data-entry or OCR error in the differential count.")
    return None


def _infection_differential_sum_low(values):
    total = values["neutrophils"] + values["lymphocytes"]
    if total < 70:
        return (f"Neutrophils + lymphocytes ({total}%) leaves an unusually large remainder — "
                f"verify monocyte/eosinophil/basophil counts or check for a data-entry error.")
    return None


def _infection_crp_without_fever(values):
    temp, crp = values["temperature"], values["crp"]
    if temp < 37 and crp > 100:
        return (f"CRP is markedly elevated ({crp} mg/L) without fever ({temp} degC) — atypical for acute "
                f"bacterial infection; consider a non-infectious inflammatory cause or verify the readings.")
    return None


def _cholesterol_friedewald(values):
    tc, ldl, hdl, trig = values["total_cholesterol"], values["ldl"], values["hdl"], values["triglycerides"]
    expected_ldl = tc - hdl - trig / 5.0
    if abs(ldl - expected_ldl) > 15:
        return (f"LDL ({ldl} mg/dL) is inconsistent with the Friedewald estimate "
                f"(~{round(expected_ldl, 1)} mg/dL from TC - HDL - TG/5).")
    return None


def _cholesterol_ratio(values):
    tc, hdl, ratio = values["total_cholesterol"], values["hdl"], values["cholesterol_ratio"]
    if hdl <= 0:
        return None
    expected_ratio = tc / hdl
    if expected_ratio > 0 and abs(ratio - expected_ratio) / expected_ratio > 0.15:
        return (f"Cholesterol ratio ({ratio}) is inconsistent with TC/HDL "
                f"(expected ~{round(expected_ratio, 1)}).")
    return None


def _cholesterol_vldl(values):
    trig, vldl = values["triglycerides"], values["vldl"]
    expected_vldl = trig / 5.0
    if expected_vldl > 0 and abs(vldl - expected_vldl) / expected_vldl > 0.20:
        return (f"VLDL ({vldl} mg/dL) is inconsistent with triglycerides "
                f"(expected ~{round(expected_vldl, 1)} mg/dL from TG/5).")
    return None


RULES = (
    ConsistencyRule("anemia_hct_hgb_rule_of_three", "anemia", ("hemoglobin", "hematocrit"),
                     _anemia_rule_of_three, "warning"),
    ConsistencyRule("anemia_mch_definition_check", "anemia", ("hemoglobin", "rbc", "mch"),
                     _anemia_mch_definition, "warning"),
    ConsistencyRule("anemia_rbc_mcv_hct_check", "anemia", ("rbc", "mcv", "hematocrit"),
                     _anemia_rbc_mcv_hct, "info"),
    ConsistencyRule("diabetes_glucose_hba1c_mismatch", "diabetes", ("glucose", "hba1c"),
                     _diabetes_glucose_hba1c_mismatch, "warning"),
    ConsistencyRule("diabetes_glucose_hba1c_mismatch_severe", "diabetes", ("glucose", "hba1c"),
                     _diabetes_glucose_hba1c_mismatch_severe, "critical"),
    ConsistencyRule("diabetes_hba1c_glucose_reverse_mismatch", "diabetes", ("glucose", "hba1c"),
                     _diabetes_hba1c_glucose_reverse_mismatch, "warning"),
    ConsistencyRule("infection_differential_sum_high", "infection", ("neutrophils", "lymphocytes"),
                     _infection_differential_sum_high, "critical"),
    ConsistencyRule("infection_differential_sum_low", "infection", ("neutrophils", "lymphocytes"),
                     _infection_differential_sum_low, "warning"),
    ConsistencyRule("infection_crp_without_fever", "infection", ("temperature", "crp"),
                     _infection_crp_without_fever, "info"),
    ConsistencyRule("cholesterol_friedewald_consistency", "cholesterol",
                     ("total_cholesterol", "ldl", "hdl", "triglycerides"),
                     _cholesterol_friedewald, "warning",
                     applicability_fn=lambda values: values["triglycerides"] < 400),
    ConsistencyRule("cholesterol_ratio_consistency", "cholesterol",
                     ("total_cholesterol", "hdl", "cholesterol_ratio"), _cholesterol_ratio, "warning"),
    ConsistencyRule("cholesterol_vldl_triglyceride_check", "cholesterol",
                     ("triglycerides", "vldl"), _cholesterol_vldl, "info"),
)


def run_consistency_checks(disease, normalized_values, impossible_fields=()):
    """
    normalized_values: {field: float_or_None} in canonical units.
    impossible_fields: set/iterable of field names classified as IMPOSSIBLE (skip rules touching them).
    Returns list[{rule_id, disease, severity, fields_involved, message}].
    """
    warnings = []
    impossible = set(impossible_fields)
    for rule in RULES:
        if rule.disease != disease:
            continue
        if any(normalized_values.get(f) is None or f in impossible for f in rule.fields_required):
            continue
        values = {f: normalized_values[f] for f in rule.fields_required}
        if not rule.applicability_fn(values):
            continue
        message = rule.check_fn(values)
        if message:
            warnings.append({
                "rule_id": rule.rule_id,
                "disease": disease,
                "severity": rule.severity,
                "fields_involved": list(rule.fields_required),
                "message": message,
            })
    return warnings
