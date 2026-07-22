"""
Physiological bounds table — Module 1, "Parameter Validation".

Distinguishes three tiers for every lab value:
  NORMAL    — within the reference range
  EXTREME   — outside reference range but survivable/documented in the medical
              literature; already surfaced by models/disease_models.py's
              per-disease `_safety_override` alerts, so this module does NOT
              duplicate an alert for it, it just tags the tier.
  IMPOSSIBLE — outside what a living human could have. Almost certainly an
              OCR misread or data-entry error. Blocks prediction unless the
              caller explicitly confirms the value (see engine.py overrides).

Bounds are deliberately generous on the IMPOSSIBLE side (rare but real
case-report extremes), never on the side of masking a genuine data error.
"""

from enum import Enum


class Tier(Enum):
    NORMAL = "normal"
    EXTREME = "extreme"
    IMPOSSIBLE = "impossible"


# field -> (hard_min, hard_max, normal_min, normal_max)
# hard_min/hard_max: physiological viability bounds (outside => IMPOSSIBLE)
# normal_min/normal_max: reference range (outside but within hard bounds => EXTREME)
FIELD_BOUNDS = {
    "anemia": {
        "hemoglobin":  (2.0, 25.0, 12.0, 17.5),   # g/dL   — survival case reports ~1.4-2 g/dL; polycythemia ceiling ~24-25
        "rbc":         (1.0, 8.0, 4.2, 6.1),       # M/uL   — aplastic crisis floor ~1.0; polycythemia vera ceiling ~7-8
        "mcv":         (40, 140, 80, 100),         # fL     — severe thalassemic microcytosis ~40-50; megaloblastic macrocytosis ~130-140
        "mch":         (10, 50, 27, 33),           # pg     — severe hypochromia ~10-14; severe macrocytic anemia ~40-45
        "hematocrit":  (9, 65, 36, 52),            # %      — survivable floor ~9%; polycythemia vera ceiling ~63-65%
        "ferritin":    (0, 3000, 15, 200),         # ng/mL  — cannot be negative; assay ceiling ~2000-3000 without dilution
    },
    "diabetes": {
        "glucose":         (10, 1500, 70, 99),     # mg/dL  — severe hypoglycemia survival floor ~10-20; HHS crisis ceiling ~1200-1500
        "hba1c":           (3.0, 20.0, 4.0, 5.6),  # %      — physiologic floor ~3%; extreme chronic hyperglycemia lab ceiling ~18-20%
        "bmi":             (8, 100, 18.5, 24.9),   # kg/m2  — extreme cachexia case-report floor ~8-10; super-obesity ceiling ~100
        "age":             (0, 120, 18, 90),       # years  — oldest verified human lifespan 122y; 120 practical ceiling
        "insulin":         (0, 1000, 2, 25),       # uU/mL  — 0 valid (T1D); insulinoma/severe resistance up to ~500-1000
        "blood_pressure":  (20, 180, 60, 80),      # mmHg (diastolic) — profound shock floor ~20-30; hypertensive emergency ceiling ~160-180
    },
    "infection": {
        "wbc":          (0, 500000, 4000, 11000),  # cells/uL — 0 valid (agranulocytosis); leukemic blast-crisis up to ~400-500k
        "neutrophils":  (0, 100, 40, 75),          # %
        "lymphocytes":  (0, 100, 20, 45),          # %
        "crp":          (0, 1000, 0, 10),          # mg/L   — cannot be negative; extreme sepsis ~500-700, assay ceiling ~1000
        "esr":          (0, 150, 0, 20),           # mm/hr  — Westergren tube physically limited to ~140-150mm
        "temperature":  (30, 45, 36.1, 37.5),      # degC   — hypothermia/hyperthermia viability window for a reporting patient
    },
    "cholesterol": {
        "total_cholesterol":  (50, 1500, 100, 199),  # mg/dL — cell-membrane floor ~40-50; homozygous FH case reports ~1000-1200
        "ldl":                (0, 1000, 40, 129),     # mg/dL — cannot be negative; homozygous FH ceiling ~600-1000
        "hdl":                (5, 200, 40, 100),      # mg/dL — Tangier disease floor ~1-5; hyperalphalipoproteinemia ceiling ~150-200
        "triglycerides":      (10, 15000, 30, 149),   # mg/dL — hypobetalipoproteinemia floor ~10-20; chylomicronemia ceiling ~10000-15000
        "vldl":               (0, 3000, 2, 30),       # mg/dL — derived from triglycerides/5, bound tracks TG bound
        "cholesterol_ratio":  (1.0, 40, 2.0, 5.0),    # TC/HDL — mathematically TC >= HDL always, so ratio >= 1
    },
}


def get_bounds(disease, field):
    """Return (hard_min, hard_max, normal_min, normal_max) or None if unbounded."""
    return FIELD_BOUNDS.get(disease, {}).get(field)


def classify_value(disease, field, value):
    """Classify a single canonical-unit value into a Tier."""
    bounds = get_bounds(disease, field)
    if bounds is None:
        return Tier.NORMAL
    hard_min, hard_max, normal_min, normal_max = bounds
    if value < hard_min or value > hard_max:
        return Tier.IMPOSSIBLE
    if value < normal_min or value > normal_max:
        return Tier.EXTREME
    return Tier.NORMAL


def check_field_range(disease, field, value):
    """
    Return a dict describing the tier check result:
      {field, value, tier, blocking, message}
    `message` is None for NORMAL/EXTREME (EXTREME is silently tagged — the
    existing safety-override alerts already cover clinically dangerous but
    plausible values).
    """
    tier = classify_value(disease, field, value)
    message = None
    if tier is Tier.IMPOSSIBLE:
        bounds = get_bounds(disease, field)
        hard_min, hard_max, _, _ = bounds
        message = (
            f"{field} value {value} is physiologically impossible "
            f"(must be between {hard_min} and {hard_max})."
        )
    return {
        "field": field,
        "value": value,
        "tier": tier.value,
        "blocking": tier is Tier.IMPOSSIBLE,
        "message": message,
    }
