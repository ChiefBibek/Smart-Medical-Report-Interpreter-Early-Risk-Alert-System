"""
Unit validation — Module 1, "Unit Validation".

Recognizes an explicitly declared unit (always trusted) or auto-detects the
unit from the value's magnitude when no explicit unit is given. Never
silently guesses when the magnitude is genuinely ambiguous between two
plausible units — passes the value through unconverted with a low-confidence
note instead.
"""

from validation.ranges import Tier, classify_value

# One entry per field that has a common alternate unit in real-world reports.
# canonical_value = raw_value * factor   (factor converts alt_unit -> canonical_unit)
# auto_convert_if / assume_canonical_if / ambiguous_if operate on the RAW value.
UNIT_CONFIG = {
    "glucose": {
        "alt_names": ("mmol/l", "mmol"), "canonical_names": ("mg/dl", "mg/dL"),
        "canonical_unit": "mg/dL", "alt_unit": "mmol/L", "factor": 18.0182,
        "auto_convert_if": lambda v: v <= 25,
        "ambiguous_if": lambda v: 25 < v < 45,
    },
    "total_cholesterol": {
        "alt_names": ("mmol/l",), "canonical_names": ("mg/dl",),
        "canonical_unit": "mg/dL", "alt_unit": "mmol/L", "factor": 38.67,
        "auto_convert_if": lambda v: v <= 15,
        "ambiguous_if": lambda v: 15 < v < 45,
    },
    "ldl": {
        "alt_names": ("mmol/l",), "canonical_names": ("mg/dl",),
        "canonical_unit": "mg/dL", "alt_unit": "mmol/L", "factor": 38.67,
        "auto_convert_if": lambda v: v <= 15,
        "ambiguous_if": lambda v: 15 < v < 45,
    },
    "hdl": {
        "alt_names": ("mmol/l",), "canonical_names": ("mg/dl",),
        "canonical_unit": "mg/dL", "alt_unit": "mmol/L", "factor": 38.67,
        "auto_convert_if": lambda v: v <= 3.5,
        "ambiguous_if": lambda v: 3.5 < v < 15,
    },
    "triglycerides": {
        "alt_names": ("mmol/l",), "canonical_names": ("mg/dl",),
        "canonical_unit": "mg/dL", "alt_unit": "mmol/L", "factor": 88.57,
        "auto_convert_if": lambda v: v <= 10,
        "ambiguous_if": lambda v: 10 < v < 40,
    },
    "vldl": {
        # VLDL is reported as a cholesterol fraction (VLDL-C) -> cholesterol factor, not the TG factor.
        "alt_names": ("mmol/l",), "canonical_names": ("mg/dl",),
        "canonical_unit": "mg/dL", "alt_unit": "mmol/L", "factor": 38.67,
        "auto_convert_if": lambda v: v <= 2,
        "ambiguous_if": lambda v: 2 < v < 8,
    },
    "hemoglobin": {
        "alt_names": ("g/l",), "canonical_names": ("g/dl",),
        "canonical_unit": "g/dL", "alt_unit": "g/L", "factor": 0.1,
        "auto_convert_if": lambda v: v > 25,
        "ambiguous_if": lambda v: 20 <= v <= 25,
    },
    "wbc": {
        "alt_names": ("x10^9/l", "10^9/l", "10e9/l"), "canonical_names": ("cells/ul", "/ul"),
        "canonical_unit": "cells/uL", "alt_unit": "x10^9/L", "factor": 1000,
        "auto_convert_if": lambda v: v < 100,
        "ambiguous_if": lambda v: 100 <= v < 1000,
    },
    "temperature": {
        "alt_names": ("f", "fahrenheit"), "canonical_names": ("c", "celsius"),
        "canonical_unit": "C", "alt_unit": "F",
        "convert": lambda v: (v - 32) * 5.0 / 9.0,
        "auto_convert_if": lambda v: 45 < v <= 115,
        "ambiguous_if": lambda v: False,  # unambiguous: >45 is impossible in C
    },
    "crp": {
        # Overlap at low values is genuinely ambiguous (5 mg/L vs 5 mg/dL=50 mg/L
        # are both plausible) -> auto-detection is not attempted for this field.
        "alt_names": ("mg/dl",), "canonical_names": ("mg/l",),
        "canonical_unit": "mg/L", "alt_unit": "mg/dL", "factor": 10,
        "no_auto": True,
    },
}


def _convert(field, cfg, raw_value):
    if "convert" in cfg:
        return cfg["convert"](raw_value)
    return raw_value * cfg["factor"]


def normalize_unit(disease, field, raw_value, explicit_unit=None):
    """
    Return a UnitConversion dict:
      {field, raw_value, raw_unit, canonical_value, canonical_unit, method, confidence, note}
    method: "explicit" | "auto" | "as_is" | "ambiguous_as_is"
    """
    cfg = UNIT_CONFIG.get(field)

    if cfg is None:
        return {
            "field": field, "raw_value": raw_value, "raw_unit": None,
            "canonical_value": raw_value, "canonical_unit": None,
            "method": "as_is", "confidence": "high", "note": None,
        }

    canonical_unit = cfg["canonical_unit"]

    # 1) Explicit unit always wins when recognized.
    if explicit_unit:
        unit_norm = str(explicit_unit).strip().lower()
        if unit_norm in cfg["alt_names"]:
            return {
                "field": field, "raw_value": raw_value, "raw_unit": cfg["alt_unit"],
                "canonical_value": _convert(field, cfg, raw_value), "canonical_unit": canonical_unit,
                "method": "explicit", "confidence": "high", "note": None,
            }
        if unit_norm in cfg["canonical_names"]:
            return {
                "field": field, "raw_value": raw_value, "raw_unit": canonical_unit,
                "canonical_value": raw_value, "canonical_unit": canonical_unit,
                "method": "explicit", "confidence": "high", "note": None,
            }
        # Unrecognized unit string -> ignore it, fall through to auto-detection,
        # but leave a note since the caller's stated unit couldn't be honored.
        fallback_note = f"Unrecognized unit '{explicit_unit}' for {field}; falling back to auto-detection."
    else:
        fallback_note = None

    if cfg.get("no_auto"):
        # No note in the common case (no explicit unit given) -- assuming the canonical
        # unit by convention is not itself noteworthy and would otherwise spam
        # unit_conversions_applied on every request that includes this field.
        return {
            "field": field, "raw_value": raw_value, "raw_unit": canonical_unit,
            "canonical_value": raw_value, "canonical_unit": canonical_unit,
            "method": "as_is", "confidence": "high" if fallback_note is None else "low",
            "note": fallback_note,
        }

    # 2) No (usable) explicit unit -> try both interpretations, prefer whichever is plausible.
    as_is_tier = classify_value(disease, field, raw_value)
    converted_value = _convert(field, cfg, raw_value)
    converted_tier = classify_value(disease, field, converted_value)

    as_is_ok = as_is_tier is not Tier.IMPOSSIBLE
    converted_ok = converted_tier is not Tier.IMPOSSIBLE

    if as_is_ok and not converted_ok:
        return {
            "field": field, "raw_value": raw_value, "raw_unit": canonical_unit,
            "canonical_value": raw_value, "canonical_unit": canonical_unit,
            "method": "as_is", "confidence": "high", "note": fallback_note,
        }
    if converted_ok and not as_is_ok:
        return {
            "field": field, "raw_value": raw_value, "raw_unit": cfg["alt_unit"],
            "canonical_value": converted_value, "canonical_unit": canonical_unit,
            "method": "auto", "confidence": "high",
            "note": fallback_note or f"Auto-detected {field}={raw_value} as {cfg['alt_unit']}, converted to {round(converted_value, 2)} {canonical_unit}.",
        }

    # Both impossible -> not a unit problem at all (e.g. a negative value). Check this
    # BEFORE the magnitude heuristics below, which are only meant to disambiguate
    # between two PLAUSIBLE interpretations, not to be applied when neither is plausible.
    if not as_is_ok and not converted_ok:
        note = (
            f"{field}={raw_value} is implausible under both {canonical_unit} and {cfg['alt_unit']} "
            f"interpretations; likely a data-entry or OCR error, not a unit mismatch."
        )
        return {
            "field": field, "raw_value": raw_value, "raw_unit": canonical_unit,
            "canonical_value": raw_value, "canonical_unit": canonical_unit,
            "method": "as_is", "confidence": "low", "note": ((fallback_note or "") + " " + note).strip(),
        }

    # Both plausible -> fall back to magnitude heuristics to pick the likelier unit.
    if cfg["auto_convert_if"](raw_value):
        return {
            "field": field, "raw_value": raw_value, "raw_unit": cfg["alt_unit"],
            "canonical_value": converted_value, "canonical_unit": canonical_unit,
            "method": "auto", "confidence": "high",
            "note": fallback_note or f"Auto-detected {field}={raw_value} as {cfg['alt_unit']}, converted to {round(converted_value, 2)} {canonical_unit}.",
        }
    if cfg["ambiguous_if"](raw_value):
        note = (fallback_note or "") + (
            f" Ambiguous unit for {field}={raw_value}: could be {canonical_unit} or {cfg['alt_unit']}; "
            f"passed through unconverted. Provide '{field}_unit' explicitly to resolve."
        )
        return {
            "field": field, "raw_value": raw_value, "raw_unit": canonical_unit,
            "canonical_value": raw_value, "canonical_unit": canonical_unit,
            "method": "ambiguous_as_is", "confidence": "low", "note": note.strip(),
        }

    # Neither auto nor ambiguous predicate fired (value sits above both bands) -> assume canonical.
    return {
        "field": field, "raw_value": raw_value, "raw_unit": canonical_unit,
        "canonical_value": raw_value, "canonical_unit": canonical_unit,
        "method": "as_is", "confidence": "high", "note": fallback_note,
    }
