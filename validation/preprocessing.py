"""
Preprocessing — alias resolution + tolerant numeric coercion.

Runs first in the validation pipeline (engine.py), before unit normalization
or range checks ever see a value. This is the ONLY place field-name aliases
are resolved (e.g. the CholesterolModel's old "cholesterol" -> "total_cholesterol"
rename), since validation must see the canonical field name before it can
look up bounds/units for it.
"""

import re

# disease -> {alias_field_name: canonical_field_name}
ALIASES = {
    "cholesterol": {"cholesterol": "total_cholesterol"},
}

_NUMERIC_TOKEN = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def coerce_numeric(value):
    """
    Tolerantly parse a lab value into a float, or None if genuinely missing/unparseable.
    Returns (float_or_None, note_or_None). Never raises.
    """
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"Boolean value '{value}' is not a valid lab value; treated as missing."
    if isinstance(value, (int, float)):
        return float(value), None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, None
        try:
            return float(text), None
        except ValueError:
            pass
        try:
            return float(text.replace(",", ".")), None
        except ValueError:
            pass
        match = _NUMERIC_TOKEN.search(text)
        if match:
            token = match.group(0).replace(",", ".")
            try:
                return float(token), f"Parsed numeric value {token} out of raw text '{value}'."
            except ValueError:
                pass
        return None, f"Unparseable value '{value}' treated as missing."
    return None, f"Unsupported value type for field value '{value}'; treated as missing."


def resolve_aliases(disease, input_data):
    """Return a shallow copy of input_data with known field aliases renamed to canonical names."""
    working = dict(input_data)
    for alias, canonical in ALIASES.get(disease, {}).items():
        if alias in working and canonical not in working:
            working[canonical] = working.pop(alias)
    return working


def resolve_aliases_and_coerce(disease, input_data, known_fields):
    """
    Alias-resolve, then tolerantly coerce every field in `known_fields` to float|None.

    Returns (coerced, parse_notes):
      coerced: {field_name: float_or_None} for every field in known_fields
      parse_notes: list[str] describing any lossy/ambiguous parses
    Sidecar keys ("<field>_unit", "<field>_ocr_confidence", "<field>_confirmed",
    "disease", "patient_history") are left untouched in the original input_data
    for the caller to read directly.
    """
    working = resolve_aliases(disease, input_data)
    coerced, parse_notes = {}, []
    for field in known_fields:
        value, note = coerce_numeric(working.get(field))
        coerced[field] = value
        if note:
            parse_notes.append(f"{field}: {note}")
    return coerced, parse_notes
