"""
ValidationEngine — Module 1 orchestrator (Medical Validation Engine).

Pipeline order (must be preserved — later steps assume earlier ones ran):
  1. preprocessing  — alias resolution + tolerant numeric coercion
  2. units          — explicit unit (trusted) or magnitude auto-detect -> canonical value
  3. ranges         — physiological tiering; IMPOSSIBLE (unless "<field>_confirmed") blocks
  4. completeness   — 0-100% score -> Excellent/Good/Fair/Poor
  5. consistency    — cross-parameter contradiction rules
  6. confidence      — combine base confidence x completeness x OCR x consistency
  7. context/tests  — missing-test context -> optional shared semantic retrieval

`ValidationEngine` has ZERO import-time dependency on the recommendation/
knowledge_base packages (and therefore on sentence-transformers) — the
optional `test_suggestion_client` is dependency-injected by predictor.py, so
this package stays independently usable/testable without the heavier ML
dependency installed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from validation.completeness import compute_completeness
from validation.confidence import adjust_confidence
from validation.consistency import run_consistency_checks
from validation.context_builder import build_missing_test_context
from validation.preprocessing import resolve_aliases_and_coerce
from validation.ranges import Tier, check_field_range
from validation.units import normalize_unit


@dataclass
class ValidationResult:
    disease: str
    blocked: bool
    normalized_input: Dict[str, Any] = field(default_factory=dict)
    unit_conversions_applied: List[dict] = field(default_factory=list)
    blocking_errors: List[dict] = field(default_factory=list)
    completeness_score: float = 0.0
    completeness_label: str = "Poor"
    consistency_warnings: List[dict] = field(default_factory=list)
    missing_test_context: dict = field(default_factory=dict)
    suggested_tests: List[dict] = field(default_factory=list)
    base_confidence: float = 0.0
    confidence_penalty: float = 0.0
    adjusted_confidence: float = 0.0
    overrides_applied: List[str] = field(default_factory=list)
    parse_notes: List[str] = field(default_factory=list)
    mandatory_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)

    def to_public_dict(self):
        """JSON-serializable summary attached to the API response as result["validation"]."""
        return {
            "completeness_score": self.completeness_score,
            "completeness_label": self.completeness_label,
            "unit_conversions_applied": self.unit_conversions_applied,
            "consistency_warnings": self.consistency_warnings,
            "blocking_errors": self.blocking_errors,
            "suggested_tests": self.suggested_tests,
            "base_confidence": self.base_confidence,
            "adjusted_confidence": self.adjusted_confidence,
            "confidence_penalty": self.confidence_penalty,
            "overrides_applied": self.overrides_applied,
            "parse_notes": self.parse_notes,
        }

    def to_blocking_error_dict(self):
        """
        Strict superset of the existing "missing mandatory fields" error shape
        (see models/disease_models.py::DiseaseModel._run_prediction), so any
        existing .NET error-handling path isn't surprised by a structurally
        different shape.
        """
        invalid_fields = [e["field"] for e in self.blocking_errors]
        reasons = "; ".join(e["message"] for e in self.blocking_errors)
        return {
            "error": "Invalid field value(s)",
            "error_type": "validation_blocking",
            "disease": self.disease,
            "missing_mandatory": [],
            "mandatory_fields": self.mandatory_fields,
            "optional_fields": self.optional_fields,
            "invalid_fields": invalid_fields,
            "blocking_errors": self.blocking_errors,
            "message": f"Cannot predict {self.disease} risk: {reasons}",
        }


class ValidationEngine:
    def __init__(self, field_schema, test_suggestion_client=None):
        """
        field_schema: the FIELD_SCHEMA dict from predictor.py
                      ({disease: {"mandatory": {field: label}, "optional": {field: label}}}).
        test_suggestion_client: optional object exposing
                      .suggest_tests(context: str, disease: str) -> list[dict]
                      (recommendation.retrieval_engine.RecommendationEngine satisfies this).
                      Injected by predictor.py after the embedder/KB are warmed up.
        """
        self.field_schema = field_schema
        self.test_suggestion_client = test_suggestion_client

    def validate(self, disease, input_data):
        schema = self.field_schema[disease]
        mandatory_fields = list(schema["mandatory"].keys())
        optional_fields = list(schema["optional"].keys())
        all_fields = mandatory_fields + optional_fields
        field_labels = {**schema["mandatory"], **schema["optional"]}

        coerced, parse_notes = resolve_aliases_and_coerce(disease, input_data, all_fields)

        # --- unit normalization ---
        unit_conversions = []
        canonical = {}
        for f in all_fields:
            v = coerced.get(f)
            if v is None:
                canonical[f] = None
                continue
            explicit_unit = input_data.get(f"{f}_unit")
            conv = normalize_unit(disease, f, v, explicit_unit)
            canonical[f] = conv["canonical_value"]
            if conv["method"] != "as_is" or conv["note"]:
                unit_conversions.append(conv)

        # --- physiological tiering + blocking ---
        blocking_errors = []
        overrides_applied = []
        tiers = {}
        for f in all_fields:
            v = canonical.get(f)
            if v is None:
                continue
            check = check_field_range(disease, f, v)
            tiers[f] = Tier(check["tier"])
            if check["blocking"]:
                if bool(input_data.get(f"{f}_confirmed")):
                    tiers[f] = Tier.EXTREME
                    overrides_applied.append(f"{f}_confirmed bypassed impossible-value block (value={v})")
                else:
                    blocking_errors.append(check)

        if blocking_errors:
            return ValidationResult(
                disease=disease, blocked=True,
                blocking_errors=blocking_errors,
                overrides_applied=overrides_applied,
                parse_notes=parse_notes,
                mandatory_fields=mandatory_fields, optional_fields=optional_fields,
            )

        impossible_fields = {f for f, t in tiers.items() if t is Tier.IMPOSSIBLE}

        # --- completeness ---
        mandatory_provided = sum(1 for f in mandatory_fields if canonical.get(f) is not None)
        optional_provided = sum(1 for f in optional_fields if canonical.get(f) is not None)
        completeness = compute_completeness(len(mandatory_fields), mandatory_provided,
                                             len(optional_fields), optional_provided)

        # --- clinical consistency ---
        consistency_warnings = run_consistency_checks(disease, canonical, impossible_fields=impossible_fields)

        # --- confidence adjustment ---
        base_confidence = 100.0 if not optional_fields else round((optional_provided / len(optional_fields)) * 100, 1)
        ocr_confidences = [
            float(input_data[f"{f}_ocr_confidence"]) for f in all_fields
            if isinstance(input_data.get(f"{f}_ocr_confidence"), (int, float))
        ]
        severities = [w["severity"] for w in consistency_warnings]
        confidence_adj = adjust_confidence(base_confidence, completeness["label"], ocr_confidences, severities)

        # --- missing-test context + shared semantic retrieval (Modules 2/3) ---
        missing_optional = [f for f in optional_fields if canonical.get(f) is None]
        context_dict, context_str = build_missing_test_context(
            disease, canonical, tiers, missing_optional, field_labels, consistency_warnings, completeness["label"])

        suggested_tests = []
        if self.test_suggestion_client is not None and mandatory_provided == len(mandatory_fields):
            try:
                suggested_tests = self.test_suggestion_client.suggest_tests(context_str, disease)
            except Exception:
                suggested_tests = []

        normalized_input = {"disease": disease, **{f: canonical.get(f) for f in all_fields}}

        return ValidationResult(
            disease=disease, blocked=False,
            normalized_input=normalized_input,
            unit_conversions_applied=unit_conversions,
            blocking_errors=[],
            completeness_score=completeness["score"], completeness_label=completeness["label"],
            consistency_warnings=consistency_warnings,
            missing_test_context=context_dict,
            suggested_tests=suggested_tests,
            base_confidence=confidence_adj["base_confidence"],
            confidence_penalty=confidence_adj["penalty_points"],
            adjusted_confidence=confidence_adj["adjusted_confidence"],
            overrides_applied=overrides_applied,
            parse_notes=parse_notes,
            mandatory_fields=mandatory_fields, optional_fields=optional_fields,
        )
