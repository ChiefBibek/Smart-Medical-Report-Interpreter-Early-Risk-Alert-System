"""
Disease-Specific Model Trainers — with Mandatory / Optional Field Support

Each disease defines:
  MANDATORY_FIELDS  — minimum required to run the model (prediction blocked without these)
  OPTIONAL_FIELDS   — enrich prediction when present (imputed from training mean if absent)

Missing optional fields are filled with the training-set mean for that feature.
A confidence score (0-100%) reflects how many optional fields were actually provided.
"""

import numpy as np
from models.logistic_regression import LogisticRegressionScratch
from data.data_generator import (
    MedicalDataGenerator, MinMaxNormalizer,
    train_test_split, compute_metrics
)


class DiseaseModel:
    """Base class for all disease-specific models."""

    MANDATORY_FIELDS = []
    OPTIONAL_FIELDS  = []

    def __init__(self, disease_name, learning_rate=0.05, n_iterations=3000, l2_lambda=0.01):
        self.disease_name    = disease_name
        self.model           = LogisticRegressionScratch(learning_rate=learning_rate,
                                                         n_iterations=n_iterations,
                                                         l2_lambda=l2_lambda)
        self.normalizer      = MinMaxNormalizer()
        self.feature_names   = []
        self.training_means  = {}
        self.is_trained      = False
        self.metrics         = {}

    def train(self, X, y, feature_names):
        self.feature_names  = feature_names
        self.training_means = {fname: float(np.mean(X[:, i]))
                               for i, fname in enumerate(feature_names)}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
        X_train_norm = self.normalizer.fit_transform(X_train)
        X_test_norm  = self.normalizer.transform(X_test)
        self.model.fit(X_train_norm, y_train)
        y_pred       = self.model.predict(X_test_norm)
        y_prob       = self.model.predict_proba(X_test_norm)
        self.metrics = compute_metrics(y_test, y_pred, y_prob)
        self.is_trained = True
        return self.metrics

    def validate_input(self, input_dict):
        missing_mandatory = [f for f in self.MANDATORY_FIELDS
                             if f not in input_dict or input_dict[f] is None]
        missing_optional  = [f for f in self.OPTIONAL_FIELDS
                             if f not in input_dict or input_dict[f] is None]
        return (len(missing_mandatory) == 0), missing_mandatory, missing_optional

    def _build_feature_array(self, input_dict):
        values         = []
        imputed_fields = []
        for fname in self.feature_names:
            if fname in input_dict and input_dict[fname] is not None:
                values.append(float(input_dict[fname]))
            else:
                values.append(self.training_means.get(fname, 0.0))
                imputed_fields.append(fname)
        return np.array(values), imputed_fields

    def _compute_confidence(self, missing_optional):
        if not self.OPTIONAL_FIELDS:
            return 100.0
        provided = len(self.OPTIONAL_FIELDS) - len(missing_optional)
        return round((provided / len(self.OPTIONAL_FIELDS)) * 100, 1)

    def _get_risk_level(self, probability):
        if probability < 0.2:
            return "Low", "🟢"
        elif probability < 0.5:
            return "Borderline", "⚠️"
        elif probability < 0.75:
            return "Moderate", "🟡"
        else:
            return "High", "🔴"

    def _compute_contributions(self, norm_array, imputed_fields):
        weights, _ = self.model.get_weights()
        contributions = norm_array * weights
        total = np.sum(np.abs(contributions)) + 1e-10
        importances = (np.abs(contributions) / total) * 100
        result = []
        for i, fname in enumerate(self.feature_names):
            result.append({
                "feature":        fname,
                "importance_pct": round(float(importances[i]), 2),
                "weight":         round(float(weights[i]), 4),
                "imputed":        fname in imputed_fields,
            })
        result.sort(key=lambda x: x["importance_pct"], reverse=True)
        return result

    def _generate_explanation(self, top_factors, imputed_fields, confidence):
        actual = [f for f in top_factors if not f["imputed"]][:2]
        imputed_note = ""
        if imputed_fields:
            names = ", ".join(f.replace("_", " ") for f in imputed_fields)
            verb  = "was" if len(imputed_fields) == 1 else "were"
            imputed_note = (f" Note: {names} {verb} not provided and estimated from "
                            f"population averages (prediction confidence: {confidence}%).")
        if actual:
            factors_str = " and ".join(f["feature"].replace("_", " ") for f in actual)
            return f"Risk is primarily driven by {factors_str}.{imputed_note}"
        return f"Prediction based on available mandatory data.{imputed_note}"

    def _run_prediction(self, input_dict):
        is_valid, missing_mandatory, missing_optional = self.validate_input(input_dict)
        if not is_valid:
            return {
                "error":             "Missing mandatory fields",
                "disease":           self.disease_name,
                "missing_mandatory": missing_mandatory,
                "mandatory_fields":  self.MANDATORY_FIELDS,
                "optional_fields":   self.OPTIONAL_FIELDS,
                "message": (
                    f"Cannot predict {self.disease_name} risk without: "
                    + ", ".join(missing_mandatory) + ". These are the minimum "
                    "required lab values for this disease."
                )
            }

        raw_array, imputed_fields = self._build_feature_array(input_dict)
        confidence = self._compute_confidence(missing_optional)

        norm_array  = self.normalizer.transform(raw_array.reshape(1, -1))[0]
        probability = float(self.model.predict_proba(norm_array.reshape(1, -1))[0])

        probability, alerts = self._safety_override(input_dict, probability)

        risk_level, risk_emoji = self._get_risk_level(probability)
        contributions = self._compute_contributions(norm_array, imputed_fields)
        explanation   = self._generate_explanation(contributions, imputed_fields, confidence)

        return {
            "disease":          self.disease_name,
            "risk_probability": round(probability, 4),
            "risk_level":       risk_level,
            "risk_emoji":       risk_emoji,
            "fields_provided":   sorted(set(self.feature_names) - set(imputed_fields)),
            "fields_imputed":    imputed_fields,
            "fields_missing_optional": missing_optional,
            "prediction_confidence": confidence,
            "top_factors": [
                {"feature": c["feature"], "contribution": c["importance_pct"],
                 "imputed": c["imputed"]}
                for c in contributions[:3]
            ],
            "all_factors":   contributions,
            "explanation":   explanation,
            "alerts":          alerts,
            "disclaimer": ("This system is for awareness only and is not a replacement "
                           "for professional medical advice."),
        }

    def predict(self, input_dict):
        raise NotImplementedError

    def _safety_override(self, input_dict, probability):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# ANEMIA  — Mandatory: hemoglobin | Optional: rbc, mcv, mch, hematocrit, ferritin
# ─────────────────────────────────────────────────────────────────────────────

class AnemiaModel(DiseaseModel):
    MANDATORY_FIELDS = ["hemoglobin"]
    OPTIONAL_FIELDS  = ["rbc", "mcv", "mch", "hematocrit", "ferritin"]

    def __init__(self):
        super().__init__("Anemia", learning_rate=0.05, n_iterations=3000, l2_lambda=0.005)

    def train_on_generated_data(self):
        X, y, feature_names = MedicalDataGenerator.generate_anemia_dataset(n_samples=2000)
        return self.train(X, y, feature_names)

    def _safety_override(self, input_dict, probability):
        alerts, forced_high = [], False
        hgb  = input_dict.get("hemoglobin")
        rbc  = input_dict.get("rbc")
        hct  = input_dict.get("hematocrit")
        ferr = input_dict.get("ferritin")

        if hgb < 7:
            alerts.append("🔴 CRITICAL: Hemoglobin < 7 g/dL — Severe anemia, transfusion may be needed")
            forced_high = True
        elif hgb < 9:
            alerts.append("🔴 CRITICAL: Hemoglobin < 9 g/dL — Moderate-to-severe anemia")
            forced_high = True
        elif hgb < 12:
            alerts.append("⚠️ Hemoglobin below normal range (< 12 g/dL)")

        if rbc is not None:
            if rbc < 3.5:
                alerts.append("🔴 CRITICAL: RBC count critically low (< 3.5 M/uL)")
                forced_high = True
            elif rbc < 4.2:
                alerts.append("⚠️ RBC count below normal range")

        if hct is not None:
            if hct < 30:
                alerts.append("🔴 CRITICAL: Hematocrit critically low (< 30%)")
                forced_high = True
            elif hct < 36:
                alerts.append("⚠️ Hematocrit below normal range")

        if ferr is not None and ferr < 10:
            alerts.append("🔴 LOW FERRITIN: Iron stores depleted (< 10 ng/mL)")

        if forced_high:
            probability = max(probability, 0.85)
        return probability, alerts

    def predict(self, input_dict):
        return self._run_prediction(input_dict)


# ─────────────────────────────────────────────────────────────────────────────
# DIABETES  — Mandatory: glucose | Optional: hba1c, bmi, age, insulin, blood_pressure
# ─────────────────────────────────────────────────────────────────────────────

class DiabetesModel(DiseaseModel):
    MANDATORY_FIELDS = ["glucose"]
    OPTIONAL_FIELDS  = ["hba1c", "bmi", "age", "insulin", "blood_pressure"]

    def __init__(self):
        # Pima Indians is a moderately hard dataset — lower LR + more iterations + L2
        super().__init__("Diabetes", learning_rate=0.03, n_iterations=5000, l2_lambda=0.01)

    def train_on_generated_data(self):
        X, y, feature_names = MedicalDataGenerator.generate_diabetes_dataset(n_samples=2000)
        return self.train(X, y, feature_names)

    def _safety_override(self, input_dict, probability):
        alerts, forced_high = [], False
        glucose = input_dict.get("glucose")
        hba1c   = input_dict.get("hba1c")
        bmi     = input_dict.get("bmi")

        if glucose > 300:
            alerts.append("🔴 CRITICAL: Blood glucose > 300 mg/dL — Possible hyperglycemic crisis")
            forced_high = True
        elif glucose > 200:
            alerts.append("🔴 CRITICAL: Blood glucose > 200 mg/dL — Diabetic range (symptomatic)")
            forced_high = True
        elif glucose > 126:
            alerts.append("⚠️ Fasting glucose above diabetic threshold (> 126 mg/dL)")
        elif glucose > 100:
            alerts.append("⚠️ Pre-diabetic fasting glucose range (100–125 mg/dL)")

        if hba1c is not None:
            if hba1c > 10:
                alerts.append("🔴 CRITICAL: HbA1c > 10% — Very poor long-term glucose control")
                forced_high = True
            elif hba1c > 8:
                alerts.append("🔴 HbA1c > 8% — Poor long-term control")
                forced_high = True
            elif hba1c > 6.5:
                alerts.append("⚠️ HbA1c in diabetic range (> 6.5%)")
            elif hba1c > 5.7:
                alerts.append("⚠️ HbA1c in pre-diabetic range (5.7–6.4%)")

        if bmi is not None and bmi > 35:
            alerts.append("⚠️ BMI > 35 — Obesity significantly increases diabetes risk")

        if forced_high:
            probability = max(probability, 0.85)
        return probability, alerts

    def predict(self, input_dict):
        return self._run_prediction(input_dict)


# ─────────────────────────────────────────────────────────────────────────────
# INFECTION  — Mandatory: wbc | Optional: neutrophils, lymphocytes, crp, esr, temperature
# ─────────────────────────────────────────────────────────────────────────────

class InfectionModel(DiseaseModel):
    MANDATORY_FIELDS = ["wbc"]
    OPTIONAL_FIELDS  = ["neutrophils", "lymphocytes", "crp", "esr", "temperature"]

    def __init__(self):
        super().__init__("Infection", learning_rate=0.05, n_iterations=3000, l2_lambda=0.005)

    def train_on_generated_data(self):
        X, y, feature_names = MedicalDataGenerator.generate_infection_dataset(n_samples=2000)
        return self.train(X, y, feature_names)

    def _safety_override(self, input_dict, probability):
        alerts, forced_high = [], False
        wbc   = input_dict.get("wbc")
        crp   = input_dict.get("crp")
        temp  = input_dict.get("temperature")
        neut  = input_dict.get("neutrophils")
        lymph = input_dict.get("lymphocytes")

        if wbc > 30000:
            alerts.append("🔴 CRITICAL: WBC > 30,000 — Possible leukemia or severe sepsis")
            forced_high = True
        elif wbc > 20000:
            alerts.append("🔴 CRITICAL: WBC > 20,000 — Severe leukocytosis, possible sepsis")
            forced_high = True
        elif wbc > 12000:
            alerts.append("⚠️ Elevated WBC (> 12,000) — Possible bacterial infection")
        elif wbc < 2000:
            alerts.append("🔴 CRITICAL: WBC < 2,000 — Severe leukopenia, immune compromise")
            forced_high = True
        elif wbc < 4000:
            alerts.append("⚠️ Low WBC (< 4,000) — Possible viral infection or bone marrow issue")

        if crp is not None:
            if crp > 200:
                alerts.append("🔴 CRITICAL: CRP > 200 mg/L — Severe systemic infection/sepsis")
                forced_high = True
            elif crp > 100:
                alerts.append("🔴 CRP > 100 mg/L — Severe inflammation")
                forced_high = True
            elif crp > 10:
                alerts.append("⚠️ Elevated CRP (> 10 mg/L) — Active inflammation detected")

        if temp is not None:
            if temp > 40:
                alerts.append("🔴 CRITICAL: Fever > 40°C — Hyperpyrexia, seek emergency care")
                forced_high = True
            elif temp > 39.5:
                alerts.append("🔴 High fever > 39.5°C — Possible serious infection")
                forced_high = True
            elif temp > 38:
                alerts.append("⚠️ Fever detected (> 38°C)")
            elif temp < 36:
                alerts.append("⚠️ Low temperature (< 36°C) — Possible hypothermia or sepsis")

        if neut is not None and neut > 80:
            alerts.append("⚠️ Elevated neutrophils (> 80%) — Bacterial infection suspected")
        if lymph is not None and lymph < 10:
            alerts.append("⚠️ Low lymphocytes (< 10%) — Possible viral infection or immune stress")

        if forced_high:
            probability = max(probability, 0.85)
        return probability, alerts

    def predict(self, input_dict):
        return self._run_prediction(input_dict)


# ─────────────────────────────────────────────────────────────────────────────
# CHOLESTEROL  — Mandatory: total_cholesterol | Optional: ldl, hdl, triglycerides, vldl, cholesterol_ratio
# ─────────────────────────────────────────────────────────────────────────────

class CholesterolModel(DiseaseModel):
    MANDATORY_FIELDS = ["total_cholesterol"]
    OPTIONAL_FIELDS  = ["ldl", "hdl", "triglycerides", "vldl", "cholesterol_ratio"]

    def __init__(self):
        super().__init__("Cholesterol", learning_rate=0.04, n_iterations=4000, l2_lambda=0.01)

    def train_on_generated_data(self):
        X, y, feature_names = MedicalDataGenerator.generate_cholesterol_dataset(n_samples=2000)
        return self.train(X, y, feature_names)

    def _safety_override(self, input_dict, probability):
        alerts, forced_high = [], False
        tc   = input_dict.get("total_cholesterol")
        ldl  = input_dict.get("ldl")
        hdl  = input_dict.get("hdl")
        trig = input_dict.get("triglycerides")

        if tc > 320:
            alerts.append("🔴 CRITICAL: Total cholesterol > 320 mg/dL — Very high cardiovascular risk")
            forced_high = True
        elif tc > 280:
            alerts.append("🔴 Total cholesterol > 280 mg/dL — High risk, medical review needed")
            forced_high = True
        elif tc > 240:
            alerts.append("⚠️ Total cholesterol > 240 mg/dL — Borderline-high range")

        if ldl is not None:
            if ldl > 190:
                alerts.append("🔴 CRITICAL: LDL > 190 mg/dL — Possible familial hypercholesterolemia")
                forced_high = True
            elif ldl > 160:
                alerts.append("⚠️ LDL in high risk range (> 160 mg/dL)")
            elif ldl > 130:
                alerts.append("⚠️ LDL borderline-high (130–159 mg/dL)")

        if hdl is not None:
            if hdl < 35:
                alerts.append("🔴 Very low HDL (< 35 mg/dL) — Significant cardiovascular risk")
            elif hdl < 40:
                alerts.append("⚠️ Low HDL (< 40 mg/dL) — Increases heart disease risk")

        if trig is not None:
            if trig > 1000:
                alerts.append("🔴 CRITICAL: Triglycerides > 1000 mg/dL — Acute pancreatitis risk")
                forced_high = True
            elif trig > 500:
                alerts.append("🔴 CRITICAL: Triglycerides > 500 mg/dL — Pancreatitis risk")
                forced_high = True
            elif trig > 200:
                alerts.append("⚠️ Elevated triglycerides (> 200 mg/dL)")

        if forced_high:
            probability = max(probability, 0.85)
        return probability, alerts

    def predict(self, input_dict):
        # Alias resolution ("cholesterol" -> "total_cholesterol") now happens
        # upstream in validation/preprocessing.py, before this model ever sees
        # the input, so no local alias handling is needed here.
        return self._run_prediction(input_dict)
