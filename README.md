# AI-Assisted Clinical Decision Support System
### Final Year Project — Python ML + CDSS System

---

## System Overview

A disease-focused AI system that predicts risk for 4 diseases using Logistic Regression
implemented **FROM SCRATCH** using NumPy only (no sklearn, no pre-built ML libraries for the
core prediction models), wrapped in a full Clinical Decision Support System pipeline:

```
Report Upload + OCR (external .NET service)
        ↓
Medical Validation Engine   (validation/)   — unit normalization, physiological-bounds
        ↓                                      checking, completeness scoring, clinical
        ↓                                      consistency checks, confidence adjustment
Disease Prediction           (models/)      — logistic regression from scratch, unchanged
        ↓
SHAP-style Explainability    (models/)      — linear attribution, unchanged
        ↓
Clinical Decision Support     (recommendation/)  — Sentence-BERT context embedding +
        ↓                                          semantic similarity search
Medical Knowledge Base        (knowledge_base/)  — 24 evidence-tagged clinical scenarios
        ↓
Traceability / Audit Trail    (audit/)      — every decision logged to SQLite
```

**4 Supported Diseases:** Anemia · Diabetes · Infection (Bacterial) · Cholesterol (Cardiovascular)

OCR and PDF/image parameter extraction are **not** part of this repository — they run in a
separate .NET Hospital Management System backend. This service only ever receives already
structured lab values (optionally with a per-field OCR confidence score), and returns risk
prediction, explanation, and a personalized, semantically-retrieved recommendation.

---

## Project Structure

```
medical_ai/
├── predictor.py                ← MAIN ENTRY POINT (run this) — orchestrates the full pipeline
├── models/
│   ├── logistic_regression.py  ← LR from scratch (NumPy only)
│   └── disease_models.py       ← 4 disease models + safety overrides + explainability
├── data/
│   └── data_generator.py       ← Synthetic/real data generation + metrics
├── validation/                 ← Module 1: Medical Validation Engine
│   ├── ranges.py                 physiological bounds (NORMAL/EXTREME/IMPOSSIBLE tiers)
│   ├── units.py                  unit auto-detection + conversion (mg/dL <-> mmol/L, etc.)
│   ├── completeness.py           report completeness score (Excellent/Good/Fair/Poor)
│   ├── consistency.py            cross-parameter clinical consistency rule table
│   ├── confidence.py             confidence adjustment formula
│   └── engine.py                 ValidationEngine orchestrator
├── recommendation/             ← Module 2: Clinical Decision Support Engine
│   ├── context_builder.py        patient clinical context -> natural-language paragraph
│   ├── retrieval_engine.py       Sentence-BERT semantic search over the knowledge base
│   └── embedding_store.py        disk-cached KB embeddings
├── knowledge_base/             ← Module 3: Medical Knowledge Base
│   └── entries/*.json            24 structured, evidence-tagged clinical scenarios
├── audit/                      ← Module 3: Traceability / audit trail
│   └── db.py                     SQLite-backed decision log + doctor-feedback API
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
This pulls in `sentence-transformers` (for the semantic recommendation engine), which in turn
installs PyTorch — expect a multi-minute, several-hundred-MB install. The first `train_all()`
run also downloads the `all-MiniLM-L6-v2` embedding model (~90MB) from the Hugging Face Hub,
which needs one-time outbound internet access (or a pre-warmed `~/.cache/huggingface` for
air-gapped/hospital deployments).

### 2. Run demo
```bash
cd medical_ai
python predictor.py
```

### 3. Run as Flask API (for .NET backend)
```bash
python predictor.py --serve
# → API starts at http://localhost:5000
```

---

## API Endpoints (Flask)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + training status |
| POST | `/predict` | Single patient prediction (full CDSS pipeline) |
| POST | `/predict/batch` | Batch predictions |
| GET | `/metrics` | Training metrics for all models |
| GET | `/decisions/<id>` | Fetch the full logged record for one prediction (Module 3 traceability) |
| POST | `/decisions/<id>/feedback` | Record a doctor's correction/override against a decision |

---

## Input Format (from .NET OCR backend)

```json
{
  "disease": "anemia",
  "hemoglobin": 10.2,
  "rbc": 3.8,
  "mcv": 74,
  "mch": 23,
  "hematocrit": 31,
  "ferritin": 7
}
```

**Supported field names per disease:**

| Disease | Fields |
|---------|--------|
| anemia | hemoglobin, rbc, mcv, mch, hematocrit, ferritin |
| diabetes | glucose, hba1c, bmi, age, insulin, blood_pressure |
| infection | wbc, neutrophils, lymphocytes, crp, esr, temperature |
| cholesterol | total_cholesterol (or cholesterol), ldl, hdl, triglycerides, vldl, cholesterol_ratio |

**Optional sidecar fields** (any lab field name `<field>` may have these siblings):

| Field | Purpose |
|-------|---------|
| `<field>_unit` | Explicit unit declaration (e.g. `"glucose_unit": "mmol/L"`) — always trusted over auto-detection |
| `<field>_ocr_confidence` | Per-field OCR confidence, 0-100, from the upstream OCR service — feeds the confidence adjustment |
| `<field>_confirmed` | `true` to bypass a physiologically-impossible-value block for a genuine rare-extreme case |
| `patient_history` | Optional `{"conditions": [...]}` object, folded into the recommendation context |

---

## Output Format (to .NET backend)

All fields below are unchanged from the original API and remain byte-compatible. New fields
(`validation`, `recommendation_detail`, `decision_id`) are strictly additive; `recommendations`
keeps its original `list[str]` shape but its content is now retrieved from the knowledge base
instead of a hardcoded per-disease list.

```json
{
  "disease": "Anemia",
  "risk_probability": 0.89,
  "risk_level": "High",
  "risk_emoji": "🔴",
  "top_factors": [
    {"feature": "hematocrit", "contribution": 24.54},
    {"feature": "mch", "contribution": 22.08},
    {"feature": "hemoglobin", "contribution": 18.33}
  ],
  "explanation": "Risk is primarily influenced by hematocrit and mch.",
  "alerts": [
    "🔴 CRITICAL: Hemoglobin critically low (< 9 g/dL) — Severe anemia",
    "🔴 CRITICAL: RBC count critically low (< 3.5 M/uL)"
  ],
  "recommendations": [
    "Initiate iron repletion and investigate the underlying cause of iron loss...",
    "Diet: Increase iron-rich foods and pair with Vitamin C...",
    "Lifestyle: Take iron supplements on an empty stomach if tolerated..."
  ],
  "recommendation_detail": {
    "recommendation": "Initiate iron repletion and investigate the underlying cause of iron loss...",
    "diet_advice": "Increase iron-rich foods and pair with Vitamin C...",
    "lifestyle_advice": "Take iron supplements on an empty stomach if tolerated...",
    "suggested_follow_up_tests": ["Serum iron and TIBC / transferrin saturation", "Peripheral blood smear"],
    "urgency_level": "Soon",
    "evidence_tags": ["Kassebaum NJ et al., Hematol Oncol Clin North Am 2016"],
    "primary_entry_id": "anemia_iron_deficiency_low_ferritin",
    "similarity_score": 0.62,
    "match_confidence": "high",
    "is_fallback": false
  },
  "validation": {
    "completeness_score": 100.0,
    "completeness_label": "Excellent",
    "unit_conversions_applied": [],
    "consistency_warnings": [],
    "blocking_errors": [],
    "suggested_tests": [],
    "adjusted_confidence": 100.0
  },
  "disclaimer": "This system is for awareness only and not a replacement for professional medical advice.",
  "model_metrics": {
    "accuracy": 1.0,
    "f1_score": 1.0,
    "auc": 1.0
  },
  "decision_id": 42
}
```

A physiologically impossible value (likely an OCR misread) short-circuits before the model
ever runs, returning a superset of the existing "missing mandatory field" error shape:

```json
{
  "error": "Invalid field value(s)",
  "error_type": "validation_blocking",
  "disease": "diabetes",
  "missing_mandatory": [],
  "mandatory_fields": ["glucose"],
  "optional_fields": ["hba1c", "bmi", "age", "insulin", "blood_pressure"],
  "invalid_fields": ["glucose"],
  "blocking_errors": [{"field": "glucose", "value": -50, "tier": "impossible", "blocking": true,
                        "message": "glucose value -50 is physiologically impossible (must be between 10 and 1500)."}],
  "message": "Cannot predict diabetes risk: glucose value -50 is physiologically impossible (must be between 10 and 1500)."
}
```

---

## Risk Categorization

| Probability | Level | Emoji |
|-------------|-------|-------|
| p < 0.20 | Low Risk | 🟢 |
| 0.20 – 0.50 | Borderline | ⚠️ |
| 0.50 – 0.75 | Moderate | 🟡 |
| p > 0.75 | High | 🔴 |

---

## ML Implementation (Logistic Regression FROM SCRATCH)

### Core Math
```
z = w1*x1 + w2*x2 + ... + wn*xn + b     (linear combination)
p = 1 / (1 + e^(-z))                     (sigmoid activation)
L = -[y*log(p) + (1-y)*log(1-p)]         (binary cross-entropy loss)
w = w - lr * (dL/dw)                     (gradient descent)
```

### Features per disease
- Separate feature sets per disease
- Min-Max normalization (from scratch)
- 80/20 train/test split (from scratch)
- Metrics: Accuracy, Precision, Recall, F1, AUC (all from scratch)

---

## Safety Layer (Critical Overrides)

Each disease has rule-based overrides. Example:
- Hemoglobin < 9 → Force HIGH risk (anemia)
- Glucose > 200 → Force HIGH risk (diabetes)
- WBC > 20,000 → Force HIGH risk (infection)
- Total Cholesterol > 280 → Force HIGH risk (cholesterol)

**Philosophy: Prefer false positives. Safety first.**

---

## Explainability

Feature importance computed as:
```
contribution = feature_value × weight
importance = |contribution| / sum(|contributions|) × 100
```

Returns human-readable explanation of top contributing factors.

---

## Training Results (on 2000 synthetic samples per disease)

| Disease | Accuracy | F1 Score | AUC |
|---------|----------|----------|-----|
| Anemia | 100.00% | 100.00% | 1.000 |
| Diabetes | 100.00% | 100.00% | 0.998 |
| Infection | 100.00% | 100.00% | 0.999 |
| Cholesterol | 100.00% | 100.00% | 0.999 |

---

## Viva Preparation Summary

**Why logistic regression?**
- Interpretable and medically acceptable
- Works well for binary classification with small-medium datasets
- Weights directly show feature importance

**Why synthetic data?**
- Real medical data is limited, imbalanced, and privacy-restricted
- Synthetic data generated from clinical reference ranges
- Allows controlled class balance and edge case coverage

**Why separate models?**
- Different diseases use completely different biomarkers
- Separate models = better accuracy + cleaner explainability
- Modular design allows independent updates

**Why not deep learning?**
- Not explainable (black box)
- Requires massive datasets
- Not medically acceptable without clinical validation

**How is safety ensured?**
- Rule-based overrides for critical values
- Conservative thresholds (prefer false positives)
- Safety layer runs AFTER ML prediction and can override it

---

## Integration with .NET Backend

```python
# In your Flask app (predictor.py --serve)
# POST to http://localhost:5000/predict
# with JSON body containing disease + lab values
# Returns structured JSON response

# .NET can call this with HttpClient:
# var response = await httpClient.PostAsJsonAsync("http://localhost:5000/predict", labData);
```

---

*This system is for academic and awareness purposes only.*
*Not a replacement for professional medical diagnosis.*
