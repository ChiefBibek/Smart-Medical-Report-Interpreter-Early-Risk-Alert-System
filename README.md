# AI-Based Smart Medical Report Interpreter & Early Risk Alert System

## Project Structure

```
medical_risk_system/
├── ml_model.py              ← Logistic Regression FROM SCRATCH (NumPy only)
├── preprocessing.py         ← Min-max normalization + reference ranges
├── risk_engine.py           ← Risk categorization + safety overrides
├── explainability.py        ← SHAP-like feature contributions (no library)
├── alerts_recommendations.py ← Rule-based alerts + recommendation engine
├── ocr_extractor.py         ← PDF/image OCR + regex lab value extraction
├── train_model.py           ← Synthetic data generation + model training
├── pipeline.py              ← End-to-end pipeline (all modules connected)
├── app.py                   ← Streamlit dashboard
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# Also install Tesseract OCR binary:
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS:         brew install tesseract
# Windows:       https://github.com/UB-Mannheim/tesseract/wiki
```

## Run

```bash
# Step 1: Train the model
python train_model.py

# Step 2: Run end-to-end demo
python pipeline.py

# Step 3: Launch Streamlit dashboard
streamlit run app.py
```

---

## Viva Q&A Preparation

### Q: Why not use ChatGPT/LLMs for the core prediction?
**A:** LLMs are not reproducible, not explainable at the parameter level, and cannot provide
deterministic risk probabilities. Our logistic regression gives:
- A precise probability score (0–1)
- Per-feature weights that explain exactly why a prediction was made
- Reproducible results — same input always gives same output
- No hallucination risk

### Q: Why use ML instead of pure rules?
**A:** Rule-based systems only catch known patterns (e.g., hemoglobin < 9).
ML captures combined, non-linear interactions between features.
Example: slightly low hemoglobin + slightly elevated WBC + slightly high glucose together
may signal risk even though each value alone wouldn't trigger a rule.

### Q: Where is your ML algorithm?
**A:**
- `ml_model.py` — sigmoid, gradient descent, binary cross-entropy, weight updates — all from scratch
- `preprocessing.py` — min-max normalization from scratch
- `explainability.py` — SHAP-like contributions: contribution_i = normalized_value_i × weight_i

### Q: What if the prediction is wrong?
**A:** The system is designed to fail safely:
1. **Safety layer** (`risk_engine.py`): critical rule-based overrides force High Risk regardless of ML
2. **Disclaimer**: prominently shown — "not a replacement for medical advice"
3. **Borderline category**: uncertainty is explicitly communicated
4. **False positive preference**: thresholds set to catch more cases than miss them
5. **Recommendations always advise** consulting a doctor

### Q: How does explainability work?
**A:** For logistic regression: `z = Σ(weight_i × feature_i) + bias`
Each term (weight_i × feature_i) is that feature's contribution to z.
We normalize these contributions to percentages and display a ranked bar chart.
This is mathematically equivalent to SHAP linear explanations — no library required.

### Q: How was the model trained?
**A:** On synthetically generated data with medically plausible ranges.
For deployment, real anonymized patient data from labs would be used with IRB approval.
The synthetic data correctly captures the statistical distributions of each parameter.

### Q: What is your business model?
**A:**
- **Primary (B2B):** Sell to labs/clinics on a per-report or subscription basis
- **Secondary:** Patient-facing mobile app (freemium), hospital API integration

---

## Algorithm Reference (for viva)

### Logistic Regression from Scratch

```
Forward pass:
    z = w1·hemoglobin + w2·glucose + w3·rbc + w4·wbc + w5·platelets + w6·creatinine + b
    p = sigmoid(z) = 1 / (1 + e^-z)

Loss (binary cross-entropy):
    L = -[y·log(p) + (1-y)·log(1-p)]

Gradients:
    dL/dw = X^T · (p - y) / n
    dL/db = mean(p - y)

Update rule:
    w ← w - lr · dL/dw
    b ← b - lr · dL/db
```

### Risk Thresholds
| Probability | Category   | Alert |
|-------------|-----------|-------|
| < 0.20      | Low        | 🟢   |
| 0.20–0.50   | Borderline | ⚠️   |
| 0.50–0.75   | Moderate   | 🟡   |
| ≥ 0.75      | High       | 🔴   |

### Safety Override Rules (Critical)
| Condition              | Action             |
|------------------------|--------------------|
| Hemoglobin < 9 g/dL    | Force → High Risk  |
| Glucose > 200 mg/dL    | Force → High Risk  |
| WBC > 20 K/uL          | Force → High Risk  |
| Platelets < 50 K/uL    | Force → High Risk  |
| Creatinine > 3.0 mg/dL | Force → High Risk  |
