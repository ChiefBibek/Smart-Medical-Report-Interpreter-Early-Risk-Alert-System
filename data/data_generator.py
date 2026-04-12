"""
Dataset Generation & Preprocessing
Generates realistic synthetic + real-world-like medical data for each disease.
"""

import numpy as np


class MinMaxNormalizer:
    """Min-Max Normalization from scratch."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        range_ = self.max_ - self.min_
        range_[range_ == 0] = 1  # Avoid division by zero
        return (X - self.min_) / range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MedicalDataGenerator:
    """
    Generates synthetic + realistic medical datasets for each disease.
    Based on clinical reference ranges from medical literature.
    """

    @staticmethod
    def generate_anemia_dataset(n_samples=1000, random_state=42):
        """
        Anemia Detection Dataset
        Features: hemoglobin, rbc, mcv, mch, hematocrit, ferritin
        
        Clinical Reference Ranges:
        - Hemoglobin: Normal 12-17 g/dL | Anemic < 12 (women) / < 13 (men)
        - RBC: Normal 4.2-5.9 M/uL | Low < 4.2
        - MCV: Normal 80-100 fL | Low < 80 (microcytic)
        - MCH: Normal 27-33 pg | Low < 27
        - Hematocrit: Normal 36-50% | Low < 36
        - Ferritin: Normal 12-300 ng/mL | Low < 12
        """
        np.random.seed(random_state)
        n_pos = n_samples // 2  # Balanced classes

        # Negative (Healthy)
        healthy_hemoglobin = np.random.normal(14.0, 1.2, n_pos)
        healthy_rbc = np.random.normal(4.8, 0.4, n_pos)
        healthy_mcv = np.random.normal(90, 5, n_pos)
        healthy_mch = np.random.normal(30, 2, n_pos)
        healthy_hematocrit = np.random.normal(43, 3, n_pos)
        healthy_ferritin = np.random.normal(100, 40, n_pos)
        healthy_labels = np.zeros(n_pos)

        # Positive (Anemia)
        anemic_hemoglobin = np.random.normal(9.5, 1.5, n_pos)
        anemic_rbc = np.random.normal(3.5, 0.5, n_pos)
        anemic_mcv = np.random.normal(72, 8, n_pos)
        anemic_mch = np.random.normal(22, 3, n_pos)
        anemic_hematocrit = np.random.normal(30, 4, n_pos)
        anemic_ferritin = np.random.normal(8, 4, n_pos)
        anemic_labels = np.ones(n_pos)

        # Combine
        hemoglobin = np.concatenate([healthy_hemoglobin, anemic_hemoglobin])
        rbc = np.concatenate([healthy_rbc, anemic_rbc])
        mcv = np.concatenate([healthy_mcv, anemic_mcv])
        mch = np.concatenate([healthy_mch, anemic_mch])
        hematocrit = np.concatenate([healthy_hematocrit, anemic_hematocrit])
        ferritin = np.concatenate([healthy_ferritin, anemic_ferritin])
        labels = np.concatenate([healthy_labels, anemic_labels])

        X = np.column_stack([hemoglobin, rbc, mcv, mch, hematocrit, ferritin])
        feature_names = ["hemoglobin", "rbc", "mcv", "mch", "hematocrit", "ferritin"]

        # Shuffle
        idx = np.random.permutation(len(labels))
        return X[idx], labels[idx], feature_names

    @staticmethod
    def generate_diabetes_dataset(n_samples=1000, random_state=42):
        """
        Diabetes Risk Dataset
        Features: glucose, hba1c, bmi, age, insulin, blood_pressure
        
        Clinical Reference:
        - Glucose: Normal < 100 mg/dL | Pre-diabetic 100-125 | Diabetic > 126
        - HbA1c: Normal < 5.7% | Pre-diabetic 5.7-6.4 | Diabetic > 6.5
        - BMI: Normal 18.5-24.9 | Overweight 25-29.9 | Obese > 30
        """
        np.random.seed(random_state)
        n_pos = n_samples // 2

        # Healthy
        h_glucose = np.random.normal(85, 10, n_pos)
        h_hba1c = np.random.normal(5.2, 0.3, n_pos)
        h_bmi = np.random.normal(23, 2.5, n_pos)
        h_age = np.random.normal(35, 10, n_pos)
        h_insulin = np.random.normal(12, 4, n_pos)
        h_bp = np.random.normal(75, 8, n_pos)
        h_labels = np.zeros(n_pos)

        # Diabetic
        d_glucose = np.random.normal(160, 30, n_pos)
        d_hba1c = np.random.normal(7.5, 1.0, n_pos)
        d_bmi = np.random.normal(31, 4, n_pos)
        d_age = np.random.normal(52, 12, n_pos)
        d_insulin = np.random.normal(180, 60, n_pos)
        d_bp = np.random.normal(90, 12, n_pos)
        d_labels = np.ones(n_pos)

        glucose = np.concatenate([h_glucose, d_glucose])
        hba1c = np.concatenate([h_hba1c, d_hba1c])
        bmi = np.concatenate([h_bmi, d_bmi])
        age = np.concatenate([h_age, d_age])
        insulin = np.concatenate([h_insulin, d_insulin])
        bp = np.concatenate([h_bp, d_bp])
        labels = np.concatenate([h_labels, d_labels])

        X = np.column_stack([glucose, hba1c, bmi, age, insulin, bp])
        feature_names = ["glucose", "hba1c", "bmi", "age", "insulin", "blood_pressure"]

        idx = np.random.permutation(len(labels))
        return X[idx], labels[idx], feature_names

    @staticmethod
    def generate_infection_dataset(n_samples=1000, random_state=42):
        """
        Infection Risk Dataset
        Features: wbc, neutrophils, lymphocytes, crp, esr, temperature
        
        Clinical Reference:
        - WBC: Normal 4,000-11,000 | Infection > 11,000
        - Neutrophils: Normal 40-70% | Elevated > 75%
        - CRP: Normal < 10 mg/L | Elevated > 10 (infection marker)
        - ESR: Normal < 20 mm/hr | Elevated > 20
        - Temperature: Normal 36.5-37.5°C | Fever > 38
        """
        np.random.seed(random_state)
        n_pos = n_samples // 2

        # Healthy
        h_wbc = np.random.normal(7000, 1500, n_pos)
        h_neut = np.random.normal(60, 5, n_pos)
        h_lymph = np.random.normal(30, 5, n_pos)
        h_crp = np.random.normal(3, 2, n_pos)
        h_esr = np.random.normal(10, 5, n_pos)
        h_temp = np.random.normal(37.0, 0.3, n_pos)
        h_labels = np.zeros(n_pos)

        # Infected
        i_wbc = np.random.normal(15000, 3000, n_pos)
        i_neut = np.random.normal(80, 6, n_pos)
        i_lymph = np.random.normal(15, 5, n_pos)
        i_crp = np.random.normal(60, 20, n_pos)
        i_esr = np.random.normal(45, 15, n_pos)
        i_temp = np.random.normal(38.8, 0.5, n_pos)
        i_labels = np.ones(n_pos)

        wbc = np.concatenate([h_wbc, i_wbc])
        neut = np.concatenate([h_neut, i_neut])
        lymph = np.concatenate([h_lymph, i_lymph])
        crp = np.concatenate([h_crp, i_crp])
        esr = np.concatenate([h_esr, i_esr])
        temp = np.concatenate([h_temp, i_temp])
        labels = np.concatenate([h_labels, i_labels])

        X = np.column_stack([wbc, neut, lymph, crp, esr, temp])
        feature_names = ["wbc", "neutrophils", "lymphocytes", "crp", "esr", "temperature"]

        idx = np.random.permutation(len(labels))
        return X[idx], labels[idx], feature_names

    @staticmethod
    def generate_cholesterol_dataset(n_samples=1000, random_state=42):
        """
        Cholesterol Risk Dataset
        Features: total_cholesterol, ldl, hdl, triglycerides, vldl, cholesterol_ratio
        
        Clinical Reference:
        - Total Cholesterol: Desirable < 200 mg/dL | Borderline 200-239 | High > 240
        - LDL: Optimal < 100 | Borderline 130-159 | High > 160
        - HDL: Low risk > 60 | High risk < 40
        - Triglycerides: Normal < 150 | Borderline 150-199 | High > 200
        """
        np.random.seed(random_state)
        n_pos = n_samples // 2

        # Healthy
        h_tc = np.random.normal(175, 20, n_pos)
        h_ldl = np.random.normal(90, 15, n_pos)
        h_hdl = np.random.normal(65, 10, n_pos)
        h_trig = np.random.normal(100, 25, n_pos)
        h_vldl = np.random.normal(15, 5, n_pos)
        h_ratio = h_tc / h_hdl
        h_labels = np.zeros(n_pos)

        # High Risk
        r_tc = np.random.normal(265, 25, n_pos)
        r_ldl = np.random.normal(175, 20, n_pos)
        r_hdl = np.random.normal(35, 8, n_pos)
        r_trig = np.random.normal(250, 50, n_pos)
        r_vldl = np.random.normal(45, 10, n_pos)
        r_ratio = r_tc / r_hdl
        r_labels = np.ones(n_pos)

        tc = np.concatenate([h_tc, r_tc])
        ldl = np.concatenate([h_ldl, r_ldl])
        hdl = np.concatenate([h_hdl, r_hdl])
        trig = np.concatenate([h_trig, r_trig])
        vldl = np.concatenate([h_vldl, r_vldl])
        ratio = np.concatenate([h_ratio, r_ratio])
        labels = np.concatenate([h_labels, r_labels])

        X = np.column_stack([tc, ldl, hdl, trig, vldl, ratio])
        feature_names = ["total_cholesterol", "ldl", "hdl", "triglycerides", "vldl", "cholesterol_ratio"]

        idx = np.random.permutation(len(labels))
        return X[idx], labels[idx], feature_names


def train_test_split(X, y, test_ratio=0.2, random_state=42):
    """Simple train-test split from scratch."""
    np.random.seed(random_state)
    n = len(y)
    n_test = int(n * test_ratio)
    idx = np.random.permutation(n)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def compute_metrics(y_true, y_pred, y_prob):
    """Compute accuracy, precision, recall, F1, AUC from scratch."""
    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # Confusion matrix components
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # Simple AUC approximation
    thresholds = np.linspace(0, 1, 100)
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred_t = (y_prob >= t).astype(int)
        tp_t = np.sum((pred_t == 1) & (y_true == 1))
        fp_t = np.sum((pred_t == 1) & (y_true == 0))
        tn_t = np.sum((pred_t == 0) & (y_true == 0))
        fn_t = np.sum((pred_t == 0) & (y_true == 1))
        tpr_list.append(tp_t / (tp_t + fn_t + 1e-10))
        fpr_list.append(fp_t / (fp_t + tn_t + 1e-10))

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    sorted_idx = np.argsort(fpr_arr)
    auc = np.trapezoid(tpr_arr[sorted_idx], fpr_arr[sorted_idx]) if hasattr(np, 'trapezoid') else np.trapz(tpr_arr[sorted_idx], fpr_arr[sorted_idx])

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "auc": round(float(auc), 4),
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
    }
