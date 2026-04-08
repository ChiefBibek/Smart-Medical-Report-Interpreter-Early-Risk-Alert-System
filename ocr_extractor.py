"""
OCR Report Extractor
=====================
Extracts lab values from PDF or image medical reports.
Uses pytesseract for OCR + regex pattern matching.

Install: pip install pytesseract pdf2image Pillow
         sudo apt-get install tesseract-ocr  (Linux)
         brew install tesseract              (macOS)
"""

import re
import os


# ─────────────────────────────────────────────
# Regex patterns for lab values
# Covers formats: "Hemoglobin: 8.2 g/dL" or "HGB  8.2"
# ─────────────────────────────────────────────
LAB_PATTERNS = {
    "hemoglobin": [
        r"h[ae]moglobin[:\s]*([0-9]+\.?[0-9]*)",
        r"\bhgb[:\s]*([0-9]+\.?[0-9]*)",
        r"\bhb[:\s]*([0-9]+\.?[0-9]*)",
    ],
    "glucose": [
        r"glucose[:\s]*([0-9]+\.?[0-9]*)",
        r"blood\s+sugar[:\s]*([0-9]+\.?[0-9]*)",
        r"\bfbs[:\s]*([0-9]+\.?[0-9]*)",   # fasting blood sugar
        r"\brbs[:\s]*([0-9]+\.?[0-9]*)",   # random blood sugar
    ],
    "rbc": [
        r"rbc[:\s]*([0-9]+\.?[0-9]*)",
        r"red\s+blood\s+cells?[:\s]*([0-9]+\.?[0-9]*)",
        r"erythrocytes?[:\s]*([0-9]+\.?[0-9]*)",
    ],
    "wbc": [
        r"wbc[:\s]*([0-9]+\.?[0-9]*)",
        r"white\s+blood\s+cells?[:\s]*([0-9]+\.?[0-9]*)",
        r"leukocytes?[:\s]*([0-9]+\.?[0-9]*)",
        r"tlc[:\s]*([0-9]+\.?[0-9]*)",    # total leukocyte count
    ],
    "platelets": [
        r"platelets?[:\s]*([0-9]+\.?[0-9]*)",
        r"\bplt[:\s]*([0-9]+\.?[0-9]*)",
        r"thrombocytes?[:\s]*([0-9]+\.?[0-9]*)",
    ],
    "creatinine": [
        r"creatinine[:\s]*([0-9]+\.?[0-9]*)",
        r"\bcr[e]?a?t?[:\s]*([0-9]+\.?[0-9]*)",
    ],
}


def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image file using Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        # Grayscale + threshold improves OCR accuracy on lab reports
        img = img.convert("L")
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text.lower()
    except ImportError:
        raise ImportError(
            "pytesseract and Pillow are required for image OCR.\n"
            "Install: pip install pytesseract Pillow\n"
            "Also install Tesseract binary: https://tesseract-ocr.github.io/"
        )


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=200)
        all_text = ""
        for page in pages:
            import pytesseract
            all_text += pytesseract.image_to_string(page, config="--psm 6").lower()
        return all_text
    except ImportError:
        raise ImportError(
            "pdf2image and pytesseract required for PDF OCR.\n"
            "Install: pip install pdf2image pytesseract"
        )


def parse_lab_values(text: str) -> dict:
    """
    Extract lab parameter values from raw OCR text using regex.

    Returns dict {feature: float_value} for matched parameters.
    """
    text = text.lower()
    extracted = {}

    for feature, patterns in LAB_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1))
                    extracted[feature] = val
                    break  # stop at first match for this feature
                except ValueError:
                    continue

    return extracted


def extract_from_report(file_path: str) -> dict:
    """
    Main entry point — accepts PDF or image, returns lab values.

    Parameters
    ----------
    file_path : str  — path to .pdf, .jpg, .jpeg, or .png

    Returns
    -------
    dict: { feature_name: float_value, ... }
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
        text = extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    values = parse_lab_values(text)

    if not values:
        raise ValueError(
            "No lab parameters could be extracted from the report. "
            "Ensure the file contains readable text for: "
            "hemoglobin, glucose, RBC, WBC, platelets, creatinine."
        )

    return values


# ─── Fallback: manual entry ──────────────────
def validate_manual_entry(values_dict: dict) -> dict:
    """
    Validate manually entered lab values.
    Raises ValueError for out-of-range physiological values.
    """
    from preprocessing import REFERENCE_RANGES

    validated = {}
    for feature, val in values_dict.items():
        if feature not in REFERENCE_RANGES:
            continue
        r = REFERENCE_RANGES[feature]
        lo, hi = r["global_min"], r["global_max"]
        if not (lo <= float(val) <= hi):
            raise ValueError(
                f"{feature} value {val} is outside physiological range "
                f"({lo}–{hi} {r['unit']}). Please check the value."
            )
        validated[feature] = float(val)
    return validated


if __name__ == "__main__":
    # Demo: simulate OCR output
    sample_ocr_text = """
    COMPLETE BLOOD COUNT (CBC) REPORT
    Patient: John Doe    Date: 2024-01-15
    
    Hemoglobin (HGB):     8.2 g/dL         Low
    RBC Count:            3.1 M/uL          Low
    WBC Count:            11.8 K/uL         High
    Platelets (PLT):      155 K/uL          Normal
    
    BIOCHEMISTRY
    Glucose (FBS):        210 mg/dL         High
    Creatinine:           0.9 mg/dL         Normal
    """

    print("Parsing simulated OCR text...")
    values = parse_lab_values(sample_ocr_text)
    print("Extracted lab values:")
    for k, v in values.items():
        print(f"  {k}: {v}")
