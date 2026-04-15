"""
OCR Ticket Extractor for SkyCast Analytics
-------------------------------------------
Accepts a PIL Image or PDF path, preprocesses with OpenCV,
runs EasyOCR (pure Python, no Tesseract binary needed) to extract text,
and uses regex patterns to extract flight-related fields.
Returns a dict of {field_name: {"value": str, "confidence": float, "low_confidence": bool}}.
"""

import re
import os
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import imutils
except ImportError:
    imutils = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# ---------------------------------------------------------------------------
# IATA code reference for validation
# ---------------------------------------------------------------------------
INDIAN_IATA_CODES = {
    "DEL", "BOM", "BLR", "MAA", "CCU", "HYD", "COK", "GOI", "AMD",
    "PNQ", "JAI", "LKO", "PAT", "GAU", "IXC", "SXR", "ATQ", "VNS",
    "IXB", "BBI", "IDR", "IXR", "RPR", "VTZ", "TRV", "NAG", "IMF",
    "IXA", "IXZ", "DED", "IXM", "UDR", "JDH", "BDQ", "RAJ", "DIB",
    "IXJ", "IXE", "CNN", "CJB", "TRZ", "DMU", "AJL", "SHL", "IXS",
}

INDIAN_AIRLINES_MAP = {
    "AI": "Air India", "6E": "IndiGo", "SG": "SpiceJet",
    "UK": "Vistara", "G8": "Go First", "QP": "Akasa Air",
    "IX": "Air India Express", "I5": "AirAsia India",
}

CONFIDENCE_THRESHOLD = 0.75

# Lazy-loaded EasyOCR reader (loaded once, reused)
_easyocr_reader = None


def _get_easyocr_reader():
    """Lazy-load the EasyOCR reader so it doesn't slow down import time."""
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader


# ---------------------------------------------------------------------------
# Image preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Converts a PIL Image into a clean, OCR-ready grayscale numpy array.
    Steps: grayscale → denoise → adaptive threshold → deskew.
    """
    if cv2 is None:
        raise ImportError("opencv-python-headless is required. Install it with: pip install opencv-python-headless")

    # Convert PIL → OpenCV BGR
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 3. Adaptive thresholding for better contrast
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 4. Deskew using imutils (if available)
    if imutils is not None:
        try:
            coords = np.column_stack(np.where(gray > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:
                gray = imutils.rotate_bound(gray, angle)
        except Exception:
            pass  # Skip deskew if it fails on a particular image

    return gray


# ---------------------------------------------------------------------------
# PDF → Image conversion
# ---------------------------------------------------------------------------
def pdf_to_image(pdf_path: str) -> Image.Image:
    """Convert the first page of a PDF to a PIL Image."""
    if convert_from_path is None:
        raise ImportError("pdf2image is required for PDF support. Install it with: pip install pdf2image")
    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
    if not pages:
        raise ValueError("Could not extract any page from the PDF.")
    return pages[0]


# ---------------------------------------------------------------------------
# OCR runners (EasyOCR primary, Tesseract fallback)
# ---------------------------------------------------------------------------
def run_easyocr(processed_img: np.ndarray) -> tuple:
    """
    Run EasyOCR on a preprocessed image.
    Returns (raw_text, avg_confidence).
    No external binary needed — EasyOCR is pure Python + PyTorch.
    """
    reader = _get_easyocr_reader()
    results = reader.readtext(processed_img)

    # results is a list of (bbox, text, confidence)
    texts = []
    confidences = []
    for (bbox, text, conf) in results:
        texts.append(text)
        confidences.append(conf)

    raw_text = " ".join(texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return raw_text, avg_conf


def run_tesseract(processed_img: np.ndarray) -> tuple:
    """
    Fallback: Run Tesseract on a preprocessed image.
    Requires Tesseract binary installed on the system.
    Returns (raw_text, avg_confidence).
    """
    if not TESSERACT_AVAILABLE:
        raise ImportError("pytesseract is not installed.")

    # Get per-word confidence data
    data = pytesseract.image_to_data(
        processed_img, output_type=Output.DICT, config="--psm 6"
    )

    raw_text = pytesseract.image_to_string(processed_img, config="--psm 6")

    # Average confidence
    confs = [int(c) for c in data.get("conf", []) if int(c) > 0]
    avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0

    return raw_text, avg_conf


def run_ocr(processed_img: np.ndarray) -> tuple:
    """
    Smart OCR runner: tries EasyOCR first (no binary needed),
    falls back to Tesseract if EasyOCR is unavailable.
    """
    if EASYOCR_AVAILABLE:
        return run_easyocr(processed_img)
    elif TESSERACT_AVAILABLE:
        return run_tesseract(processed_img)
    else:
        raise ImportError(
            "No OCR engine found! Install one:\n"
            "  • pip install easyocr          (recommended, no external binary)\n"
            "  • pip install pytesseract       (requires Tesseract binary on system)"
        )


# ---------------------------------------------------------------------------
# Regex field extractors
# ---------------------------------------------------------------------------
def _extract_flight_number(text: str) -> dict:
    """Extract flight number like AI302, 6E1234, UK835."""
    patterns = [
        r'\b([A-Z]{2})\s*(\d{1,4})\b',         # AI 302, 6E1234
        r'\b(\d[A-Z])\s*(\d{1,4})\b',           # 6E 1234
    ]
    for pattern in patterns:
        match = re.search(pattern, text.upper())
        if match:
            code = match.group(1).strip()
            num = match.group(2).strip()
            flight_no = f"{code}{num}"
            # Check if the airline code is a known Indian airline
            conf = 0.95 if code in INDIAN_AIRLINES_MAP else 0.70
            return {"value": flight_no, "confidence": conf}
    return {"value": "", "confidence": 0.0}


def _extract_iata_codes(text: str) -> list:
    """Extract all potential 3-letter IATA codes."""
    candidates = re.findall(r'\b([A-Z]{3})\b', text.upper())
    results = []
    for c in candidates:
        if c in INDIAN_IATA_CODES:
            results.append({"value": c, "confidence": 0.95})
        else:
            results.append({"value": c, "confidence": 0.60})
    return results


def _extract_date(text: str) -> dict:
    """Extract date in formats: DD MMM YYYY, DD-MMM-YYYY, YYYY-MM-DD, DD/MM/YYYY."""
    patterns = [
        (r'\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{4})\b', 0.90),
        (r'\b(\d{1,2})-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)-(\d{4})\b', 0.90),
        (r'\b(\d{4})-(\d{2})-(\d{2})\b', 0.85),
        (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', 0.80),
        (r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', 0.75),
    ]
    text_upper = text.upper()
    for pattern, conf in patterns:
        match = re.search(pattern, text_upper)
        if match:
            return {"value": match.group(0), "confidence": conf}
    return {"value": "", "confidence": 0.0}


def _extract_time(text: str) -> dict:
    """Extract time in HH:MM format (24h or 12h with AM/PM)."""
    patterns = [
        (r'\b(\d{1,2}:\d{2})\s*(AM|PM)\b', 0.90),
        (r'\b(\d{1,2}:\d{2})\b', 0.85),
        (r'\b(\d{4})\s*(?:HRS|hrs|H)\b', 0.80),  # 1430 HRS
    ]
    for pattern, conf in patterns:
        match = re.search(pattern, text.upper())
        if match:
            raw = match.group(0).strip()
            # Normalize "1430 HRS" → "14:30"
            if re.match(r'\d{4}\s*(?:HRS|hrs|H)', raw, re.IGNORECASE):
                digits = re.search(r'(\d{4})', raw).group(1)
                raw = f"{digits[:2]}:{digits[2:]}"
            return {"value": raw, "confidence": conf}
    return {"value": "", "confidence": 0.0}


def _extract_airline_name(text: str, flight_number_result: dict) -> dict:
    """Try to resolve airline name from flight number code or text search."""
    # First try from flight number
    if flight_number_result.get("value"):
        code = flight_number_result["value"][:2]
        if code in INDIAN_AIRLINES_MAP:
            return {"value": INDIAN_AIRLINES_MAP[code], "confidence": 0.95}

    # Search for airline name directly in text
    text_upper = text.upper()
    for code, name in INDIAN_AIRLINES_MAP.items():
        if name.upper() in text_upper:
            return {"value": name, "confidence": 0.90}

    return {"value": "", "confidence": 0.0}


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------
def extract_ticket_fields(source) -> dict:
    """
    Main entry point. Accepts:
      - A PIL Image object (from camera/upload)
      - A string path to a PDF file

    Returns a dict:
    {
        "flight_number": {"value": "AI302", "confidence": 0.95, "low_confidence": False},
        "airline":       {"value": "Air India", "confidence": 0.95, "low_confidence": False},
        "origin":        {"value": "DEL", "confidence": 0.90, "low_confidence": False},
        "destination":   {"value": "BOM", "confidence": 0.85, "low_confidence": False},
        "date":          {"value": "15 APR 2026", "confidence": 0.90, "low_confidence": False},
        "time":          {"value": "14:30", "confidence": 0.85, "low_confidence": False},
        "raw_text":      {"value": "...", "confidence": 1.0,  "low_confidence": False},
        "overall_ocr_confidence": {"value": "0.82", "confidence": 0.82, "low_confidence": False},
    }
    """
    # --- Step 1: Get the PIL image ---
    if isinstance(source, str):
        if source.lower().endswith(".pdf"):
            pil_img = pdf_to_image(source)
        else:
            pil_img = Image.open(source)
    elif isinstance(source, Image.Image):
        pil_img = source
    else:
        raise TypeError(f"Expected PIL Image or file path, got {type(source)}")

    # --- Step 2: Preprocess ---
    processed = preprocess_image(pil_img)

    # --- Step 3: Run OCR (EasyOCR preferred, Tesseract fallback) ---
    raw_text, overall_conf = run_ocr(processed)

    # --- Step 4: Extract fields ---
    flight_no = _extract_flight_number(raw_text)
    airline = _extract_airline_name(raw_text, flight_no)
    iata_codes = _extract_iata_codes(raw_text)
    date = _extract_date(raw_text)
    time_field = _extract_time(raw_text)

    # Assign origin & destination from IATA codes (first two found)
    origin = iata_codes[0] if len(iata_codes) >= 1 else {"value": "", "confidence": 0.0}
    destination = iata_codes[1] if len(iata_codes) >= 2 else {"value": "", "confidence": 0.0}

    # --- Step 5: Build result with low_confidence flags ---
    def _flag(field: dict) -> dict:
        field["low_confidence"] = field["confidence"] < CONFIDENCE_THRESHOLD
        return field

    result = {
        "flight_number": _flag(flight_no),
        "airline": _flag(airline),
        "origin": _flag(origin),
        "destination": _flag(destination),
        "date": _flag(date),
        "time": _flag(time_field),
        "raw_text": {"value": raw_text.strip(), "confidence": 1.0, "low_confidence": False},
        "overall_ocr_confidence": {
            "value": f"{overall_conf:.2f}",
            "confidence": overall_conf,
            "low_confidence": overall_conf < CONFIDENCE_THRESHOLD,
        },
    }

    return result
