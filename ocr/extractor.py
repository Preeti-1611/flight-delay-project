"""
OCR Ticket Extractor for SkyCast Analytics
-------------------------------------------
Accepts a PIL Image or PDF path, preprocesses with OpenCV,
runs Tesseract OCR, and uses regex patterns to extract
flight-related fields (flight number, IATA codes, dates, times).
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
    import pytesseract
    from pytesseract import Output
except ImportError:
    pytesseract = None

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
# Tesseract OCR runner
# ---------------------------------------------------------------------------
def run_tesseract(processed_img: np.ndarray) -> tuple:
    """
    Run Tesseract on a preprocessed image.
    Returns (raw_text, per_word_data_dict).
    """
    if pytesseract is None:
        raise ImportError("pytesseract is required. Install it with: pip install pytesseract")

    # Get per-word confidence data
    data = pytesseract.image_to_data(
        processed_img, output_type=Output.DICT, config="--psm 6"
    )

    # Build raw text from confident words
    raw_text = pytesseract.image_to_string(processed_img, config="--psm 6")

    return raw_text, data


def _avg_confidence(data: dict) -> float:
    """Calculate average Tesseract word confidence (0-1 scale)."""
    confs = [int(c) for c in data.get("conf", []) if int(c) > 0]
    if not confs:
        return 0.0
    return sum(confs) / len(confs) / 100.0


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
    # Look for 3-letter uppercase sequences
    candidates = re.findall(r'\b([A-Z]{3})\b', text.upper())
    # Filter to known IATA codes, or keep unknowns with lower confidence
    results = []
    for c in candidates:
        if c in INDIAN_IATA_CODES:
            results.append({"value": c, "confidence": 0.95})
        else:
            # Could still be a valid international IATA code
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
        # It's a file path — assume PDF
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

    # --- Step 3: Run OCR ---
    raw_text, word_data = run_tesseract(processed)
    overall_conf = _avg_confidence(word_data)

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
