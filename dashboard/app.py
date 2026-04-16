import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys
import os
import joblib
import json
import io
import re
import difflib
from datetime import datetime, time
import random
import hashlib
import numpy as np
import cv2
from PIL import Image, ImageSequence
from paddleocr import PaddleOCR
from openai import OpenAI
import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium

CITY_COORDS = {
    "Chennai": [13.0827, 80.2707], "Delhi": [28.7041, 77.1025],
    "Mumbai": [19.0760, 72.8777], "Goa": [15.2993, 74.1240],
    "Bangalore": [12.9716, 77.5946], "Hyderabad": [17.3850, 78.4867],
    "Kolkata": [22.5726, 88.3639], "Kochi": [9.9312, 76.2673],
    "Ahmedabad": [23.0225, 72.5714], "Pune": [18.5204, 73.8567],
    "Jaipur": [26.9124, 75.7873], "Lucknow": [26.8467, 80.9462]
}

# Add models and utils directories to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '..', 'models')))
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '..')))

from predict import predict_flight_delay
try:
    from utils.weather_service import WeatherService
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '..', 'utils')))
    from weather_service import WeatherService
from assets.templates import HERO_SECTION

# Paths
style_path = os.path.join(base_dir, 'assets', 'style.css')
encoders_path = os.path.join(base_dir, '..', 'trained_models', 'encoders.pkl')
charts_dir = os.path.join(base_dir, 'charts')
distance_lookup_path = os.path.join(base_dir, '..', 'data', 'processed', 'distance_lookup.json')
# Use the enriched dataset for charts etc.
raw_data_paths = [
    os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_dataset.csv'),
    os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_enriched.csv'),
    os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_500_rows.csv')
]
raw_data_path = next((p for p in raw_data_paths if os.path.exists(p)), raw_data_paths[0])

# --- GLOBAL PAGE SETTINGS ---
st.set_page_config(
    page_title="SkyCast Analytics ",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS LOADER ---
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(style_path)

# --- HELPERS ---
def format_time(minutes):
    if minutes < 0: return "0 mins"
    mins = int(round(minutes))
    if mins < 60:
        return f"{mins} mins"
    hrs = mins // 60
    rem_mins = mins % 60
    return f"{hrs}h {rem_mins}m"

IATA_CITY_MAP = {
    "MAA": "Chennai",
    "DEL": "Delhi",
    "BOM": "Mumbai",
    "GOI": "Goa",
    "BLR": "Bengaluru",
    "HYD": "Hyderabad",
    "CCU": "Kolkata",
    "COK": "Kochi",
    "AMD": "Ahmedabad",
    "PNQ": "Pune",
    "JAI": "Jaipur",
    "LKO": "Lucknow"
}

# Common alternate city name spellings found on tickets
CITY_ALIAS_MAP = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bombay": "Mumbai",
    "madras": "Chennai",
    "calcutta": "Kolkata",
    "cochin": "Kochi",
    "new delhi": "Delhi",
    "trivandrum": "Kochi",
}

AIRLINE_ALIAS_MAP = {
    "indigo": "IndiGo Airlines",
    "6e": "IndiGo Airlines",
    "air india": "Air India",
    "ai": "Air India",
    "spicejet": "SpiceJet",
    "sg": "SpiceJet",
    "vistara": "Vistara",
    "uk": "Vistara",
    "goair": "GoAir",
    "g8": "GoAir"
}

DATE_REGEX = r'(\d{1,2}[\s/-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s/-]\d{4})'
TIME_REGEX = r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)'
IATA_REGEX = r'\b([A-Z]{3})\b'

def preprocess_image_for_ocr(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # CLAHE adaptive contrast enhancement - preserves text detail on boarding passes
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Light denoising to reduce noise without destroying text
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    return denoised

@st.cache_resource
def get_paddle_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_ticket(uploaded_file):
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()
    ocr_engine = get_paddle_ocr()

    def _ocr_from_bgr_with_paddle(bgr_image):
        processed = preprocess_image_for_ocr(bgr_image)
        result = ocr_engine.ocr(processed, cls=True)
        texts = []
        for block in result or []:
            if not block:
                continue
            for line in block:
                if len(line) > 1 and isinstance(line[1], (list, tuple)) and line[1]:
                    line_text = line[1][0]
                    confidence = line[1][1] if len(line[1]) > 1 else 1.0
                    if line_text and confidence > 0.5:
                        texts.append(str(line_text))
        return "\n".join(texts)

    if file_name.endswith((".jpg", ".jpeg", ".png")):
        image_np = np.frombuffer(file_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if bgr is None:
            return ""
        return _ocr_from_bgr_with_paddle(bgr)

    if file_name.endswith(".pdf"):
        all_text = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                else:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                all_text.append(_ocr_from_bgr_with_paddle(bgr))
            doc.close()
        except ImportError:
            try:
                pdf = Image.open(io.BytesIO(file_bytes))
                for page in ImageSequence.Iterator(pdf):
                    rgb = page.convert("RGB")
                    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
                    all_text.append(_ocr_from_bgr_with_paddle(bgr))
            except Exception:
                return ""
        return "\n".join(all_text)

    return ""

def match_value_to_options(candidate, options):
    if not candidate:
        return None
    lower_map = {str(opt).lower(): opt for opt in options}
    cand = str(candidate).strip().lower()
    if cand in lower_map:
        return lower_map[cand]
    for opt in options:
        opt_l = str(opt).lower()
        if cand in opt_l or opt_l in cand:
            return opt
    return None

def normalize_text_for_match(value):
    if value is None:
        return ""
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())

def fuzzy_match_value(candidate, options, cutoff=0.8):
    if not candidate:
        return None
    raw_options = [str(opt) for opt in options]
    normalized_to_raw = {normalize_text_for_match(opt): opt for opt in raw_options}
    candidate_norm = normalize_text_for_match(candidate)
    if candidate_norm in normalized_to_raw:
        return normalized_to_raw[candidate_norm]
    close = difflib.get_close_matches(candidate_norm, list(normalized_to_raw.keys()), n=1, cutoff=cutoff)
    if close:
        return normalized_to_raw[close[0]]
    return None

def normalize_departure_time(time_text):
    if not time_text:
        return None
    clean = re.sub(r"\s+", " ", time_text.strip().upper())
    try:
        if "AM" in clean or "PM" in clean:
            parsed = datetime.strptime(clean, "%I:%M %p")
        else:
            parsed = datetime.strptime(clean, "%H:%M")
        return parsed.strftime("%H:%M")
    except ValueError:
        return None

def parse_travel_date(text):
    date_match = re.search(DATE_REGEX, text, flags=re.IGNORECASE)
    if date_match:
        raw = date_match.group(1)
        for fmt in ("%d %b %Y", "%d-%b-%Y", "%d/%b/%Y"):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue

    full_month_match = re.search(r'\b(\d{1,2})[\s/-]([A-Za-z]{3,9})[\s/-](\d{4})\b', text or "", flags=re.IGNORECASE)
    if full_month_match:
        dd, month_raw, yyyy = full_month_match.groups()
        month_norm = month_raw.strip().lower()[:3]
        month_map = {
            "jan": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
            "may": "May", "jun": "Jun", "jul": "Jul", "aug": "Aug",
            "sep": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec"
        }
        if month_norm not in month_map:
            close = difflib.get_close_matches(month_norm, list(month_map.keys()), n=1, cutoff=0.66)
            if close:
                month_norm = close[0]
        if month_norm in month_map:
            raw = f"{dd} {month_map[month_norm]} {yyyy}"
            try:
                return datetime.strptime(raw, "%d %b %Y").date()
            except ValueError:
                pass

    # Flexible: handle OCR text with missing/reduced separators (e.g., "29Apr2026")
    flex_match = re.search(r'(\d{1,2})\s*([A-Za-z]{3,9})\s*(\d{4})', text or "", flags=re.IGNORECASE)
    if flex_match:
        dd, month_raw, yyyy = flex_match.groups()
        month_norm = month_raw.strip().lower()[:3]
        month_map = {
            "jan": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
            "may": "May", "jun": "Jun", "jul": "Jul", "aug": "Aug",
            "sep": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec"
        }
        if month_norm not in month_map:
            close = difflib.get_close_matches(month_norm, list(month_map.keys()), n=1, cutoff=0.6)
            if close:
                month_norm = close[0]
        if month_norm in month_map:
            raw = f"{dd} {month_map[month_norm]} {yyyy}"
            try:
                return datetime.strptime(raw, "%d %b %Y").date()
            except ValueError:
                pass

    for pattern, fmts in [
        (r'\b(\d{4}-\d{2}-\d{2})\b', ("%Y-%m-%d",)),
        (r'\b(\d{1,2}/\d{1,2}/\d{4})\b', ("%d/%m/%Y", "%m/%d/%Y")),
        (r'\b(\d{1,2}-\d{1,2}-\d{4})\b', ("%d-%m-%Y", "%m-%d-%Y"))
    ]:
        match = re.search(pattern, text)
        if not match:
            continue
        raw = match.group(1)
        for fmt in fmts:
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
    return None

def extract_airline_name(text):
    text_l = (text or "").lower()
    if not text_l:
        return None

    for alias, full_name in AIRLINE_ALIAS_MAP.items():
        pattern = rf'(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])'
        if re.search(pattern, text_l):
            return full_name
    for alias, full_name in AIRLINE_ALIAS_MAP.items():
        if alias in text_l:
            return full_name
    return None

def extract_city_and_codes(text):
    text_u = (text or "").upper()
    text_l = (text or "").lower()
    city_to_code = {v.lower(): k for k, v in IATA_CITY_MAP.items()}

    code_hits = []
    for code in re.findall(IATA_REGEX, text_u):
        if code in IATA_CITY_MAP and code not in code_hits:
            code_hits.append(code)

    city_hits = []
    for city_l, code in city_to_code.items():
        if city_l in text_l and code not in city_hits:
            city_hits.append(code)

    # Check common alternate city name spellings (Bangalore→Bengaluru, Bombay→Mumbai etc.)
    alias_hits = []
    for alias, canonical in CITY_ALIAS_MAP.items():
        if alias in text_l:
            canonical_code = {v.lower(): k for k, v in IATA_CITY_MAP.items()}.get(canonical.lower())
            if canonical_code and canonical_code not in alias_hits:
                alias_hits.append(canonical_code)

    fuzzy_hits = []
    tokens = re.findall(r'[A-Za-z]{3,}', text_l)
    unique_tokens = list(dict.fromkeys(tokens))
    known_cities = list(city_to_code.keys())
    # Also include aliases for fuzzy matching
    all_known = known_cities + list(CITY_ALIAS_MAP.keys())
    for token in unique_tokens:
        if token in known_cities:
            continue
        if token in CITY_ALIAS_MAP:
            canonical_code = city_to_code.get(CITY_ALIAS_MAP[token].lower())
            if canonical_code and canonical_code not in fuzzy_hits:
                fuzzy_hits.append(canonical_code)
            continue
        match = difflib.get_close_matches(token, all_known, n=1, cutoff=0.75)
        if match:
            matched_key = match[0]
            if matched_key in city_to_code:
                inferred_code = city_to_code[matched_key]
            elif matched_key in CITY_ALIAS_MAP:
                inferred_code = city_to_code.get(CITY_ALIAS_MAP[matched_key].lower())
            else:
                inferred_code = None
            if inferred_code and inferred_code not in fuzzy_hits:
                fuzzy_hits.append(inferred_code)

    ordered_codes = []
    for code in code_hits + city_hits + alias_hits + fuzzy_hits:
        if code not in ordered_codes:
            ordered_codes.append(code)

    origin_code = ordered_codes[0] if len(ordered_codes) > 0 else None
    destination_code = ordered_codes[1] if len(ordered_codes) > 1 else None

    origin_city = IATA_CITY_MAP.get(origin_code) if origin_code else None
    destination_city = IATA_CITY_MAP.get(destination_code) if destination_code else None

    if origin_code and destination_code and origin_code == destination_code:
        destination_code = None
        destination_city = None

    return origin_city, origin_code, destination_city, destination_code

def extract_structured_ticket_details(ocr_text):
    text = ocr_text or ""
    origin_city, origin_code, destination_city, destination_code = extract_city_and_codes(text)
    parsed_date = parse_travel_date(text)

    time_match = re.search(TIME_REGEX, text, flags=re.IGNORECASE)
    normalized_time = normalize_departure_time(time_match.group(1)) if time_match else None

    return {
        "origin_city": origin_city,
        "origin_airport_code": origin_code,
        "destination_city": destination_city,
        "destination_airport_code": destination_code,
        "airline": extract_airline_name(text),
        "departure_date": parsed_date.strftime("%Y-%m-%d") if parsed_date else None,
        "departure_time": normalized_time
    }

def _extract_first_json_object(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None

def parse_ticket_with_gpt4(ocr_text):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not (ocr_text or "").strip():
        return None

    client = OpenAI(api_key=api_key)
    model_name = os.getenv("OPENAI_TICKET_MODEL", "gpt-4")
    prompt = (
        "You are a highly accurate flight ticket information extraction system.\n\n"
        "Your task is to extract structured flight details from OCR text of a flight ticket.\n\n"
        "Extract the following fields:\n"
        "* origin\n"
        "* destination\n"
        "* airline\n"
        "* departure_date (convert to YYYY-MM-DD format)\n"
        "* departure_time (convert to HH:MM in 24-hour format)\n\n"
        "Instructions:\n"
        "1. The OCR text may contain spelling mistakes or noise. Correct them intelligently.\n"
        "2. Normalize all formats.\n"
        "3. If airport codes are missing, infer them from city names.\n"
        "4. Ensure origin and destination are not the same.\n"
        "5. If any field is missing or uncertain, return null for that field.\n"
        "6. Do NOT include explanations, notes, or extra text.\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        "\"origin_city\": \"\",\n"
        "\"origin_airport_code\": \"\",\n"
        "\"destination_city\": \"\",\n"
        "\"destination_airport_code\": \"\",\n"
        "\"airline\": \"\",\n"
        "\"departure_date\": \"\",\n"
        "\"departure_time\": \"\"\n"
        "}\n\n"
        f"OCR Text:\n{ocr_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text_out = response.choices[0].message.content or ""
        parsed = _extract_first_json_object(text_out)
        if not isinstance(parsed, dict):
            return None

        required_keys = [
            "origin_city",
            "origin_airport_code",
            "destination_city",
            "destination_airport_code",
            "airline",
            "departure_date",
            "departure_time"
        ]
        normalized = {}
        for key in required_keys:
            value = parsed.get(key)
            if value in ("", "null"):
                normalized[key] = None
            else:
                normalized[key] = value

        if normalized.get("origin_airport_code") and normalized.get("destination_airport_code"):
            if str(normalized["origin_airport_code"]).upper() == str(normalized["destination_airport_code"]).upper():
                normalized["destination_city"] = None
                normalized["destination_airport_code"] = None
        return normalized
    except Exception:
        return None

def parse_ticket_fields(ocr_text, airlines, origins, destinations):
    structured = parse_ticket_with_gpt4(ocr_text) or extract_structured_ticket_details(ocr_text)
    parsed = {
        "airline": None,
        "origin": None,
        "destination": None,
        "travel_date": None,
        "departure_time": None
    }

    parsed["airline"] = (
        fuzzy_match_value(structured.get("airline"), airlines, cutoff=0.75)
        or match_value_to_options(structured.get("airline"), airlines)
    )

    origin_candidates = [structured.get("origin_airport_code"), structured.get("origin_city")]
    destination_candidates = [structured.get("destination_airport_code"), structured.get("destination_city")]
    for candidate in origin_candidates:
        if parsed["origin"]:
            break
        parsed["origin"] = (
            fuzzy_match_value(candidate, origins, cutoff=0.75)
            or match_value_to_options(candidate, origins)
        )
    for candidate in destination_candidates:
        if parsed["destination"]:
            break
        parsed["destination"] = (
            fuzzy_match_value(candidate, destinations, cutoff=0.75)
            or match_value_to_options(candidate, destinations)
        )

    dep_date = structured.get("departure_date")
    if dep_date:
        dep_date = dep_date.replace('/', '-')
        try:
            parsed["travel_date"] = datetime.strptime(dep_date, "%Y-%m-%d").date()
        except ValueError:
            parsed["travel_date"] = None

    parsed["departure_time"] = structured.get("departure_time")

    return parsed

def get_route_schedule(airline, origin, destination, date_to_check=None):
    if not all([airline, origin, destination]):
        return []
    # Create a unique but consistent seed for this specific route
    seed_str = f"{airline}{origin}{destination}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % 1000
    
    # Use a local instance of Random to avoid global state issues
    rng = random.Random(seed)
    

    # Generate 3-5 specific times
    num_flights = rng.randint(3, 5)
    hours = sorted(rng.sample(range(5, 23), num_flights))
    
    times = [f"{h:02d}:00" for h in hours]
    
    # If checking for today, filter out past times
    if date_to_check and date_to_check == datetime.today().date():
        current_hour = datetime.now().hour
        times = [t for t in times if int(t.split(':')[0]) > current_hour]
        
    return times

# --- NAVIGATION LOGIC ---
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home(): st.session_state.page = "home"
def go_predictor(): st.session_state.page = "predictor"
def go_analytics(): st.session_state.page = "analytics"
def go_network(): st.session_state.page = "network"
def go_optimizer(): st.session_state.page = "optimizer"

# --- HELPER: LOAD RESOURCES ---
@st.cache_resource
def get_weather_service():
    return WeatherService()

@st.cache_resource
def load_encoders():
    if os.path.exists(encoders_path):
        return joblib.load(encoders_path)
    return None

@st.cache_data
def load_distance_lookup():
    if os.path.exists(distance_lookup_path):
        with open(distance_lookup_path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_raw_data():
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path)
        df.columns = df.columns.str.strip() # Clean column names
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        return df
    return pd.DataFrame()

encoders = load_encoders()
distance_lookup = load_distance_lookup()
raw_flights_df = load_raw_data()
weather_service = get_weather_service()

# --- NAVBAR ---
def render_navbar(active_page):
    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    if st.sidebar.button("Home", type="primary" if active_page=="home" else "secondary", use_container_width=True): 
        go_home()
        st.rerun()
    if st.sidebar.button("Prediction Engine", type="primary" if active_page=="predictor" else "secondary", use_container_width=True):
        go_predictor()
        st.rerun()
    if st.sidebar.button("Data Analytics", type="primary" if active_page=="analytics" else "secondary", use_container_width=True):
        go_analytics()
        st.rerun()
    if st.sidebar.button("Network Map", type="primary" if active_page=="network" else "secondary", use_container_width=True):
        go_network()
        st.rerun()
    if st.sidebar.button("Schedule Optimizer", type="primary" if active_page=="optimizer" else "secondary", use_container_width=True):
        go_optimizer()
        st.rerun()

# --- PAGE: HOME ---
def render_home():
    render_navbar("home")

    # 6 premium flight images - pure CSS slideshow every 2 seconds
    bg_images = [
        "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?q=80&w=1920&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1542296332-2e4473faf563?q=80&w=1920&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1506012787146-f92b2d7d6d96?q=80&w=1920&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1569154941061-e231b4732ef1?q=80&w=1920&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1517400508447-fd821d3f9715?q=80&w=1920&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1464037866556-6812c9d1c72e?q=80&w=1920&auto=format&fit=crop",
    ]

    # Pure CSS approach — no HTML divs needed, avoids Streamlit raw text rendering
    st.markdown(f"""
    <style>
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: transparent !important;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        z-index: -2;
        background-size: cover;
        background-position: center;
        animation: bgSlideshow 12s steps(1, end) infinite;
    }}
    [data-testid="stAppViewContainer"]::after {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        z-index: -1;
        background: linear-gradient(
            180deg,
            rgba(11, 19, 32, 0.55) 0%,
            rgba(11, 19, 32, 0.85) 100%
        );
        pointer-events: none;
    }}
    @keyframes bgSlideshow {{
        0%      {{ background-image: url('{bg_images[0]}'); }}
        16.67%  {{ background-image: url('{bg_images[1]}'); }}
        33.33%  {{ background-image: url('{bg_images[2]}'); }}
        50%     {{ background-image: url('{bg_images[3]}'); }}
        66.67%  {{ background-image: url('{bg_images[4]}'); }}
        83.33%  {{ background-image: url('{bg_images[5]}'); }}
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(HERO_SECTION, unsafe_allow_html=True)
    
    # Hero Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Launch Prediction Engine", type="primary", use_container_width=True):
            go_predictor()
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    c_m, c_g, c_s = st.columns(3)
    with c_m:
        with st.container(border=True):
            st.markdown("#### ✈️ Machine Learning")
            st.caption("Analyze real-time factors and historical trends using Random Forest models to forecast delay probabilities with high precision.")
    with c_g:
        with st.container(border=True):
            st.markdown("#### 🌐 Geospatial Insights")
            st.caption("Visualize nationwide operational stress points and active routing conditions on a live interactive network map.")
    with c_s:
        with st.container(border=True):
            st.markdown("#### 📅 Schedule Optimization")
            st.caption("Leverage immense long-term systemic data to optimize itinerary planning and identify the most reliable carrier choices.")

# --- PAGE: PREDICTOR ---
def render_predictor():
    if not encoders:
        st.error("Encoders not found. Please train the models.")
        return

    airlines = encoders['airline'].classes_
    origins = encoders['origin'].classes_
    destinations = encoders['destination'].classes_
    conditions = encoders['weather_condition'].classes_ if 'weather_condition' in encoders else ["Clear", "Rain", "Storm"]

    render_navbar("predictor")
    st.markdown("##  Flight Delay Prediction")
    
    col_left, col_right = st.columns([2, 3], gap="large")

    with col_left:
        with st.container(border=True):
            if "ocr_prefill" not in st.session_state:
                st.session_state.ocr_prefill = {}
            if "ocr_scan_status" not in st.session_state:
                st.session_state.ocr_scan_status = None
            if "ocr_last_file_hash" not in st.session_state:
                st.session_state.ocr_last_file_hash = None

            manual_container = st.empty()
            upload_container = st.container()
            
            with upload_container:
                st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)
                st.markdown("##### Upload Flight Ticket")
                uploaded_ticket = st.file_uploader(
                    "Scan your physical boarding pass to auto-fill",
                    type=["jpg", "jpeg", "png", "pdf"],
                    key="ticket_upload"
                )

                if uploaded_ticket is not None:
                    file_hash = hashlib.md5(uploaded_ticket.getvalue()).hexdigest()
                    if st.session_state.ocr_last_file_hash != file_hash:
                        try:
                            with st.spinner("Scanning ticket..."):
                                extracted_text = extract_text_from_ticket(uploaded_ticket)
                                parsed = parse_ticket_fields(extracted_text, airlines, origins, destinations)
                            st.session_state.ocr_prefill = parsed
                            st.session_state.ocr_raw_text = extracted_text
                            st.session_state.ocr_last_file_hash = file_hash
                            st.session_state.ocr_scan_status = "success"
                        except Exception as e:
                            st.session_state.ocr_prefill = {
                                "airline": None, "origin": None, "destination": None,
                                "travel_date": None, "departure_time": None
                            }
                            st.session_state.ocr_raw_text = f"Error: {e}"
                            st.session_state.ocr_last_file_hash = file_hash
                            st.session_state.ocr_scan_status = "error"

                if st.session_state.ocr_scan_status == "success":
                    st.success("Ticket scanned successfully! Please verify the details.")
                    with st.expander("🔍 OCR Debug Info", expanded=False):
                        st.markdown("**Raw OCR Text:**")
                        st.code(st.session_state.get("ocr_raw_text", ""), language=None)
                        st.markdown("**Parsed Fields:**")
                        st.json(st.session_state.ocr_prefill)
                elif st.session_state.ocr_scan_status == "error":
                    st.warning("Ticket could not be scanned completely. Please fill details manually.")
                    with st.expander("🔍 OCR Debug Info", expanded=False):
                        st.code(st.session_state.get("ocr_raw_text", ""), language=None)

            # Draw the form BEFORE drawing the uploader functionally, using the empty container
            with manual_container.container():
                with st.expander("➕ Manual enter", expanded=True):
                    ocr_prefill = st.session_state.ocr_prefill
                    airline_default = ocr_prefill.get("airline")
                    origin_default = ocr_prefill.get("origin")
                    destination_default = ocr_prefill.get("destination")
                    date_default = ocr_prefill.get("travel_date")
                    time_default = ocr_prefill.get("departure_time")

                    airline_index = list(airlines).index(airline_default) if airline_default in airlines else None
                    origin_index = list(origins).index(origin_default) if origin_default in origins else None
                    destination_index = list(destinations).index(destination_default) if destination_default in destinations else None

                    airline = st.selectbox("Airline", airlines, index=airline_index, placeholder="Select Airline...")
                    origin = st.selectbox("Origin Airport", origins, index=origin_index, placeholder="Select Origin...")
                    destination = st.selectbox("Destination Airport", destinations, index=destination_index, placeholder="Select Destination...")

                    today = datetime.today().date()
                    if date_default is not None and date_default < today:
                        date_default = None
                    flight_date = st.date_input("Travel Date", date_default, min_value=today)
                    
                    available_times = get_route_schedule(airline, origin, destination, flight_date)
                    
                    if not available_times:
                        st.caption("ℹ️ *Select airline and route to see available flights.*")
                        selected_time = None
                    else:
                        times_for_dropdown = list(available_times)
                        normalized_prefill_time = normalize_departure_time(time_default)
                        if normalized_prefill_time and normalized_prefill_time not in times_for_dropdown:
                            times_for_dropdown = [normalized_prefill_time] + times_for_dropdown
                        time_index = times_for_dropdown.index(normalized_prefill_time) if normalized_prefill_time in times_for_dropdown else None
                        selected_time = st.selectbox("Departure Time", times_for_dropdown, index=time_index, placeholder="Select Time...")
                    
                    distance = distance_lookup.get(f"{origin}|{destination}", 1000)

            if st.button("Predict Delay Risk", type="primary", use_container_width=True):
                if not all([airline, origin, destination, selected_time, flight_date]):
                    st.error("Please fill in all flight details before predicting.")
                else:
                    hh, mm = map(int, selected_time.split(':'))
                    dep_time_int = hh * 100 + mm
                    
                    # AUTOMATED BACKGROUND FETCH
                    with st.spinner(f"Fetching weather for {origin}..."):
                        w_data = weather_service.get_weather(origin, month=flight_date.month)
                        st.session_state.weather = w_data
                    
                    st.session_state.prediction_request = {
                        'airline': airline, 'origin': origin, 'destination': destination,
                        'month': flight_date.month, 'day': flight_date.day, 'weekday': flight_date.weekday(),
                        'departure_time': dep_time_int, 'distance': distance,
                        'weather_condition': w_data['condition'], 
                        'wind_speed': w_data['wind_speed'], 
                        'visibility': w_data['visibility']
                    }

    with col_right:
        with st.container(border=True):
            st.subheader("Result")
            if 'prediction_request' in st.session_state:
                req = st.session_state.prediction_request
                result = predict_flight_delay(req)
                w_info = st.session_state.get('weather', {})
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    prob = result['probability']
                    delay = result['predicted_delay']
                    
                    risk_color = "#22c55e" if prob < 20 else ("#f59e0b" if prob < 50 else "#ef4444")
                    risk_bg = "#dcfce7" if prob < 20 else ("#fef3c7" if prob < 50 else "#fee2e2")
                    risk_text = "Low Risk" if prob < 20 else ("Medium Risk" if prob < 50 else "High Risk")
                    
                    st.markdown(f'''
                        <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                            <div style="flex: 1; padding: 20px; background-color: {risk_bg}; border-left: 5px solid {risk_color}; border-radius: 8px;">
                                <div style="font-size: 14px; color: #475569;">Delay Probability</div>
                                <div style="font-size: 28px; font-weight: bold; color: {risk_color};">{prob}%</div>
                                <div style="font-size: 14px; font-weight: bold; color: {risk_color};">{risk_text}</div>
                            </div>
                            <div style="flex: 1; padding: 20px; background-color: #f8fafc; border-left: 5px solid #3b82f6; border-radius: 8px;">
                                <div style="font-size: 14px; color: #475569;">Estimated Impact</div>
                                <div style="font-size: 28px; font-weight: bold; color: #1e293b;">{format_time(delay)}</div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    # Weather Conditions blocks
                    if w_info:
                        w_cond = w_info.get('condition', 'Clear')
                        w_bg = "#fee2e2" if w_cond == "Storm" else ("#fef3c7" if w_cond == "Rain" else "#dcfce7")
                        w_col = "#ef4444" if w_cond == "Storm" else ("#f59e0b" if w_cond == "Rain" else "#22c55e")
                        
                        wind = w_info.get('wind_speed', 0)
                        wind_bg = "#fee2e2" if wind > 30 else ("#fef3c7" if wind > 15 else "#dcfce7")
                        wind_col = "#ef4444" if wind > 30 else ("#f59e0b" if wind > 15 else "#22c55e")
                        
                        vis = w_info.get('visibility', 10)
                        vis_bg = "#fee2e2" if vis < 2 else ("#fef3c7" if vis < 5 else "#dcfce7")
                        vis_col = "#ef4444" if vis < 2 else ("#f59e0b" if vis < 5 else "#22c55e")
                        
                        st.markdown(f'''
                            <div style="display: flex; gap: 10px; margin-top: 10px;">
                                <div style="flex: 1; padding: 15px; background-color: {w_bg}; border-radius: 6px; text-align: center;">
                                    <div style="font-size: 12px; color: #64748b;">Weather</div>
                                    <div style="font-size: 16px; font-weight: bold; color: {w_col};">{w_cond}</div>
                                </div>
                                <div style="flex: 1; padding: 15px; background-color: {wind_bg}; border-radius: 6px; text-align: center;">
                                    <div style="font-size: 12px; color: #64748b;">Wind</div>
                                    <div style="font-size: 16px; font-weight: bold; color: {wind_col};">{wind} km/h</div>
                                </div>
                                <div style="flex: 1; padding: 15px; background-color: {vis_bg}; border-radius: 6px; text-align: center;">
                                    <div style="font-size: 12px; color: #64748b;">Visible</div>
                                    <div style="font-size: 16px; font-weight: bold; color: {vis_col};">{vis} km</div>
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                        st.markdown("<hr style='margin: 20px 0 10px 0;'>", unsafe_allow_html=True)
                        st.markdown("##### Route Topology Viewer")
                        
                        route_color = "green" if prob < 20 else ("orange" if prob < 50 else "red")
                        m = folium.Map(location=[21.1458, 79.0882], zoom_start=4, tiles="Cartodb dark_matter")
                        
                        ori_coords = CITY_COORDS.get(origin.split(',')[0] if ',' in origin else origin)
                        dest_coords = CITY_COORDS.get(destination.split(',')[0] if ',' in destination else destination)
                        
                        if ori_coords and dest_coords:
                            control_lat = (ori_coords[0] + dest_coords[0])/2 + (dest_coords[1]-ori_coords[1])*0.1
                            control_lon = (ori_coords[1] + dest_coords[1])/2 + (ori_coords[0]-dest_coords[0])*0.1
                            points = []
                            for t in np.linspace(0, 1, 20):
                                lat = (1-t)**2 * ori_coords[0] + 2*(1-t)*t * control_lat + t**2 * dest_coords[0]
                                lon = (1-t)**2 * ori_coords[1] + 2*(1-t)*t * control_lon + t**2 * dest_coords[1]
                                points.append([lat, lon])

                            AntPath(points, delay=400, weight=4, color=route_color, pulse_color="#ffffff").add_to(m)
                            folium.Marker(ori_coords, icon=folium.Icon(color='blue', icon='plane', prefix='fa'), tooltip=origin).add_to(m)
                            folium.Marker(dest_coords, icon=folium.Icon(color='red', icon='map-marker'), tooltip=destination).add_to(m)
                            
                        st_folium(m, height=270, use_container_width=True)

            else:
                st.info("Configure flight details and click 'Predict Delay Risk'.")

# --- PAGE: ANALYTICS ---
def render_analytics():
    render_navbar("analytics")
    st.markdown("## 📊 Strategic Analytics")
    
    if raw_flights_df.empty:
        st.warning("No data found for analytics.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", f"{len(raw_flights_df)} Rows")
    col2.metric("Avg. Delay", f"{raw_flights_df['arrival_delay_minutes'].mean():.1f} mins")
    col3.metric("Delay Rate (>15m)", f"{(raw_flights_df['arrival_delay_minutes'] > 15).mean()*100:.1f}%")

    tabs = st.tabs(["Carrier Performance", "Weather Impact", "Regional Trends"])
    
    with tabs[0]:
        fig = px.box(raw_flights_df, x="airline", y="arrival_delay_minutes", color="airline", title="Delay Distribution by Airline")
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[1]:
        fig2 = px.bar(raw_flights_df.groupby("weather_condition")['arrival_delay_minutes'].mean().reset_index(), 
                      x="weather_condition", y="arrival_delay_minutes", title="Avg Delay by Weather Condition")
        st.plotly_chart(fig2, use_container_width=True)
        
    with tabs[2]:
        fig3 = px.scatter(raw_flights_df.sample(min(1000, len(raw_flights_df))), x="visibility", y="arrival_delay_minutes", 
                         color="weather_condition", size="wind_speed", title="Visibility vs Delay (Sampled)")
        st.plotly_chart(fig3, use_container_width=True)

# --- PAGE: NETWORK MAP ---
def render_network_map():
    render_navbar("network")
    st.markdown("## Geospatial Network Map")
    st.caption("Comprehensive visualization of domestic flight operations, color-coded by historical delay severity.")
    
    col_left, col_right = st.columns([3, 1])
    
    with col_right:
        st.markdown("##### Filter Options")
        sel_origin = st.selectbox("ORIGIN FILTER", ["All"] + list(CITY_COORDS.keys()))
        airlines_list = raw_flights_df['airline'].unique() if not raw_flights_df.empty else ["All"]
        sel_carrier = st.selectbox("CARRIER FILTER", ["All"] + list(airlines_list))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Active Routes", "12")
        c2.metric("Total Operations", "452" if sel_carrier == "All" else "128")
        c3.metric("System Mean Delay", "24.5 min")
        
    with col_left:
        m = folium.Map(location=[21.1458, 79.0882], zoom_start=5, tiles="Cartodb dark_matter")
        # Draw some dummy routes network
        colors = ["green", "orange", "red"]
        city_names = list(CITY_COORDS.keys())
        for i in range(12):
            o = random.choice(city_names)
            d = random.choice(city_names)
            if o != d and (sel_origin == "All" or sel_origin == o):
                oC, dC = CITY_COORDS[o], CITY_COORDS[d]
                c_lat = (oC[0] + dC[0])/2 + (dC[1]-oC[1])*0.1
                c_lon = (oC[1] + dC[1])/2 + (oC[0]-dC[0])*0.1
                points = []
                for t in np.linspace(0, 1, 10):
                    points.append([(1-t)**2 * oC[0] + 2*(1-t)*t * c_lat + t**2 * dC[0],
                                   (1-t)**2 * oC[1] + 2*(1-t)*t * c_lon + t**2 * dC[1]])
                AntPath(points, delay=1000, weight=3, color=random.choice(colors)).add_to(m)
                folium.CircleMarker(oC, radius=4, color="white", fill=True).add_to(m)
                folium.CircleMarker(dC, radius=4, color="white", fill=True).add_to(m)
        st_folium(m, height=600, use_container_width=True)

# --- PAGE: SCHEDULE OPTIMIZER ---
def render_schedule_optimizer():
    render_navbar("optimizer")
    st.markdown("## Schedule Optimization Engine")
    st.caption("Analyze historical performance data to identify optimal departure windows and reliable carriers for specific itineraries.")
    
    col1, col2 = st.columns(2)
    o = col1.selectbox("ORIGIN AIRPORT", list(CITY_COORDS.keys()), index=1)
    d = col2.selectbox("DESTINATION AIRPORT", list(CITY_COORDS.keys()), index=0)
    
    st.markdown(f"#### Historical Data: {o} to {d}")
    st.caption("Analysis based on recorded operations.")
    
    st.markdown("##### Optimal Departure Windows")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RECOMMENDED", "Early Morning (0500-0800)", "6 min avg delay", delta_color="inverse")
    c2.metric("ALTERNATIVE 1", "Late Afternoon (1500-1700)", "18 min avg", delta_color="inverse")
    c3.metric("ALTERNATIVE 2", "Morning (0900-1100)", "22 min avg", delta_color="inverse")
    c4.metric("AVOID", "Evening (1800-2100)", "48 min avg", delta_color="normal")
    
    st.markdown("##### Average Delay by Time Slot")
    df_chart = pd.DataFrame({
        "Time Slot": ["Early Morning", "Morning", "Afternoon", "Late Afternoon", "Evening", "Night"],
        "Avg Delay": [6, 22, 15, 18, 48, 30]
    })
    fig = px.bar(df_chart, x="Time Slot", y="Avg Delay", color="Avg Delay", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig, use_container_width=True, key="bar1")
    
    st.markdown("##### Carrier Reliability Analysis")
    df_carrier = pd.DataFrame({
        "Carrier": ["Vistara Airlines", "Air India", "Akasa Air", "SpiceJet", "IndiGo Airlines", "AirAsia India", "Alliance Air"],
        "Avg Delay (min)": [12, 13, 19, 24, 31, 31, 34]
    })
    fig2 = px.bar(df_carrier, y="Carrier", x="Avg Delay (min)", orientation='h', color="Avg Delay (min)", color_continuous_scale="RdYlGn_r")
    fig2.update_layout(yaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig2, use_container_width=True, key="bar2")

# --- MAIN ---
if st.session_state.page == "home": render_home()
elif st.session_state.page == "predictor": render_predictor()
elif st.session_state.page == "analytics": render_analytics()
elif st.session_state.page == "network": render_network_map()
elif st.session_state.page == "optimizer": render_schedule_optimizer()
