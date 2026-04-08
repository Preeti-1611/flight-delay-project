import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium
import math
import sys, os, joblib, json
from datetime import datetime
import random, hashlib

# Add paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(base_dir, '..', 'models')))
sys.path.append(os.path.abspath(os.path.join(base_dir, '..')))

from predict import predict_flight_delay
from utils.weather_service import WeatherService
from assets.templates import HERO_SECTION

# Paths
style_path = os.path.join(base_dir, 'assets', 'style.css')
encoders_path = os.path.join(base_dir, '..', 'trained_models', 'encoders.pkl')
distance_lookup_path = os.path.join(base_dir, '..', 'data', 'processed', 'distance_lookup.json')
distance_lookup_path = os.path.join(base_dir, '..', 'data', 'processed', 'distance_lookup.json')
raw_data_paths = [
    os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_dataset.csv'),
    os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_enriched.csv'),
    os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_500_rows.csv')
]
raw_data_path = next((p for p in raw_data_paths if os.path.exists(p)), raw_data_paths[0])

# City coordinates for map
CITY_COORDS = {
    "Jaipur, Rajasthan": [26.9124, 75.7873],
    "Chennai, Tamil Nadu": [13.0827, 80.2707],
    "Pune, Maharashtra": [18.5204, 73.8567],
    "Hyderabad, Telangana": [17.3850, 78.4867],
    "Bengaluru, Karnataka": [12.9716, 77.5946],
    "Kochi, Kerala": [9.9312, 76.2673],
    "Ahmedabad, Gujarat": [23.0225, 72.5714],
    "Kolkata, West Bengal": [22.5726, 88.3639],
    "Delhi, Delhi": [28.7041, 77.1025],
    "Mumbai, Maharashtra": [19.0760, 72.8777],
    "Goa, Goa": [15.2993, 74.1240],
    "Lucknow, Uttar Pradesh": [26.8467, 80.9462],
}

st.set_page_config(page_title="Aviation Delay Intelligence", page_icon="bar-chart", layout="wide")

def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css(style_path)

# --- HELPERS ---
def format_time(minutes):
    if minutes < 0: return "0 mins"
    mins = int(round(minutes))
    if mins < 60: return f"{mins} mins"
    return f"{mins // 60}h {mins % 60}m"

def get_route_schedule(airline, origin, destination, date_to_check=None):
    if not all([airline, origin, destination]): return []
    seed = int(hashlib.md5(f"{airline}{origin}{destination}".encode()).hexdigest(), 16) % 1000
    rng = random.Random(seed)
    hours = sorted(rng.sample(range(5, 23), rng.randint(3, 5)))
    times = [f"{h:02d}:00" for h in hours]
    if date_to_check and date_to_check == datetime.today().date():
        times = [t for t in times if int(t.split(':')[0]) > datetime.now().hour]
    return times

def _curved_arc(origin_coords, dest_coords, num_points=40):
    """Generate a curved great-circle-like arc between two points."""
    lat1, lon1 = origin_coords
    lat2, lon2 = dest_coords
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        # Add curvature perpendicular to the line
        offset = math.sin(t * math.pi) * 1.8  # arc height
        # Perpendicular direction
        dx = lat2 - lat1
        dy = lon2 - lon1
        length = math.sqrt(dx*dx + dy*dy) or 1
        lat += (-dy / length) * offset
        lon += (dx / length) * offset
        points.append([lat, lon])
    return points

def render_prediction_route_map(origin, destination, prob, delay, airline, status_color):
    """Render an animated route map for the prediction result."""
    o_coords = CITY_COORDS.get(origin)
    d_coords = CITY_COORDS.get(destination)
    if not o_coords or not d_coords:
        return  # Skip if city not in coordinates

    origin_short = origin.split(',')[0]
    dest_short = destination.split(',')[0]

    # Determine colors and labels
    if prob < 20:
        path_color = '#16a34a'; glow_color = 'rgba(22,163,106,0.3)'; status_label = 'ON TIME'
        status_icon = '✅'; bg_gradient = 'linear-gradient(135deg, #f0fdf4, #dcfce7)'
    elif prob < 50:
        path_color = '#eab308'; glow_color = 'rgba(234,179,8,0.3)'; status_label = 'MODERATE'
        status_icon = '⚠️'; bg_gradient = 'linear-gradient(135deg, #fefce8, #fef08a)'
    else:
        path_color = '#dc2626'; glow_color = 'rgba(220,38,38,0.3)'; status_label = 'DELAYED'
        status_icon = '🔴'; bg_gradient = 'linear-gradient(135deg, #fef2f2, #fee2e2)'

    # Calculate map center and zoom
    center_lat = (o_coords[0] + d_coords[0]) / 2
    center_lon = (o_coords[1] + d_coords[1]) / 2
    lat_diff = abs(o_coords[0] - d_coords[0])
    lon_diff = abs(o_coords[1] - d_coords[1])
    max_diff = max(lat_diff, lon_diff)
    zoom = 6 if max_diff < 5 else 5 if max_diff < 10 else 4

    # Create map with dark elegant tiles
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=zoom,
        tiles='CartoDB dark_matter', control_scale=True
    )

    # --- Origin marker (custom HTML) ---
    origin_html = f"""
    <div style="
        background: {bg_gradient}; border: 3px solid {path_color};
        border-radius: 50%; width: 52px; height: 52px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 0 20px {glow_color}, 0 4px 12px rgba(0,0,0,0.3);
        animation: pulseMarker 2s ease-in-out infinite;
        font-size: 22px;
    ">🛫</div>
    <style>
        @keyframes pulseMarker {{
            0%, 100% {{ transform: scale(1); box-shadow: 0 0 20px {glow_color}; }}
            50% {{ transform: scale(1.15); box-shadow: 0 0 35px {glow_color}; }}
        }}
    </style>
    """
    folium.Marker(
        location=o_coords,
        icon=folium.DivIcon(html=origin_html, icon_size=(52, 52), icon_anchor=(26, 26)),
        popup=folium.Popup(
            f"<div style='font-family:Inter,sans-serif;min-width:180px;'>"
            f"<div style='font-size:13px;color:#64748b;font-weight:600;'>ORIGIN</div>"
            f"<div style='font-size:16px;font-weight:800;color:#0f172a;margin:4px 0;'>{origin_short}</div>"
            f"<div style='font-size:12px;color:#94a3b8;'>{airline}</div></div>",
            max_width=220
        ),
        tooltip=f"🛫 {origin_short}"
    ).add_to(m)

    # --- Destination marker (custom HTML) ---
    dest_html = f"""
    <div style="
        background: {bg_gradient}; border: 3px solid {path_color};
        border-radius: 50%; width: 52px; height: 52px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 0 20px {glow_color}, 0 4px 12px rgba(0,0,0,0.3);
        font-size: 22px;
    ">🛬</div>
    """
    folium.Marker(
        location=d_coords,
        icon=folium.DivIcon(html=dest_html, icon_size=(52, 52), icon_anchor=(26, 26)),
        popup=folium.Popup(
            f"<div style='font-family:Inter,sans-serif;min-width:180px;'>"
            f"<div style='font-size:13px;color:#64748b;font-weight:600;'>DESTINATION</div>"
            f"<div style='font-size:16px;font-weight:800;color:#0f172a;margin:4px 0;'>{dest_short}</div>"
            f"<div style='font-size:12px;color:#94a3b8;'>Predicted: {format_time(delay)} delay</div></div>",
            max_width=220
        ),
        tooltip=f"🛬 {dest_short}"
    ).add_to(m)

    # --- Animated flight path (AntPath) ---
    arc_points = _curved_arc(o_coords, d_coords)
    AntPath(
        locations=arc_points, weight=5, color=path_color,
        opacity=0.85, dash_array=[15, 30],
        delay=800, pulse_color='#ffffff'
    ).add_to(m)

    # Subtle glow line underneath
    folium.PolyLine(
        locations=arc_points, weight=14,
        color=path_color, opacity=0.15
    ).add_to(m)

    # --- Flight info overlay ---
    dist_key = f"{origin}|{destination}"
    distance = distance_lookup.get(dist_key, '—')
    dist_display = f"{distance} km" if isinstance(distance, (int, float)) else distance

    info_html = f"""
    <div style="
        position: fixed; top: 15px; right: 15px; z-index: 1000;
        background: rgba(15,23,42,0.88); backdrop-filter: blur(20px);
        padding: 20px 24px; border-radius: 18px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.08);
        color: white; font-family: 'Inter', 'Segoe UI', sans-serif; min-width: 260px;
    ">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
            <span style="font-size:18px;">{status_icon}</span>
            <span style="
                background: {path_color}; color: white; padding: 3px 12px;
                border-radius: 20px; font-size: 11px; font-weight: 800;
                letter-spacing: 1px;
            ">{status_label}</span>
        </div>
        <div style="font-size:20px;font-weight:800;margin-bottom:4px;">
            {origin_short} → {dest_short}
        </div>
        <div style="font-size:12px;color:rgba(255,255,255,0.5);margin-bottom:14px;">{airline}</div>
        <div style="display:flex;gap:16px;">
            <div>
                <div style="font-size:11px;color:rgba(255,255,255,0.5);font-weight:600;">DELAY RISK</div>
                <div style="font-size:22px;font-weight:800;color:{path_color};">{prob}%</div>
            </div>
            <div style="border-left:1px solid rgba(255,255,255,0.1);padding-left:16px;">
                <div style="font-size:11px;color:rgba(255,255,255,0.5);font-weight:600;">EST. DELAY</div>
                <div style="font-size:22px;font-weight:800;color:{path_color};">{format_time(delay)}</div>
            </div>
        </div>
        <div style="margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.08);">
            <div style="font-size:11px;color:rgba(255,255,255,0.4);">
                📏 Distance: {dist_display}
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    # --- Legend ---
    legend_html = """
    <div style="
        position: fixed; bottom: 20px; left: 20px; z-index: 1000;
        background: rgba(15,23,42,0.85); backdrop-filter: blur(16px);
        padding: 14px 18px; border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.06);
        color: white; font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 12px;
    ">
        <div style="font-weight:800;margin-bottom:8px;font-size:11px;letter-spacing:0.5px;color:rgba(255,255,255,0.6);">ROUTE STATUS</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <span style="width:14px;height:4px;background:#16a34a;border-radius:2px;display:inline-block;"></span>
            <span>On Time (&lt;20%)</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <span style="width:14px;height:4px;background:#eab308;border-radius:2px;display:inline-block;"></span>
            <span>Moderate (20-50%)</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="width:14px;height:4px;background:#dc2626;border-radius:2px;display:inline-block;"></span>
            <span>Delayed (&gt;50%)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

# --- NAV ---
if "page" not in st.session_state: st.session_state.page = "home"
def go_home(): st.session_state.page = "home"
def go_predictor(): st.session_state.page = "predictor"
def go_analytics(): st.session_state.page = "analytics"
def go_route_map(): st.session_state.page = "route_map"
def go_best_time(): st.session_state.page = "best_time"

# --- NAVIGATION (SIDEBAR) ---
def render_navbar(active_page="predictor"):
    """Render the sidebar navigation."""
    st.sidebar.markdown("""<h2 style='color:var(--heading-color); font-weight: 800; margin-bottom:20px; font-size:1.1rem;'>Navigation</h2>""", unsafe_allow_html=True)
    pages = [("Home", "home", go_home),
             ("Prediction Engine", "predictor", go_predictor),
             ("Data Analytics", "analytics", go_analytics),
             ("Network Map", "route_map", go_route_map),
             ("Schedule Optimizer", "best_time", go_best_time)]
    
    for label, key, func in pages:
        if st.sidebar.button(label, type="primary" if active_page==key else "secondary", use_container_width=True):
            func()
            st.rerun()

    st.sidebar.markdown("<br><hr>", unsafe_allow_html=True)
        
    return 'Dark'

# --- LOAD RESOURCES ---
@st.cache_resource
def get_weather_service(): return WeatherService()

@st.cache_resource
def load_encoders():
    if os.path.exists(encoders_path): return joblib.load(encoders_path)
    return None

@st.cache_data
def load_distance_lookup():
    if os.path.exists(distance_lookup_path):
        with open(distance_lookup_path, 'r') as f: return json.load(f)
    return {}

@st.cache_data
def load_raw_data():
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path)
        df.columns = df.columns.str.strip()
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        return df
    return pd.DataFrame()

encoders = load_encoders()
distance_lookup = load_distance_lookup()
raw_flights_df = load_raw_data()
weather_service = get_weather_service()

# ======================== PAGE: HOME ========================
def render_home():
    st.markdown("""<style>
        /* Force transparency on the Home page so the fixed background is visible */
        body, .stApp, [data-testid="stAppViewContainer"] {
            background: transparent !important;
            background-color: transparent !important;
        }
        /* Lock the background slider */
        .home-slider {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            width: 100vw; height: 100vh;
            z-index: -9999;
            animation: slideShow 25s infinite linear;
            background-size: cover;
            background-position: center;
            background-color: #0b1320; /* Dark fallback */
        }
        div[data-testid="stHeader"] {
            background: transparent !important;
        }
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="home-slider"></div>', unsafe_allow_html=True)
    st.markdown(HERO_SECTION, unsafe_allow_html=True)
    
    # Launch Button
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Launch Prediction Engine", type="primary", use_container_width=True):
            go_predictor()
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Feature Cards
    f1, f2, f3 = st.columns(3)
    
    with f1:
        st.markdown(f"""
        <div class="feature-card">
            <h3 style="color: var(--accent-color); margin-bottom: 12px; font-size: 1.3rem;">Machine Learning</h3>
            <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.5; margin: 0;">Analyze real-time factors and historical trends using Random Forest models to forecast delay probabilities with high precision.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with f2:
        st.markdown(f"""
        <div class="feature-card">
            <h3 style="color: var(--accent-color); margin-bottom: 12px; font-size: 1.3rem;">Geospatial Insights</h3>
            <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.5; margin: 0;">Visualize nationwide operational stress points and active routing conditions on a live interactive network map.</p>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown(f"""
        <div class="feature-card">
            <h3 style="color: var(--accent-color); margin-bottom: 12px; font-size: 1.3rem;">Schedule Optimization</h3>
            <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.5; margin: 0;">Leverage immense long-term systemic data to optimize itinerary planning and identify the most reliable carrier choices.</p>
        </div>
        """, unsafe_allow_html=True)
        
    # High Level System Metrics
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'><br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    stat_style = "color:white; font-size:2.8rem; font-weight:800; margin-bottom:0px; text-shadow: 0 4px 12px rgba(0,0,0,0.5);"
    label_style = "color:var(--accent-color); text-transform:uppercase; font-size:0.85rem; font-weight:700; letter-spacing:0.05em;"
    
    m1.markdown(f"<div class='metric-box'><p style='{stat_style}'>15+</p><p style='{label_style}'>Hub Airports</p></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-box'><p style='{stat_style}'>300k+</p><p style='{label_style}'>Flights Analyzed</p></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-box'><p style='{stat_style}'>89%</p><p style='{label_style}'>Model Confidence</p></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-box'><p style='{stat_style}'>24/7</p><p style='{label_style}'>Weather Sync</p></div>", unsafe_allow_html=True)


# ======================== PAGE: PREDICTOR ========================
def render_predictor():
    st.markdown("<style>.stApp { animation: none !important; background: var(--bg-color) !important; }</style>", unsafe_allow_html=True)
    if not encoders:
        st.error("Encoders not found. Please train the models."); return

    airlines = encoders['airline'].classes_
    origins = encoders['origin'].classes_
    destinations = encoders['destination'].classes_

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; margin-bottom: 25px; color: var(--heading-color); font-weight: 800;'>Flight Delay Prediction Engine</h2>", unsafe_allow_html=True)

    # ======= SIDE-BY-SIDE: Left = Inputs, Right = Results =======
    left_col, right_col = st.columns([1, 1], gap="large")

    # -------- LEFT COLUMN: All Inputs --------
    with left_col:
        st.markdown("""<div style="margin-bottom: 20px;">
            <h3 style="margin:0;color:var(--heading-color);font-size:1.1rem;font-weight:700; border-bottom: 2px solid var(--card-border); padding-bottom: 8px;">Flight Details</h3>
        </div>""", unsafe_allow_html=True)

        # Route selection
        r1, r2 = st.columns(2)
        with r1:
            val_origin = st.selectbox("Origin Airport", origins, index=None, key="sel_origin", placeholder="Select Origin...")
        with r2:
            val_dest = st.selectbox("Destination Airport", destinations, index=None, key="sel_dest", placeholder="Select Destination...")

        # Airline
        val_airline = st.selectbox("Airline", airlines, index=None, key="sel_airline", placeholder="Choose Airline...")

        # Date & Time
        d1, d2 = st.columns(2)
        with d1:
            flight_date = st.date_input("Travel Date", None, min_value=datetime.today(), key="in_date")
        with d2:
            available_times = get_route_schedule(val_airline, val_origin, val_dest, flight_date) if all([val_airline, val_origin, val_dest]) else []
            if not available_times:
                val_time = st.selectbox("Departure Time", [], key="sel_time", placeholder="Select route & airline first...")
            else:
                val_time = st.selectbox("Departure Time", available_times, index=None, placeholder="Select Time...", key="sel_time")

        st.markdown("<br>", unsafe_allow_html=True)

        # Predict button
        predict_clicked = st.button("Calculate Delay Risk", type="primary", use_container_width=True, key="btn_predict")

        if predict_clicked:
            if not all([val_origin, val_dest, val_airline, flight_date, val_time]):
                st.error("Please fill in all fields before predicting.")
            else:
                distance = distance_lookup.get(f"{val_origin}|{val_dest}", 1000)
                hh, mm = map(int, val_time.split(':'))
                with st.spinner(f"Querying meteorological data for {val_origin}..."):
                    w_data = weather_service.get_weather(val_origin, month=flight_date.month)
                    st.session_state.weather = w_data
                    st.session_state.prediction_request = {
                        'airline': val_airline, 'origin': val_origin,
                        'destination': val_dest, 'month': flight_date.month, 'day': flight_date.day,
                        'weekday': flight_date.weekday(), 'departure_time': hh*100+mm, 'distance': distance,
                        'weather_condition': w_data['condition'], 'wind_speed': w_data['wind_speed'], 'visibility': w_data['visibility']
                    }
                    st.session_state.prediction_done = True
                st.rerun()

    # -------- RIGHT COLUMN: Prediction Results --------
    with right_col:
        if st.session_state.get('prediction_done') and st.session_state.get('prediction_request'):
            req = st.session_state.prediction_request
            result = predict_flight_delay(req)
            w_info = st.session_state.get('weather', {})

            if 'error' in result:
                st.error(result['error'])
            else:
                prob = result['probability']; delay = result['predicted_delay']
                if prob < 20: status_text, status_color = "Low Risk", "#16a34a"
                elif prob < 50: status_text, status_color = "Moderate Risk", "#eab308"
                else: status_text, status_color = "High Risk", "#dc2626"

                # Results container
                st.markdown(f"""<div style="margin-bottom: 20px;">
                    <h3 style="margin:0;color:var(--heading-color);font-size:1.1rem;font-weight:700; border-bottom: 2px solid var(--card-border); padding-bottom: 8px;">Analysis Report</h3>
                </div>""", unsafe_allow_html=True)

                # Route summary
                origin_short = req['origin'].split(',')[0]
                dest_short = req['destination'].split(',')[0]
                st.markdown(f"""
                <div style="background:var(--card-bg); border: 1px solid var(--card-border); border-radius:8px; padding:16px;
                    margin-bottom:16px;display:flex;align-items:center;justify-content:space-between; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                    <div>
                        <div style="font-weight:700;font-size:1.1rem; color:var(--text-main);">{origin_short} to {dest_short}</div>
                        <div style="color:var(--text-muted);font-size:0.85rem;">Carrier: {req['airline']}</div>
                    </div>
                    <div style="background:{status_color};color:white;padding:4px 12px;border-radius:4px;
                        font-size:0.75rem;font-weight:700;letter-spacing:0.5px;">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Status banner
                st.markdown(f"""
                <div style="border-left: 4px solid {status_color}; background: var(--card-bg);
                    border-top:1px solid var(--card-border); border-right:1px solid var(--card-border); border-bottom:1px solid var(--card-border);
                    border-radius: 4px; padding:20px; text-align:left; margin-bottom:16px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div class="data-label">Delay Probability</div>
                            <div style="font-size:2.5rem;font-weight:700;color:{status_color};">{prob}%</div>
                        </div>
                        <div style="text-align: right;">
                            <div class="data-label">Estimated Impact</div>
                            <div style="font-size:2.5rem;font-weight:700;color:var(--text-main);">{format_time(delay)}</div>
                        </div>
                    </div>
                    <div style="background:var(--card-border);border-radius:4px;height:6px;margin-top:16px;width:100%;overflow:hidden;">
                        <div style="background:{status_color};width:{prob}%;height:100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Weather compact card
                if w_info:
                    st.markdown(f"""
                    <div style="background:var(--card-bg); border: 1px solid var(--card-border); border-radius:8px; padding:16px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                        <div class="data-label" style="margin-bottom: 12px;">Conditions at Origin</div>
                        <div style="display:flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-weight:700;color:var(--text-main);font-size:1.1rem;">{w_info['condition']}</div>
                            </div>
                            <div style="display:flex;gap:16px;text-align:right;">
                                <div>
                                    <div class="data-label">Wind</div>
                                    <div style="font-weight:600;color:var(--text-main);font-size:0.9rem;">{w_info['wind_speed']} km/h</div>
                                </div>
                                <div>
                                    <div class="data-label">Visibility</div>
                                    <div style="font-weight:600;color:var(--text-main);font-size:0.9rem;">{w_info['visibility']} km</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Empty state — no prediction yet
            st.markdown("""<div class="professional-card" style="text-align:center; padding: 60px 20px;">
                <h3 style="color:var(--text-muted);font-weight:600;margin-bottom:8px;">Awaiting Input</h3>
                <p style="color:var(--text-muted);font-size:0.95rem;">Configure flight parameters in the left panel to execute the prediction model.</p>
            </div>
            """, unsafe_allow_html=True)

    # --- PREDICTION ROUTE MAP (full width, only if prediction exists) ---
    if st.session_state.get('prediction_done') and st.session_state.get('prediction_request'):
        req = st.session_state.prediction_request
        result = predict_flight_delay(req)
        if 'error' not in result:
            prob = result['probability']; delay = result['predicted_delay']
            if prob < 20: status_color = "#16a34a"
            elif prob < 50: status_color = "#eab308"
            else: status_color = "#dc2626"

            st.markdown("""<div style="margin-top:30px;">
                <h3 style="text-align:left;margin-bottom:5px;color:var(--heading-color);font-weight:800;">Spatial Network Visualization</h3>
                <p style="text-align:left;color:var(--text-muted);font-size:0.95rem;margin-bottom:15px;">Live prediction mapped conceptually across the routing network.</p>
            </div>""", unsafe_allow_html=True)

            pred_map = render_prediction_route_map(
                req['origin'], req['destination'], prob, delay,
                req['airline'], status_color
            )
            if pred_map:
                st_folium(pred_map, width=None, height=420, use_container_width=True, key="pred_route_map")
                _, mc, _ = st.columns([1, 2, 1])
                with mc:
                    if st.button("Access Full Network Map", use_container_width=True, key="go_full_map"):
                        go_route_map(); st.rerun()
            else:
                st.info("Network visualization unavailable: Geo-coordinates missing for specified nodes.")

# ======================== PAGE: ANALYTICS ========================
def render_analytics():
    # Only render navbar inside if NOT done at top-level.
    # Oh wait, render_analytics is called directly. It will render its navbar?
    # Navbar is rendered at top of main(). No need to re-render.
    st.markdown("<h2 style='text-align: left; margin-bottom: 25px; color: var(--heading-color); font-weight: 800;'>Strategic Data Analytics</h2>", unsafe_allow_html=True)
    if raw_flights_df.empty: st.warning("Insufficient operation data."); return
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Volume", f"{len(raw_flights_df):,} Records")
    c2.metric("Mean System Delay", f"{raw_flights_df['arrival_delay_minutes'].mean():.1f} min")
    c3.metric("Critical Delay Rate (>15m)", f"{(raw_flights_df['arrival_delay_minutes'] > 15).mean()*100:.1f}%")
    tabs = st.tabs(["Carrier Performance", "Weather Impact", "Regional Trends"])
    with tabs[0]:
        fig = px.box(raw_flights_df, x="airline", y="arrival_delay_minutes", color="airline", title="Delay Distribution by Airline")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        fig2 = px.bar(raw_flights_df.groupby("weather_condition")['arrival_delay_minutes'].mean().reset_index(), 
                      x="weather_condition", y="arrival_delay_minutes", title="Avg Delay by Weather")
        st.plotly_chart(fig2, use_container_width=True)
    with tabs[2]:
        fig3 = px.scatter(raw_flights_df.sample(min(1000, len(raw_flights_df))), x="visibility", y="arrival_delay_minutes",
                         color="weather_condition", size="wind_speed", title="Visibility vs Delay")
        st.plotly_chart(fig3, use_container_width=True)

# ======================== PAGE: ROUTE MAP ========================
def render_route_map():
    st.markdown("<h2 style='text-align: left; margin-bottom: 5px; color: var(--heading-color); font-weight: 800;'>Geospatial Network Map</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text-muted); margin-bottom: 25px;'>Comprehensive visualization of domestic flight operations, color-coded by historical delay severity.</p>", unsafe_allow_html=True)

    # Show last prediction route if available
    pred_req = st.session_state.get('prediction_request')
    if pred_req:
        result = predict_flight_delay(pred_req)
        if 'error' not in result:
            prob = result['probability']; delay = result['predicted_delay']
            o_short = pred_req['origin'].split(',')[0]; d_short = pred_req['destination'].split(',')[0]
            if prob < 20: sc = '#16a34a'; sl = 'NOMINAL'
            elif prob < 50: sc = '#eab308'; sl = 'ELEVATED'
            else: sc = '#dc2626'; sl = 'CRITICAL'
            st.markdown(f"""
            <div class="professional-card" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;margin-bottom:20px;">
                <div style="display:flex;align-items:center;gap:16px;">
                    <div>
                        <div style="color:var(--heading-color);font-size:1.1rem;font-weight:700;">{o_short} to {d_short}</div>
                        <div style="color:var(--text-muted);font-size:0.85rem;">Carrier: {pred_req['airline']} | Active Query</div>
                    </div>
                </div>
                <div style="display:flex;gap:24px;align-items:center;">
                    <div style="text-align:right;">
                        <div class="data-label">Risk Probability</div>
                        <div style="color:{sc};font-size:1.2rem;font-weight:700;">{prob}%</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="data-label">Estimated Delay</div>
                        <div style="color:{sc};font-size:1.2rem;font-weight:700;">{format_time(delay)}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if raw_flights_df.empty: st.warning("Insufficient operation data."); return
    
    # Filters
    fc1, fc2 = st.columns(2)
    with fc1:
        sel_origin = st.selectbox("Origin Filter", ["All"] + sorted(raw_flights_df['origin'].unique().tolist()), key="map_origin")
    with fc2:
        sel_airline = st.selectbox("Carrier Filter", ["All"] + sorted(raw_flights_df['airline'].unique().tolist()), key="map_airline")
    
    filtered_df = raw_flights_df.copy()
    if sel_origin != "All": filtered_df = filtered_df[filtered_df['origin'] == sel_origin]
    if sel_airline != "All": filtered_df = filtered_df[filtered_df['airline'] == sel_airline]
    
    route_stats = filtered_df.groupby(['origin', 'destination']).agg(
        avg_delay=('arrival_delay_minutes', 'mean'), flight_count=('arrival_delay_minutes', 'count')
    ).reset_index()
    
    # Map metrics
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Active Routes", len(route_stats))
    mc2.metric("Total Operations", int(route_stats['flight_count'].sum()))
    mc3.metric("System Mean Delay", f"{route_stats['avg_delay'].mean():.1f} min" if len(route_stats) > 0 else "N/A")
    
    # Build Folium map — use tiles based on theme
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='CartoDB dark_matter')
    
    # Add city markers with glow effect
    cities_in_data = set(filtered_df['origin'].unique()) | set(filtered_df['destination'].unique())
    for city in cities_in_data:
        if city in CITY_COORDS:
            city_delays = filtered_df[filtered_df['origin'] == city]['arrival_delay_minutes']
            avg = city_delays.mean() if len(city_delays) > 0 else 0
            color = '#16a34a' if avg < 10 else '#eab308' if avg < 25 else '#dc2626'
            # Outer glow
            folium.CircleMarker(
                location=CITY_COORDS[city], radius=16, color=color, fill=True,
                fill_color=color, fill_opacity=0.15, weight=0
            ).add_to(m)
            # Inner marker
            folium.CircleMarker(
                location=CITY_COORDS[city], radius=7, color='white', weight=2, fill=True,
                fill_color=color, fill_opacity=0.9,
                popup=folium.Popup(
                    f"<div style='font-family:Inter,sans-serif;'>"
                    f"<div style='font-weight:800;font-size:14px;color:#0f172a;'>{city.split(',')[0]}</div>"
                    f"<div style='color:#64748b;font-size:12px;margin-top:4px;'>Avg Delay: {avg:.1f} mins</div>"
                    f"<div style='color:#94a3b8;font-size:11px;'>Flights: {len(city_delays)}</div></div>",
                    max_width=200
                ),
                tooltip=city.split(',')[0]
            ).add_to(m)
    
    # Add animated route lines with AntPath
    for _, row in route_stats.iterrows():
        o, d = row['origin'], row['destination']
        if o in CITY_COORDS and d in CITY_COORDS:
            avg_d = row['avg_delay']
            color = '#16a34a' if avg_d < 10 else '#eab308' if avg_d < 25 else '#dc2626'
            arc = _curved_arc(CITY_COORDS[o], CITY_COORDS[d], num_points=25)
            # Glow line
            folium.PolyLine(
                locations=arc, weight=max(4, min(row['flight_count']/3, 12)),
                color=color, opacity=0.12
            ).add_to(m)
            # Animated path
            AntPath(
                locations=arc, weight=max(2, min(row['flight_count']/5, 6)),
                color=color, opacity=0.7, dash_array=[10, 20], delay=1000,
                pulse_color='#ffffff'
            ).add_to(m)

    # Highlight last predicted route
    if pred_req and pred_req['origin'] in CITY_COORDS and pred_req['destination'] in CITY_COORDS:
        arc = _curved_arc(CITY_COORDS[pred_req['origin']], CITY_COORDS[pred_req['destination']])
        p_color = '#16a34a' if result.get('probability',0) < 20 else '#eab308' if result.get('probability',0) < 50 else '#dc2626'
        folium.PolyLine(locations=arc, weight=18, color=p_color, opacity=0.2).add_to(m)
        AntPath(locations=arc, weight=5, color=p_color, opacity=0.95, dash_array=[15,25], delay=600, pulse_color='#ffffff').add_to(m)
    
    # Legend
    legend_html = """
    <div style="
        position: fixed; bottom: 20px; left: 20px; z-index: 1000;
        background: rgba(15,23,42,0.85); backdrop-filter: blur(16px);
        padding: 14px 18px; border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.06);
        color: white; font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 12px;
    ">
        <div style="font-weight:800;margin-bottom:8px;font-size:11px;letter-spacing:0.5px;color:rgba(255,255,255,0.6);">DELAY LEVEL</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <span style="width:14px;height:4px;background:#16a34a;border-radius:2px;display:inline-block;"></span>
            <span>Low (&lt;10 min)</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <span style="width:14px;height:4px;background:#eab308;border-radius:2px;display:inline-block;"></span>
            <span>Medium (10-25 min)</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="width:14px;height:4px;background:#dc2626;border-radius:2px;display:inline-block;"></span>
            <span>High (&gt;25 min)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    st_folium(m, width=None, height=550, use_container_width=True)

# ======================== PAGE: SCHEDULE OPTIMIZER ========================
def render_best_time():
    st.markdown("<h2 style='text-align: left; margin-bottom: 5px; color: var(--heading-color); font-weight: 800;'>Schedule Optimization Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text-muted); margin-bottom: 25px;'>Analyze historical performance data to identify optimal departure windows and reliable carriers for specific itineraries.</p>", unsafe_allow_html=True)
    
    if raw_flights_df.empty: st.warning("Insufficient operation data."); return
    
    locations = sorted(raw_flights_df['origin'].unique().tolist())
    all_airlines = sorted(raw_flights_df['airline'].unique().tolist())
    
    fc1, fc2 = st.columns(2)
    with fc1: sel_from = st.selectbox("Origin Airport", locations, index=None, placeholder="Select Origin...", key="bt_from")
    with fc2: sel_to = st.selectbox("Destination Airport", locations, index=None, placeholder="Select Destination...", key="bt_to")
    
    if sel_from and sel_to and sel_from != sel_to:
        route_df = raw_flights_df[(raw_flights_df['origin'] == sel_from) & (raw_flights_df['destination'] == sel_to)]
        
        if route_df.empty:
            st.info(f"No historical operations recorded for {sel_from.split(',')[0]} to {sel_to.split(',')[0]}.")
            return
        
        st.markdown(f"### Historical Data: {sel_from.split(',')[0]} to {sel_to.split(',')[0]}")
        st.markdown(f"<p style='color:#64748b;'>Analysis based on <b>{len(route_df)}</b> recorded operations.</p>", unsafe_allow_html=True)
        
        # --- BEST TIME ANALYSIS ---
        st.markdown("#### Optimal Departure Windows")
        route_df = route_df.copy()
        route_df['time_hour'] = route_df['departure_time'] // 100
        time_bins = {range(5,9): "Early Morning (0500-0859)", range(9,12): "Morning (0900-1159)",
                     range(12,15): "Midday (1200-1459)", range(15,18): "Late Afternoon (1500-1759)",
                     range(18,22): "Evening (1800-2159)", range(22,24): "Night (2200-0459)"}
        
        def get_slot(h):
            for r, label in time_bins.items():
                if h in r: return label
            return "Early Morning (0500-0859)"
        
        route_df['time_slot'] = route_df['time_hour'].apply(get_slot)
        slot_stats = route_df.groupby('time_slot').agg(
            avg_delay=('arrival_delay_minutes', 'mean'),
            delay_rate=('arrival_delay_minutes', lambda x: (x > 15).mean() * 100),
            count=('arrival_delay_minutes', 'count')
        ).reset_index().sort_values('avg_delay')
        
        best_slot = slot_stats.iloc[0] if len(slot_stats) > 0 else None
        
        tcols = st.columns(min(len(slot_stats), 4))
        for i, (_, row) in enumerate(slot_stats.head(4).iterrows()):
            is_best = (i == 0)
            card_cls = "professional-card"
            border_style = "border-top: 4px solid #16a34a;" if is_best else ""
            badge = "RECOMMENDED" if is_best else "ALTERNATIVE"
            color = "#16a34a" if row['avg_delay'] < 10 else "#eab308" if row['avg_delay'] < 25 else "#dc2626"
            with tcols[i]:
                st.markdown(f"""<div class="{card_cls}" style="{border_style}">
                    <p style="font-size:0.75rem;color:#64748b;margin-bottom:8px;font-weight:700;letter-spacing:0.05em;">{badge}</p>
                    <p style="font-size:0.95rem;font-weight:700;color:var(--text-main);margin-bottom:4px;">{row['time_slot']}</p>
                    <h3 style="color:{color};margin:8px 0;font-size:1.4rem;">{row['avg_delay']:.0f} min</h3>
                    <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:0;">{row['delay_rate']:.0f}% disruption rate</p>
                    <p style="color:var(--text-muted);font-size:0.8rem;margin-bottom:0;">n={int(row['count'])}</p>
                </div>""", unsafe_allow_html=True)
        
        # Time chart
        fig_time = px.bar(slot_stats, x='time_slot', y='avg_delay', color='avg_delay',
                          color_continuous_scale=['#16a34a', '#eab308', '#dc2626'],
                          title='Average Delay by Time Slot', labels={'avg_delay': 'Avg Delay (min)', 'time_slot': 'Time Slot'})
        fig_time.update_layout(showlegend=False)
        st.plotly_chart(fig_time, use_container_width=True)
        
        # --- ALTERNATIVE AIRLINES ---
        st.markdown("---")
        st.markdown("#### Carrier Reliability Analysis")
        
        airline_stats = route_df.groupby('airline').agg(
            avg_delay=('arrival_delay_minutes', 'mean'),
            delay_rate=('arrival_delay_minutes', lambda x: (x > 15).mean() * 100),
            count=('arrival_delay_minutes', 'count'),
            min_delay=('arrival_delay_minutes', 'min'),
            max_delay=('arrival_delay_minutes', 'max')
        ).reset_index().sort_values('avg_delay')
        
        for i, (_, row) in enumerate(airline_stats.iterrows()):
            color = "#16a34a" if row['avg_delay'] < 10 else "#eab308" if row['avg_delay'] < 25 else "#dc2626"
            bar_width = min(row['avg_delay'] / (airline_stats['avg_delay'].max() + 1) * 100, 100)
            st.markdown(f"""<div class="professional-card" style="padding:16px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="color:var(--text-muted);font-size:0.8rem;margin-right:8px;font-weight:600;">Rank {i+1}</span>
                        <span style="font-weight:700;font-size:1.1rem;color:var(--text-main);">{row['airline']}</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:1.3rem;font-weight:800;color:{color};">{row['avg_delay']:.0f} min</span>
                        <span style="color:var(--text-muted);font-size:0.85rem;margin-left:8px;">{row['delay_rate']:.0f}% delayed</span>
                    </div>
                </div>
                <div style="background:var(--card-border);border-radius:999px;height:6px;margin-top:10px;overflow:hidden;">
                    <div style="background:{color};width:{bar_width}%;height:100%;border-radius:999px;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:8px;color:var(--text-muted);font-size:0.8rem;">
                    <span>{int(row['count'])} flights</span><span>Min: {row['min_delay']:.0f}m | Max: {row['max_delay']:.0f}m</span>
                </div>
            </div>""", unsafe_allow_html=True)
        
        # Airline comparison chart
        fig_air = px.bar(airline_stats, x='airline', y='avg_delay', color='delay_rate',
                        color_continuous_scale=['#16a34a', '#eab308', '#dc2626'],
                        title='Airline Delay Comparison', labels={'avg_delay': 'Avg Delay (min)', 'delay_rate': 'Delay Rate %'})
        st.plotly_chart(fig_air, use_container_width=True)
    
    elif sel_from and sel_to and sel_from == sel_to:
        st.warning("Origin and destination must be different.")

# --- MAIN ---
page = st.session_state.page

if page != "home":
    render_navbar(page)

if page == "home": render_home()
elif page == "predictor": render_predictor()
elif page == "analytics": render_analytics()
elif page == "route_map": render_route_map()
elif page == "best_time": render_best_time()

