import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys
import os
import joblib
import json
from datetime import datetime, time
import random
import hashlib

# Add models and utils directories to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(base_dir, '..', 'models')))
sys.path.append(os.path.abspath(os.path.join(base_dir, '..')))

from predict import predict_flight_delay
from utils.weather_service import WeatherService
from assets.templates import HERO_SECTION, get_prediction_reasons

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
    layout="wide"
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
    st.markdown("""
        <style>
        /* Uniform Fixed-Size Buttons */
        div[data-testid="stHorizontalBlock"] button {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            color: #475569 !important;
            height: 45px !important;
            width: 100% !important;
            padding: 0px !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Hover State */
        div[data-testid="stHorizontalBlock"] button:hover {
            border-color: #2563eb !important;
            color: #2563eb !important;
            background-color: #f8fafc !important;
            transform: translateY(-1px);
        }

        /* Active Button Highlight */
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background-color: #eff6ff !important;
            border-color: #2563eb !important;
            color: #1d4ed8 !important;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.1) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Equal columns for buttons, larger spacer on right
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.5, 5])
    with c1: 
        if st.button(" Home", type="primary" if active_page=="home" else "secondary", use_container_width=True): 
            go_home()
            st.rerun()
    with c2:
        if st.button(" Predictor", type="primary" if active_page=="predictor" else "secondary", use_container_width=True):
            go_predictor()
            st.rerun()
    with c3:
        if st.button("📊 Analytics", type="primary" if active_page=="analytics" else "secondary", use_container_width=True):
            go_analytics()
            st.rerun()

# --- PAGE: HOME ---
def render_home():
    # Premium High-Res Flight Image (Unsplash)
    bg_url = "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?q=80&w=2074&auto=format&fit=crop"
    
    # Inject background image CSS with a bright Light Theme overlay
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.85)), 
                        url("{bg_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-color: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(HERO_SECTION, unsafe_allow_html=True)
    
    # Hero Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Explore Predictor", type="primary", use_container_width=True):
            go_predictor()
            st.rerun()

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
            st.subheader(" Flight Details")
            airline = st.selectbox("Airline", airlines, index=None, placeholder="Select Airline...")
            origin = st.selectbox("Origin Airport", origins, index=None, placeholder="Select Origin...")
            destination = st.selectbox("Destination Airport", destinations, index=None, placeholder="Select Destination...")
            flight_date = st.date_input("Travel Date", None, min_value=datetime.today())
            
            # Simulated schedules - filtered by airline and route
            available_times = get_route_schedule(airline, origin, destination, flight_date)
            
            if not available_times:
                st.caption("ℹ️ *Select airline and route to see available flights.*")
                selected_time = None
            else:
                selected_time = st.selectbox("Departure Time", available_times, index=None, placeholder="Select Time...")
            
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
            st.subheader("Results")
            if 'prediction_request' in st.session_state:
                req = st.session_state.prediction_request
                result = predict_flight_delay(req)
                w_info = st.session_state.get('weather', {})
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    prob = result['probability']
                    delay = result['predicted_delay']
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Delay Probability", f"{prob}%")
                    c2.metric("Estimated Delay", format_time(delay))
                    
                    if prob < 20: 
                        st.success("**Great News!** Your flight is predicted to be on time. No major disruptions expected based on current data.")
                    elif prob < 50: 
                        st.warning("**Attention Needed.** There is a moderate risk of a minor delay. We recommend monitoring your airline's live status.")
                    else: 
                        st.error("**High Risk Alert!** Significant delays are very likely. Please check with your airline for possible schedule changes before leaving for the airport.")
                    
                    # Live Weather Badge
                    if w_info:
                        source = w_info.get('api_source', 'Live Data')
                        st.caption(f" **Weather Insight ({source}):** {w_info['condition']} | Wind: {w_info['wind_speed']} km/h | Vis: {w_info['visibility']} km")
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

# --- MAIN ---
if st.session_state.page == "home": render_home()
elif st.session_state.page == "predictor": render_predictor()
elif st.session_state.page == "analytics": render_analytics()
