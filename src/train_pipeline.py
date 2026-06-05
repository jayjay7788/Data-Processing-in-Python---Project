import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- IMPORTANT: IMPORT YOUR WRAPPER FUNCTIONS ---
# Ensure your src folder is accessible. If your structure requires it, 
# you can add your root to sys.path, or import relatively like this:
from src.model_wrapper import load_model_bundle, predict_bundle

# ==============================================================================
# 🌍 1. DYNAMIC FILE PATH RESOLUTION (Using Pathlib)
# ==============================================================================
# This solves the file-not-found issues by referencing locations from the project root,
# regardless of which folder you launch Streamlit from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "results" / "models"

# Set up the page configuration layout (Must be called before any other st. widgets)
st.set_page_config(page_title="Prague Air Quality Hub", layout="wide")


# ==============================================================================
# 📦 2. CACHED DATA & MODEL LOADING (With Error Protections)
# ==============================================================================
@st.cache_data
def load_historical_data():
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find processed data at: {PROCESSED_DATA_PATH}. "
            "Please execute your preprocessing script first!"
        )
    
    df = pd.read_csv(PROCESSED_DATA_PATH) 
    df['startTime'] = pd.to_datetime(df['startTime'])
    df = df.sort_values(by='startTime').reset_index(drop=True)
    
    # Ensure temperature column is cast cleanly to numeric values
    if 'weather_TMA' in df.columns:
        df['weather_TMA'] = pd.to_numeric(df['weather_TMA'], errors='coerce')
    return df

@st.cache_resource
def get_model_bundle():
    if not MODEL_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"Could not find model output directory at: {MODEL_OUTPUT_DIR}. "
            "Please ensure you run train_pipeline.py first!"
        )
    return load_model_bundle(MODEL_OUTPUT_DIR)

# Run safe initialization sequence
try:
    df_historical = load_historical_data()
    bundle = get_model_bundle()
except FileNotFoundError as fnf_error:
    st.error(f"📁 **Missing Files:** {fnf_error}")
    st.stop()
except Exception as e:
    st.error(f"💥 **Initialization Error:** {e}")
    st.stop()


# ==============================================================================
# 🗂️ 3. TOP-LEVEL NAVIGATION STRUCTURE
# ==============================================================================
st.title("🏙️ Prague Urban Air Quality Explorer")
st.write("An interactive interface combining machine learning forecasts with historical measurements.")

tab1, tab2 = st.tabs(["🔮 Microclimate Simulator", "📊 Historical Visualisations"])


# ==============================================================================
# 🔮 TAB 1: MICROCLIMATE SIMULATOR
# ==============================================================================
with tab1:
    st.header("Real-Time Scenario Simulator")
    st.write("Adjust meteorological features to observe how local pollution targets adjust in real-time.")
    
    # --- Sidebar Grouping A: Simulation Parameters ---
    st.sidebar.header("Simulation Inputs")
    temperature = st.sidebar.slider("Current Temperature (°C)", -10.0, 40.0, 15.0, step=0.5)
    humidity = st.sidebar.slider("Relative Humidity (%)", 10, 100, 50, step=1)
    pressure = st.sidebar.slider("Atmospheric Pressure (hPa)", 950, 1050, 1013, step=1)
    wind_speed = st.sidebar.slider("Wind Speed (m/s) [weather_F]", 0.0, 15.0, 2.5, step=0.1)
    
    # New Wind Direction element mapping to vectors
    wind_direction = st.sidebar.slider(
        "Wind Direction (Degrees °)", 
        min_value=0, max_value=360, value=180, step=5, 
        help="0°=North, 90°=East, 180°=South, 270°=West"
    )
    rain = st.sidebar.slider("Precipitation Intensity (mm)", 0.0, 10.0, 0.0, step=0.1)
    
    station = st.sidebar.selectbox(
        "Target Air Monitoring Station", 
        options=["Praha 2-Legerova (hot spot)", "Praha 10-Průmyslová", "Praha 8-Kobylisy", "Praha 9-Vysočany", "Praha 6-Suchdol", "Praha 7-Holešovice"]
    )

    # --- SIMULATOR MATHEMATICS ENGINE ---
    # Convert direction angle into continuous grid vectors matching your pipeline features
    now = pd.Timestamp.now()
    radians = np.deg2rad(wind_direction)
    u_vector = -wind_speed * np.sin(radians)
    v_vector = -wind_speed * np.cos(radians)

    # Build comprehensive weather profiles, scaling lags dynamically to eliminate NaN errors
    payload = {
        "weather_T": temperature, "weather_H": humidity, "weather_P": pressure, "weather_F": wind_speed,
        "wind_u": u_vector, "wind_v": v_vector,
        
        # Scaling highly critical features (Rank 1 and Rank 3 Feature Importances)
        "weather_Fmax": wind_speed * 1.3, 
        "weather_TMA": temperature + 2.0, 
        "weather_TMI": temperature - 2.0,
        
        # Mapping wind parameters cleanly 
        "weather_D": wind_direction, "weather_Dprum": wind_direction, "weather_Dmax": (wind_direction + 15) % 360,
        "wind_u_lag1": u_vector * 0.9, "wind_u_mean3": u_vector * 0.95, "wind_u_mean6": u_vector,
        "wind_v_lag1": v_vector * 0.9, "wind_v_mean3": v_vector * 0.95, "wind_v_mean6": v_vector,
        
        # Weather historical offsets
        "weather_F_lag1": wind_speed * 0.9, "weather_F_mean3": wind_speed * 0.95, "weather_F_mean6": wind_speed,
        "weather_T_lag1": temperature - 0.5, "weather_T_mean3": temperature, "weather_T_mean6": temperature + 0.5,
        "weather_H_lag1": humidity, "weather_H_mean3": humidity, "weather_H_mean6": humidity,
        "weather_P_lag1": pressure, "weather_P_mean3": pressure, "weather_P_mean6": pressure,
        "weather_P_change6": -0.5 if wind_speed > 5 else 0.0,
        
        # Safe precipitation mapping preventing linear compounding spikes
        "weather_SRA10M": rain, "precip_sum3": rain, "precip_sum6": rain, "precip_sum12": rain, "precip_sum24": rain,
        
        # Human time contexts
        "hour": now.hour, "dayofweek": now.dayofweek, "is_weekend": 1 if now.dayofweek in [5, 6] else 0,
        "month": now.month, "dayofyear": now.dayofyear, "distance_km": 1.2
    }

    # --- MANUAL ONE-HOT ENCODING MAPPING ---
    # Manually maps strings to 1.0 or 0.0 binary vectors to fulfill model columns
    all_stations = ["Praha 2-Legerova (hot spot)", "Praha 10-Průmyslová", "Praha 8-Kobylisy", "Praha 9-Vysočany", "Praha 6-Suchdol", "Praha 7-Holešovice"]
    for s in all_stations:
        payload[f"air_station_name_{s}"] = 1.0 if s == station else 0.0
        
    all_weather_stations = ["Praha, Klementinum", "Praha, Libuš"]
    for ws in all_weather_stations:
        if "Legerova" in station or "Holešovice" in station:
            payload[f"weather_station_name_{ws}"] = 1.0 if "Klementinum" in ws else 0.0
        else:
            payload[f"weather_station_name_{ws}"] = 0.0 if "Klementinum" in ws else 1.0

    # Execute predictions through your wrapper logic
    input_df = pd.DataFrame([payload])
    predictions = predict_bundle(bundle, input_df).iloc[0]

    # --- DISPLAY SIMULATION METRIC BLOCKS ---
    st.markdown("### Simulated Air Quality Estimates")
    col1, col2 = st.columns(2)
    
    pm10_value = predictions.get('air_PM10', 0.0)
    no2_value = predictions.get('air_NO2', 0.0)
    
    col1.metric(
        label="Predicted PM10 Concentration", 
        value=f"{pm10_value:.2f} µg/m³",
        delta="- Good Air" if pm10_value < 50 else "+ Warning: High Dust",
        delta_color="inverse"
    )
    col2.metric(
        label="Predicted NO2 Concentration", 
        value=f"{no2_value:.2f} µg/m³",
        delta="- Safe" if no2_value < 40 else "+ Traffic Threshold Danger",
        delta_color="inverse"
    )


# ==============================================================================
# 📊 TAB 2: HISTORICAL VISUALISATIONS
# ==============================================================================
with tab2:
    st.header("Historical Observations Viewer")
    st.write("Plot long-term metrics directly from aggregated dataset tables.")

    # --- Sidebar Grouping B: Graph Controls ---
    st.sidebar.markdown("---")
    st.sidebar.header("Visualization Filters")

    freq_choice = st.sidebar.selectbox(
        "Select Time Aggregation Frequency:",
        options=[('Original (Hourly)', 'h'), ('Daily Average', 'D'), ('Weekly Average', 'W'), ('Monthly Average', 'ME')],
        format_func=lambda x: x[0]
    )
    frequency = freq_choice[1]

    # Configure chronological bounds dynamically based on historical files
    min_date = df_historical['startTime'].min().date()
    max_date = df_historical['startTime'].max().date()

    st.sidebar.subheader("Select Calendar Range")
    date_range = st.sidebar.date_input(
        "Choose a single day or range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Clean date picker output variations safely
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0] if isinstance(date_range, tuple) else date_range

    # Execute time window masking
    mask = (df_historical['startTime'].dt.date >= start_date) & (df_historical['startTime'].dt.date <= end_date)
    filtered_df = df_historical[mask].copy()

    if filtered_df.empty:
        st.warning("No data records match the specified calendar timeline window!")
    else:
        # Prevent TypeError by calling numeric_only=True during aggregation loops
        resampled_df = filtered_df.set_index('startTime').resample(frequency).mean(numeric_only=True).reset_index()

        # Build structural Matplotlib plot figure
        fig, ax = plt.subplots(figsize=(12, 4.5))
        
        # Automatically toggle markers on to retain visibility during small time scales
        use_marker = 'o' if (len(resampled_df) < 50 or frequency in ['W', 'ME']) else None
        
        ax.plot(
            resampled_df['startTime'], 
            resampled_df['weather_TMA'], 
            color='#1B5E20', 
            linestyle='-', 
            marker=use_marker, 
            linewidth=2
        )
        
        ax.set_title(f"Max Daily Air Temperature Pattern ({start_date} to {end_date})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Timeline")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        plt.xticks(rotation=35)
        plt.tight_layout()

        # Send figure natively to the Streamlit window frame
        st.pyplot(fig)
        
        # Display contextual window summary statistics
        v1, v2 = st.columns(2)
        v1.metric("Highest Temperature in Period", f"{resampled_df['weather_TMA'].max():.1f} °C")
        v2.metric("Lowest Temperature in Period", f"{resampled_df['weather_TMA'].min():.1f} °C")