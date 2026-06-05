from pathlib import Path
import sys

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_wrapper import load_model_bundle, predict_bundle

# Set the page configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Prague Microclimate Engine", page_icon="🌤️", layout="wide")

st.sidebar.header("⚙️ Microclimate Simulator")
st.sidebar.markdown("Adjust the meteorological conditions to see how pollution disperses or stagnates.")

# Create sliders for the weather variables based on Prague's typical ranges
wind_speed = st.sidebar.slider("Wind Speed (m/s) [weather_F]", min_value=0.0, max_value=15.0, value=2.5, step=0.1)
temperature = st.sidebar.slider("Temperature (°C) [weather_T]", min_value=-15.0, max_value=35.0, value=10.0, step=0.5)
humidity = st.sidebar.slider("Relative Humidity (%) [weather_H]", min_value=20, max_value=100, value=65, step=1)
pressure = st.sidebar.slider("Pressure (hPa) [weather_P]", min_value=980.0, max_value=1040.0, value=1015.0, step=1.0)
rain = st.sidebar.slider("Precipitation (mm/10m) [weather_SRA10M]", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

# Dropdown for the station
station = st.sidebar.selectbox(
    "Select Target Location", 
    ["Praha 2-Legerova (hot spot)", "Praha 4-Libuš", "Praha 1-n. Republiky", "Praha 8-Karlín"]
)


# Dashboard
st.title("Prague Air Quality & Microclimate Dashboard")
st.markdown("""
This interactive engine uses a **Random Forest Machine Learning model** to predict pollution spikes based on localized weather conditions.
Move the sliders in the sidebar to simulate different atmospheric scenarios.
""")

st.divider()

@st.cache_resource
def get_model_bundle():
    model_path = PROJECT_ROOT / "results" / "models"
    return load_model_bundle(model_path)


def predict_pollution(w_speed, temp, hum, pressure, rain, stat):
    bundle = get_model_bundle()
    
    # Calculate the exact time variables the model was trained on
    now = pd.Timestamp.now()
    hour = now.hour
    dayofweek = now.dayofweek
    is_weekend = 1 if dayofweek in [5, 6] else 0
    month = now.month
    dayofyear = now.dayofyear

    # FIX: Change the keys to use the 'weather_' prefix so the model recognizes them!
    input_data = pd.DataFrame([
        {
            "weather_T": temp,
            "weather_TMA": temp + 1.0,
            "weather_TMI": temp - 1.0,
            "weather_P": pressure,
            "weather_H": hum,
            "weather_F": w_speed,
            "weather_Fmax": min(15.0, w_speed + 2.0),
            "weather_D": 0.0,
            "weather_Dprum": 0.0,
            "weather_Dmax": 0.0,
            "weather_SRA10M": rain,
            "hour": hour,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "month": month,
            "dayofyear": dayofyear,
            "air_station_name": stat,
            "weather_station_name": stat,
            "distance_km": 1.2
        }
    ])

    # Pass the correctly formatted data into the bundle
    predictions = predict_bundle(bundle, input_data).iloc[0]
    return float(predictions.get("air_PM10", 0.0)), float(predictions.get("air_NO2", 0.0))

# Run the prediction
pred_pm10, pred_no2 = predict_pollution(wind_speed, temperature, humidity, pressure, rain, station)

# Display results
st.subheader(f"Predicted Airborne Concentrations for: {station}")

# Use Streamlit columns to display numbers side-by-side
col1, col2 = st.columns(2)

with col1:
    # EU limit for PM10 is 50. We color it red if it crosses the threshold.
    delta_color = "inverse" if pred_pm10 > 50 else "normal"
    st.metric(
        label="PM10 Concentration (µg/m³)", 
        value=f"{pred_pm10:.1f}", 
        delta="Above EU Limit (50)" if pred_pm10 > 50 else "Within Safe Limits",
        delta_color=delta_color
    )

with col2:
    delta_color_no2 = "inverse" if pred_no2 > 40 else "normal"
    st.metric(
        label="NO2 Concentration (µg/m³)", 
        value=f"{pred_no2:.1f}", 
        delta="Above EU Limit (40)" if pred_no2 > 40 else "Within Safe Limits",
        delta_color=delta_color_no2
    )

st.caption("Note: These predictions come from the saved trained model bundle and are loaded once per session.")


st.subheader("Prague Monitoring Network")
st.markdown("Zoom and pan to explore the spatial distribution of our active sensor network.")

# Create a tiny dataframe of your 10 final stations and their coordinates
# (You can extract this dynamically from your CHMI metadata table)
map_data = pd.DataFrame({
    'station': ["Legerova", "Libuš", "N. Republiky", "Karlín"],
    'LAT': [50.0718, 50.0076, 50.0886, 50.0934],  # Replace with actual latitudes
    'LON': [14.4312, 14.4468, 14.4285, 14.4533]   # Replace with actual longitudes
})

# Streamlit automatically plots these coordinates on a dark-mode interactive map
st.map(map_data, zoom=11)