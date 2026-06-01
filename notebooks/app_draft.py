import streamlit as st
import pandas as pd
import numpy as np

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

# Predictions
# TODO: We will replace this function with joblib.load('your_model.pkl') later!
def predict_pollution_dummy(w_speed, temp, hum, rain, stat):
    # This is fake math just to make the dashboard react to your sliders today
    base_pm10 = 45.0
    base_no2 = 35.0
    
    # Simulate wind dilution (higher wind = lower pollution)
    pm10_pred = base_pm10 - (w_speed * 2) - (rain * 3) + (hum * 0.1)
    no2_pred = base_no2 - (w_speed * 3) - (temp * 0.2)
    
    # Simulate the Legerova street canyon effect
    if "Legerova" in stat:
        pm10_pred += 15
        no2_pred += 25
        
    return max(5.0, pm10_pred), max(5.0, no2_pred) # Prevent negative numbers

# Run the prediction
pred_pm10, pred_no2 = predict_pollution_dummy(wind_speed, temperature, humidity, rain, station)

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

st.caption("Note: Predictions are generated in real-time based on the meteorological parameters provided in the sidebar.")


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