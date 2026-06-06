from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydeck as pdk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_wrapper import load_model_bundle, predict_bundle

# Set the page configuration
st.set_page_config(page_title="Prague Microclimate Engine", page_icon="🌤️", layout="wide")

# CENTRALIZED VARIABLE CONFIGURATION MAP ----
POLLUTANT_MAP = {
    "air_PM10": {"label": "PM10 Concentration (µg/m³)", "short": "PM10", "limit": 50.0, "status_text": "Above EU Limit (50)"}, # daily limit
    "air_NO2":  {"label": "NO2 Concentration (µg/m³)",  "short": "NO2",  "limit": 200, "status_text": "Above EU Limit (200)"}, # daily limit
    "air_NOx":  {"label": "NOx Concentration (µg/m³)",  "short": "NOx",  "limit": 200, "status_text": "High Emissions"} # EU does not declare the safety limit directly
}

# LOAD HISTORICAL DATA AT THE TOP ----
# define path
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

# get the data
@st.cache_data
def load_historical_data():
    if not PROCESSED_DATA_PATH.exists():
        st.error(f"Could not find processed data at: {PROCESSED_DATA_PATH}")
        st.stop()
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['startTime'] = pd.to_datetime(df['startTime'])
    df = df.sort_values(by='startTime').reset_index(drop=True)
    return df

df_historical = load_historical_data()
available_stations = df_historical['air_station_name'].unique()


# MASTER DASHBOARD NAVIGATION ----
st.title("Prague Air Quality Hub")
current_view = st.radio(
    "Navigate Workspace Views:",
    options=["🔮 Microclimate Simulator", "📊 Historical Visualizations"],
    horizontal=True,
    label_visibility="collapsed" 
)
st.write("---")


# VIEW 1: SCENARIO SIMULATOR ----
if current_view == "🔮 Microclimate Simulator":
    st.header("Real-Time Scenario Simulator")
    st.write("Adjust the weather parameters in the sidebar to simulate pollution levels.")

    ## SIDEBAR CONTROLS ----
    st.sidebar.markdown("## Simulation Controls")
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", min_value=0.0, max_value=15.0, value=2.5, step=0.1)
    wind_direction = st.sidebar.slider("Wind Direction (Degrees °)", min_value=0, max_value=360, value=180, step=5, help="0°=North, 90°=East, 180°=South, 270°=West")
    temperature = st.sidebar.slider("Temperature (°C)", min_value=-15.0, max_value=35.0, value=10.0, step=0.5)
    humidity = st.sidebar.slider("Relative Humidity (%)", min_value=20, max_value=100, value=65, step=1)
    pressure = st.sidebar.slider("Pressure (hPa)", min_value=980.0, max_value=1040.0, value=1015.0, step=1.0)
    rain = st.sidebar.slider("Precipitation (mm/10m)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

    station = st.sidebar.selectbox(
        "Select Target Location", 
        options=available_stations
    )

    # caption for the sliders
    st.markdown("""
    This interactive engine uses a **Random Forest Machine Learning model** to predict pollution spikes based on localized weather conditions.
    Move the sliders in the sidebar to simulate different atmospheric scenarios.
    """)

    st.divider()

    @st.cache_resource
    def get_model_bundle():
        model_path = PROJECT_ROOT / "results" / "models"
        return load_model_bundle(model_path)

    ## PREDICTION FUNCTION ----
    def predict_pollution(w_speed, wind_dir, temp, hum, press, rain_val, stat):
        # refer to the model bundle
        bundle = get_model_bundle()
        # preprocess data
        now = pd.Timestamp.now()
        hour = now.hour
        dayofweek = now.dayofweek
        is_weekend = 1 if dayofweek in [5, 6] else 0
        month = now.month
        dayofyear = now.dayofyear
        radians = np.deg2rad(wind_dir)
        u_vector = -w_speed * np.sin(radians)
        v_vector = -w_speed * np.cos(radians)

        # define the values that will be used for predictions
        payload = {
            "weather_T": temp, "weather_H": hum, "weather_P": press, "weather_F": w_speed,
            "wind_u": u_vector, "wind_v": v_vector,
            "weather_Fmax": w_speed * 1.3, "weather_TMA": temp + 2.0, "weather_TMI": temp - 2.0,
            "weather_D": wind_dir, "weather_Dprum": wind_dir, "weather_Dmax": (wind_dir + 15) % 360,
            "wind_u_lag1": u_vector * 0.9, "wind_u_mean3": u_vector * 0.95, "wind_u_mean6": u_vector,
            "wind_v_lag1": v_vector * 0.9, "wind_v_mean3": v_vector * 0.95, "wind_v_mean6": v_vector,
            "weather_F_lag1": w_speed * 0.9, "weather_F_mean3": w_speed * 0.95, "weather_F_mean6": w_speed,
            "weather_T_lag1": temp - 0.5, "weather_T_mean3": temp, "weather_T_mean6": temp + 0.5,
            "weather_H_lag1": hum, "weather_H_mean3": hum, "weather_H_mean6": hum,
            "weather_P_lag1": press, "weather_P_mean3": press, "weather_P_mean6": press,
            "weather_P_change6": -0.5 if w_speed > 5 else 0.0,
            "weather_SRA10M": rain_val, "precip_sum3": rain_val, "precip_sum6": rain_val,
            "precip_sum12": rain_val, "precip_sum24": rain_val,
            "hour": hour, "dayofweek": dayofweek, "is_weekend": is_weekend, "month": month, "dayofyear": dayofyear,
            "distance_km": 1.2
        }

        matched_rows = df_historical[df_historical['air_station_name'] == stat]
        actual_weather_station = matched_rows['weather_station_name'].iloc[0]
        expected_features = bundle.get("numeric_features", []) + bundle.get("categorical_features", [])

        for feature in expected_features:
            if feature.startswith("air_station_name_"):
                station_name_in_bundle = feature.replace("air_station_name_", "")
                payload[feature] = 1.0 if station_name_in_bundle == stat else 0.0
            elif feature.startswith("weather_station_name_"):
                w_station_name_in_bundle = feature.replace("weather_station_name_", "")
                payload[feature] = 1.0 if w_station_name_in_bundle == actual_weather_station else 0.0

        input_data = pd.DataFrame([payload])
        predictions = predict_bundle(bundle, input_data).iloc[0]
        
        # return predictions
        return float(predictions.get("air_PM10", 0.0)), float(predictions.get("air_NO2", 0.0)), float(predictions.get("air_NOx", 0.0))

    pred_pm10, pred_no2, pred_nox = predict_pollution(wind_speed, wind_direction, temperature, humidity, pressure, rain, station)

    ## PRINT THE RESULTS ----
    st.subheader(f"Predicted Airborne Concentrations for: {station}")
    col1, col2, col3 = st.columns(3)
    with col1:
        meta_pm10 = POLLUTANT_MAP["air_PM10"]
        delta_color_pm10 = "inverse" if pred_pm10 > meta_pm10["limit"] else "normal"
        st.metric(label=meta_pm10["label"], value=f"{pred_pm10:.1f}", delta=meta_pm10["status_text"] if pred_pm10 > meta_pm10["limit"] else "Within Safe Limits", delta_color=delta_color_pm10)
    with col2:
        meta_no2 = POLLUTANT_MAP["air_NO2"]
        delta_color_no2 = "inverse" if pred_no2 > meta_no2["limit"] else "normal"
        st.metric(label=meta_no2["label"], value=f"{pred_no2:.1f}", delta=meta_no2["status_text"] if pred_no2 > meta_no2["limit"] else "Within Safe Limits", delta_color=delta_color_no2)
    with col3:
        meta_nox = POLLUTANT_MAP["air_NOx"]
        delta_color_nox = "inverse" if pred_nox > meta_nox["limit"] else "normal"
        st.metric(label=meta_nox["label"], value=f"{pred_nox:.1f}", delta=meta_nox["status_text"] if pred_nox > meta_nox["limit"] else "Normal Baseline", delta_color=delta_color_nox)
   
    # add disclaimer 
    with st.expander("Disclaimer & Model Approximations"):
        st.markdown("""
        **About These Predictions:**
        These values are synthesized in real-time using a saved **Random Forest Regressor** bundle trained on historical Prague microclimate observations. 
        
        **Methodology & Feature Abstraction:**
        To provide a clean user experience, the control sidebar exposes only the primary meteorological variables. However, the underlying machine learning framework relies on complex spatiotemporal dependencies, including:
        * **Lagged Features:** Prior 1-hour atmospheric states ($t-1$).
        * **Rolling Metrics:** 3-hour and 6-hour moving averages for wind vectors ($u, v$) and thermodynamic metrics.
        * **Precipitation Washout Accumulations:** Trailing 3, 6, 12, and 24-hour rainfall sums.
        
        **Simulation Assumptions:**
        Because a user cannot manually input past conditions, the engine assumes a **quasi-steady state simulation scenario**. Lagged and rolling metrics are dynamically extrapolated as mathematical functions of your active slider inputs.
        
        Therefore, these outputs represent **approximate localized atmospheric responses**.
        """)

    ## MAP OF THE AIR STATIONS ----
    st.subheader("Prague Monitoring Network")
    st.markdown("Explore the exact location of the selected monitoring station across our active sensor network.")

    lat_col = 'air_lat' if 'air_lat' in df_historical.columns else 'latitude' if 'latitude' in df_historical.columns else 'LAT'
    lon_col = 'air_lon' if 'air_lon' in df_historical.columns else 'longitude' if 'longitude' in df_historical.columns else 'LON'

    # map all stations from the prepared all_stations_df
    all_stations_df = df_historical[["air_station_name", lat_col, lon_col]].dropna().drop_duplicates(subset=["air_station_name"]).copy()
    all_stations_df["is_selected"] = all_stations_df["air_station_name"] == station
    all_stations_df["radius"] = all_stations_df["is_selected"].map(lambda x: 180 if x else 60)
    all_stations_df["color_r"] = all_stations_df["is_selected"].map(lambda x: 216 if x else 30)
    all_stations_df["color_g"] = all_stations_df["is_selected"].map(lambda x: 27 if x else 136)
    all_stations_df["color_b"] = all_stations_df["is_selected"].map(lambda x: 96 if x else 229)
    all_stations_df["color_a"] = all_stations_df["is_selected"].map(lambda x: 255 if x else 140)

    selected_row = all_stations_df[all_stations_df["is_selected"]].iloc[0]
    center_lat = float(selected_row[lat_col])
    center_lon = float(selected_row[lon_col])

    # adjust the visual 
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11.8, pitch=0)
    layer = pdk.Layer(
        "ScatterplotLayer", all_stations_df, pickable=True, opacity=0.8, stroked=True, filled=True,
        radius_scale=1, radius_min_pixels=5, radius_max_pixels=25, line_width_min_pixels=1,
        get_position=[lon_col, lat_col], get_radius="radius",
        get_fill_color=["color_r", "color_g", "color_b", "color_a"], get_line_color=[17, 17, 17, 200],
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=pdk.map_styles.CARTO_LIGHT, tooltip={"text": "Station: {air_station_name}"}))



# 📊 VIEW 2: HISTORICAL VISUALIZATIONS & SPATIAL COMPARISON ----
else:
    # get rid of the slider for this view
    st.sidebar.markdown("### ℹ️ Navigation info")
    st.sidebar.write("Simulation options are hidden while parsing historical timelines.")

    st.header("Historical Air Quality & Weather Analysis")

    # add description
    st.write("Analyze how changing weather conditions interact with city pollution levels over time.")

    ## FIGURE 1: DUAL AXIS CHARTS FOR THE WEATHER AND POLLUTION TIME SERIES ----
    st.subheader("Weather Conditions and Air Pollution Co-movements")
   
    # define what the user can adjust
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        selected_viz_station = st.selectbox("Filter by Station", options=available_stations, key="viz_station")
    with col2:
        pollutant = st.selectbox("Select Target Pollutant", options=list(POLLUTANT_MAP.keys()), format_func=lambda x: POLLUTANT_MAP[x]["label"])
    with col3:
        weather_var = st.selectbox(
            "Select Weather Overlay", 
            options=[
                ('Temperature (°C)', 'weather_T'), ('Wind Speed (m/s)', 'weather_F'), 
                ('Humidity (%)', 'weather_H'), ('Atmospheric Pressure (hPa)', 'weather_P')
            ],
            format_func=lambda x: x[0]
        )
        selected_weather_col = weather_var[1]
        selected_weather_label = weather_var[0]
    with col4:
        freq_choice = st.selectbox(
            "Time Aggregation Frequency",
            options=[('Hourly (Raw Data)', 'RAW'), ('Daily Average', 'D'), ('Weekly Average', 'W'), ('Monthly Average', 'ME')],
            format_func=lambda x: x[0]
        )
        frequency = freq_choice[1]
    with col5:
        min_date = df_historical['startTime'].min().date()
        max_date = df_historical['startTime'].max().date()
        date_range = st.date_input("Select Date Period", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    viz_mask = (df_historical['air_station_name'] == selected_viz_station) & (df_historical['startTime'].dt.date >= start_date) & (df_historical['startTime'].dt.date <= end_date)
    viz_df = df_historical[viz_mask].copy()

    # warning just to be careful
    if viz_df.empty:
        st.warning(f"No historical records found for {selected_viz_station} within the selected window.")
    else:
        if frequency == 'RAW':
            resampled_viz = viz_df.sort_values('startTime').reset_index(drop=True)
        else:
            resampled_viz = viz_df.set_index('startTime').resample(frequency).mean(numeric_only=True).reset_index()

        fig, ax1 = plt.subplots(figsize=(14, 6))
        use_marker = 'o' if (frequency != 'RAW' and len(resampled_viz) < 60) else None
        
        current_meta = POLLUTANT_MAP[pollutant]
        pollutant_label = current_meta["label"]
        legal_threshold = current_meta["limit"]
        line_color1 = '#1E88E5' if pollutant == "air_PM10" else '#D81B60' if pollutant == "air_NO2" else '#8E24AA'

        line1 = ax1.plot(resampled_viz['startTime'], resampled_viz[pollutant], color=line_color1, linestyle='-', marker=use_marker, linewidth=1.2 if frequency == 'RAW' else 2.5, label=pollutant_label)
        ax1.set_xlabel("Timeline Window", fontweight='bold', labelpad=10)
        ax1.set_ylabel(pollutant_label, color=line_color1, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=line_color1)
        ax1.grid(True, linestyle='--', alpha=0.3)
        thresh_line = ax1.axhline(y=legal_threshold, color='#E53935', linestyle=':', linewidth=1.5, label=f"Benchmark Limit ({int(legal_threshold)})")
        
        ax2 = ax1.twinx() 
        line_color2 = '#FF8F00' 
        line2 = ax2.plot(resampled_viz['startTime'], resampled_viz[selected_weather_col], color=line_color2, linestyle='--', marker=use_marker, linewidth=1.0 if frequency == 'RAW' else 1.8, alpha=0.5 if frequency == 'RAW' else 0.8, label=selected_weather_label)
        ax2.set_ylabel(selected_weather_label, color=line_color2, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=line_color2)
        
        all_lines = line1 + line2 + [thresh_line]
        all_labels = [l.get_label() for l in all_lines]
        ax1.legend(all_lines, all_labels, loc="upper right")
        plt.title(f"Correlation Analysis: {current_meta['short']} vs {selected_weather_label} at {selected_viz_station}\n({start_date} to {end_date})", fontsize=14, fontweight='bold', pad=15)
        ax1.tick_params(axis='x', rotation=25)
        fig.tight_layout()
        st.pyplot(fig)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(f"Avg {current_meta['short']} Level", f"{resampled_viz[pollutant].mean():.1f} µg/m³")
        unit = selected_weather_label.split("(")[-1].split(")")[0] if "(" in selected_weather_label else ""
        clean_weather_name = selected_weather_label.split("(")[0].strip()
        m_col2.metric(label=f"Avg {clean_weather_name}", value=f"{resampled_viz[selected_weather_col].mean():.1f} {unit}".strip())
        correlation = resampled_viz[pollutant].corr(resampled_viz[selected_weather_col])
        m_col3.metric("Correlation Coefficient (r)", f"{correlation:.2f}" if not np.isnan(correlation) else "N/A")

        # FIGURE 2: MULTI-STATION COMPARISON PLOT ----
        st.write("---")
        st.subheader("Cross-City Neighborhood Comparison")
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            comp_pollutant = st.selectbox("Select Pollutant to Compare", options=list(POLLUTANT_MAP.keys()), format_func=lambda x: POLLUTANT_MAP[x]["label"], key="comp_poll")
        with comp_col2:
            comp_frequency = st.selectbox("Aggregation Frequency", options=[('Hourly (Raw Data)', 'RAW'), ('Daily Average', 'D'), ('Weekly Average', 'W'), ('Monthly Average', 'ME')], format_func=lambda x: x[0], key="comp_freq")[1]
        with comp_col3:
            comp_date_range = st.date_input("Select Comparison Period", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="comp_date_range")

        comp_start, comp_end = comp_date_range if isinstance(comp_date_range, tuple) and len(comp_date_range) == 2 else (min_date, max_date)
        comp_mask = (df_historical['startTime'].dt.date >= comp_start) & (df_historical['startTime'].dt.date <= comp_end)
        comp_df = df_historical[comp_mask].copy()

        if not comp_df.empty:
            fig_comp, ax_comp = plt.subplots(figsize=(14, 6))
            all_city_stations = comp_df['air_station_name'].unique()
            colors = plt.cm.get_cmap('tab10', len(all_city_stations))

            for idx, current_station in enumerate(all_city_stations):
                station_slice = comp_df[comp_df['air_station_name'] == current_station].copy()
                station_resampled = station_slice.sort_values('startTime').reset_index(drop=True) if comp_frequency == 'RAW' else station_slice.set_index('startTime').resample(comp_frequency).mean(numeric_only=True).reset_index()
                comp_marker = 'o' if (comp_frequency != 'RAW' and len(station_resampled) < 40) else None
                ax_comp.plot(station_resampled['startTime'], station_resampled[comp_pollutant], label=current_station, color=colors(idx), linewidth=1.2 if comp_frequency == 'RAW' else 2.2, marker=comp_marker, alpha=0.7 if comp_frequency == 'RAW' else 0.9)

            comp_meta = POLLUTANT_MAP[comp_pollutant]
            ax_comp.set_title(f"Comparative Spatial Analysis: {comp_meta['short']} Levels Across Prague Districts\n({comp_start} to {comp_end})", fontsize=14, fontweight='bold', pad=15)
            ax_comp.set_xlabel("Timeline Window", fontweight='bold', labelpad=10)
            ax_comp.set_ylabel(comp_meta["label"], fontweight='bold')
            ax_comp.grid(True, linestyle='--', alpha=0.3)
            ax_comp.axhline(y=comp_meta["limit"], color='#E53935', linestyle=':', linewidth=1.5, label=f"Benchmark Boundary ({int(comp_meta['limit'])})")
            ax_comp.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="Monitoring Stations")
            plt.xticks(rotation=25)
            fig_comp.tight_layout()
            st.pyplot(fig_comp)

        # FIGURE 3: POLLUTANT WIND ROSE ANALYSIS ---
        try:
            from windrose import WindroseAxes
            st.write("---")
            st.subheader("Meteorological Dispersion Analysis (Wind Rose)")
            rose_col1, rose_col2, rose_col3 = st.columns(3)
            with rose_col1:
                rose_station = st.selectbox("Select Station for Wind Rose", options=available_stations, key="rose_stat")
            with rose_col2:
                rose_pollutant = st.selectbox("Select Target Pollutant", options=list(POLLUTANT_MAP.keys()), format_func=lambda x: POLLUTANT_MAP[x]["label"], key="rose_poll")
            with rose_col3:
                rose_date_range = st.date_input("Select Wind Rose Period", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="rose_date_range")

            rose_start, rose_end = rose_date_range if isinstance(rose_date_range, tuple) and len(rose_date_range) == 2 else (min_date, max_date)
            rose_mask = (df_historical['air_station_name'] == rose_station) & (df_historical['startTime'].dt.date >= rose_start) & (df_historical['startTime'].dt.date <= rose_end)
            rose_df = df_historical[rose_mask].dropna(subset=['weather_Dprum', rose_pollutant]).copy()

            if not rose_df.empty:
                corresponding_weather_station = rose_df["weather_station_name"].iloc[0]
                fig_rose = plt.figure(figsize=(7, 7))
                ax_rose = WindroseAxes.from_ax(fig=fig_rose)
                ax_rose.bar(rose_df['weather_Dprum'], rose_df[rose_pollutant], normed=True, opening=0.85, edgecolor='white', cmap=cm.YlOrRd, nsector=16)
                rose_meta = POLLUTANT_MAP[rose_pollutant]
                ax_rose.set_legend(title=f"{rose_meta['short']} ($\mu g/m^3$)", bbox_to_anchor=(1.15, 0.95))
                plt.title(f"Pollution Wind Rose: {rose_station}\nWeather Station Profile: {corresponding_weather_station}\n({rose_start} to {rose_end})", fontsize=8, fontweight='bold', pad=25)
                st.pyplot(fig_rose)
        # make sure package is installed
        except ImportError:
            st.error("The `windrose` package is not installed. Run `pip install windrose` to enable this feature.")