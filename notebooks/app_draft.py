from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from windrose import WindroseAxes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_wrapper import load_model_bundle, predict_bundle

# Set the page configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Prague Microclimate Engine", page_icon="🌤️", layout="wide")
st.title("Prague Air Quality Hub")

# --- 🛠️ LOAD HISTORICAL DATA AT THE TOP ---
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

@st.cache_data
def load_historical_data():
    if not PROCESSED_DATA_PATH.exists():
        st.error(f"Could not find processed data at: {PROCESSED_DATA_PATH}")
        st.stop()
    df = pd.read_csv(PROCESSED_DATA_PATH)
    # Ensure startTime is datetime format for visualization logic
    df['startTime'] = pd.to_datetime(df['startTime'])
    df = df.sort_values(by='startTime').reset_index(drop=True)
    return df

df_historical = load_historical_data()


# Create the main navigation tabs
tab1, tab2 = st.tabs(["🔮 Microclimate Simulator", "📊 Historical Visualizations"])

# ==============================================================================
# 🔮 TAB 1: SIMULATOR (Fully complete and optimized)
# ==============================================================================
with tab1:
    st.header("Real-Time Scenario Simulator")
    st.write("Adjust the weather parameters in the sidebar to simulate pollution levels.")

    st.sidebar.markdown("## Simulation Controls")
    wind_speed = st.sidebar.slider("Wind Speed (m/s) [weather_F]", min_value=0.0, max_value=15.0, value=2.5, step=0.1)
    wind_direction = st.sidebar.slider("Wind Direction (Degrees °)", min_value=0, max_value=360, value=180, step=5, help="0°=North, 90°=East, 180°=South, 270°=West")
    temperature = st.sidebar.slider("Temperature (°C) [weather_T]", min_value=-15.0, max_value=35.0, value=10.0, step=0.5)
    humidity = st.sidebar.slider("Relative Humidity (%) [weather_H]", min_value=20, max_value=100, value=65, step=1)
    pressure = st.sidebar.slider("Pressure (hPa) [weather_P]", min_value=980.0, max_value=1040.0, value=1015.0, step=1.0)
    rain = st.sidebar.slider("Precipitation (mm/10m) [weather_SRA10M]", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

    available_stations = df_historical['air_station_name'].unique()

    station = st.sidebar.selectbox(
        "Select Target Location", 
        options=available_stations
    )

    st.markdown("""
    This interactive engine uses a **Random Forest Machine Learning model** to predict pollution spikes based on localized weather conditions.
    Move the sliders in the sidebar to simulate different atmospheric scenarios.
    """)

    st.divider()

    @st.cache_resource
    def get_model_bundle():
        model_path = PROJECT_ROOT / "results" / "models"
        return load_model_bundle(model_path)

    def predict_pollution(w_speed, wind_dir, temp, hum, press, rain_val, stat):
        bundle = get_model_bundle()
        now = pd.Timestamp.now()
        hour = now.hour
        dayofweek = now.dayofweek
        is_weekend = 1 if dayofweek in [5, 6] else 0
        month = now.month
        dayofyear = now.dayofyear

        radians = np.deg2rad(wind_dir)
        u_vector = -w_speed * np.sin(radians)
        v_vector = -w_speed * np.cos(radians)

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
            "weather_SRA10M": rain_val,
            "precip_sum3": rain_val,   
            "precip_sum6": rain_val,
            "precip_sum12": rain_val,
            "precip_sum24": rain_val,
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
        
        return float(predictions.get("air_PM10", 0.0)), float(predictions.get("air_NO2", 0.0))

    pred_pm10, pred_no2 = predict_pollution(wind_speed, wind_direction, temperature, humidity, pressure, rain, station)

    st.subheader(f"Predicted Airborne Concentrations for: {station}")
    col1, col2 = st.columns(2)
    with col1:
        delta_color = "inverse" if pred_pm10 > 50 else "normal"
        st.metric(label="PM10 Concentration (µg/m³)", value=f"{pred_pm10:.1f}", delta="Above EU Limit (50)" if pred_pm10 > 50 else "Within Safe Limits", delta_color=delta_color)
    with col2:
        delta_color_no2 = "inverse" if pred_no2 > 40 else "normal"
        st.metric(label="NO2 Concentration (µg/m³)", value=f"{pred_no2:.1f}", delta="Above EU Limit (40)" if pred_no2 > 40 else "Within Safe Limits", delta_color=delta_color_no2)

    st.caption("Note: These predictions come from the saved trained model bundle and are loaded once per session.")

    st.subheader("Prague Monitoring Network")
    st.markdown("Explore the exact location of the selected monitoring station across our active sensor network.")

    station_rows = df_historical[df_historical['air_station_name'] == station]
    lat_col = 'air_lat' if 'air_lat' in df_historical.columns else 'latitude' if 'latitude' in df_historical.columns else 'LAT'
    lon_col = 'air_lon' if 'air_lon' in df_historical.columns else 'longitude' if 'longitude' in df_historical.columns else 'LON'

    actual_lat = float(station_rows[lat_col].iloc[0])
    actual_lon = float(station_rows[lon_col].iloc[0])

    map_data = pd.DataFrame({'lat': [actual_lat], 'lon': [actual_lon]})
    st.map(map_data, zoom=13)


# ==============================================================================
# 📊 TAB 2: HISTORICAL VISUALIZATIONS & SPATIAL COMPARISON
# ==============================================================================
with tab2:
    st.header("Historical Air Quality & Weather Analysis")
    st.write("Explore long-term air quality trends and weather relationships in Prague.")

    st.write("---")
    st.subheader("Weather Conditions and Air Pollution Co-movements")
   
    # 1. UI Filter Controls for Page Layout (Primary Double-Axis Chart Filters)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        selected_viz_station = st.selectbox("Filter by Station", options=available_stations, key="viz_station")
    with col2:
        pollutant = st.selectbox("Select Target Pollutant", ["air_PM10", "air_NO2"])
    with col3:
        weather_var = st.selectbox(
            "Select Weather Overlay", 
            options=[
                ('Temperature (°C)', 'weather_T'), 
                ('Wind Speed (m/s)', 'weather_F'), 
                ('Humidity (%)', 'weather_H'), 
                ('Atmospheric Pressure (hPa)', 'weather_P')
            ],
            format_func=lambda x: x[0]
        )
        selected_weather_col = weather_var[1]
        selected_weather_label = weather_var[0]
    with col4:
        freq_choice = st.selectbox(
            "Time Aggregation Frequency",
            options=[
                ('Hourly (Raw Data)', 'RAW'),
                ('Daily Average', 'D'), 
                ('Weekly Average', 'W'), 
                ('Monthly Average', 'ME')
            ],
            format_func=lambda x: x[0]
        )
        frequency = freq_choice[1]
    with col5:
        # Extract dataset date boundaries dynamically from your dataframe
        min_date = df_historical['startTime'].min().date()
        max_date = df_historical['startTime'].max().date()
        
        # Streamlit Date Input configured to accept a range selection [start, end]
        date_range = st.date_input(
            "Select Date Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    # 2. Slice Data based on station AND selected date boundaries
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Build the filter mask for the single-station view
    viz_mask = (
        (df_historical['air_station_name'] == selected_viz_station) &
        (df_historical['startTime'].dt.date >= start_date) &
        (df_historical['startTime'].dt.date <= end_date)
    )
    viz_df = df_historical[viz_mask].copy()

    if viz_df.empty:
        st.warning(f"No historical records found for {selected_viz_station} within the selected window: {start_date} to {end_date}.")
    else:
        # 3. Handle Datetime Resampling Safely or Keep Raw
        if frequency == 'RAW':
            resampled_viz = viz_df.sort_values('startTime').reset_index(drop=True)
        else:
            resampled_viz = viz_df.set_index('startTime').resample(frequency).mean(numeric_only=True).reset_index()

        # 4. Generate the Double Axis Figure
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        use_marker = 'o' if (frequency != 'RAW' and len(resampled_viz) < 60) else None
        
        # --- AXIS 1: POLLUTION (Primary Left Y-Axis) ---
        line_color1 = '#1E88E5' if pollutant == "air_PM10" else '#D81B60'
        pollutant_label = "PM10 Concentration (µg/m³)" if pollutant == "air_PM10" else "NO2 Concentration (µg/m³)"

        line1 = ax1.plot(
            resampled_viz['startTime'], 
            resampled_viz[pollutant], 
            color=line_color1, 
            linestyle='-', 
            marker=use_marker, 
            linewidth=1.2 if frequency == 'RAW' else 2.5, 
            label=pollutant_label
        )
        
        ax1.set_xlabel("Timeline Window", fontweight='bold', labelpad=10)
        ax1.set_ylabel(pollutant_label, color=line_color1, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=line_color1)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        legal_threshold = 50.0 if pollutant == "air_PM10" else 40.0
        thresh_line = ax1.axhline(y=legal_threshold, color='#E53935', linestyle=':', linewidth=1.5, label=f"EU Limit ({int(legal_threshold)})")
        
        # --- AXIS 2: WEATHER (Secondary Right Y-Axis) ---
        ax2 = ax1.twinx() 
        
        line_color2 = '#FF8F00' 
        line2 = ax2.plot(
            resampled_viz['startTime'], 
            resampled_viz[selected_weather_col], 
            color=line_color2, 
            linestyle='--', 
            marker=use_marker, 
            linewidth=1.0 if frequency == 'RAW' else 1.8,
            alpha=0.5 if frequency == 'RAW' else 0.8, 
            label=selected_weather_label
        )
        
        ax2.set_ylabel(selected_weather_label, color=line_color2, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=line_color2)
        
        all_lines = line1 + line2 + [thresh_line]
        all_labels = [l.get_label() for l in all_lines]
        ax1.legend(all_lines, all_labels, loc="upper right")
        
        plt.title(f"Correlation Analysis: {pollutant} vs {selected_weather_label} at {selected_viz_station}\n({start_date} to {end_date})", fontsize=14, fontweight='bold', pad=15)
        ax1.tick_params(axis='x', rotation=25)
        fig.tight_layout()

        st.pyplot(fig)
        
        # --- 5. Statistical Info Columns underneath ---
        m_col1, m_col2, m_col3 = st.columns(3)
        
        m_col1.metric(f"Avg {pollutant_label.split()[0]} Level", f"{resampled_viz[pollutant].mean():.1f} µg/m³")
        
        if "(" in selected_weather_label and ")" in selected_weather_label:
            unit = selected_weather_label.split("(")[-1].split(")")[0]
            clean_weather_name = selected_weather_label.split("(")[0].strip()
        else:
            unit = "" 
            clean_weather_name = selected_weather_label
            
        m_col2.metric(
            label=f"Avg {clean_weather_name}", 
            value=f"{resampled_viz[selected_weather_col].mean():.1f} {unit}".strip()
        )
        
        correlation = resampled_viz[pollutant].corr(resampled_viz[selected_weather_col])
        corr_val = f"{correlation:.2f}" if not np.isnan(correlation) else "N/A"
        m_col3.metric("Correlation Coefficient (r)", corr_val, help="Closer to -1 or 1 means a very strong relationship.")


        # ==============================================================================
        # 📊 SECTION 2: MULTI-STATION COMPARISON PLOT (ALL STATIONS OVERLAID)
        # ==============================================================================
        st.write("---")
        st.subheader("Cross-City Neighborhood Comparison")
        st.write("Compare raw or aggregated pollution profiles across all monitoring stations simultaneously.")

        # 1. Horizontal filters for the comparison plot
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            comp_pollutant = st.selectbox("Select Pollutant to Compare", ["air_PM10", "air_NO2"], key="comp_poll")
        with comp_col2:
            comp_freq_choice = st.selectbox(
                "Aggregation Frequency",
                options=[
                    ('Hourly (Raw Data)', 'RAW'),
                    ('Daily Average', 'D'), 
                    ('Weekly Average', 'W'), 
                    ('Monthly Average', 'ME')
                ],
                format_func=lambda x: x[0],
                key="comp_freq"
            )
            comp_frequency = comp_freq_choice[1]
        with comp_col3:
            # Extract dataset date boundaries dynamically from your dataframe
            min_date = df_historical['startTime'].min().date()
            max_date = df_historical['startTime'].max().date()
            
            # Independent calendar for the comparison graph
            comp_date_range = st.date_input(
                "Select Comparison Period",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="comp_date_range"
            )

        # Break down the tuple safely into unique specific variables for Section 2
        if isinstance(comp_date_range, tuple) and len(comp_date_range) == 2:
            comp_start, comp_end = comp_date_range
        else:
            comp_start, comp_end = min_date, max_date

        # 2. Slice the historical dataset using the local section 2 variables
        comp_mask = (
            (df_historical['startTime'].dt.date >= comp_start) &
            (df_historical['startTime'].dt.date <= comp_end)
        )
        comp_df = df_historical[comp_mask].copy()

        if comp_df.empty:
            st.warning(f"No data found within the selected comparison window: {comp_start} to {comp_end}.")
        else:
            # 3. Create the multi-line figure
            fig_comp, ax_comp = plt.subplots(figsize=(14, 6))
            
            all_city_stations = comp_df['air_station_name'].unique()
            colors = plt.cm.get_cmap('tab10', len(all_city_stations))

            # 4. Loop through each station and plot its unique timeline row-by-row
            for idx, current_station in enumerate(all_city_stations):
                station_slice = comp_df[comp_df['air_station_name'] == current_station].copy()
                
                if comp_frequency == 'RAW':
                    station_resampled = station_slice.sort_values('startTime').reset_index(drop=True)
                else:
                    station_resampled = station_slice.set_index('startTime').resample(comp_frequency).mean(numeric_only=True).reset_index()
                
                comp_marker = 'o' if (comp_frequency != 'RAW' and len(station_resampled) < 40) else None
                ax_comp.plot(
                    station_resampled['startTime'],
                    station_resampled[comp_pollutant],
                    label=current_station,
                    color=colors(idx),
                    linewidth=1.2 if comp_frequency == 'RAW' else 2.2,
                    marker=comp_marker,
                    alpha=0.7 if comp_frequency == 'RAW' else 0.9
                )

            # 5. Graph Chart Aesthetics & Labeling
            pollutant_title_label = "PM10" if comp_pollutant == "air_PM10" else "NO2"
            ax_comp.set_title(f"Comparative Spatial Analysis: {pollutant_title_label} Levels Across Prague Districts\n({comp_start} to {comp_end})", fontsize=14, fontweight='bold', pad=15)
            ax_comp.set_xlabel("Timeline Window", fontweight='bold', labelpad=10)
            ax_comp.set_ylabel(f"Concentration (µg/m³)", fontweight='bold')
            ax_comp.grid(True, linestyle='--', alpha=0.3)
            
            comp_legal_limit = 50.0 if comp_pollutant == "air_PM10" else 40.0
            ax_comp.axhline(y=comp_legal_limit, color='#E53935', linestyle=':', linewidth=1.5, label=f"EU Safety Boundary ({int(comp_legal_limit)})")
            
            ax_comp.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="Monitoring Stations")
            
            plt.xticks(rotation=25)
            fig_comp.tight_layout()

            st.pyplot(fig_comp)


        # ==============================================================================
        # 🌪️ SECTION 3: POLLUTANT WIND ROSE ANALYSIS
        # ==============================================================================
        try:
            from windrose import WindroseAxes
            import matplotlib.cm as cm
            
            st.write("---")
            st.subheader("Meteorological Dispersion Analysis (Wind Rose)")
            st.write("Examine how different wind vectors affect localized pollution spikes within a selected timeframe.")

            # 1. UI Filters for the Wind Rose view
            rose_col1, rose_col2, rose_col3 = st.columns(3)
            with rose_col1:
                rose_station = st.selectbox("Select Station for Wind Rose", options=available_stations, key="rose_stat")
            with rose_col2:
                rose_pollutant = st.selectbox("Select Target Pollutant", ["air_PM10", "air_NO2"], key="rose_poll")
            with rose_col3:
                # 🌟 FIX: Independent calendar input for the wind rose layout section
                rose_date_range = st.date_input(
                    "Select Wind Rose Period",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="rose_date_range"
                )

            # 🌟 FIX: Safely break down the tuple into specific wind rose timeframe limits
            if isinstance(rose_date_range, tuple) and len(rose_date_range) == 2:
                rose_start, rose_end = rose_date_range
            else:
                rose_start, rose_end = min_date, max_date

            # 2. Extract and filter rows matching user options using local wind rose masks
            rose_mask = (
                (df_historical['air_station_name'] == rose_station) &
                (df_historical['startTime'].dt.date >= rose_start) &
                (df_historical['startTime'].dt.date <= rose_end)
            )
            rose_df = df_historical[rose_mask].copy()

            if rose_df.empty:
                st.warning(f"No records found for {rose_station} within the specified window: {rose_start} to {rose_end}.")
            else:
                corresponding_weather_station = rose_df["weather_station_name"].iloc[0]
                
                # Check for NaNs or missing entries in columns required for the polar calculations
                rose_df = rose_df.dropna(subset=['weather_Dprum', rose_pollutant])
                
                if rose_df.empty:
                    st.warning("No valid wind direction or pollutant metrics found in this specific timeframe slice.")
                else:
                    # 3. Initialize the dedicated Wind Rose structure
                    fig_rose = plt.figure(figsize=(7, 7))
                    ax_rose = WindroseAxes.from_ax(fig=fig_rose)

                    # 4. Render the polar bar distributions
                    ax_rose.bar(
                        rose_df['weather_Dprum'], 
                        rose_df[rose_pollutant], 
                        normed=True,             
                        opening=0.85,            
                        edgecolor='white',       
                        cmap=cm.YlOrRd,          
                        nsector=16               
                    )

                    # 5. Visual styling and legends
                    clean_lbl = "PM10" if rose_pollutant == "air_PM10" else "NO2"
                    ax_rose.set_legend(title=f"{clean_lbl} ($\mu g/m^3$)", bbox_to_anchor=(1.15, 0.95))
                    
                    plt.title(
                        f"Pollution Wind Rose: {rose_station}\nWeather Station Profile: {corresponding_weather_station}\n({rose_start} to {rose_end})", 
                        fontsize=12, 
                        fontweight='bold', 
                        pad=25
                    )

                    # 6. Push the rendering into the Streamlit tab structure
                    st.pyplot(fig_rose)

        except ImportError:
            st.error("The `windrose` package is not installed. Please run `pip install windrose` in your terminal to enable this visualization feature.")