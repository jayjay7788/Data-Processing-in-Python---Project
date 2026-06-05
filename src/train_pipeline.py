from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# project root is to the system path to allow importing wrapper
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import model wrapper functions
from src.model_wrapper import save_model_bundle

def run_training_pipeline():
    # Define paths based on your structure
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
    model_output_dir = PROJECT_ROOT / "results" / "models"
    
    print(f"Loading processed data from: {processed_data_path}")
    if not processed_data_path.exists():
        raise FileNotFoundError(
            f"Could not find processed data. Please run your preprocessing script first!"
        )
        
    df = pd.read_csv(processed_data_path)
    
    # Define features that match Streamlit App Sliders exactly
    numeric_features = [
        "hour", "dayofweek", "is_weekend", "month", "dayofyear",
        "weather_T", "weather_H", "weather_P", "weather_F", "wind_u", "wind_v",
        "weather_F_lag1","weather_F_mean3","weather_F_mean6",
        "wind_u_lag1","wind_u_mean3","wind_u_mean6","wind_v_lag1","wind_v_mean3","wind_v_mean6",
        "weather_T_lag1","weather_T_mean3","weather_T_mean6",
        "weather_H_lag1","weather_H_mean3","weather_H_mean6",
        "weather_P_lag1","weather_P_mean3","weather_P_mean6","weather_P_change6",
        "precip_sum3","precip_sum6","precip_sum12","precip_sum24"
    ]
    targets = ["air_NO2", "air_PM10"]
    
    # Clean up any missing rows in our training target features
    df_clean = df.dropna(subset=numeric_features + targets)
    X = df_clean[numeric_features]
    print("Sample station names in training data:", df["air_station_name"].unique())
    trained_models = {}
    
    # Train a distinct model for each air pollutant
    for target in targets:
        print(f" Training Random Forest Regressor for {target}...")
        y = df_clean[target]
        
        # Using a fast configuration; adjust hyperparameters as needed
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        trained_models[target] = model
    
    # Use model wrapper to save the bundle precisely where the app expects it
    print(f" Saving trained models into bundle...")
    bundle_path = save_model_bundle(
        models=trained_models,
        save_path=model_output_dir,
        numeric_features=numeric_features
    )
    print(f"🎉 Success! Model bundle successfully exported to: {bundle_path}")

if __name__ == "__main__":
    run_training_pipeline()




def predict_pollution(w_speed, temp, hum, pressure, rain, stat):
    bundle = get_model_bundle()
    
    # Calculate time features to feed the model
    now = pd.Timestamp.now()
    hour = now.hour
    dayofweek = now.dayofweek
    is_weekend = 1 if dayofweek in [5, 6] else 0
    month = now.month
    dayofyear = now.dayofyear

    # Map the UI sliders to the EXACT column names your models were trained on
    input_data = pd.DataFrame([
        {
            "weather_T": temp,
            "weather_H": hum,
            "weather_P": pressure,
            "weather_F": w_speed,
            "hour": hour,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "month": month,
            "dayofyear": dayofyear
        }
    ])

    predictions = predict_bundle(bundle, input_data).iloc[0]
    return float(predictions.get("air_PM10", 0.0)), float(predictions.get("air_NO2", 0.0))