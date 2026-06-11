from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# project root 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# import model wrapper functions
from src.model_wrapper import save_model_bundle

# the training function
def run_training_pipeline():
    # define paths based on the structure
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
    model_output_dir = PROJECT_ROOT / "results" / "models"
    
    # print process info and error message
    print(f"Loading processed data from: {processed_data_path}")
    if not processed_data_path.exists():
        raise FileNotFoundError(
            f"Could not find processed data. Please run your preprocessing script first!"
        )
    # read the data    
    df = pd.read_csv(processed_data_path)
    
    # define features that will be used in the model
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
    targets = ["air_NO2", "air_NOx", "air_PM10"]
    
    # clean up any missing rows in the training target features
    df_clean = df.dropna(subset=numeric_features + targets)
    X_numeric = df_clean[numeric_features] # add numeric features
     # encode station identities
    station_dummies = pd.get_dummies(df_clean['air_station_name'], prefix='air_station_name')
    weather_dummies = pd.get_dummies(df_clean['weather_station_name'], prefix='weather_station_name')
    X = pd.concat([X_numeric, station_dummies, weather_dummies], axis=1)
    # capture the exact dummy columns produced
    categorical_features = station_dummies.columns.tolist() + weather_dummies.columns.tolist()
    trained_models = {}
    
    # train a distinct model for each air pollutant
    for target in targets:
        print(f" Training Random Forest Regressor for {target}...")
        y = df_clean[target]
        
        # fit the RF model
        model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
        model.fit(X, y)
        trained_models[target] = model
    
    # use the predefined structure from model_wrapper and app_draft to save the model outputs
    bundle_path = save_model_bundle(
        models=trained_models,
        save_path=model_output_dir,
        numeric_features=numeric_features,
        categorical_features=categorical_features 
    )
    print(f"Model bundle successfully exported to: {bundle_path}")

def main():
    print("Initiating Model Training...")
    run_training_pipeline()
    print("Model Training Finished!")

if __name__ == "__main__":
    main()
