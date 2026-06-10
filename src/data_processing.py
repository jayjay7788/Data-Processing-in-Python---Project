
from pathlib import Path
import os
import sys
import csv
import json
import pandas as pd
import numpy as np

import warnings

import geopandas as gpd
from shapely.geometry import LineString


def get_paths():
    repo_root = Path(__file__).resolve().parents[1]
    data_raw_path = repo_root / "data" / "raw"
    data_processed_path = repo_root / "data" / "processed"
    data_processed_path.mkdir(parents=True, exist_ok=True)
    return repo_root, data_raw_path, data_processed_path


def process_weather_stations(data_raw_path, keep_contains="Praha", drop_wsi=None, window_start=None, window_end=None):
    if window_start is None:
        window_start = pd.Timestamp("2025-01-01", tz="UTC")
    elif isinstance(window_start, str):
        window_start = pd.Timestamp(window_start, tz="UTC")
    if window_end is None:
        window_end = pd.Timestamp("2025-12-31", tz="UTC")
    elif isinstance(window_end, str):
        window_end = pd.Timestamp(window_end, tz="UTC")

    if window_start > window_end:
        raise ValueError("window_start must be <= window_end")

    df = pd.read_csv(data_raw_path / "chmi_weather_stations_metadata.csv")
    df = df[df["FULL_NAME"].str.contains(keep_contains, case=False, na=False)]
    if drop_wsi is None:
        drop_wsi = [
            '0-203-0-11201020001',
            '0-203-0-11202007001',
            '0-203-0-11105048001',
            '0-203-0-11201020003',
            '0-203-0-11515'
        ]
    df = df[~df["WSI"].isin(drop_wsi)].copy()
    df["BEGIN_DATE_DT"] = pd.to_datetime(df["BEGIN_DATE"], utc=True, errors="coerce")
    df["END_DATE_DT"] = pd.to_datetime(df["END_DATE"], utc=True, errors="coerce")
    df = df.sort_values("END_DATE_DT").drop_duplicates(subset="WSI", keep="last")
    if window_start is not None and window_end is not None:
        df = df[(df["BEGIN_DATE_DT"] <= window_start) & (df["END_DATE_DT"] >= window_end)].copy()
    
    weather_stations_meta = (
        df.sort_values("END_DATE_DT")
        .drop_duplicates(subset="WSI", keep="last")
    )
    return weather_stations_meta


# def write_dict_csv(d: dict, path: Path):
#     with open(path, "w", encoding="utf-8-sig", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["key", "value"])
#         writer.writerows(d.items())


# def build_wsi_and_var_dicts(df_chmi_stat, df_chmi_vars, data_raw_path):
#     wsi_dict = dict(
#         zip(df_chmi_stat["WSI"].astype(str),
#              df_chmi_stat["FULL_NAME"].astype(str)))
#     write_dict_csv(wsi_dict, data_raw_path / "wsi_dict.csv")

#     chmi_vars_dict = dict(
#         zip(df_chmi_vars["EG_EL_ABBREVIATION"].astype(str),
#              df_chmi_vars["NAME"].astype(str)))
#     write_dict_csv(chmi_vars_dict, data_raw_path / "chmi_vars_dict.csv")
#     return wsi_dict, chmi_vars_dict


def aggregate_weather_10min_to_hourly(data_raw_path):
    df = pd.read_csv(data_raw_path / "weather_data_10min.csv")

    #load dicts
    wsi_df = pd.read_csv(data_raw_path / "wsi_dict.csv")
    chmi_vars_df = pd.read_csv(data_raw_path / "chmi_vars_dict.csv")

    wsi_dict = wsi_df.set_index("key")["value"].to_dict()
    chmi_vars_dict = chmi_vars_df.set_index("key")["value"].to_dict()

    # filter out bad quality
    df = df[df["QUALITY"] != 4]

    df["DT"] = pd.to_datetime(df["DT"], utc=True, errors="coerce").dt.floor("h")
    for col in ["STATION", "WSI", "ELEMENT"]:
        df[col] = df[col].astype("category")

    base = ["STATION", "WSI", "ELEMENT", "DT"]

    sum_elements = {"RGLB10", "SRA10M", "SSV10M"}
    max_elements = {"TMA", "Fmax"}
    min_elements = {"TMI"}
    dir_elements = {"Dprum", "D"}

    parts = []
    parts.append(
        df[df["ELEMENT"].isin(sum_elements)]
        .groupby(base, as_index=False, sort=False, observed=True)["VAL"]
        .sum()
        )
    parts.append(
        df[df["ELEMENT"].isin(max_elements)]
        .groupby(base, as_index=False, sort=False, observed=True)["VAL"]
        .max()
        )
    parts.append(
        df[df["ELEMENT"].isin(min_elements)]
        .groupby(base, as_index=False, sort=False, observed=True)["VAL"]
        .min()
        )

    df_dir = df[df["ELEMENT"].isin(dir_elements)].copy()
    df_dir["_sin"] = np.sin(np.deg2rad(df_dir["VAL"]))
    df_dir["_cos"] = np.cos(np.deg2rad(df_dir["VAL"]))

    dir_agg = (
        df_dir.groupby(base, as_index=False, sort=False, observed=True)
    .agg(_sin=("_sin", "sum"), _cos=("_cos", "sum"))
         )
    dir_agg["VAL"] = (np.degrees(np.arctan2(dir_agg["_sin"], dir_agg["_cos"])) % 360)
    dir_agg = dir_agg.drop(columns=["_sin", "_cos"])
    parts.append(dir_agg)

    other = ~df["ELEMENT"].isin(sum_elements | max_elements | min_elements | dir_elements)
    parts.append(
        df[other]
        .groupby(base, as_index=False, sort=False, observed=True)["VAL"]
        .mean()
        )

    df_weather_hourly = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["STATION", "WSI", "DT", "ELEMENT"], kind="mergesort")
        .reset_index(drop=True)
        )
    df_weather_hourly["YEAR"] = df_weather_hourly["DT"].dt.year
    df_weather_hourly["MONTH"] = df_weather_hourly["DT"].dt.month

     # remove unused categories
    cat_cols = df_weather_hourly.select_dtypes(include=["category"]).columns
    df_weather_hourly[cat_cols] = df_weather_hourly[cat_cols].apply(
        lambda s: s.cat.remove_unused_categories().astype(object)
        )

    # map names
    df_weather_hourly["ELEMENT_NAME"] = df_weather_hourly["ELEMENT"].map(chmi_vars_dict)
    df_weather_hourly["WSI_NAME"] = df_weather_hourly["WSI"].map(wsi_dict)

    return df_weather_hourly


def load_and_merge_weather_station_metadata(df_weather_hourly, weather_stations_meta):
    # stations_meta = pd.read_csv(data_raw_path / "chmi_weather_stations_metadata.csv")
    # stations_meta["END_DATE_DT"] = pd.to_datetime(stations_meta["END_DATE"], utc=True, errors="coerce")
    # stations_meta = (
    #     stations_meta.sort_values("END_DATE_DT")
    #     .drop_duplicates(subset="WSI", keep="last")
    # )
    stations_sel = weather_stations_meta[["WSI", "FULL_NAME", "ELEVATION", "GEOGR1", "GEOGR2"]].copy()
    stations_sel = stations_sel.rename(columns={
        "GEOGR1": "LON"
        , "GEOGR2": "LAT"
        })
    # ensure string types
    df_weather_hourly["WSI"] = df_weather_hourly["WSI"].astype(str)
    stations_sel["WSI"] = stations_sel["WSI"].astype(str)
    df_weather_ext = df_weather_hourly.merge(stations_sel, on="WSI", how="left")
    return df_weather_ext


def load_air_quality_meta(data_raw_path):
    air_stations_meta = pd.read_csv(data_raw_path / "airquality_CHMI_stations_metadata.csv")
    air_stations_meta = air_stations_meta[
        air_stations_meta['locality_name'].str.contains('Praha', case=False, na=False)
        ]
    air_stations_meta["id_registration"] = air_stations_meta["id_registration"].astype(str)

    return air_stations_meta

def load_and_merge_air_quality_data(air_stations_meta, data_raw_path):
    df_air = pd.read_csv(data_raw_path / "airquality_CHMI_1hour.csv")
    df_air = df_air.rename(columns={"idRegistration": "id_registration"})
    df_air["id_registration"] = df_air["id_registration"].astype(str)

    df_air_qual = df_air.merge(air_stations_meta, on="id_registration", how="right")

    return df_air_qual


def spatial_join_stations(air_stations_meta, weather_stations_meta):

    weather_stations = (
        weather_stations_meta
        .loc[:, ["WSI", "FULL_NAME", "GEOGR1", "GEOGR2", "ELEVATION"]]
        .drop_duplicates()
        .rename(columns={
            "WSI": "weather_station_id",
            "FULL_NAME": "weather_station_name",
            "GEOGR1": "weather_lon",
            "GEOGR2": "weather_lat",
            "ELEVATION": "weather_alt"
        })
    )

    air_stations = (
        air_stations_meta
        .loc[
            air_stations_meta["locality_name"].str.contains("Praha", case=False, na=False),
            ["station_code", "locality_name", "lon", "lat", "alt"]
        ]
        .drop_duplicates()
        .rename(columns={
            "station_code": "air_station_code",
            "locality_name": "air_station_name",
            "lon": "air_lon",
            "lat": "air_lat",
            "alt": "air_alt"
        })
    )

    air_gdf = gpd.GeoDataFrame(
        air_stations,
        geometry=gpd.points_from_xy(
            air_stations["air_lon"],
            air_stations["air_lat"]
            ),
             crs="EPSG:4326"
    )
    weather_gdf = gpd.GeoDataFrame(
        weather_stations,
        geometry=gpd.points_from_xy(
            weather_stations["weather_lon"],
            weather_stations["weather_lat"]
        ),
        crs="EPSG:4326"
        )
    
    #convert to meters
    air_gdf_m = air_gdf.to_crs("EPSG:5514")
    weather_gdf_m = weather_gdf.to_crs("EPSG:5514")
    #find nearest match
    nearest_match = gpd.sjoin_nearest(
        air_gdf_m,
        weather_gdf_m,
        how="left", 
        distance_col="distance_m"
        )
    #add km column
    nearest_match["distance_km"] = nearest_match["distance_m"] / 1000
    #clean cols
    nearest_match = nearest_match[
        ["air_station_code", "air_station_name", 
         "air_lon", "air_lat", "air_alt", 
         "weather_station_id", "weather_station_name", 
         "weather_lon", "weather_lat", "weather_alt", 
         "distance_m", "distance_km"
         ]
    ].copy()

    return nearest_match


def process_and_merge_air_weather_data(df_air_qual, df_weather_ext, nearest_match, max_distance_km=5):
    air = df_air_qual
    weather = df_weather_ext
    station_match = nearest_match
    # clean column names
    air.columns = air.columns.str.strip()
    weather.columns = weather.columns.str.strip()
    station_match.columns = station_match.columns.str.strip()
    # set time columns
    air_time_col = "startTime"
    weather_time_col = "DT"
    # clean station names
    air["locality_name"] = air["locality_name"].astype(str).str.strip()
    weather["WSI_NAME"] = weather["WSI_NAME"].astype(str).str.strip()
    station_match["air_station_name"] = station_match["air_station_name"].astype(str).str.strip()
    station_match["weather_station_name"] = station_match["weather_station_name"].astype(str).str.strip()
    # convert times
    air[air_time_col] = pd.to_datetime(air[air_time_col]).dt.tz_localize(None)
    weather[weather_time_col] = pd.to_datetime(weather[weather_time_col]).dt.tz_localize(None)
    # keep one nearest weather station per air station
    station_match_nearest = (
        station_match
        .sort_values("distance_m")
        .drop_duplicates(subset=["air_station_name"], keep="first")
        .copy()
    )
    
    # pivot air wide
    air_wide = (
        air.pivot_table(
            index=[air_time_col, "locality_name"],
            columns="component_code", 
            values="value", 
            aggfunc="mean"
            )
            .reset_index()
    )
    
    air_wide.columns.name = None
    air_wide = air_wide.rename(
        columns={
            col: f"air_{col}" for col in air_wide.columns
              if col not in [air_time_col, "locality_name"]
        }
    )

    air_wide_with_match = air_wide.merge(
        station_match_nearest,
        left_on="locality_name", 
        right_on="air_station_name", 
        how="left", 
        validate="many_to_one"
    )

    # pivot weather wide
    weather_wide = (
        weather.pivot_table(
        index=[weather_time_col, "WSI_NAME"], 
        columns="ELEMENT", 
        values="VAL", 
        aggfunc="mean"
        )
        .reset_index()
    )
    weather_wide.columns.name = None
    weather_wide = weather_wide.rename(
        columns={
            col: f"weather_{col}" for col in weather_wide.columns 
            if col not in [weather_time_col, "WSI_NAME"]
        }
    )

    # merge using station matching table if available

    df_merged = air_wide_with_match.merge(
        weather_wide,
        left_on=[air_time_col, "weather_station_name"], 
        right_on=[weather_time_col, "WSI_NAME"], 
        how="left", 
        validate="many_to_one"
    )

    # drop distant matches
    df_merged = df_merged[df_merged["distance_km"] <= max_distance_km]

    # basic cleaning
    cols_to_keep = [
        'startTime','weather_station_name','air_station_name', 'air_NO2', 'air_NOx', 'air_PM10',  'weather_D', 'weather_Dmax', 'weather_Dprum', 'weather_F', 'weather_Fmax', 'weather_Fprum', 'weather_H', 'weather_P','weather_SRA10M', 'weather_SSV10M', 'weather_T','weather_TMA', 'weather_TMI', 'air_lon', 'air_lat', 'air_alt', 'weather_lon', 'weather_lat', 'weather_alt', 'distance_km'
    ]
    existing = [c for c in cols_to_keep if c in df_merged.columns]
    df_merged = df_merged[existing]

    # replace negative air measurements with NaN
    for col in ["air_NO2", "air_NOx", "air_PM10"]:
        df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")
        df_merged.loc[df_merged[col] < 0, col] = np.nan

    # drop some columns with too many nans if present
    df_merged = df_merged[~df_merged["air_station_name"].isin(["Praha 6-Břevnov" , "Praha 5-Stodůlky"])]

    df_merged = df_merged.drop(columns=["weather_SSV10M"], errors="ignore")
    df_merged = df_merged.drop(columns=["air_alt"], errors="ignore")

    return df_merged

def clean_data(df_merged):
    cols_to_keep = [
        'startTime','weather_station_name','air_station_name', 
        'air_NO2', 'air_NOx', 'air_PM10', 
        'weather_D', 'weather_Dmax', 'weather_Dprum', 
        'weather_F', 'weather_Fmax', 'weather_Fprum', 
        'weather_H', 'weather_P','weather_SRA10M', 
        'weather_T','weather_TMA', 'weather_TMI', 
        'air_lon', 'air_lat', 'weather_lon', 
        'weather_lat', 'weather_alt', 'distance_km'
    ]
    df_merged = df_merged[cols_to_keep]

    # replace those with NA's
    df_merged[[
        "air_NO2", "air_NOx", "air_PM10"
    ]] = df_merged[[
        "air_NO2", "air_NOx", "air_PM10"
    ]].mask(
        df_merged[[
        "air_NO2", "air_NOx", "air_PM10"
    ]] < 0,
        np.nan
    )

    # drop stations with no weather data
    df_merged = df_merged[~df_merged["air_station_name"].isin(["Praha 6-Břevnov" , "Praha 5-Stodůlky"])]

    df_cleaned = df_merged

    return df_cleaned


def engineer_features(df_merged):
    df = df_merged
    # df = df.rename(columns={"startTime": "startTime"})
    # df["startTime"] = pd.to_datetime(df["startTime"], errors='coerce')

    # date features
    df["startTime"] = pd.to_datetime(df["startTime"])
    df["hour"] = df["startTime"].dt.hour
    df["dayofweek"] = df["startTime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["date"] = df["startTime"].dt.date
    df["month"] = df["startTime"].dt.month
    df["dayofyear"] = df["startTime"].dt.dayofyear
    df["weekday_name"] = df["startTime"].dt.day_name()

    target_cols = ["air_NO2", "air_NOx", "air_PM10"]
    weather_cols = [
    "weather_D", "weather_Dmax", "weather_Dprum",
    "weather_F", "weather_Fmax", "weather_Fprum",
    "weather_H", "weather_P",
    "weather_SRA10M",
    "weather_T", "weather_TMA", "weather_TMI"
    ]
    spatial_cols = [
    "air_lon", "air_lat",
    "weather_lon", "weather_lat", "weather_alt",
    "distance_km"
    ]
    id_cols = [
        "startTime", "weather_station_name", "air_station_name"
    ]

    numeric_should_be = target_cols + weather_cols + spatial_cols
    df[numeric_should_be] = df[numeric_should_be].apply(pd.to_numeric, errors="coerce")

    # wind vector
    theta = np.deg2rad(df["weather_D"])
    df["wind_u"] = -df["weather_F"] * np.sin(theta)
    df["wind_v"] = -df["weather_F"] * np.cos(theta)

    # lags and rolling means per station
    df = df.sort_values(["air_station_name", "startTime"]).reset_index(drop=True)
    meteo_vars = ["weather_F", "wind_u", "wind_v", "weather_T", "weather_H", "weather_P"]

    for var in meteo_vars:
        df[f"{var}_lag1"] = df.groupby("air_station_name")[var].shift(1)
        df[f"{var}_mean3"] = (
            df.groupby("air_station_name")[var]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        df[f"{var}_mean6"] = (
            df.groupby("air_station_name")[var]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )

    for window in [3, 6, 12, 24]:
        df[f"precip_sum{window}"] = (
            df.groupby("air_station_name")["weather_SRA10M"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
        )

    df["weather_P_change6"] = df["weather_P"] - df.groupby("air_station_name")["weather_P"].shift(6)

    df_final = df

    return df_final


def save_processed(df_final, data_processed_path, fname="processed_data.csv"):
    path = data_processed_path / fname
    df_final.to_csv(path, index=False)
    return path

### alternative using parquet compression

# def save_processed(df_final, data_processed_path, fname="processed_data.parquet"):
#     path = data_processed_path / fname
    
#     # Save as compressed Parquet instead of CSV
#     df_final.to_parquet(path, index=False)
    
#     print(f"Successfully saved compressed data to: {path}")
#     return path


def main():
    repo_root, data_raw_path, data_processed_path = get_paths()

    # process station metadata
    weather_stations_meta = process_weather_stations(data_raw_path)

    # aggregate weather
    df_weather_hourly = aggregate_weather_10min_to_hourly(data_raw_path)
    df_weather_ext = load_and_merge_weather_station_metadata(df_weather_hourly, weather_stations_meta)

    # load air quality and station metadata
    air_stations_meta = load_air_quality_meta(data_raw_path)
    df_air_qual = load_and_merge_air_quality_data(air_stations_meta, data_raw_path)

    #spatial join
    nearest_match = spatial_join_stations(air_stations_meta, weather_stations_meta)

    # preprocess and merge
    df_merged = process_and_merge_air_weather_data(df_air_qual, df_weather_ext, nearest_match)

    # data cleaning and engineering
    df_cleaned = clean_data(df_merged)
    df_final = engineer_features(df_cleaned)

    out_path = save_processed(df_final, data_processed_path)
    print(f"Saved processed data to: {out_path}")


if __name__ == "__main__":
    main()
