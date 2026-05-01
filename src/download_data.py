# packages
import os
import json
import io
import csv
import requests
import time

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/ -> project root
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# API key
def set_api_key(token="api_key.env"):
    load_dotenv(token)
    api_key = os.getenv("GOLEMIO_API_KEY")
    if api_key:
        print("API key loaded")
    else:
        print("API key not found")
    return api_key

# CHMI weather metadata
def get_chmi_weather_stations_metadata(out_path: Path|str = RAW_DIR / "chmi_weather_stations_metadata.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://opendata.chmi.cz/"
    route = "/meteorology/climate/historical/metadata/meta1.json"
    headers = {
        "accept": "application/json",
        "User-Agent": "JEM207 DataProcessingCourse (Educational access; contact: 19658413@fsv.cuni.cz)",
    }
    resp = requests.get(f"{url}{route}", headers=headers, timeout=60)
    resp.raise_for_status()
    response = resp.json()
    data_response = response.get('data', {}).get('data', {})
    headers = data_response.get('header', '').split(',')
    values = data_response.get('values', [])
    df = pd.DataFrame(values, columns=headers)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df

def get_chmi_weather_variables_metadata(out_path: Path|str = RAW_DIR / "chmi__weather_variables_metadata.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://opendata.chmi.cz/"
    route = "/meteorology/climate/historical/metadata/meta2.json"
    headers = {
        "accept": "application/json",
        "User-Agent": "JEM207 DataProcessingCourse (Educational access; contact: 19658413@fsv.cuni.cz)",
    }
    resp = requests.get(f"{url}{route}", headers=headers, timeout=60)
    resp.raise_for_status()
    response = resp.json()
    data_response = response.get('data', {}).get('data', {})
    headers = data_response.get('header', '').split(',')
    values = data_response.get('values', [])
    df = pd.DataFrame(values, columns=headers)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df

# CHMI weather data (10min)
def get_chmi_weather_data(start_year=2025, end_year=2025, wsi_csv = RAW_DIR / "wsi_dict.csv", out_path: Path|str = RAW_DIR / "weather_data_10min.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = "https://opendata.chmi.cz/"
    route_template = "/meteorology/climate/historical/data/10min/{year}/10m-{wsi}-{ym}.json"
    url_header = {
        "accept": "application/json",
        "User-Agent": "JEM207 DataProcessingCourse (Educational access; contact: 19658413@fsv.cuni.cz)",
    }
    # we set to dowload only data from selected stations to not get unnecessary big dataset
    wsi_dict = pd.read_csv(wsi_csv, encoding="utf-8-sig").set_index("key")["value"].to_dict()
    years = [f"{y}" for y in range(start_year, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    results = []
    for wsi in wsi_dict:
        for year in years:
            for month in months:
                ym = f"{year}{month}"
                route = route_template.format(year=year, wsi=wsi, ym=ym)
                try:
                    time.sleep(0.5)
                    response = requests.get(f"{base_url}{route}", headers=url_header, timeout=90)
                    response.raise_for_status()
                except requests.exceptions.RequestException as exc:
                    print(f"Request failed for {wsi} {wsi_dict.get(wsi)} {year} {month}: {exc}")
                    continue
                response = response.json()
                data_response = response.get("data", {}).get("data", {})
                headers = data_response.get("header", "").split(",")
                values = data_response.get("values", [])
                if not values:
                    continue
                df_part = pd.DataFrame(values, columns=headers)
                df_part["WSI"] = wsi
                df_part["YEAR"] = year
                df_part["MONTH"] = month
                results.append(df_part)
    if not results:
        return pd.DataFrame()
    df = pd.concat(results, ignore_index=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df

# CHMI air quality CSV downloader
def download_airquality_data(data_dir_url="https://opendata.chmi.cz/air_quality/recent/data/", out_path: Path|str = RAW_DIR / "airquality_CHMI_stations_data.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(data_dir_url, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    csv_files = [link.get("href") for link in soup.find_all("a") if link.get("href", "").endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the directory.")
    latest_file = sorted(csv_files)[-1]
    file_url = f"{data_dir_url}{latest_file}"
    print(f"Downloading raw data from: {latest_file}")
    data_response = requests.get(file_url)
    data_response.raise_for_status()
    df_data = pd.read_csv(io.StringIO(data_response.text))
    df_data.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df_data

def download_airquality_metadata(metadata_url="https://opendata.chmi.cz/air_quality/recent/metadata/metadata.json", out_path: Path|str = RAW_DIR / "airquality_CHMI_stations_metadata.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(metadata_url, timeout=60)
    resp.raise_for_status()
    metadata = resp.json()
    mapping_list = []
    localities = metadata.get("data", {}).get("Localities", [])
    for locality in localities:
        loc_code = locality.get("LocalityCode", {})
        loc_name = locality.get("Name", {})
        loc = locality.get("Localization", {})
        lon = loc.get("LonAsNumber")
        lat = loc.get("LatAsNumber")
        alt = loc.get("Alt")
        addr = locality.get("Address", {})
        street = addr.get("Street")
        city = addr.get("City")
        programs = locality.get("MeasuringPrograms", [])
        for program in programs:
            station_code = program.get("Code")
            measurements = program.get("Measurements", [])
            for measurement in measurements:
                row = {
                    "id_registration": measurement.get("IdRegistration"),
                    "station_code": station_code,
                    "locality_code": loc_code,
                    "locality_name": loc_name,
                    "street": street,
                    "city": city,
                    "lon": lon,
                    "lat": lat,
                    "alt": alt,
                    "component_code": measurement.get("ComponentCode"),
                    "component_name": measurement.get("ComponentName"),
                    "unit": measurement.get("UnitAsASCII"),
                }
                mapping_list.append(row)
    df_mapping = pd.DataFrame(mapping_list)
    df_mapping.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df_mapping

# Golemio airquality metadata
def get_airquality_stations_metadata(api_key, out_path: Path|str = RAW_DIR / "airquality_stations_metadata.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://api.golemio.cz/"
    route = "/v2/airqualitystations"
    headers = {
        "X-access-token": api_key,
        "accept": "application/json",
        "User-Agent": "JEM207 DataProcessingCourse (Educational access; contact: 19658413@fsv.cuni.cz)",
    }
    resp = requests.get(f"{url}{route}", headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.json_normalize(data.get("features", {}))
    tmp = df.explode("properties.measurement.components", ignore_index=True)
    dirs = pd.json_normalize(tmp["properties.measurement.components"]).add_prefix("properties.measurement.components.")
    df = pd.concat([tmp.drop(columns=["properties.measurement.components"]), dirs], axis=1)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


# dictionaries

# this need to run after dowloading metadata dn before downloading data 

def build_and_save_wsi_dict(stations_metadata_csv=RAW_DIR / "chmi_stations_metadata.csv", out_path=RAW_DIR / "wsi_dict.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(stations_metadata_csv)
    df = df[df['FULL_NAME'].str.contains('Praha', case=False, na=False)]
    wsi_to_drop = [
        '0-203-0-11201020001', #Praha, Vinohrady - Flora	
        '0-203-0-11202007001', #Praha, Suchdol
        '0-203-0-11105048001', #Praha, Zadní Kopanina
        '0-203-0-11201020003' #Praha, Chodov
        ]
    df = df[~df['WSI'].isin(wsi_to_drop)]
    df["END_DATE_DT"] = pd.to_datetime(df["END_DATE"], utc=True, errors="coerce")
    df = (
        df.sort_values("END_DATE_DT")
        .drop_duplicates(subset="WSI", keep="last")
    )
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[(df["END_DATE_DT"] >= now_utc)]

    wsi_dict = dict(zip(df["WSI"].astype(str), df["FULL_NAME"].astype(str)))
    pd.DataFrame(list(wsi_dict.items()), columns=["key","value"]).to_csv(out_path, index=False, encoding="utf-8-sig")
    return wsi_dict