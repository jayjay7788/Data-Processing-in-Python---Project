# packages
import os
import json
import io
import csv

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# utility
def print_json_structure(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{pad}{key}: {type(value).__name__}")
            print_json_structure(value, indent + 1)
    elif isinstance(obj, list):
        print(f"{pad}[list] len={len(obj)}")
        if obj:
            print_json_structure(obj[0], indent + 1)

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
def get_chmi_weather_stations_metadata(out_path="data/raw/chmi_weather_stations_metadata.csv"):
    url = "https://opendata.chmi.cz/"
    route = "/meteorology/climate/historical/metadata/meta1.json"
    headers = {
        "accept": "application/json",
        "User-Agent": "JEM207 DataProcessingCourse (Educational access; contact: 19658413@fsv.cuni.cz)",
    }
    resp = requests.get(f"{url}{route}", headers=headers, timeout=60)
    resp.raise_for_status()
    response = response.json()
    data_response = response.get('data', {}).get('data', {})
    headers = data_response.get('header', '').split(',')
    values = data_response.get('values', [])
    df = pd.DataFrame(values, columns=headers)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df

def get_chmi_weather_variables_metadata(out_path="data/raw/chmi__weather_variables_metadata.csv"):
    url = "https://opendata.chmi.cz/"
    route = "/meteorology/climate/historical/metadata/meta2.json"
    headers = {
        "accept": "application/json",
        "User-Agent": "JEM207 DataProcessingCourse (Educational access; contact: 19658413@fsv.cuni.cz)",
    }
    resp = requests.get(f"{url}{route}", headers=headers, timeout=60)
    resp.raise_for_status()
    response = response.json()
    data_response = response.get('data', {}).get('data', {})
    headers = data_response.get('header', '').split(',')
    values = data_response.get('values', [])
    df = pd.DataFrame(values, columns=headers)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df

# CHMI weather data (10min)
def get_chmi_weather_data(start_year=2025, end_year=2025, wsi_csv="wsi_dict.csv", out_path="data/raw/weather_data_10min.csv"):
    base_url = "https://opendata.chmi.cz/"
    route_template = "/meteorology/climate/historical/data/10min/{year}/10m-{wsi}-{ym}.json"
    url_header = {
        "accept": "application/json",
        "User-Agent": "JEM207 DataProcessingCourse (Educational access; contact: 19658413@fsv.cuni.cz)",
    }
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
                    response = requests.get(f"{base_url}{route}", headers=url_header, timeout=60)
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df

# CHMI air quality CSV downloader
def download_latest_data(data_dir_url, out_path="data/raw/airquality_CHMI_stations_data.csv"):
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_data.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df_data

def download_metadata(metadata_url, out_path="data/raw/airquality_CHMI_stations_metadata.csv"):
    resp = requests.get(metadata_url, timeout=60)
    resp.raise_for_status()
    metadata = resp.json()
    mapping_list = []
    localities = metadata.get("data", {}).get("Localities", [])
    for locality in localities:
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_mapping.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df_mapping

# Golemio airquality metadata
def get_airquality_stations_metadata(api_key, out_path="data/raw/airquality_stations_metadata.csv"):
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


# dictionaries

def build_and_save_wsi_dict(df_chmi, out_path="data/raw/wsi_dict.csv"):
    wsi_dict = dict(zip(df_chmi["WSI"].astype(str), df_chmi["FULL_NAME"].astype(str)))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame(list(wsi_dict.items()), columns=["key","value"]).to_csv(out_path, index=False, encoding="utf-8-sig")
    return wsi_dict