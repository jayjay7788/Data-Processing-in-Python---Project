import io
import time
from pathlib import Path

import csv

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CHMI_HEADERS = {
    "accept": "application/json",
    "User-Agent": "JEM207 DataProcessingCourse (Educational access; contact: 19658413@fsv.cuni.cz)",
}


def _ensure_parent(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _extract_chmi_table(payload: dict) -> pd.DataFrame:
    data_response = payload.get("data", {}).get("data", {})
    headers = data_response.get("header", "").split(",")
    values = data_response.get("values", [])
    return pd.DataFrame(values, columns=headers)


def download_chmi_weather_stations_metadata(
    out_path: Path | str = RAW_DIR / "chmi_weather_stations_metadata.csv",
):
    out_path = _ensure_parent(out_path)
    resp = requests.get(
        "https://opendata.chmi.cz/meteorology/climate/historical/metadata/meta1.json",
        headers=CHMI_HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    df = _extract_chmi_table(resp.json())
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def download_chmi_weather_variables_metadata(
    out_path: Path | str = RAW_DIR / "chmi_weather_variables_metadata.csv",
):
    out_path = _ensure_parent(out_path)
    resp = requests.get(
        "https://opendata.chmi.cz/meteorology/climate/historical/metadata/meta2.json",
        headers=CHMI_HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    df = _extract_chmi_table(resp.json())
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def build_and_save_wsi_dict(
    stations_metadata_csv: Path | str = RAW_DIR / "chmi_weather_stations_metadata.csv",
    out_path: Path | str = RAW_DIR / "wsi_dict.csv",
):
    out_path = _ensure_parent(out_path)
    df = pd.read_csv(stations_metadata_csv)
    df = df[df["FULL_NAME"].str.contains("Praha", case=False, na=False)]
    wsi_to_drop = [
        "0-203-0-11201020001",
        "0-203-0-11202007001",
        "0-203-0-11105048001",
        "0-203-0-11201020003",
    ]
    df = df[~df["WSI"].isin(wsi_to_drop)].copy()
    df["END_DATE_DT"] = pd.to_datetime(df["END_DATE"], utc=True, errors="coerce")
    df = df.sort_values("END_DATE_DT").drop_duplicates(subset="WSI", keep="last")
    df = df[df["END_DATE_DT"] >= pd.Timestamp.now(tz="UTC")]

    wsi_dict = dict(zip(df["WSI"].astype(str), df["FULL_NAME"].astype(str)))
    pd.DataFrame(list(wsi_dict.items()), columns=["key", "value"]).to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    return wsi_dict


def write_dict_csv(d: dict, path: Path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerows(d.items())

def build_var_dict(data_raw_path):
    df_chmi_vars = pd.read_csv(RAW_DIR / "chmi_weather_variables_metadata.csv")
    chmi_vars_dict = dict(
        zip(df_chmi_vars["EG_EL_ABBREVIATION"].astype(str),
             df_chmi_vars["NAME"].astype(str)))
    write_dict_csv(chmi_vars_dict, data_raw_path / "chmi_vars_dict.csv")
    return chmi_vars_dict


def download_chmi_weather_data(
    start_year: int = 2025,
    end_year: int = 2025,
    wsi_csv: Path | str = RAW_DIR / "wsi_dict.csv",
    out_path: Path | str = RAW_DIR / "weather_data_10min.csv",
):
    out_path = _ensure_parent(out_path)
    wsi_dict = pd.read_csv(wsi_csv, encoding="utf-8-sig").set_index("key")["value"].to_dict()
    years = [f"{year}" for year in range(start_year, end_year + 1)]
    months = [f"{month:02d}" for month in range(1, 13)]
    route_template = "/meteorology/climate/historical/data/10min/{year}/10m-{wsi}-{ym}.json"

    results = []
    for wsi, station_name in wsi_dict.items():
        for year in years:
            for month in months:
                ym = f"{year}{month}"
                route = route_template.format(year=year, wsi=wsi, ym=ym)
                try:
                    time.sleep(0.5)
                    resp = requests.get(
                        f"https://opendata.chmi.cz{route}",
                        headers=CHMI_HEADERS,
                        timeout=90,
                    )
                    resp.raise_for_status()
                except requests.exceptions.RequestException as exc:
                    print(f"Request failed for {wsi} {station_name} {year}-{month}: {exc}")
                    continue

                df_part = _extract_chmi_table(resp.json())
                if df_part.empty:
                    continue
                df_part["WSI"] = wsi
                df_part["YEAR"] = year
                df_part["MONTH"] = month
                results.append(df_part)
        print(f"Loaded {wsi} {station_name}")

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df

def download_airquality_metadata(metadata_url="https://opendata.chmi.cz/air_quality/recent/metadata/metadata.json",
                                  out_path: Path|str = RAW_DIR / "airquality_CHMI_stations_metadata.csv"):
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
    df_mapping = df_mapping[df_mapping['locality_name'].str.contains('Praha', case=False, na=False)]
    df_mapping.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df_mapping

def build_and_save_air_stations_dict(
    metadata_csv: Path | str = RAW_DIR / "airquality_CHMI_stations_metadata.csv",
    out_path: Path | str = RAW_DIR / "air_stat_dict.csv",
):
    out_path = _ensure_parent(out_path)
    
    try:
        df = pd.read_csv(metadata_csv, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_csv}")
        return {}
    
    df = df[["station_code", "locality_name"]].drop_duplicates()
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    ids_dict = dict(zip(df["station_code"], df["locality_name"]))
    return ids_dict
    

def download_airquality_data(
    start_year: int = 2025,
    end_year: int = 2025,
    data_dir_url: str = "https://opendata.chmi.cz/air_quality/recent/data/",
    out_path: Path | str = RAW_DIR / "airquality_CHMI_1hour.csv",
    ids_csv: Path | str = RAW_DIR / "air_stat_dict.csv",
):
    out_path = _ensure_parent(out_path)

    try:
        df_ids = pd.read_csv(ids_csv, encoding="utf-8-sig")
        ids_to_keep = df_ids["station_code"].tolist()
        print(f"Loaded {len(ids_to_keep)} station IDs from {ids_csv.name} for filtering.")
    except FileNotFoundError:
        print(f"Warning: {ids_csv} not found. Proceeding without ID filtering.")
        ids_to_keep = None

    resp = requests.get(data_dir_url, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    csv_files = [link.get("href") for link in soup.find_all("a") if link.get("href", "").endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the directory.")

    years = [f"{year}" for year in range(start_year, end_year + 1)]
    months = [f"{month:02d}" for month in range(1, 13)]
    file_prefixes = [f"airquality_1h_avg_CZ_{year}{month}" for year in years for month in months]
    files_period = sorted(
        file for file in csv_files if any(file.startswith(prefix) for prefix in file_prefixes)
    )
    if not files_period:
        raise FileNotFoundError(
            f"No air-quality CSV files found for years {start_year}-{end_year} and months {months}."
        )

    print(f"Found {len(files_period)} hourly air-quality files.")
    results = []
    for file in files_period:
        try:
            time.sleep(0.1)
            resp = requests.get(f"{data_dir_url}{file}", timeout=60)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"Request failed for {file}: {exc}")
            continue

        df_part = pd.read_csv(io.StringIO(resp.text))
        if ids_to_keep is not None:
            df_part = df_part[df_part["idRegistration"].isin(ids_to_keep)].copy()
        if df_part.empty:
            continue
        df_part["source_file"] = file
        results.append(df_part)

    if not results:
        return pd.DataFrame()
    
    print(f"Downloaded {len(results)} air-quality files.")

    df = pd.concat(results, ignore_index=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def main():
    download_chmi_weather_stations_metadata()
    download_chmi_weather_variables_metadata()
    build_and_save_wsi_dict()

    print("Downloading Weather Data (2025)...")
    download_chmi_weather_data()
    
    download_airquality_metadata()
    build_and_save_air_stations_dict()

    print("Downloading Air Quality Data (2025)...")
    download_airquality_data()

    print("All downloads completed.")


if __name__ == "__main__":
    main()