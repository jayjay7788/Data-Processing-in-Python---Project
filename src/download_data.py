import argparse
import io
import time
from pathlib import Path

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
                print(f"Loaded {wsi} {station_name} {year}-{month}: {len(df_part)} rows")

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def download_airquality_ids_dict(
    metadata_csv: Path | str = RAW_DIR / "airquality_CHMI_stations_metadata.csv",
    out_path: Path | str = RAW_DIR / "air_ids_dict.csv",
):
    out_path = _ensure_parent(out_path)
    
    try:
        df = pd.read_csv(metadata_csv, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_csv}")
        return {}
    
    df = df[["id_registration", "locality_name"]].drop_duplicates()
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    ids_dict = dict(zip(df["id_registration"], df["locality_name"]))
    return ids_dict
    


def download_airquality_data_period(
    start_year: int = 2025,
    end_year: int = 2025,
    months: list[int] | None = None,
    data_dir_url: str = "https://opendata.chmi.cz/air_quality/recent/data/",
    out_path: Path | str = RAW_DIR / "airquality_CHMI_1hour.csv",
    ids_to_keep: list[str] | None = None,
):
    out_path = _ensure_parent(out_path)
    resp = requests.get(data_dir_url, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    csv_files = [link.get("href") for link in soup.find_all("a") if link.get("href", "").endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the directory.")

    years = [f"{year}" for year in range(start_year, end_year + 1)]
    months = [f"{month:02d}" for month in (months or range(1, 13))]
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

    df = pd.concat(results, ignore_index=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download raw CHMI and air-quality data.")
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--months", nargs="*", type=int, default=None)
    args = parser.parse_args()

    download_chmi_weather_stations_metadata()
    download_chmi_weather_variables_metadata()
    build_and_save_wsi_dict()
    download_chmi_weather_data(start_year=args.start_year, end_year=args.end_year)
    
    # Download air-quality IDs dictionary
    air_ids_dict = download_airquality_ids_dict()
    ids_to_keep = list(air_ids_dict.keys()) if air_ids_dict else None
    
    download_airquality_data_period(
        start_year=args.start_year,
        end_year=args.end_year,
        months=args.months,
        ids_to_keep=ids_to_keep,
    )
    print("All downloads completed.")


if __name__ == "__main__":
    main()