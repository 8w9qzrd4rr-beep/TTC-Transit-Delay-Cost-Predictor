"""
download_weather.py
-------------------
Downloads hourly weather data from Environment Canada for the Toronto City Centre
station and saves it as a single unified CSV.

Run this once before opening the modeling notebook:
    python download_weather.py

Output: data/weather_2023_2025.csv
"""

import io
import time
import requests
import pandas as pd

STATION_ID = 51459          # Toronto City Centre (Environment Canada)
YEARS      = [2023, 2024, 2025]
OUT_PATH   = "data/weather_2023_2025.csv"
TIMEOUT    = 15             # seconds per request
SLEEP      = 1.0            # polite delay between requests


def download_weather(station_id: int, years: list[int]) -> pd.DataFrame:
    all_months = []

    for year in years:
        for month in range(1, 13):
            url = (
                "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
                f"?format=csv&stationID={station_id}"
                f"&Year={year}&Month={month}&Day=1&timeframe=1&submit=Download+Data"
            )
            try:
                r = requests.get(url, timeout=TIMEOUT)
                r.raise_for_status()
                df_w = pd.read_csv(io.StringIO(r.text), skiprows=0)
                all_months.append(df_w)
                print(f"  ✓  {year}-{month:02d}  ({len(df_w):,} rows)")
            except Exception as e:
                print(f"  ✗  {year}-{month:02d}  FAILED: {e}")

            time.sleep(SLEEP)

    return pd.concat(all_months, ignore_index=True)


def clean_weather(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={
        "Time (LST)":          "Hour",
        "Temp (°C)":           "Temp",
        "Precip. Amount (mm)": "Precipitation",
        "Wind Spd (km/h)":    "WindSpeed",
        "Rel Hum (%)":         "Humidity",
        "Wind Chill":          "WindChill",
    })

    df = df[["Year", "Month", "Day", "Hour",
             "Temp", "Precipitation", "WindSpeed", "Humidity", "WindChill"]]

    df["Hour"]          = pd.to_datetime(df["Hour"], format="%H:%M").dt.hour
    df["Precipitation"] = df["Precipitation"].fillna(0)
    df["WindSpeed"]     = df["WindSpeed"].fillna(0)
    df["WindChill"]     = df["WindChill"].fillna(df["Temp"])
    df["Humidity"]      = df["Humidity"].fillna(df["Humidity"].median())
    df                  = df.dropna(subset=["Temp"])

    df["IsSnowCondition"] = (
        (df["Temp"] <= 2) & (df["Precipitation"] > 0)
    ).astype(int)

    return df


if __name__ == "__main__":
    print(f"Downloading weather data for years: {YEARS}")
    print(f"Station ID: {STATION_ID} (Toronto City Centre)\n")

    raw     = download_weather(STATION_ID, YEARS)
    weather = clean_weather(raw)

    weather.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(weather):,} rows → {OUT_PATH}")