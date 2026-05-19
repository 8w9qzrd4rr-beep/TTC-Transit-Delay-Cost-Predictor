"""
clean_bus.py
------------
Cleans and consolidates TTC bus delay records (2020-2025).

Input:  raw/bus/bus_2020.csv ... bus_2025.csv
        raw/bus/bus_code.csv
Output: data/clean_bus.csv

Run:
    python clean_bus.py
"""

import numpy as np
import pandas as pd

RAW_DIR  = 'raw/bus'
CODE_CSV = f'{RAW_DIR}/bus_code.csv'
OUT_CSV  = 'data/clean_bus.csv'

DIRECTION_MAP = {
    'N': 'NB', 'n': 'NB', 'NB': 'NB', 'nb': 'NB', 'N/B': 'NB', 'n/b': 'NB', 'BN': 'NB',
    'S': 'SB', 's': 'SB', 'SB': 'SB', 'S/B': 'SB',
    'E': 'EB', 'ee': 'EB', 'EB': 'EB', 'E/B': 'EB', 'e/b': 'EB',
    'W': 'WB', 'WB': 'WB', 'wb': 'WB', 'W/B': 'WB',
    'OB': 'OB', 'ob': 'OB', 'O/B': 'OB', 'o/b': 'OB',
    'B': 'BOTH', 'BW': 'BOTH', 'B/W': 'BOTH',
}

NOISE_VALUES = {
    'UNKNOWN', 'Service adjsuted.', 'Chan', 'Cab', 'St',
    'RAD', 'nn', '/', '\\', '`',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'I', 'J', 'Q', 'G', 'D', 'M', 'A', 'L', 'H', 'T',
}


def normalize_time(t: str) -> str:
    t = str(t).strip()
    if 'T' in t or (len(t) > 8 and '-' in t):
        return pd.to_datetime(t).strftime('%H:%M')
    elif len(t) == 8:
        return t[:5]
    return t


def clean_direction(series: pd.Series) -> pd.Series:
    return (
        series
        .str.strip()
        .map(lambda x: DIRECTION_MAP.get(x, None) if x not in NOISE_VALUES else None)
    )


def load_raw() -> pd.DataFrame:
    # 2021-2024: consistent schema, concat as-is
    unified = pd.concat(
        [pd.read_csv(f'{RAW_DIR}/bus_{y}.csv') for y in [2021, 2022, 2023, 2024]],
        ignore_index=True,
    )

    # 2020: different column names
    df2020 = pd.read_csv(f'{RAW_DIR}/bus_2020.csv')
    df2020.columns = ['Date', 'Route', 'Time', 'Day', 'Location',
                      'Incident', 'Min Delay', 'Min Gap', 'Direction', 'Vehicle']
    unified = pd.concat([unified, df2020], ignore_index=True)

    # 2025: has _id, uses Code instead of Incident
    df2025 = pd.read_csv(f'{RAW_DIR}/bus_2025.csv').drop(columns=['_id'])
    codes = pd.read_csv(CODE_CSV)
    code_map = dict(zip(codes['CODE'], codes['DESCRIPTION']))
    df2025['Incident'] = df2025['Code'].map(code_map)
    df2025 = df2025.drop(columns=['Code'])
    df2025.columns = ['Date', 'Route', 'Time', 'Day', 'Location',
                      'Min Delay', 'Min Gap', 'Direction', 'Vehicle', 'Incident']

    return pd.concat([unified, df2025], ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows where key features are missing
    df = df.dropna(subset=['Route', 'Location'])
    df = df.dropna(subset=['Vehicle', 'Incident', 'Min Delay', 'Min Gap'])

    # Direction: fill nulls, create known/unknown flag
    df['Direction'] = df['Direction'].fillna('UNKNOWN')
    df['Direction_known'] = (df['Direction'] != 'UNKNOWN').astype(int)

    # Vehicle 0 means no vehicle recorded
    df['Vehicle'] = df['Vehicle'].replace(0, np.nan)

    # Normalize time and extract features
    df['Time']   = df['Time'].apply(normalize_time)
    df['Time']   = pd.to_datetime(df['Time'], format='%H:%M').dt.time
    df['Hour']   = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S').dt.hour
    df['Minute'] = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S').dt.minute
    df = df.drop(columns=['Time'])

    df['Date']      = pd.to_datetime(df['Date'], format='mixed')
    df['Month']     = df['Date'].dt.month
    df['Year']      = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Day']       = df['Date'].dt.day
    df = df.drop(columns=['Date'])

    df['Direction_clean'] = clean_direction(df['Direction'])

    return df


if __name__ == '__main__':
    print('Loading raw bus data...')
    df = load_raw()
    print(f'  Raw rows: {len(df):,}')

    df = clean(df)
    print(f'  Clean rows: {len(df):,}')

    df.to_csv(OUT_CSV, index=False)
    print(f'  Saved to {OUT_CSV}')