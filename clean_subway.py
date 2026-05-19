"""
clean_subway.py
---------------
Cleans and consolidates TTC subway delay records (2020-2025).

Input:  raw/subway/subway_2020.csv ... subway_2025.csv
        raw/subway/subway_code.csv
Output: data/clean_sub.csv

Run:
    python clean_subway.py
"""

import pandas as pd

RAW_DIR  = 'raw/subway'
CODE_CSV = f'{RAW_DIR}/subway_code.csv'
OUT_CSV  = 'data/clean_sub.csv'

DIRECTION_MAP = {
    'N': 'NB', 'n': 'NB', 'NB': 'NB', 'nb': 'NB', 'N/B': 'NB', 'n/b': 'NB', 'BN': 'NB',
    'S': 'SB', 's': 'SB', 'SB': 'SB', 'S/B': 'SB',
    'E': 'EB', 'ee': 'EB', 'EB': 'EB', 'E/B': 'EB', 'e/b': 'EB',
    'W': 'WB', 'WB': 'WB', 'wb': 'WB', 'W/B': 'WB',
    'OB': 'OB', 'ob': 'OB', 'O/B': 'OB', 'o/b': 'OB',
    'B': 'BOTH', 'BW': 'BOTH', 'B/W': 'BOTH',
}


def load_raw() -> pd.DataFrame:
    dfs = [pd.read_csv(f'{RAW_DIR}/subway_{y}.csv') for y in range(2020, 2026)]

    # 2025 has an extra _id column
    dfs[5] = dfs[5].drop(columns=['_id'])

    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Map incident codes to descriptions
    codes = pd.read_csv(CODE_CSV)
    code_map = dict(zip(codes['CODE'], codes['DESCRIPTION']))
    df['Incident'] = df['Code'].map(code_map)
    df = df.drop(columns=['Code'])

    # Drop rows with missing key features
    df = df.dropna(subset=['Incident', 'Line'])

    # Direction: fill nulls, create known/unknown flag
    df['Bound'] = df['Bound'].fillna('UNKNOWN')
    df['Direction_known'] = (df['Bound'] != 'UNKNOWN').astype(int)

    # Standardize column names
    df = df.rename(columns={
        'Station': 'Location',
        'Bound':   'Direction',
        'Line':    'Route',
    })

    # Extract time features
    df['Hour']      = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    df['Minute']    = pd.to_datetime(df['Time'], format='%H:%M').dt.minute
    df['Month']     = pd.to_datetime(df['Date']).dt.month
    df['Year']      = pd.to_datetime(df['Date']).dt.year
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Day']       = pd.to_datetime(df['Date']).dt.day
    df = df.drop(columns=['Date', 'Time'])

    df['Direction_clean'] = df['Direction'].str.strip().map(pd.Series(DIRECTION_MAP))

    return df


if __name__ == '__main__':
    print('Loading raw subway data...')
    df = load_raw()
    print(f'  Raw rows: {len(df):,}')

    df = clean(df)
    print(f'  Clean rows: {len(df):,}')

    df.to_csv(OUT_CSV, index=False)
    print(f'  Saved to {OUT_CSV}')