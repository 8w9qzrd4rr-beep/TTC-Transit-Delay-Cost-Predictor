"""
clean_streetcar.py
------------------
Cleans and consolidates TTC streetcar delay records (2020-2025).

Input:  raw/streetcar/streetcar_2020.csv ... streetcar_2025.csv
        raw/streetcar/streetcar_code.csv
Output: data/clean_streetcar.csv

Run:
    python clean_streetcar.py
"""

import pandas as pd

RAW_DIR  = 'raw/streetcar'
CODE_CSV = f'{RAW_DIR}/streetcar_code.csv'
OUT_CSV  = 'data/clean_streetcar.csv'

DIRECTION_MAP = {
    'N': 'NB', 'n': 'NB', 'NB': 'NB', 'nb': 'NB', 'N/B': 'NB',
    'S': 'SB', 's': 'SB', 'SB': 'SB', 'S/B': 'SB',
    'E': 'EB', 'EB': 'EB', 'E/B': 'EB', 'e/b': 'EB', 'ew': 'EB',
    'W': 'WB', 'WB': 'WB', 'wb': 'WB', 'W/B': 'WB',
    'B': 'BOTH', 'B/W': 'BOTH', 'bw': 'BOTH', 'btw': 'BOTH',
}

NOISE_VALUES = {'UNKNOWN', 'r', 'T', 'Q', 'C', '5', '`', '1', '8'}


def normalize_time(t: str) -> str:
    t = str(t).strip()
    if len(t) == 8 and t[2] == ':' and t[5] == ':':  # HH:MM:SS
        return t[:5]
    return t  # already HH:MM


def clean_direction(series: pd.Series) -> pd.Series:
    return (
        series
        .str.strip()
        .map(lambda x: DIRECTION_MAP.get(x, None) if x not in NOISE_VALUES else None)
    )


def load_raw() -> pd.DataFrame:
    # 2020: different column names
    df2020 = pd.read_csv(f'{RAW_DIR}/streetcar_2020.csv')
    df2020 = df2020.rename(columns={
        'Report Date': 'Date',
        'Route':       'Line',
        'Delay':       'Min Delay',
        'Gap':         'Min Gap',
        'Direction':   'Bound',
    })

    # 2021-2024: consistent schema
    middle = pd.concat(
        [pd.read_csv(f'{RAW_DIR}/streetcar_{y}.csv') for y in [2021, 2022, 2023, 2024]],
        ignore_index=True,
    )

    # 2025: has _id, uses Code instead of Incident, route name appended to line number
    df2025 = pd.read_csv(f'{RAW_DIR}/streetcar_2025.csv').drop(columns=['_id'])
    codes = pd.read_csv(CODE_CSV)
    code_map = dict(zip(codes['CODE'], codes['DESCRIPTION']))
    df2025['Incident'] = df2025['Code'].map(code_map)
    df2025 = df2025.drop(columns=['Code'])
    df2025 = df2025.rename(columns={'Station': 'Location'})
    df2025['Line'] = df2025['Line'].str.extract(r'^(\d+)').astype('float')

    return pd.concat([df2020, middle, df2025], ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing key features
    df = df.dropna(subset=['Line', 'Location', 'Incident'])
    df = df.dropna(subset=['Vehicle', 'Min Delay', 'Min Gap'])

    # Direction: fill nulls, create known/unknown flag
    df['Bound'] = df['Bound'].fillna('UNKNOWN')
    df['Direction_known'] = (df['Bound'] != 'UNKNOWN').astype(int)

    # Normalize time and extract features
    df['Time']   = df['Time'].apply(normalize_time)
    df['Hour']   = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    df['Minute'] = pd.to_datetime(df['Time'], format='%H:%M').dt.minute
    df = df.drop(columns=['Time'])

    df['Date']      = pd.to_datetime(df['Date'], format='mixed')
    df['Month']     = df['Date'].dt.month
    df['Year']      = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Day']       = df['Date'].dt.day
    df = df.drop(columns=['Date'])

    df['Direction_clean'] = clean_direction(df['Bound'])

    return df


if __name__ == '__main__':
    print('Loading raw streetcar data...')
    df = load_raw()
    print(f'  Raw rows: {len(df):,}')

    df = clean(df)
    print(f'  Clean rows: {len(df):,}')

    df.to_csv(OUT_CSV, index=False)
    print(f'  Saved to {OUT_CSV}')