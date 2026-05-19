"""
model.py
--------
Feature engineering, model training, evaluation, and inference
for TTC transit delay cost prediction.

Trains a separate Gradient Boosting model per transit mode
(Bus, Subway, Streetcar) and outputs predicted delay in minutes
and estimated cost in CAD.

Run:
    python model.py

Prerequisites:
    - data/clean_bus.csv
    - data/clean_sub.csv
    - data/clean_streetcar.csv
    - data/weather_2023_2025.csv

    Run the clean_*.py scripts and download_weather.py first if these
    files are missing.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Cost per delay minute by transit mode (CAD)
# Source: TTC Annual Report — operating cost per vehicle hour, divided by 60
COST_PER_MIN = {
    'Bus':       1.23,
    'Subway':    2.87,
    'Streetcar': 1.95,
}

TARGET = 'Min Delay'

# Min Gap and GapDeviation are intentionally excluded — gap size is partly caused
# by the delay itself (circular leakage). Removing them dropped R² from 0.95 to 0.27.
FEATURES = [
    'Hour', 'Month', 'DayOfWeek',
    'IsRushHour', 'IsWeekend', 'IsNight',
    'Direction_known',
    'Season', 'CovidPhase', 'Incident', 'Vehicle', 'Route',
    'RouteAvgDelay', 'RouteDelayStd',
    'VehiclePrevDelay',
    'Temp', 'Precipitation', 'WindSpeed',
    'Humidity', 'WindChill', 'IsSnowCondition',
]

# Loaded once, reused across all three modes in engineer_features()
weather = None


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_bus       = pd.read_csv('data/clean_bus.csv')
    df_sub       = pd.read_csv('data/clean_sub.csv')
    df_streetcar = pd.read_csv('data/clean_streetcar.csv')

    print(f'Bus rows      : {len(df_bus):,}')
    print(f'Subway rows   : {len(df_sub):,}')
    print(f'Streetcar rows: {len(df_streetcar):,}')

    return df_bus, df_sub, df_streetcar


def load_weather() -> pd.DataFrame:
    global weather
    weather = pd.read_csv('data/weather_2023_2025.csv')
    print(f'Weather rows  : {len(weather):,}')
    return weather


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline to a transit delay DataFrame.

    Steps:
        1. Normalize route column name (Line -> Route for subway/streetcar)
        2. Engineer time-based binary flags
        3. Assign season from month
        4. Assign COVID phase from year
        5. Join hourly weather data
        6. Compute route-level historical aggregates
        7. Compute vehicle lag feature (previous delay)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transit delay dataframe with columns:
        Hour, DayOfWeek, Month, Year, Day, Route/Line, Vehicle, Min Delay, Min Gap

    Returns
    -------
    pd.DataFrame with all engineered features added
    """
    df = df.copy()

    # Normalize route column: subway and streetcar use 'Line', bus uses 'Route'
    if 'Line' in df.columns and 'Route' not in df.columns:
        df = df.rename(columns={'Line': 'Route'})

    # Time-based binary flags
    df['IsRushHour'] = np.where(
        df['Hour'].between(7, 9) | df['Hour'].between(16, 18), 1, 0
    )
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsNight']   = np.where(df['Hour'].between(0, 5), 1, 0)

    # Season from month
    season_conditions = [
        df['Month'].isin([12, 1, 2]),
        df['Month'].between(3, 5),
        df['Month'].between(6, 8),
        df['Month'].between(9, 11),
    ]
    df['Season'] = np.select(
        season_conditions,
        ['Winter', 'Spring', 'Summer', 'Fall'],
        default='Unknown',
    )

    # COVID phase flag — keeps 2020-2022 in training with a discount signal
    df['CovidPhase'] = 'Normal'
    df.loc[df['Year'] <= 2021, 'CovidPhase'] = 'Lockdown'
    df.loc[df['Year'] == 2022, 'CovidPhase'] = 'Recovery'

    # Join hourly weather — rows with no match (COVID years) get median-filled
    df = df.merge(weather, on=['Year', 'Month', 'Day', 'Hour'], how='left')
    for col in ['Precipitation', 'WindSpeed', 'IsSnowCondition']:
        df[col] = df[col].fillna(0)
    for col in ['Temp', 'Humidity', 'WindChill']:
        df[col] = df[col].fillna(df[col].median())

    # Route-level historical aggregates
    route_stats = df.groupby('Route')[TARGET].agg(
        RouteAvgDelay='mean',
        RouteDelayStd='std',
    ).reset_index()
    df = df.merge(route_stats, on='Route', how='left')

    # Gap deviation (computed for reference, not used as model feature)
    route_avg_gap      = df.groupby('Route')['Min Gap'].transform('mean')
    df['GapDeviation'] = df['Min Gap'] - route_avg_gap

    # Vehicle lag feature — previous delay for the same vehicle
    df = df.sort_values(['Vehicle', 'Year', 'Month', 'Day', 'Hour'])
    df['VehiclePrevDelay'] = df.groupby('Vehicle')[TARGET].shift(1)
    df['VehiclePrevDelay'] = df['VehiclePrevDelay'].fillna(df['RouteAvgDelay'])

    return df


def encode(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all object-type columns.

    Parameters
    ----------
    df       : DataFrame to encode
    encoders : existing LabelEncoder objects (pass during inference)
    fit      : True = fit new encoders (training), False = transform only (inference)

    Returns
    -------
    (encoded DataFrame, encoder dictionary)
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in df.select_dtypes(include='object').columns:
        if col == TARGET:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    return df, encoders


def train_and_evaluate(datasets: dict) -> dict:
    """
    Train one GradientBoostingRegressor per transit mode and print evaluation metrics.

    Train: 2020-2023
    Test:  2024

    Returns
    -------
    dict of results keyed by mode name
    """
    results = {}

    for name, df in datasets.items():
        print(f"\n{'='*54}")
        print(f'  {name}')
        print(f"{'='*54}")

        features = [f for f in FEATURES if f in df.columns]
        data     = df[features + [TARGET, 'Year']].dropna()

        train = data[data['Year'] <= 2023]
        test  = data[data['Year'] == 2024]

        if len(test) == 0:
            print(f'  No 2024 data — skipping {name}')
            continue

        train_enc, encoders = encode(train[features + [TARGET]], fit=True)
        test_enc,  _        = encode(test[features + [TARGET]],  fit=False, encoders=encoders)

        X_train = train_enc[features]
        y_train = train_enc[TARGET]
        X_test  = test_enc[features]
        y_test  = test_enc[TARGET]

        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = np.clip(model.predict(X_test), 0, None)

        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        cost_per_min         = COST_PER_MIN[name]
        avg_predicted_cost   = y_pred.mean() * cost_per_min
        total_predicted_cost = y_pred.sum()  * cost_per_min

        print(f'  Rows (train): {len(X_train):,}  |  Rows (test): {len(X_test):,}')
        print(f'  R2  : {r2:.4f}')
        print(f'  MAE : {mae:.2f} min')
        print(f'  RMSE: {rmse:.2f} min')
        print(f'\n  Cost per minute  : ${cost_per_min:.2f} CAD')
        print(f'  Avg cost/incident: ${avg_predicted_cost:.2f} CAD')
        print(f'  Total est. cost  : ${total_predicted_cost:,.0f} CAD  (2024 test period)')

        fi    = pd.Series(model.feature_importances_, index=features)
        top10 = fi.sort_values(ascending=False).head(10)
        print('\n  Top 10 Feature Importances:')
        for feat, imp in top10.items():
            print(f'    {feat:<22} {imp:.4f}')

        results[name] = {
            'model':                 model,
            'encoders':              encoders,
            'features':              features,
            'r2':                    r2,
            'mae':                   mae,
            'rmse':                  rmse,
            'avg_cost_per_incident': avg_predicted_cost,
            'total_cost':            total_predicted_cost,
            'y_test':                y_test,
            'y_pred':                y_pred,
        }

    return results


def print_summary(results: dict) -> None:
    rows = []
    for name, v in results.items():
        rows.append({
            'Mode':              name,
            'R2':                round(v['r2'],   4),
            'MAE (min)':         round(v['mae'],  2),
            'RMSE (min)':        round(v['rmse'], 2),
            'Avg Cost/Incident': f"${v['avg_cost_per_incident']:.2f} CAD",
            'Total Cost (2024)': f"${v['total_cost']:,.0f} CAD",
        })

    summary = pd.DataFrame(rows).set_index('Mode')
    print('\n\nSummary')
    print(summary.to_string())


def predict_delay_cost(
    mode: str,
    new_df: pd.DataFrame,
    results: dict,
) -> pd.DataFrame:
    """
    Predict delay duration and estimated cost for new incident rows.

    Parameters
    ----------
    mode    : 'Bus', 'Subway', or 'Streetcar'
    new_df  : DataFrame with feature columns (weather columns joined in beforehand)
    results : output from train_and_evaluate()

    Returns
    -------
    Input DataFrame with two added columns:
        predicted_delay_min, predicted_cost_cad
    """
    if mode not in results:
        raise ValueError(f"Mode '{mode}' not found. Run train_and_evaluate() first.")

    r        = results[mode]
    features = r['features']
    encoders = r['encoders']
    model    = r['model']

    df = new_df.copy()

    if 'CovidPhase' not in df.columns:
        df['CovidPhase'] = 'Normal'
    if 'IsRushHour' not in df.columns:
        df['IsRushHour'] = np.where(
            df['Hour'].between(7, 9) | df['Hour'].between(16, 18), 1, 0
        )
    if 'IsWeekend' not in df.columns:
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    if 'IsNight' not in df.columns:
        df['IsNight'] = np.where(df['Hour'].between(0, 5), 1, 0)
    if 'IsSnowCondition' not in df.columns and 'Temp' in df.columns:
        df['IsSnowCondition'] = (
            (df['Temp'] <= 2) & (df['Precipitation'] > 0)
        ).astype(int)

    df, _ = encode(df, encoders=encoders, fit=False)

    preds = np.clip(model.predict(df[features]), 0, None)
    df['predicted_delay_min'] = preds
    df['predicted_cost_cad']  = preds * COST_PER_MIN[mode]

    return df


if __name__ == '__main__':
    print('Loading data...')
    df_bus, df_sub, df_streetcar = load_data()
    load_weather()

    print('\nEngineering features...')
    df_bus       = engineer_features(df_bus)
    df_sub       = engineer_features(df_sub)
    df_streetcar = engineer_features(df_streetcar)

    datasets = {
        'Bus':       df_bus,
        'Subway':    df_sub,
        'Streetcar': df_streetcar,
    }

    results = train_and_evaluate(datasets)
    print_summary(results)

    # Example inference
    example = pd.DataFrame([{
        'Hour': 8, 'Month': 1, 'DayOfWeek': 0,
        'Direction_known': 1, 'Season': 'Winter', 'CovidPhase': 'Normal',
        'Incident': 'Mechanical', 'Vehicle': 8001, 'Route': 29,
        'RouteAvgDelay': 4.2, 'RouteDelayStd': 3.1, 'VehiclePrevDelay': 3.0,
        'Temp': -8.0, 'Precipitation': 0.0, 'WindSpeed': 25.0,
        'Humidity': 72.0, 'WindChill': -15.0, 'IsSnowCondition': 0,
    }])

    prediction = predict_delay_cost('Bus', example, results)
    print(f"\nExample prediction (Bus Route 29, winter morning rush):")
    print(f"  Predicted delay : {prediction['predicted_delay_min'].values[0]:.1f} minutes")
    print(f"  Estimated cost  : ${prediction['predicted_cost_cad'].values[0]:.2f} CAD")