# TTC Transit Delay & Cost Predictor

Predicts delay duration and estimates the economic cost of Toronto Transit Commission (TTC) incidents across Bus, Subway, and Streetcar using Gradient Boosting and 6 years of real operational data.

---

## The Problem

TTC operates over 200 bus routes, 4 subway lines, and 10 streetcar routes across Toronto. Every delay has a cost — to riders, to operators, and to the city. The TTC publishes raw delay logs but no tool exists to translate those logs into dollar figures by route, incident type, or time of day.

This project builds that tool.

---

## What It Does

- Ingests 6 years of TTC delay records (2020–2025) across all three transit modes
- Joins hourly weather data from Environment Canada to capture temperature, precipitation, wind, and snow conditions
- Engineers 20+ features including route risk profiles, vehicle wear signals, COVID phase flags, and time-of-day patterns
- Trains a separate Gradient Boosting model per transit mode
- Outputs predicted delay in minutes and estimated cost in CAD per incident
- Identifies which routes, incident types, and conditions drive the highest economic cost

---

## Results

| Mode       | MAE        | RMSE       | Avg Cost / Incident | Total Est. Cost (2024)  |
|------------|------------|------------|---------------------|-------------------------|
| Bus        | 11.77 min  | 42.82 min  | $24.32 CAD          | $1,377,055 CAD          |
| Subway     | 2.48 min   | 9.27 min   | $8.28 CAD           | $217,723 CAD            |
| Streetcar  | 13.02 min  | 34.07 min  | $33.25 CAD          | $466,820 CAD            |

**A note on R²:** The model achieves R² of ~0.27 for bus and streetcar, consistent with published transit delay prediction research (typical range: 0.15–0.35 without real-time AVL data). Transit delays have high inherent randomness — a signal failure and a passenger medical emergency can produce identical feature vectors but vastly different outcomes. The model is designed for **aggregate cost forecasting**, not individual incident prediction, where errors average out across thousands of records.

An initial run produced R² of 0.95 — which was caught and removed as data leakage (see below).

---

## Key Technical Decisions

**Time-based train/test split**
Train: 2020–2023 | Test: 2024. Random splitting leaks future delay patterns into training, inflating metrics without improving real-world performance.

**Gap column exclusion (leakage detection)**
`Min Gap` (time to next vehicle) and `GapDeviation` are excluded from model features. Testing showed R² dropped from 0.95 → 0.27 upon removal, confirming the model was learning a circular relationship: large delays cause large gaps, so the model was effectively predicting delay *from* delay. The 0.27 figure is the honest one.

**COVID phase flag instead of dropping years**
Rather than discarding 2020–2022 data entirely, a `CovidPhase` feature (Lockdown / Recovery / Normal) was added. This retains the extra training rows while letting the model discount abnormal operating periods.

**Separate models per transit mode**
Bus, subway, and streetcar have fundamentally different delay causes. Streetcar delays are dominated by overhead wire issues and traffic blockages; subway delays by signal failures and passenger incidents. A single combined model would blur these distinctions.

**Weather join at hourly granularity**
Weather data from Environment Canada was joined on Year + Month + Day + Hour. `IsSnowCondition` (temperature ≤ 2°C with active precipitation) was engineered specifically for streetcar overhead wire icing — a known high-delay scenario.

**Weather data decoupled into a standalone script**
`download_weather.py` handles all Environment Canada API calls separately. The modeling notebook reads a pre-saved `data/weather_2023_2025.csv`, ensuring 100% reproducibility when someone runs the notebook without a live internet connection or if the government server is slow.

---

## Top Predictive Features

Across all three modes, the most important features were:

1. **Incident type** — strongest predictor (0.50–0.60 importance for bus/subway)
2. **RouteAvgDelay / RouteDelayStd** — historical route risk profile
3. **VehiclePrevDelay** — vehicle wear signal (previous delay for same vehicle)
4. **Hour of day** — time-of-day operational patterns
5. **Weather (Humidity, WindChill, Temp)** — most impactful for streetcar

---

## Project Structure

```
ttc-delay-predictor/
│
├── README.md                        ← this file
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── download_weather.py              ← run once to fetch weather data before modeling
│
├── notebooks/
│   ├── 01_bus_cleaning.ipynb        ← raw bus CSVs → clean_bus.csv
│   ├── 02_subway_cleaning.ipynb     ← raw subway CSVs → clean_sub.csv
│   ├── 03_streetcar_cleaning.ipynb  ← raw streetcar CSVs → clean_streetcar.csv
│   └── 04_ttc_delay_predictor.ipynb ← feature engineering, modelling, cost estimation
│
├── data/
│   ├── clean_bus.csv                ← cleaned bus delay records (2020–2025)
│   ├── clean_sub.csv                ← cleaned subway delay records (2020–2025)
│   ├── clean_streetcar.csv          ← cleaned streetcar delay records (2020–2025)
│   └── weather_2023_2025.csv        ← hourly weather from Environment Canada
│
└── raw/                             ← not committed (see Data Sources below)
    ├── bus/
    ├── subway/
    └── streetcar/
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ttc-delay-predictor.git
cd ttc-delay-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-download raw TTC data from Toronto Open Data,
#    place files in raw/bus/, raw/subway/, raw/streetcar/,
#    then run the cleaning notebooks in order (01 → 02 → 03).
#    Pre-cleaned CSVs are already provided in data/ if you want to skip this.

# 4. Fetch weather data (run once)
python download_weather.py

# 5. Open the modelling notebook
jupyter notebook notebooks/04_ttc_delay_predictor.ipynb
```

---

## Data Sources

| Dataset | Source |
|---|---|
| TTC Bus Delay Data | [Toronto Open Data](https://open.toronto.ca/dataset/ttc-bus-delay-data/) |
| TTC Subway Delay Data | [Toronto Open Data](https://open.toronto.ca/dataset/ttc-subway-delay-data/) |
| TTC Streetcar Delay Data | [Toronto Open Data](https://open.toronto.ca/dataset/ttc-streetcar-delay-data/) |
| Hourly Weather | Environment Canada — Toronto City Centre Station (ID: 51459) |

All data is publicly available at no cost. Raw TTC files are not committed to this repo due to size; pre-cleaned CSVs are provided in `data/`.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · requests · Jupyter