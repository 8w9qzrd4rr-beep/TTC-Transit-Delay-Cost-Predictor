# TTC Transit Delay & Cost Predictor

Predicts delay duration and estimates the economic cost of Toronto Transit Commission (TTC) incidents across Bus, Subway, and Streetcar using Gradient Boosting and 6 years of real operational data.

---

## The Problem

TTC operates over 200 bus routes, 4 subway lines, and 10 streetcar routes across Toronto. Every delay has a cost — to the city, to operators, and to riders. The TTC publishes raw delay logs but no tool exists to translate those logs into dollar figures by route, incident type, or time of day.

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

| Mode | MAE | RMSE | Avg Cost/Incident | Total Estimated Cost (2024) |
|---|---|---|---|---|
| Bus | 11.77 min | 42.82 min | $24.32 CAD | $1,377,055 CAD |
| Subway | 2.48 min | 9.27 min | $8.28 CAD | $217,723 CAD |
| Streetcar | 13.02 min | 34.07 min | $33.25 CAD | $466,820 CAD |

**On R²:** The model achieves R² of ~0.27 for bus and streetcar, consistent with published transit delay prediction research (typical range: 0.15–0.35 without real-time AVL data). Transit delays have high inherent randomness — a signal failure and a passenger medical emergency can produce identical feature vectors but vastly different delays. The model is designed for **aggregate cost forecasting**, not individual incident prediction, where errors average out across thousands of records.

An initial run produced R² of 0.95 — which was identified and removed as data leakage. The `Min Gap` column (time to next vehicle) is partially caused by the delay itself, making it circular. Removing it produced the honest 0.27 figure.

---

## Key Technical Decisions

**Time-based train/test split**
Train: 2020–2023 | Test: 2024. Random splitting would leak future delay patterns into training, inflating metrics without improving real-world performance.

**Gap column exclusion (leakage detection)**
`Min Gap` and `GapDeviation` were removed after testing showed R² dropped from 0.95 → 0.27 upon removal — confirming the model was learning a circular relationship, not a predictive one.

**COVID phase flag instead of dropping years**
Rather than discarding 2020–2022 data entirely, a `CovidPhase` feature (Lockdown / Recovery / Normal) was added. This keeps the additional training rows while letting the model learn that pandemic-era patterns differ from normal operations.

**Separate models per mode**
Bus, subway, and streetcar have fundamentally different delay causes. Streetcar delays are dominated by overhead wire issues and traffic blockages; subway delays by signal failures and passenger incidents. A single combined model would blur these distinctions.

**Weather join at hourly granularity**
Weather data from Environment Canada was joined on Year + Month + Day + Hour. `IsSnowCondition` (temperature ≤ 2°C with active precipitation) was engineered specifically for streetcar overhead wire icing — a known high-delay scenario.

---

## Top Predictive Features

Across all three modes, the most important features were:

1. **Incident type** — the single strongest predictor (0.50–0.60 importance for bus/subway)
2. **RouteAvgDelay / RouteDelayStd** — historical route risk profile
3. **VehiclePrevDelay** — vehicle wear signal (previous delay for the same vehicle)
4. **Hour of day** — time-of-day operational patterns
5. **Weather** (Humidity, WindChill, Temp) — strongest for streetcar

---

## Project Structure

```
ttc-delay-predictor/
│
├── raw/
│   ├── bus/              ← TTC bus delay CSVs (2020–2025)
│   ├── subway/           ← TTC subway delay CSVs (2020–2025)
│   └── streetcar/        ← TTC streetcar delay CSVs (2020–2025)
│
├── notebooks/
│   ├── bus_cleaning.ipynb
│   ├── subway_cleaning.ipynb
│   ├── streetcar_cleaning.ipynb
│   └── ttc_delay_predictor.ipynb   ← main model notebook
│
├── clean_bus.csv
├── clean_sub.csv
├── clean_streetcar.csv
│
└── README.md
```

---

## Data Sources

- **TTC Bus Delay Data** — [Toronto Open Data](https://open.toronto.ca/dataset/ttc-bus-delay-data/)
- **TTC Subway Delay Data** — [Toronto Open Data](https://open.toronto.ca/dataset/ttc-subway-delay-data/)
- **TTC Streetcar Delay Data** — [Toronto Open Data](https://open.toronto.ca/dataset/ttc-streetcar-delay-data/)
- **Hourly Weather** — Environment Canada, Toronto City Centre Station (ID: 51459)

All data is publicly available at no cost.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · requests · Jupyter
