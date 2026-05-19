# TTC Transit Delay & Cost Predictor

Predicts delay duration and estimates the economic cost of Toronto Transit Commission (TTC) incidents across Bus, Subway, and Streetcar using Gradient Boosting and 6 years of real operational data.

## The Problem

TTC operates over 200 bus routes, 4 subway lines, and 10 streetcar routes across Toronto. Every delay has a cost to riders, to operators, and to the city. The TTC publishes raw delay logs but no tool exists to translate those logs into dollar figures by route, incident type, or time of day.

This project builds that tool.

## What It Does

Ingests 6 years of TTC delay records (2020–2025) across all three transit modes, joins hourly weather data from Environment Canada, and trains a separate Gradient Boosting model per transit mode. The output is predicted delay in minutes and estimated cost in CAD per incident, along with identifying which routes, incident types, and conditions drive the highest economic cost.

## Results

| Mode | MAE | RMSE | Avg Cost / Incident | Total Est. Cost (2024) |
|------|-----|------|---------------------|------------------------|
| Bus | 11.77 min | 42.86 min | $24.26 CAD | $1,373,563 CAD |
| Subway | 2.47 min | 9.27 min | $8.27 CAD | $217,606 CAD |
| Streetcar | 12.97 min | 34.07 min | $33.10 CAD | $464,727 CAD |

The model achieves R² of ~0.27 for bus, consistent with published transit delay prediction research (typical range: 0.15–0.35 without real-time AVL data). Transit delays have high inherent randomness — a signal failure and a passenger medical emergency can produce identical feature vectors but vastly different outcomes. The model is designed for aggregate cost forecasting, not individual incident prediction.

An initial run produced R² of 0.95, which was caught and removed as data leakage (see Key Technical Decisions).

## Key Technical Decisions

**Time-based train/test split**
Train: 2020–2023, Test: 2024. Random splitting leaks future delay patterns into training, inflating metrics without improving real-world performance.

**Gap column exclusion**
Min Gap (time to next vehicle) and Gap Deviation are excluded from model features. Testing showed R² dropped from 0.95 to 0.27 upon removal, confirming the model was learning a circular relationship: large delays cause large gaps, so the model was effectively predicting delay from delay. The 0.27 figure is the honest one.

**COVID phase flag**
Rather than discarding 2020–2022 data entirely, a CovidPhase feature (Lockdown / Recovery / Normal) was added. This retains the extra training rows while letting the model learn that pandemic-era patterns differ from normal operations.

**Separate models per transit mode**
Bus, subway, and streetcar have fundamentally different delay causes. Streetcar delays are dominated by overhead wire issues and traffic blockages while subway delays are driven by signal failures and passenger incidents. A single combined model would blur these distinctions.

**Weather joined at hourly granularity**
Weather data from Environment Canada was joined on Year + Month + Day + Hour. IsSnowCondition (temperature at or below 2°C with active precipitation) was engineered specifically for streetcar overhead wire icing, a known high-delay scenario.

**Weather data decoupled into a standalone script**
download_weather.py handles all Environment Canada API calls separately. The modeling notebook reads a pre-saved CSV, ensuring the notebook runs reliably without a live internet connection.

## Top Predictive Features

1. Incident type — strongest predictor (0.50–0.60 importance for bus/subway)
2. RouteAvgDelay / RouteDelayStd — historical route risk profile
3. VehiclePrevDelay — vehicle wear signal (previous delay for same vehicle)
4. Hour of day — time-of-day operational patterns
5. Weather (Humidity, WindChill, Temp) — most impactful for streetcar

## Project Structure
