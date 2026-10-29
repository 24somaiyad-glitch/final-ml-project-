# 🚦 Smart Traffic Congestion Prediction Using Machine Learning

Predicts traffic level (Low / Medium / High) from hour, day, weather, area, holiday, event and previous traffic (no vehicle counts needed). Live mode fills time and weather automatically and shows a **Google Maps live traffic map and a live traffic reading for the selected area**. Manual mode adds an **estimated vehicle count (Linear Regression)**.

## Run locally
```bash
pip install -r requirements.txt
python generate_dataset.py     # creates data/traffic.csv  (skip if you use a Kaggle CSV)
python train.py                # cleans data, splits, trains 5 models, saves figures/ and models/
streamlit run app.py           # opens the web UI at http://localhost:8501
python find_s.py               # Find-S concept demo
```

## Project structure
| File | Purpose |
|---|---|
| generate_dataset.py | builds a dirty synthetic dataset (missing values, duplicates, outliers) |
| features.py | shared feature engineering used by train.py and app.py |
| live_data.py | real-time clock, Open-Meteo weather, optional TomTom live traffic, Google Maps traffic map |
| train.py | cleaning -> split -> 5 classifiers + 2 Linear Regressions (delay, vehicle count) -> metrics & figures |
| app.py | Streamlit user interface |
| find_s.py | Find-S hypothesis learning demo |
| figures/ | all graphs/tables used in the report |
| models/ | saved best model, delay regressor, vehicle regressor + metadata |

## Deploy (free, works on phone/laptop browser)
1. Push this folder to a public GitHub repo (include `models/` and `figures/`).
2. Go to https://share.streamlit.io -> sign in with GitHub -> New app.
3. Pick the repo, branch `main`, main file `app.py` -> Deploy.
4. Copy the public URL into your report and README.

## Optional keys (Streamlit Cloud: App settings -> Secrets; local: .streamlit/secrets.toml, see secrets.toml.example)
- `GOOGLE_MAPS_KEY` - full live traffic layer on the Live-mode map (see REALTIME_GUIDE.md). Without it a basic map and an "Open in Google Maps" link are shown.
- `GOOGLE_ROUTES_KEY` - server key (Routes API) for the automatic Google live traffic level and delay in Live mode.
- `TOMTOM_KEY` - live road speed used as "current traffic".
