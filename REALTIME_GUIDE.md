# Real-Time Implementation Guide

## Why vehicle count was removed
Nobody types "450 vehicles" into an app. We now use inputs that are **known automatically or visible to any commuter**:

| Input | Where it comes from in real time |
|---|---|
| Hour, Day | System clock (automatic) |
| Weather | Open-Meteo API - free, no key (automatic) |
| Area / road | User picks from a dropdown |
| Previous / current traffic | TomTom Traffic API live speed (automatic), or the user picks what they see |
| Public holiday, Event nearby | Two checkboxes |
| (derived) Time_Slot, Is_Weekend, Hour_sin/cos | Computed in `features.py` |

The model also predicts **Delay_Minutes** (extra minutes for a 10 km trip) with Linear Regression. Delay can be measured from any maps travel-time API, so a real dataset can be collected.

## Architecture
```
Clock + Open-Meteo weather + (TomTom live speed)  ->  live_data.py
                         |
                 features.py (same code as training)
                         |
        best_model.pkl -> Low / Medium / High   +   delay_regressor.pkl -> minutes
                         |
                 Streamlit UI (Live mode) + 24-hour forecast
```

## What works now (tested vs not tested)
- Clock, feature engineering, model prediction, 24-hour forecast: tested.
- Open-Meteo weather: coded with a safe fallback. If the internet or API fails, the app asks the user to pick the weather. I could not test the live call from my environment, so check it on your Mac (the Weather tile shows "(live)" when it works).
- TomTom live traffic: optional and untested. Without a key the app asks "Traffic you see right now".

## Turn on TomTom live traffic (optional, free tier)
1. Create a free account at developer.tomtom.com and copy an API key.
2. Edit `AREA_POINTS` in `live_data.py` with a real road coordinate for each area, and `CITY_LAT/LON`, `TZ`.
3. Local run: `export TOMTOM_KEY="your_key"` then `streamlit run app.py`.
4. Streamlit Cloud: App settings -> Secrets -> add `TOMTOM_KEY = "your_key"`.

## Going fully real-time later (future work to mention)
1. Log live speed + weather every 15 min into a CSV/database (a scheduled script).
2. After a few weeks, retrain `train.py` on that real data.
3. Replace the synthetic CSV; no other code changes are needed.
4. Add forecast weather (Open-Meteo hourly) so the 24-hour chart uses future weather.

## How to explain it in the viva
"Instead of asking users for a vehicle count, we use inputs available in real time: the clock, a free weather API and, optionally, live road speed. The same feature code is used in training and in the app, so predictions are consistent."

## Google Maps live traffic map (Live mode)
Shown under the live tiles for the selected area (coordinates come from `AREA_POINTS` in `live_data.py` - edit them to real roads in your city).

| Setup | What you get |
|---|---|
| With `GOOGLE_MAPS_KEY` | Official Google Maps JavaScript map with the live **TrafficLayer** (green/orange/red roads) and a marker |
| Without a key | Basic keyless Google map embed (unofficial; the traffic overlay may not appear) + a link that opens Google Maps with traffic switched on |

How to get a key: Google Cloud Console -> create a project -> enable **Maps JavaScript API** -> turn on billing (Google gives a free monthly credit) -> Credentials -> Create API key.
Protect it: Application restrictions = HTTP referrers (`localhost:8501/*` and `*.streamlit.app/*`), API restriction = Maps JavaScript API only. The key is visible in the browser, so never leave it unrestricted.
Add it: local `export GOOGLE_MAPS_KEY="..."` or `.streamlit/secrets.toml`; Streamlit Cloud -> App settings -> Secrets -> `GOOGLE_MAPS_KEY = "..."`.
If the map shows an error message about the key, the API is not enabled, billing is off, or the referrer restriction does not match.
The map is for display. It is not an input to the model (the model's "current traffic" still comes from TomTom or from you).

## Estimated vehicle count (Manual mode)
Manual mode shows `~N vehicles per hour`, predicted by a second **Linear Regression** (`models/vehicle_regressor.pkl`) from the same inputs as the classifier (hour, day, weather, area, holiday, event, previous traffic). Vehicle count is never typed in by the user.
- Trained on the simulated `Vehicles` column made by `generate_dataset.py` (cleaned like the other columns: median fill, IQR cap).
- Reported in the app and `models/meta.json` as MAE, RMSE and R2. Figure: `figures/10_vehicle_regression.png`.
- Manual mode also draws the estimated vehicles for all 24 hours of the chosen day.
- It is an estimate from simulated data, not a measured count. With real counter data, replace the `Vehicles` column and re-run `train.py`.

## Live traffic reading from Google (Live mode)
When a Google key is set, the app asks the **Google Routes API** (TRAFFIC_AWARE) for a short drive inside the selected area and compares the trip time now with the free-flow time. This gives the "Current traffic" level automatically (Google first, then TomTom, then your own choice).
- ratio = free-flow time / current time: above 0.75 = Low, 0.5 to 0.75 = Medium, below 0.5 = High.
- Also shown: live delay (minutes per 10 km), trip time now, and a comparison with the delay the model expects for this hour.
- Readings are cached for 60 seconds ("Refresh live data" clears the cache). Each fresh reading is one Routes API request, which is billed after Google's free monthly allowance.
- Each area uses a ~3 km trip starting at its `AREA_POINTS` coordinate. For better readings set a real busy road per area in `AREA_ROUTES` in `live_data.py`: `{"Downtown": ((lat1, lon1), (lat2, lon2)), ...}`.

Keys (important):
| Secret | Used for | How to restrict |
|---|---|---|
| `GOOGLE_MAPS_KEY` | Map in the browser (Maps JavaScript API) | HTTP referrers: `localhost:8501/*`, `*.streamlit.app/*` |
| `GOOGLE_ROUTES_KEY` | Live reading from the server (Routes API) | API restriction = Routes API only. Do NOT use referrer restriction: server requests send no referrer and would be blocked |

If you only set `GOOGLE_MAPS_KEY` and it has a referrer restriction, the map works but the live reading shows "Google live reading unavailable: ...". Create a second key for `GOOGLE_ROUTES_KEY` (enable **Routes API** in Google Cloud for it). The app then falls back to TomTom or to your own choice.
