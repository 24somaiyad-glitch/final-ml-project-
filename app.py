"""Smart Traffic Prediction - Streamlit UI.   Run:  streamlit run app.py"""
import json, os, joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from features import make_row, forecast_day, DAYS, LEVELS
from live_data import (now_local, get_live_weather, get_live_traffic_level,
                       google_traffic_map_html, google_embed_fallback_html, google_maps_link, get_google_traffic)

st.set_page_config(page_title="Smart Traffic Prediction", page_icon="🚦", layout="centered")

# ---------------------------------------------------------------- load model
def _load_files():
    clf = joblib.load("models/best_model.pkl")
    reg = joblib.load("models/delay_regressor.pkl")
    veh = joblib.load("models/vehicle_regressor.pkl")      # Linear Regression: vehicles per hour (Manual mode)
    meta = json.load(open("models/meta.json"))
    return clf, reg, veh, meta

@st.cache_resource
def load_models():
    try:
        return _load_files()
    except Exception:
        # saved model missing or built with other library versions -> rebuild it here (about 30 seconds)
        import subprocess, sys
        os.makedirs("models", exist_ok=True); os.makedirs("figures", exist_ok=True); os.makedirs("data", exist_ok=True)
        if not os.path.exists("data/traffic.csv") or "Vehicles" not in open("data/traffic.csv").readline():
            subprocess.run([sys.executable, "generate_dataset.py"], check=True)
        subprocess.run([sys.executable, "train.py"], check=True)
        return _load_files()

try:
    clf, reg, veh_reg, meta = load_models()
except Exception as e:
    st.title("🚦 Smart Traffic Prediction System")
    st.error("Could not load the trained model. Stop the app (Ctrl + C), run `python train.py` in this folder, then start the app again.")
    st.code(str(e))
    st.stop()

choices = meta["choices"]

def secret(name):
    """Read a key from Streamlit secrets or an environment variable (None if not set)."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)

@st.cache_data(ttl=60, show_spinner=False)
def google_live(area, key):
    """Google live reading, cached for 60 s so each click does not cost an API call."""
    return get_google_traffic(area, key=key)

def rerun():
    try:
        st.rerun()
    except AttributeError:          # older Streamlit
        st.experimental_rerun()

# ---------------------------------------------------------------- header
st.title("🚦 Smart Traffic Prediction System")
st.caption("Model in use: **" + meta["best_model"] + "**  |  inputs: time, day, weather, area, holiday, event, previous traffic")

mode = st.radio("Mode", ["Live (automatic)", "Manual / what-if"], horizontal=True)
live = mode.startswith("Live")
area = st.selectbox("Road / Area", choices["Area"])

# ---------------------------------------------------------------- inputs
if live:
    now = now_local()
    hour, day = now.hour, DAYS[now.weekday()]

    w = get_live_weather()
    if w:
        weather, weather_label = w["weather"], "Weather (live)"
    else:
        weather = st.selectbox("Weather (live feed unavailable - please choose)", choices["Weather"])
        weather_label = "Weather"

    try:
        tomtom_key = st.secrets["TOMTOM_KEY"]
    except Exception:
        tomtom_key = None
    g_key = secret("GOOGLE_ROUTES_KEY") or secret("GOOGLE_MAPS_KEY")
    g = google_live(area, g_key) if g_key else None
    if g and "error" in g:
        st.warning("Google live reading unavailable: " + g["error"])
        g = None
    tt = None if g else get_live_traffic_level(area, key=tomtom_key)
    if g:
        prev = g["level"]
        source = ("Google Maps live reading for " + area + ": trip " + str(g["trip_min"]) + " min now vs " + str(g["free_min"]) +
                  " min free-flow over " + str(g["distance_km"]) + " km (" + format(g["ratio"], ".0%") + " of free-flow speed)")
    elif tt:
        prev, source = tt[0], "live road speed = " + format(tt[1], ".0%") + " of free-flow speed"
    else:
        prev = st.selectbox("Traffic you see right now", LEVELS, index=1)
        source = "entered by you"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time", now.strftime("%H:%M"))
    c2.metric("Day", day)
    c3.metric(weather_label, weather)
    c4.metric("Current traffic", prev)
    st.caption("Current traffic source: " + source)
    if st.button("Refresh live data"):
        google_live.clear()
        rerun()
    if g:
        g1, g2, g3 = st.columns(3)
        g1.metric("Google live traffic", g["level"])
        g2.metric("Live delay (Google)", "~" + str(g["delay_per_10km"]) + " min per 10 km")
        g3.metric("Trip time now", str(g["trip_min"]) + " min", delta=str(round(g["trip_min"] - g["free_min"], 1)) + " min vs free-flow", delta_color="inverse")

    # ---- Google Maps live traffic for the selected area
    st.subheader("Live Google Maps traffic - " + area)
    gmap = google_traffic_map_html(area, key=secret("GOOGLE_MAPS_KEY"))
    if gmap:
        components.html(gmap, height=440)
    else:
        components.html(google_embed_fallback_html(area), height=440)
        st.caption("Basic map shown. Add GOOGLE_MAPS_KEY (see REALTIME_GUIDE.md) for the full live traffic layer.")
    st.markdown("[Open live traffic in Google Maps](" + google_maps_link(area) + ")")
else:
    c1, c2 = st.columns(2)
    hour = c1.slider("Hour of day", 0, 23, 8)
    day = c1.selectbox("Day of week", DAYS)
    weather = c2.selectbox("Weather", choices["Weather"])
    prev = c2.selectbox("Previous traffic", LEVELS, index=2)

h1, h2 = st.columns(2)
holiday = "Yes" if h1.checkbox("Public holiday") else "No"
event = "Yes" if h2.checkbox("Event nearby (match, festival, rally)") else "No"

# ---------------------------------------------------------------- prediction
row = make_row(hour, day, weather, area, holiday, event, prev)
pred = str(clf.predict(row)[0])
delay = max(0.0, float(reg.predict(row)[0]))
icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[pred]
box = {"Low": st.success, "Medium": st.warning, "High": st.error}[pred]
box("Predicted traffic: " + icon + " **" + pred.upper() + "**   |   Expected extra delay: **~" + str(round(delay)) + " min per 10 km**")
if live and g:
    st.caption("Model expects about " + str(round(delay)) + " min extra per 10 km for this hour; Google measures about " + str(g["delay_per_10km"]) + " min right now.")

if hasattr(clf, "predict_proba"):
    proba = pd.Series(clf.predict_proba(row)[0], index=list(clf.classes_)).reindex(LEVELS)
    st.caption("Prediction confidence")
    st.bar_chart(proba, height=180)

advice = {"Low": "Roads are clear - good time to travel.",
          "Medium": "Moderate traffic - allow some extra time.",
          "High": "Heavy congestion expected - consider another route or time."}
st.info(advice[pred])

# ---------------------------------------------------------------- vehicle count (Manual mode only)
if not live:
    vehicles = max(0, round(float(veh_reg.predict(row)[0])))
    st.metric("Estimated vehicle count (Linear Regression)", "~" + format(vehicles, ",") + " vehicles per hour")
    vr = meta.get("vehicle_regression", {})
    if vr:
        st.caption("Linear Regression on simulated data: typical error about " + str(round(vr["MAE"])) + " vehicles, R2 = " + format(vr["R2"], ".2f") + ". An estimate, not a live count.")

# ---------------------------------------------------------------- 24-hour forecast
st.subheader("Forecast for " + day + " - " + area)
try:
    fc = forecast_day(clf, reg, day, weather, area, holiday, event, veh_reg=None if live else veh_reg)
    st.line_chart(fc.set_index("Hour")["Expected_Delay_min"], height=220)
    if "Expected_Vehicles" in fc.columns:
        st.caption("Estimated vehicles per hour through the day (Linear Regression)")
        st.line_chart(fc.set_index("Hour")["Expected_Vehicles"], height=200)
    high = fc.loc[fc["Predicted_Level"] == "High", "Hour"].tolist()
    st.write("**High traffic expected at:** " + (", ".join(str(h) + ":00" for h in high) if high else "no hour (a calm day)"))
except Exception as e:
    st.warning("Forecast not available: " + str(e))

# ---------------------------------------------------------------- performance
with st.expander("Model performance and figures"):
    if os.path.exists("figures/model_comparison.csv"):
        st.dataframe(pd.read_csv("figures/model_comparison.csv"))
    for f in ["figures/05_model_comparison.png", "figures/03_heatmap.png", "figures/10_vehicle_regression.png"]:
        if os.path.exists(f):
            st.image(f)
