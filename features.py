"""Shared feature code: used by train.py AND app.py so both treat inputs identically."""
import numpy as np, pandas as pd

LEVELS = ["Low", "Medium", "High"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUM = ["Hour", "Hour_sin", "Hour_cos"]
CAT = ["Day", "Weather", "Area", "Time_Slot", "Is_Weekend", "Is_Holiday", "Event_Nearby", "Previous_Traffic"]
FEATURES = NUM + CAT

def time_slot(h):
    if h <= 5:  return "Night"
    if h <= 10: return "Morning_Peak"
    if h <= 15: return "Midday"
    if h <= 20: return "Evening_Peak"
    return "Late_Evening"

def add_features(df):
    """Feature engineering: cyclic hour, time slot, weekend flag."""
    d = df.copy()
    d["Hour_sin"] = np.sin(2 * np.pi * d["Hour"] / 24)
    d["Hour_cos"] = np.cos(2 * np.pi * d["Hour"] / 24)
    d["Time_Slot"] = d["Hour"].apply(time_slot)
    d["Is_Weekend"] = np.where(d["Day"].isin(["Saturday", "Sunday"]), "Yes", "No")
    return d

def make_row(hour, day, weather, area, holiday, event, prev):
    return add_features(pd.DataFrame([{"Hour": hour, "Day": day, "Weather": weather, "Area": area,
                                       "Is_Holiday": holiday, "Event_Nearby": event, "Previous_Traffic": prev}]))

def forecast_day(clf, reg, day, weather, area, holiday, event, veh_reg=None):
    """Predict all 24 hours of a day. Previous_Traffic of hour h = predicted level of hour h-1
    (two passes so the chain settles)."""
    prev = ["Medium"] * 24
    for _ in range(2):
        rows = add_features(pd.DataFrame({"Hour": range(24), "Day": day, "Weather": weather, "Area": area,
                                          "Is_Holiday": holiday, "Event_Nearby": event,
                                          "Previous_Traffic": [prev[(h - 1) % 24] for h in range(24)]}))
        levels = list(clf.predict(rows))
        prev = levels
    delay = reg.predict(rows).clip(min=0)
    out = pd.DataFrame({"Hour": range(24), "Predicted_Level": levels, "Expected_Delay_min": delay.round(1)})
    if veh_reg is not None:                       # optional: vehicles per hour from the vehicle Linear Regression
        out["Expected_Vehicles"] = veh_reg.predict(rows).clip(min=0).round()
    return out
