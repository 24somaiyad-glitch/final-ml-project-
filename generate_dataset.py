"""Creates a synthetic traffic dataset WITHOUT vehicle counts (data/traffic.csv).
Inputs are things a commuter / city system really knows: time, day, weather, area, holiday, event, previous traffic.
Delay_Minutes = extra travel time for a 10 km trip (measurable from Google/TomTom travel-time data).
Vehicles = simulated vehicles per hour on that road (target of the Linear Regression shown in Manual mode; never typed in by the user).
Replace with a real dataset (Kaggle / city feed) keeping the same columns."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 3500
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df = pd.DataFrame({
    "Hour": rng.integers(0, 24, n), "Day": rng.choice(days, n),
    "Weather": rng.choice(["Clear", "Cloudy", "Rain", "Fog"], n, p=[.5, .25, .17, .08]),
    "Area": rng.choice(["Downtown", "Highway", "Residential", "Industrial"], n),
    "Is_Holiday": rng.choice(["No", "Yes"], n, p=[.95, .05]),
    "Event_Nearby": rng.choice(["No", "Yes"], n, p=[.92, .08]),
})
def delay(h):
    peak = 12 * np.exp(-((h - 8.5) ** 2) / 5) + 14 * np.exp(-((h - 17.5) ** 2) / 5) + 5 * np.exp(-((h - 13) ** 2) / 10)
    area = df["Area"].map({"Downtown": 3, "Highway": 4, "Residential": -1.5, "Industrial": 1}).values
    off = -5 * df["Day"].isin(["Saturday", "Sunday"]).values - 4 * (df["Is_Holiday"] == "Yes").values
    wx = df["Weather"].map({"Clear": 0, "Cloudy": .5, "Rain": 4, "Fog": 2.5}).values
    ev = 5 * (df["Event_Nearby"] == "Yes").values
    return 2 + peak + area + off + wx + ev
level = lambda d: np.where(d < 6, "Low", np.where(d < 12, "Medium", "High"))
now = np.clip(delay(df["Hour"].values) + rng.normal(0, 1.8, n), 0, None)
df["Delay_Minutes"] = now.round(1)
df["Traffic"] = level(now + rng.normal(0, 1.2, n))
df["Previous_Traffic"] = level(np.clip(delay((df["Hour"].values - 1) % 24) + rng.normal(0, 1.8, n), 0, None))
# vehicles per hour: grows with congestion, bigger roads carry more (own generator so the columns above stay unchanged)
rng2 = np.random.default_rng(7)
veh_area = df["Area"].map({"Downtown": 120, "Highway": 260, "Residential": -60, "Industrial": 40}).values
df["Vehicles"] = np.clip(150 + 26 * now + veh_area + rng2.normal(0, 30, n), 20, None).round().astype(float)
df = df[["Hour", "Day", "Weather", "Area", "Is_Holiday", "Event_Nearby", "Previous_Traffic", "Delay_Minutes", "Vehicles", "Traffic"]]
# dirty on purpose
idx = rng.choice(n, 100, replace=False)
df.loc[idx[:50], "Delay_Minutes"] = np.nan; df.loc[idx[50:], "Weather"] = np.nan
df.loc[rng.choice(n, 6, replace=False), "Delay_Minutes"] = 300          # sensor glitches
df.loc[rng2.choice(n, 40, replace=False), "Vehicles"] = np.nan            # missing vehicle counts
df.loc[rng2.choice(n, 5, replace=False), "Vehicles"] = 5000                # counter glitches
df = pd.concat([df, df.sample(30, random_state=1)], ignore_index=True)
df.to_csv("data/traffic.csv", index=False)
print("Saved data/traffic.csv", df.shape); print(df["Traffic"].value_counts())
