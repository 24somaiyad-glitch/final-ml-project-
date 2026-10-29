"""Smart Traffic Prediction - cleaning, feature engineering, split, model comparison, figures."""
import json, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score)
from features import add_features, NUM, CAT, FEATURES, LEVELS, DAYS

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv("data/traffic.csv")
print("Raw shape:", df.shape)
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("Duplicate rows:", df.duplicated().sum())
log = {"raw_rows": len(df), "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum())}

# ---------------------------------------------------------------- 2. CLEAN
df = df.drop_duplicates().reset_index(drop=True)
df["Delay_Minutes"] = df["Delay_Minutes"].fillna(df["Delay_Minutes"].median())
df["Vehicles"] = df["Vehicles"].fillna(df["Vehicles"].median())
df["Weather"] = df["Weather"].fillna(df["Weather"].mode()[0])
df = df.dropna(subset=["Traffic"])
q1, q3 = df["Delay_Minutes"].quantile([.25, .75]); iqr = q3 - q1
lo, hi = max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
log["outliers_capped"] = int(((df["Delay_Minutes"] < lo) | (df["Delay_Minutes"] > hi)).sum())
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
sns.boxplot(x=df["Delay_Minutes"], ax=ax[0], color="#e57373"); ax[0].set_title("Delay - before outlier handling")
df["Delay_Minutes"] = df["Delay_Minutes"].clip(lo, hi)
sns.boxplot(x=df["Delay_Minutes"], ax=ax[1], color="#81c784"); ax[1].set_title("Delay - after outlier handling")
plt.tight_layout(); plt.savefig("figures/01_outliers.png", dpi=150); plt.close()
vq1, vq3 = df["Vehicles"].quantile([.25, .75]); vi = vq3 - vq1          # same IQR rule for vehicle counts
vlo, vhi = max(0, vq1 - 1.5 * vi), vq3 + 1.5 * vi
log["vehicle_outliers_capped"] = int(((df["Vehicles"] < vlo) | (df["Vehicles"] > vhi)).sum())
df["Vehicles"] = df["Vehicles"].clip(vlo, vhi)

# ---------------------------------------------------------------- 3. FEATURE ENGINEERING
df = add_features(df)       # Hour_sin/cos, Time_Slot, Is_Weekend
df.to_csv("data/traffic_clean.csv", index=False)
log["clean_rows"] = len(df)
print("Clean shape:", df.shape, "| outliers capped:", log["outliers_capped"])

# ---------------------------------------------------------------- 4. EDA FIGURES
plt.figure(figsize=(5, 3.5)); sns.countplot(x="Traffic", data=df, order=LEVELS, palette=["#66bb6a", "#ffca28", "#ef5350"], hue="Traffic", legend=False)
plt.title("Class distribution"); plt.tight_layout(); plt.savefig("figures/02_class_distribution.png", dpi=150); plt.close()
pivot = df.pivot_table(index="Day", columns="Hour", values="Delay_Minutes", aggfunc="mean").reindex(DAYS)
plt.figure(figsize=(11, 3.8)); sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "Avg delay (min per 10 km)"})
plt.title("Congestion heatmap: average delay by day and hour"); plt.tight_layout(); plt.savefig("figures/03_heatmap.png", dpi=150); plt.close()
plt.figure(figsize=(7, 3.5)); df.groupby("Hour")["Delay_Minutes"].mean().plot(marker="o")
plt.ylabel("Avg delay (min)"); plt.title("Average congestion through the day"); plt.tight_layout()
plt.savefig("figures/04_hourly_trend.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 5. SPLIT
X, y = df[FEATURES], df["Traffic"]
X_tr, X_te, y_tr, y_te, d_tr, d_te, v_tr, v_te = train_test_split(X, y, df["Delay_Minutes"], df["Vehicles"], test_size=0.2, stratify=y, random_state=42)
print(f"Train: {len(X_tr)}  Test: {len(X_te)}")
log["train_rows"], log["test_rows"] = len(X_tr), len(X_te)
def prep(scale=True):
    return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])

# ---------------------------------------------------------------- 6. MODELS
models = {
    "Logistic Regression": Pipeline([("p", prep()), ("m", LogisticRegression(max_iter=2000))]),
    "Decision Tree":       Pipeline([("p", prep(False)), ("m", DecisionTreeClassifier(max_depth=7, random_state=42))]),
    "KNN":                 Pipeline([("p", prep()), ("m", KNeighborsClassifier(n_neighbors=9))]),
    "Random Forest":       Pipeline([("p", prep(False)), ("m", RandomForestClassifier(n_estimators=200, random_state=42))]),
    "Naive Bayes":         Pipeline([("p", prep()), ("m", GaussianNB())]),
}
rows, preds = [], {}
for name, pipe in models.items():
    pipe.fit(X_tr, y_tr); p = pipe.predict(X_te); preds[name] = p
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, p),
                 "Precision": precision_score(y_te, p, average="weighted", zero_division=0),
                 "Recall": recall_score(y_te, p, average="weighted"), "F1-score": f1_score(y_te, p, average="weighted"),
                 "CV Accuracy (5-fold)": cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="accuracy").mean()})
res = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
res.round(4).to_csv("figures/model_comparison.csv", index=False)
print("\n", res.round(4).to_string(index=False))
best_name = res.loc[0, "Model"]; best = models[best_name]
print("\nBest model:", best_name); print(classification_report(y_te, preds[best_name], labels=LEVELS))

m = res.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"])
plt.figure(figsize=(9, 4.5)); sns.barplot(data=m, x="Model", y="value", hue="variable")
plt.ylim(0.4, 1.0); plt.ylabel("Score"); plt.title("Model performance comparison")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4); plt.tight_layout()
plt.savefig("figures/05_model_comparison.png", dpi=150); plt.close()

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5)); axes = axes.ravel()
for ax, (name, p) in zip(axes, preds.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_te, p, labels=LEVELS), display_labels=LEVELS).plot(ax=ax, cmap="Blues", colorbar=False); ax.set_title(name)
axes[-1].axis("off"); plt.tight_layout(); plt.savefig("figures/06_confusion_matrices.png", dpi=150); plt.close()

rf = models["Random Forest"]; fn = [n.split("__")[1] for n in rf.named_steps["p"].get_feature_names_out()]
imp = pd.Series(rf.named_steps["m"].feature_importances_, index=fn).nlargest(10)
plt.figure(figsize=(7, 4)); imp[::-1].plot(kind="barh", color="#42a5f5"); plt.title("Top-10 feature importances (Random Forest)")
plt.tight_layout(); plt.savefig("figures/07_feature_importance.png", dpi=150); plt.close()

dt = Pipeline([("p", prep(False)), ("m", DecisionTreeClassifier(max_depth=3, random_state=42))]).fit(X_tr, y_tr)
plt.figure(figsize=(16, 7)); plot_tree(dt.named_steps["m"], feature_names=[n.split("__")[1] for n in dt.named_steps["p"].get_feature_names_out()],
                                      class_names=dt.named_steps["m"].classes_, filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree (depth 3) - how the model decides"); plt.tight_layout(); plt.savefig("figures/08_decision_tree.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 7. LINEAR REGRESSION (delay minutes)
reg = Pipeline([("p", prep()), ("m", LinearRegression())]).fit(X_tr, d_tr)
pr = reg.predict(X_te)
reg_res = {"MAE": mean_absolute_error(d_te, pr), "RMSE": float(np.sqrt(mean_squared_error(d_te, pr))), "R2": r2_score(d_te, pr)}
print("\nLinear Regression (delay minutes):", {k: round(v, 3) for k, v in reg_res.items()})
plt.figure(figsize=(5, 5)); plt.scatter(d_te, pr, s=8, alpha=.5); plt.plot([0, d_te.max()], [0, d_te.max()], "r--")
plt.xlabel("Actual delay (min)"); plt.ylabel("Predicted delay (min)"); plt.title("Linear Regression: actual vs predicted")
plt.tight_layout(); plt.savefig("figures/09_linear_regression.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 7b. LINEAR REGRESSION (vehicle count, shown in Manual mode)
veh_reg = Pipeline([("p", prep()), ("m", LinearRegression())]).fit(X_tr, v_tr)
pv = veh_reg.predict(X_te)
veh_res = {"MAE": mean_absolute_error(v_te, pv), "RMSE": float(np.sqrt(mean_squared_error(v_te, pv))), "R2": r2_score(v_te, pv)}
print("Linear Regression (vehicles per hour):", {k: round(v, 3) for k, v in veh_res.items()})
plt.figure(figsize=(5, 5)); plt.scatter(v_te, pv, s=8, alpha=.5); plt.plot([0, v_te.max()], [0, v_te.max()], "r--")
plt.xlabel("Actual vehicles per hour"); plt.ylabel("Predicted vehicles per hour"); plt.title("Vehicle count: actual vs predicted")
plt.tight_layout(); plt.savefig("figures/10_vehicle_regression.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 8. SAVE
joblib.dump(best, "models/best_model.pkl"); joblib.dump(reg, "models/delay_regressor.pkl"); joblib.dump(veh_reg, "models/vehicle_regressor.pkl")
log.update({"best_model": best_name, "linear_regression": reg_res, "vehicle_regression": veh_res,
            "choices": {"Weather": ["Clear", "Cloudy", "Rain", "Fog"], "Area": sorted(df["Area"].unique().tolist())}})
json.dump(log, open("models/meta.json", "w"), indent=2)
print("\nSaved models/ and figures/")
