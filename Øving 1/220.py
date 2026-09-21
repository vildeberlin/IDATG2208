import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import csv

# ---
# først skal vi laste ned og unzip datasettet
url = "https://archive.ics.uci.edu/dataset/464/superconductivty+data"
zip_path = "superconductivty+data.zip"
csv_name = "train.csv"               # navnet inni zip filen

# Hvis vi allerede har fila, dropper vi det
if not os.path.isfile(zip_path):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download finished.")
else:
    print("Zip file already present – skipping download.")

# Extract the CSV if it does not exist yet
if not os.path.isfile(csv_name):
    print("Extracting CSV from zip …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(csv_name)
    print("Extraction done.")
else:
    print("CSV already extracted – skipping.")

# ---
# Så skal vi load-e dataen
df = pd.read_csv(csv_name)

X_full = df.drop(columns=["critical_temp"]).values.astype(np.float64)   # (n,81)
y_full = df["critical_temp"].values.astype(np.float64)                # (n,)
feature_names = list(df.columns.drop("critical_temp"))

# --- 
# Finne de 10 viktigste features
from sklearn.ensemble import RandomForestRegressor

kf_tmp = KFold(n_splits=3, shuffle=True, random_state=2024)
imp_accum = np.zeros(X_full.shape[1])

for train_idx, test_idx in kf_tmp.split(X_full):
    rf = RandomForestRegressor(
        n_estimators=200,    # still fast
        max_depth=None,
        n_jobs=-1,
        random_state=42)
    rf.fit(X_full[train_idx], y_full[train_idx])
    imp_accum += rf.feature_importances_

mean_imp = imp_accum / 3
top10_idx = np.argsort(mean_imp)[-10:][::-1]      # descending
top10_names = [feature_names[i] for i in top10_idx]

print("\nTop‑10 features (by RF importance):")
for i, (name, imp) in enumerate(zip(top10_names, mean_imp[top10_idx]), 1):
    print(f"{i:2d}. {name:<30}  importance = {imp:.5f}")

# ---
# CV splitter (3‑fold, same for every experiment)
kf = KFold(n_splits=3, shuffle=True, random_state=2024)

# ---
# Containers for resultater
results = {
    "poly":   {"MSE": [], "RMSE": [], "R2": []},
    "ridge":  {"MSE": [], "RMSE": [], "R2": []},
    "lasso":  {"MSE": [], "RMSE": [], "R2": []},
    "tree":   {"MSE": [], "RMSE": [], "R2": []},
    "forest": {"MSE": [], "RMSE": [], "R2": []},
}

# ---
# (a) Polynomial regression on the top‑10 features
poly_pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
    ("lin", LinearRegression())
])

for train_idx, test_idx in kf.split(X_full):
    X_train = X_full[train_idx][:, top10_idx]   # (n_train,10)
    X_test  = X_full[test_idx][:,  top10_idx]

    poly_pipe.fit(X_train, y_full[train_idx])
    y_pred = poly_pipe.predict(X_test)

    mse = mean_squared_error(y_full[test_idx], y_pred)
    results["poly"]["MSE"].append(mse)
    results["poly"]["RMSE"].append(np.sqrt(mse))
    results["poly"]["R2"].append(r2_score(y_full[test_idx], y_pred))

# ---
# (b) Ridge & Lasso on all 81 features
ridge_alpha_grid = {"ridge__alpha": [1e-3, 1e-1, 1.0]}
lasso_alpha_grid = {"lasso__alpha": [1e-3, 1e-1, 1.0]}

for train_idx, test_idx in kf.split(X_full):
    X_tr, X_te = X_full[train_idx], X_full[test_idx]
    y_tr, y_te = y_full[train_idx], y_full[test_idx]

    # ---------- Ridge ----------
    ridge_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge())
    ])
    ridge_cv = GridSearchCV(
        ridge_pipe,
        param_grid=ridge_alpha_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1)
    ridge_cv.fit(X_tr, y_tr)
    y_pred = ridge_cv.best_estimator_.predict(X_te)

    mse = mean_squared_error(y_te, y_pred)
    results["ridge"]["MSE"].append(mse)
    results["ridge"]["RMSE"].append(np.sqrt(mse))
    results["ridge"]["R2"].append(r2_score(y_te, y_pred))

    # ---------- Lasso ----------
    lasso_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lasso", Lasso(max_iter=5_000))
    ])
    lasso_cv = GridSearchCV(
        lasso_pipe,
        param_grid=lasso_alpha_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1)
    lasso_cv.fit(X_tr, y_tr)
    y_pred = lasso_cv.best_estimator_.predict(X_te)

    mse = mean_squared_error(y_te, y_pred)
    results["lasso"]["MSE"].append(mse)
    results["lasso"]["RMSE"].append(np.sqrt(mse))
    results["lasso"]["R2"].append(r2_score(y_te, y_pred))

    # Save the coefficients from the *first* fold for later inspection
    if train_idx[0] == 0:
        lasso_coefs = lasso_cv.best_estimator_.named_steps["lasso"].coef_

# ---
# (c) Non‑linear benchmarks
for train_idx, test_idx in kf.split(X_full):
    X_tr, X_te = X_full[train_idx], X_full[test_idx]
    y_tr, y_te = y_full[train_idx], y_full[test_idx]

    # ---- Decision Tree (no pruning) ----
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_tr, y_tr)
    y_pred = tree.predict(X_te)

    results["tree"]["MSE"].append(mean_squared_error(y_te, y_pred))
    results["tree"]["RMSE"].append(np.sqrt(results["tree"]["MSE"][-1]))
    results["tree"]["R2"].append(r2_score(y_te, y_pred))

    # ---- Random Forest (100 trees, limited depth) ----
    forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,            # prevents huge trees → faster
        n_jobs=-1,
        random_state=42)
    forest.fit(X_tr, y_tr)
    y_pred = forest.predict(X_te)

    results["forest"]["MSE"].append(mean_squared_error(y_te, y_pred))
    results["forest"]["RMSE"].append(np.sqrt(results["forest"]["MSE"][-1]))
    results["forest"]["R2"].append(r2_score(y_te, y_pred))

# ---
# Summere (mean ± std)
def summarize(metric_dict):
    return {k: (np.mean(v), np.std(v, ddof=1)) for k, v in metric_dict.items()}


summary = {m: summarize(r) for m, r in results.items()}

print("\n=== 3‑fold performance (mean ± std) ===")
hdr = f"{'Model':<10} {'MSE':>20} {'RMSE':>20} {'R²':>20}"
print(hdr)
print("-" * len(hdr))
for m in ["poly", "ridge", "lasso", "tree", "forest"]:
    s = summary[m]
    print(f"{m:<10} "
          f"{s['MSE'][0]:12.1f} ± { s['MSE'][1]:5.1f} "
          f"{s['RMSE'][0]:12.2f} ± { s['RMSE'][1]:5.2f} "
          f"{s['R2'][0]:12.4f} ± { s['R2'][1]:5.4f}")

# ---
# Lasso – which features are zeroed out?
zero_idx = np.where(np.abs(lasso_coefs) < 1e-8)[0]
removed = [feature_names[i] for i in zero_idx]

print("\n=== Lasso – eliminated features (coeff ≈ 0) ===")
print(f"Number removed: {len(removed)} / 81")
print("Sample (first 15 alphabetically):")
print(", ".join(sorted(removed)[:15]))