import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import csv

csv_name = "train.csv"

# ---
# Så skal vi load-e dataen
df = pd.read_csv(csv_name)

# ---
# Finne sterk og svak predictors (based on |Pearson ρ|)
features = df.columns.drop("critical_temp")
corr = df[features].corrwith(df["critical_temp"])

weak_feat   = corr.abs().idxmin()   # smallest absolute correlation
strong_feat = corr.abs().idxmax()   # largest absolute correlation

print("\n=== Predictor selection ===")
print(f"Weak   predictor : {weak_feat:30s} (ρ = {corr[weak_feat]:+.4f})")
print(f"Strong predictor : {strong_feat:30s} (ρ = {corr[strong_feat]:+.4f})")

# ---
# Metric and optimisation utilities
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gd_simple(x_std, y, lr=2e-2, n_iter=8000, tol=1e-7):
    """GD for a *single* standardized feature → returns w, b."""
    n = len(x_std)
    rng = np.random.default_rng(seed=42)
    w = rng.normal(0, 0.01)
    b = 0.0
    for _ in range(1, n_iter + 1):
        y_hat = w * x_std + b
        loss = mse(y, y_hat)

        dw = (-2.0 / n) * np.sum(x_std * (y - y_hat))
        db = (-2.0 / n) * np.sum(y - y_hat)

        w -= lr * dw
        b -= lr * db

        if abs(loss - mse(y, w * x_std + b)) < tol:
            break
    return w, b


def gd_multiple(X_std, y, lr=1e-2, n_iter=8000, tol=1e-7):
    """GD for a multivariate standardized design matrix → returns w, b."""
    n, p = X_std.shape
    rng = np.random.default_rng(seed=42)
    w = rng.normal(0, 0.01, size=p)
    b = 0.0
    for _ in range(1, n_iter + 1):
        y_hat = X_std @ w + b
        loss = mse(y, y_hat)

        grad_w = (-2.0 / n) * (X_std.T @ (y - y_hat))
        grad_b = (-2.0 / n) * np.sum(y - y_hat)

        w -= lr * grad_w
        b -= lr * grad_b

        if abs(loss - mse(y, X_std @ w + b)) < tol:
            break
    return w, b


# ---
# 5‑fold split (identical for all three models)
rng = np.random.default_rng(seed=2024)            # reproducible folds
indices = np.arange(len(df))
rng.shuffle(indices)

n = len(df)
fold_size = int(0.2 * n)                          # 20 % test per fold

# Containers for the three modelling approaches
results = {
    "weak":   {"MSE": [], "RMSE": [], "R2": []},
    "strong": {"MSE": [], "RMSE": [], "R2": []},
    "multi":  {"MSE": [], "RMSE": [], "R2": []}
}

# Full feature matrix (81 columns)
X_full = df.drop(columns=["critical_temp"]).values.astype(np.float64)
y_full = df["critical_temp"].values.astype(np.float64)

for fold in range(5):
    # ----- define train / test indices ---------------------------------
    start = fold * fold_size
    end   = start + fold_size
    test_idx  = indices[start:end]
    train_idx = np.setdiff1d(indices, test_idx)

    # ----------=== 1) Weak predictor ==============================
    x_w_train_raw = df.loc[train_idx, weak_feat].values.astype(np.float64)
    x_w_test_raw  = df.loc[test_idx,  weak_feat].values.astype(np.float64)
    y_train       = df.loc[train_idx, "critical_temp"].values.astype(np.float64)
    y_test        = df.loc[test_idx,  "critical_temp"].values.astype(np.float64)

    mu_w = x_w_train_raw.mean()
    sigma_w = x_w_train_raw.std(ddof=0)
    x_w_train_std = (x_w_train_raw - mu_w) / sigma_w
    x_w_test_std  = (x_w_test_raw  - mu_w) / sigma_w

    w_std, b_std = gd_simple(x_w_train_std, y_train)
    w_orig = w_std / sigma_w
    b_orig = b_std - (w_std * mu_w) / sigma_w

    y_pred = w_orig * x_w_test_raw + b_orig
    results["weak"]["MSE"].append(mse(y_test, y_pred))
    results["weak"]["RMSE"].append(np.sqrt(mse(y_test, y_pred)))
    sst = np.sum((y_test - y_test.mean()) ** 2)
    sse = np.sum((y_test - y_pred) ** 2)
    results["weak"]["R2"].append(1 - sse / sst)

    # ----------=== 2) Strong predictor ============================
    x_s_train_raw = df.loc[train_idx, strong_feat].values.astype(np.float64)
    x_s_test_raw  = df.loc[test_idx,  strong_feat].values.astype(np.float64)

    mu_s = x_s_train_raw.mean()
    sigma_s = x_s_train_raw.std(ddof=0)
    x_s_train_std = (x_s_train_raw - mu_s) / sigma_s
    x_s_test_std  = (x_s_test_raw  - mu_s) / sigma_s

    w_std, b_std = gd_simple(x_s_train_std, y_train)
    w_orig = w_std / sigma_s
    b_orig = b_std - (w_std * mu_s) / sigma_s

    y_pred = w_orig * x_s_test_raw + b_orig
    results["strong"]["MSE"].append(mse(y_test, y_pred))
    results["strong"]["RMSE"].append(np.sqrt(mse(y_test, y_pred)))
    sst = np.sum((y_test - y_test.mean()) ** 2)
    sse = np.sum((y_test - y_pred) ** 2)
    results["strong"]["R2"].append(1 - sse / sst)

    # ----------=== 3) Multiple regression =========================
    X_train_raw = X_full[train_idx]   # (n_train, 81)
    X_test_raw  = X_full[test_idx]    # (n_test, 81)

    mu_X = X_train_raw.mean(axis=0)          # (81,)
    sigma_X = X_train_raw.std(axis=0, ddof=0)
    sigma_X[sigma_X == 0] = 1.0               # safety

    X_train_std = (X_train_raw - mu_X) / sigma_X
    X_test_std  = (X_test_raw  - mu_X) / sigma_X

    w_std, b_std = gd_multiple(X_train_std, y_train,
                               lr=0.01, n_iter=8000, tol=1e-7)

    # back‑transform to original space
    w_orig = w_std / sigma_X
    b_orig = b_std - np.sum(w_std * mu_X / sigma_X)

    y_pred = X_test_raw @ w_orig + b_orig
    results["multi"]["MSE"].append(mse(y_test, y_pred))
    results["multi"]["RMSE"].append(np.sqrt(mse(y_test, y_pred)))
    sst = np.sum((y_test - y_test.mean()) ** 2)
    sse = np.sum((y_test - y_pred) ** 2)
    results["multi"]["R2"].append(1 - sse / sst)

# ---
# Summere (mean ± std) for hver modelling approach
def mean_std(arr):
    return np.mean(arr), np.std(arr, ddof=1)

summary = {}
for key in results:
    mse_m, mse_s   = mean_std(results[key]["MSE"])
    rmse_m, rmse_s = mean_std(results[key]["RMSE"])
    r2_m, r2_s     = mean_std(results[key]["R2"])
    summary[key] = {
        "MSE_mean":  mse_m,   "MSE_std":  mse_s,
        "RMSE_mean": rmse_m, "RMSE_std": rmse_s,
        "R2_mean":   r2_m,   "R2_std":   r2_s
    }

# ---
# Printing
print("\n=== 5‑fold cross‑validation – mean ± std ===")
header = f"{'Model':<10} {'MSE':>20} {'RMSE':>20} {'R²':>20}"
print(header)
print("-" * len(header))

def fmt(mean, std, fmt_str="{:.2f}"):
    return f"{fmt_str.format(mean)} ± {fmt_str.format(std)}"

for model in ("weak", "strong", "multi"):
    s = summary[model]
    print(f"{model:<10} "
          f"{fmt(s['MSE_mean'],   s['MSE_std']):>20} "
          f"{fmt(s['RMSE_mean'],  s['RMSE_std']):>20} "
          f"{fmt(s['R2_mean'],    s['R2_std'], fmt_str='{:.4f}'):>20}")
