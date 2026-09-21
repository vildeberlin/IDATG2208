#  Vi skal:
#   1. Download / load the Superconductivity data set.
#   2. Identify the strong (largest |ρ|) and weak (smallest |ρ|) features.
#   3. Perform 5 random 80 %‑train / 20 %‑test splits (same seed for both
#      predictors so the folds are identical).
#   4. For each fold train a single‑feature linear model (gradient descent)
#      on the *standardised* training data, transform the coefficients back
#      to the original scale and evaluate on the test set → MSE, RMSE, R².
#   5. Compute the **mean** and **variance** (or standard deviation) of the
#      three metrics across the five folds for each predictor.

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

csv_name = "train.csv"

# ---
# Så skal vi load-e dataen
df = pd.read_csv(csv_name)

# ---
# Sterk og svak predictors (based on Pearson ρ)
features = df.columns.drop("critical_temp")
corr = df[features].corrwith(df["critical_temp"])

weak_feat   = corr.abs().idxmin()   # smallest absolute correlation
strong_feat = corr.abs().idxmax()   # largest absolute correlation

print("\n=== Predictor selection ===")
print(f"Weak   predictor : {weak_feat:30s} (ρ = {corr[weak_feat]:+.4f})")
print(f"Strong predictor : {strong_feat:30s} (ρ = {corr[strong_feat]:+.4f})")


# ---
# Gradient‑descent routine (identical for both predictors)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent(x_std, y, lr=2e-2, n_iter=5000, tol=1e-7):
    """Fit ŷ = w·x + b on *standardised* x using plain GD."""
    n = len(x_std)
    rng = np.random.default_rng(seed=42)
    w = rng.normal(0, 0.01)
    b = 0.0
    for _ in range(1, n_iter + 1):
        y_hat = w * x_std + b
        loss = mse(y, y_hat)

        # gradients of MSE
        dw = (-2.0 / n) * np.sum(x_std * (y - y_hat))
        db = (-2.0 / n) * np.sum(y - y_hat)

        w -= lr * dw
        b -= lr * db

        # early stopping
        if abs(loss - mse(y, w * x_std + b)) < tol:
            break
    return w, b


# ---
# 5‑fold random split
rng = np.random.default_rng(seed=2024)           # reproducible shuffling
indices = np.arange(len(df))
rng.shuffle(indices)

n = len(df)
fold_size = int(0.2 * n)                         # 20 % test per fold

# Containers for the two predictors
results = {
    "weak":   {"MSE": [], "RMSE": [], "R2": []},
    "strong": {"MSE": [], "RMSE": [], "R2": []}
}

for fold in range(5):
    start = fold * fold_size
    end   = start + fold_size
    test_idx  = indices[start:end]
    train_idx = np.setdiff1d(indices, test_idx)

    train = df.iloc[train_idx]
    test  = df.iloc[test_idx]

    for name, feat in [("weak", weak_feat), ("strong", strong_feat)]:
        # ---- standardise on the training set only ----
        x_tr_raw = train[feat].values.astype(np.float64)
        y_tr     = train["critical_temp"].values.astype(np.float64)

        mu = x_tr_raw.mean()
        sigma = x_tr_raw.std(ddof=0)
        x_tr_std = (x_tr_raw - mu) / sigma

        # ---- fit ----
        w_std, b_std = gradient_descent(x_tr_std, y_tr,
                                        lr=0.02, n_iter=10000)

        # ---- back‑transform to original scale ----
        w_orig = w_std / sigma
        b_orig = b_std - (w_std * mu) / sigma

        # ---- test evaluation ----
        x_te_raw = test[feat].values.astype(np.float64)
        y_te     = test["critical_temp"].values.astype(np.float64)
        y_pred   = w_orig * x_te_raw + b_orig

        fold_mse  = mse(y_te, y_pred)
        fold_rmse = np.sqrt(fold_mse)
        sst = np.sum((y_te - y_te.mean())**2)
        sse = np.sum((y_te - y_pred)**2)
        fold_r2  = 1 - sse / sst

        results[name]["MSE"].append(fold_mse)
        results[name]["RMSE"].append(fold_rmse)
        results[name]["R2"].append(fold_r2)


# ---
# Finne mean og variance for hver
summary = {}
for name in ("weak", "strong"):
    summary[name] = {
        "MSE_mean":   np.mean(results[name]["MSE"]),
        "MSE_var":    np.var(results[name]["MSE"], ddof=1),
        "RMSE_mean":  np.mean(results[name]["RMSE"]),
        "RMSE_var":   np.var(results[name]["RMSE"], ddof=1),
        "R2_mean":    np.mean(results[name]["R2"]),
        "R2_var":     np.var(results[name]["R2"], ddof=1)
    }

# ---
# Printe 
print("\n=== 5‑fold performance summary (mean ± √variance) ===")
header = f"{'Metric':<10} {'Strong predictor':>25} {'Weak predictor':>25}"
print(header)
print("-" * len(header))

def fmt(mean, var):
    std = np.sqrt(var)
    return f"{mean:8.2f} ± {std:5.2f}"

print(f"{'MSE':<10} {fmt(summary['strong']['MSE_mean'], summary['strong']['MSE_var']):>25} "
      f"{fmt(summary['weak']['MSE_mean'],   summary['weak']['MSE_var']):>25}")

print(f"{'RMSE':<10} {fmt(summary['strong']['RMSE_mean'], summary['strong']['RMSE_var']):>25} "
      f"{fmt(summary['weak']['RMSE_mean'],   summary['weak']['RMSE_var']):>25}")

print(f"{'R²':<10} {fmt(summary['strong']['R2_mean'], summary['strong']['R2_var']):>25} "
      f"{fmt(summary['weak']['R2_mean'],   summary['weak']['R2_var']):>25}")
