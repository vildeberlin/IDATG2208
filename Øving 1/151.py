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
# Feature matrise og target
X = df.drop(columns=["critical_temp"]).values.astype(np.float64)   # shape (n,81)
y = df["critical_temp"].values.astype(np.float64)                # shape (n,)

n_samples, n_features = X.shape
print(f"\nDataset: {n_samples} rows × {n_features} features")

# ---
# Gradient‑descent implementasjon (vectorised)
def mse(y_true, y_pred):
    """Mean‑squared error."""
    return np.mean((y_true - y_pred) ** 2)


def gd_multiple_linear(X_std, y, lr=1e-2, n_iter=5000, tol=1e-7, verbose=False):
    """
    Fit ŷ = X·w + b  on *standardised* features.
    Returns   w (shape (p,)), b (scalar), and the loss‑history list.
    """
    n, p = X_std.shape
    rng = np.random.default_rng(seed=42)
    w = rng.normal(0, 0.01, size=p)   # initialise weights
    b = 0.0

    loss_hist = []

    for it in range(1, n_iter + 1):
        y_hat = X_std @ w + b
        loss = mse(y, y_hat)
        loss_hist.append(loss)

        # gradients of MSE w.r.t. w and b
        # dL/dw = -(2/n) * Xᵀ (y - ŷ)
        # dL/db = -(2/n) * Σ (y - ŷ)
        grad_w = (-2.0 / n) * (X_std.T @ (y - y_hat))
        grad_b = (-2.0 / n) * np.sum(y - y_hat)

        w -= lr * grad_w
        b -= lr * grad_b

        # early stop if loss change is tiny
        if it > 1 and abs(loss_hist[-2] - loss) < tol:
            if verbose:
                print(f"Early stop at iteration {it}")
            break

        if verbose and it % 500 == 0:
            print(f"Iter {it:5d} | MSE = {loss:.4f}")

    return w, b, loss_hist


# ---
# 5‑fold split 
rng = np.random.default_rng(seed=2024)   # reproducible folding
indices = np.arange(n_samples)
rng.shuffle(indices)

fold_size = int(0.2 * n_samples)          # 20 % test per fold

fold_results = []                         # will hold dicts per fold

for fold in range(5):
    # ----- indices for this fold ---------------------------------------
    start = fold * fold_size
    end   = start + fold_size
    test_idx  = indices[start:end]
    train_idx = np.setdiff1d(indices, test_idx)

    X_train_raw = X[train_idx]    # (n_train, 81)
    y_train     = y[train_idx]

    X_test_raw  = X[test_idx]     # (n_test, 81)
    y_test      = y[test_idx]

    # ----- standardise each column on the TRAIN data only -------------
    mu   = X_train_raw.mean(axis=0)                 # (81,)
    sigma = X_train_raw.std(axis=0, ddof=0)          # (81,)

    # avoid division by zero (should not happen for this data set)
    sigma[sigma == 0] = 1.0

    X_train_std = (X_train_raw - mu) / sigma
    X_test_std  = (X_test_raw  - mu) / sigma

    # ----- fit by GD ----------------------------------------------------
    w_std, b_std, _ = gd_multiple_linear(X_train_std, y_train,
                                         lr=0.01, n_iter=8000, verbose=False)

    # ----- transform weights back to original scale ----------------------
    # ŷ = (X_std·w_std + b_std)
    # X_std = (X_raw - μ) / σ    →   ŷ = (X_raw·(w_std/σ)) + (b_std - Σ (w_std·μ/σ))
    w_orig = w_std / sigma
    b_orig = b_std - np.sum(w_std * mu / sigma)

    # ----- evaluate on the test set --------------------------------------
    y_pred = X_test_raw @ w_orig + b_orig

    fold_mse  = mse(y_test, y_pred)
    fold_rmse = np.sqrt(fold_mse)
    sst = np.sum((y_test - y_test.mean()) ** 2)
    sse = np.sum((y_test - y_pred) ** 2)
    fold_r2 = 1 - sse / sst

    fold_results.append({
        "fold":   fold + 1,
        "MSE":    fold_mse,
        "RMSE":   fold_rmse,
        "R2":     fold_r2
    })

# ---
# Printe 
print("\n=== 5‑fold performance of the *multiple* linear model ===")
header = f"{'Fold':<5} {'MSE':>12} {'RMSE':>12} {'R²':>10}"
print(header)
print("-" * len(header))
for r in fold_results:
    print(f"{r['fold']:<5d} {r['MSE']:12.2f} {r['RMSE']:12.2f} {r['R2']:10.4f}")

# ---
# Finne mean ± standard‑deviation across folds
MSEs  = np.array([r["MSE"]  for r in fold_results])
RMSEs = np.array([r["RMSE"] for r in fold_results])
R2s   = np.array([r["R2"]   for r in fold_results])

summary = {
    "MSE_mean":  MSEs.mean(),
    "MSE_std":   MSEs.std(ddof=1),
    "RMSE_mean": RMSEs.mean(),
    "RMSE_std":  RMSEs.std(ddof=1),
    "R2_mean":   R2s.mean(),
    "R2_std":    R2s.std(ddof=1)
}

print("\n=== Summary (mean ± std) ===")
print(f"MSE  : {summary['MSE_mean']:.2f} ± {summary['MSE_std']:.2f}")
print(f"RMSE : {summary['RMSE_mean']:.2f} ± {summary['RMSE_std']:.2f}")
print(f"R²   : {summary['R2_mean']:.4f} ± {summary['R2_std']:.4f}")