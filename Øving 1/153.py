import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

csv_name = "train.csv"

# ---
# Så skal vi load-e dataen
df = pd.read_csv(csv_name)

# ---
# Finne sterk predictor (largest |Pearson ρ|)
features = df.columns.drop("critical_temp")
corr = df[features].corrwith(df["critical_temp"])
strong_feat = corr.abs().idxmax()
print(f"\nStrong predictor (used for the simple model): {strong_feat}")


# ---
# Gradient‑descent utilities
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gd_simple(x_std, y, lr=2e-2, n_iter=8000, tol=1e-7):
    """
    Gradient descent for a *single* standardized feature.
    Returns (w, b, loss_history) where loss_history is a list of MSE values.
    """
    n = len(x_std)
    rng = np.random.default_rng(seed=42)
    w = rng.normal(0, 0.01)
    b = 0.0
    loss_hist = []

    for it in range(1, n_iter + 1):
        y_hat = w * x_std + b
        loss = mse(y, y_hat)
        loss_hist.append(loss)

        # gradients
        dw = (-2.0 / n) * np.sum(x_std * (y - y_hat))
        db = (-2.0 / n) * np.sum(y - y_hat)

        w -= lr * dw
        b -= lr * db

        if it > 1 and abs(loss_hist[-2] - loss) < tol:
            break
    return w, b, loss_hist


def gd_multiple(X_std, y, lr=1e-2, n_iter=8000, tol=1e-7):
    """
    Gradient descent for a multivariate standardized design matrix.
    Returns (w_vector, b, loss_history).
    """
    n, p = X_std.shape
    rng = np.random.default_rng(seed=42)
    w = rng.normal(0, 0.01, size=p)
    b = 0.0
    loss_hist = []

    for it in range(1, n_iter + 1):
        y_hat = X_std @ w + b
        loss = mse(y, y_hat)
        loss_hist.append(loss)

        # gradients
        grad_w = (-2.0 / n) * (X_std.T @ (y - y_hat))
        grad_b = (-2.0 / n) * np.sum(y - y_hat)

        w -= lr * grad_w
        b -= lr * grad_b

        if it > 1 and abs(loss_hist[-2] - loss) < tol:
            break
    return w, b, loss_hist


# ---
# 5‑fold split
rng = np.random.default_rng(seed=2024)   # reproducible folds
indices = np.arange(len(df))
rng.shuffle(indices)

n_samples = len(df)
fold_size = int(0.2 * n_samples)        # 20 % test per fold

# Containers to collect information across all folds
cost_history_simple  = []   # list of loss lists (one per fold)
cost_history_multi   = []
param_history_simple = []   # list of weight histories (scalar)
param_history_multi  = []   # list of weight‑vectors (we will look at a few)
pred_vs_actual = {"simple": [], "multi": []}
residuals      = {"simple": [], "multi": []}

# Full feature matrix (81 columns)
X_full = df.drop(columns=["critical_temp"]).values.astype(np.float64)
y_full = df["critical_temp"].values.astype(np.float64)

for fold in range(5):
    # ----- indices for this fold ---------------------------------------
    start = fold * fold_size
    end   = start + fold_size
    test_idx  = indices[start:end]
    train_idx = np.setdiff1d(indices, test_idx)

    # ----------=== 1) Simple model (strong predictor) =================
    x_train_raw = df.loc[train_idx, strong_feat].values.astype(np.float64)
    x_test_raw  = df.loc[test_idx,  strong_feat].values.astype(np.float64)
    y_train = df.loc[train_idx, "critical_temp"].values.astype(np.float64)
    y_test  = df.loc[test_idx,  "critical_temp"].values.astype(np.float64)

    mu = x_train_raw.mean()
    sigma = x_train_raw.std(ddof=0)
    x_train_std = (x_train_raw - mu) / sigma
    x_test_std  = (x_test_raw  - mu) / sigma

    w, b, loss_hist = gd_simple(x_train_std, y_train,
                                lr=0.02, n_iter=8000, tol=1e-7)
    cost_history_simple.append(loss_hist)
    param_history_simple.append([w])       # scalar, wrapped in a list for uniformity

    # back‑transform to original scale
    w_orig = w / sigma
    b_orig = b - (w * mu) / sigma
    y_pred  = w_orig * x_test_raw + b_orig

    pred_vs_actual["simple"].append((y_test, y_pred))
    residuals["simple"].append(y_test - y_pred)

    # ----------=== 2) Multiple model (all 81 features) =================
    X_train_raw = X_full[train_idx]   # (n_train, 81)
    X_test_raw  = X_full[test_idx]    # (n_test, 81)

    mu_X = X_train_raw.mean(axis=0)
    sigma_X = X_train_raw.std(axis=0, ddof=0)
    sigma_X[sigma_X == 0] = 1.0   # safety

    X_train_std = (X_train_raw - mu_X) / sigma_X
    X_test_std  = (X_test_raw  - mu_X) / sigma_X

    w_vec, b_multi, loss_hist_multi = gd_multiple(X_train_std, y_train,
                                                 lr=0.01, n_iter=8000, tol=1e-7)

    cost_history_multi.append(loss_hist_multi)
    param_history_multi.append(w_vec.copy())   # keep a copy for the fold

    # back‑transform coefficients
    w_orig_vec = w_vec / sigma_X
    b_orig_multi = b_multi - np.sum(w_vec * mu_X / sigma_X)

    y_pred_multi = X_test_raw @ w_orig_vec + b_orig_multi

    pred_vs_actual["multi"].append((y_test, y_pred_multi))
    residuals["multi"].append(y_test - y_pred_multi)

# ---
# Plot 1 – Cost (MSE) vs. Iteration
plt.figure(figsize=(9, 5))
# average loss across folds (simple)
avg_simple = np.mean([np.pad(l, (0, max(map(len, cost_history_simple)) - len(l)),
                            constant_values=np.nan) for l in cost_history_simple],
                     axis=0)
plt.plot(avg_simple, label="Simple (strong feature)", color="#4C72B0")

# average loss across folds (multiple)
avg_multi = np.mean([np.pad(l, (0, max(map(len, cost_history_multi)) - len(l)),
                           constant_values=np.nan) for l in cost_history_multi],
                    axis=0)
plt.plot(avg_multi, label="Multiple (81 features)", color="#D55E00")

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Cost vs. Iteration (average over 5 folds)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---
# Plot 2 – Parameter convergence
# Simple model – weight trajectory (scalar) across iterations
plt.figure(figsize=(9, 4))
for i, loss_hist in enumerate(cost_history_simple):
    # recover weight at each iteration by re‑running GD but storing w each step
    # (we already stored only the final w; to keep things simple we just plot the
    #  final weight as a point – the curve for the multiple model will be more
    #  informative).  Here we plot the final weight per fold:
    plt.scatter(i + 1, param_history_simple[i][0],
                color="#4C72B0", s=80, label="Simple" if i == 0 else "")
plt.xlabel("Fold")
plt.ylabel("Learned weight (strong feature)")
plt.title("Simple model – learned weight per fold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Multiple model – we visualise the magnitude of the **top 10** coefficients
# (by absolute value) after training on each fold.
top_k = 10
plt.figure(figsize=(10, 5))
for fold_idx, w_vec in enumerate(param_history_multi):
    # compute absolute values, get indices of top‑k
    idx_top = np.argsort(np.abs(w_vec))[-top_k:][::-1]
    coeffs = w_vec[idx_top]
    plt.plot(range(1, top_k + 1), coeffs,
             marker='o', label=f"Fold {fold_idx + 1}")

plt.xlabel(f"Rank (1 = largest |β| among top {top_k})")
plt.ylabel("Coefficient value (standardised space)")
plt.title(f"Multiple model – top‑{top_k} coefficients per fold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---
# Plot 3 – Predicted vs. Actual
def agg_pred_actual(pair_list):
    """Flatten a list of (y_true, y_pred) tuples."""
    y_true_all = np.concatenate([p[0] for p in pair_list])
    y_pred_all = np.concatenate([p[1] for p in pair_list])
    return y_true_all, y_pred_all


y_true_simple, y_pred_simple = agg_pred_actual(pred_vs_actual["simple"])
y_true_multi , y_pred_multi  = agg_pred_actual(pred_vs_actual["multi"])

plt.figure(figsize=(9, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_true_simple, y_pred_simple, s=6, alpha=0.5, color="#4C72B0")
plt.plot([y_true_simple.min(), y_true_simple.max()],
         [y_true_simple.min(), y_true_simple.max()],
         '--', color='gray')
plt.xlabel("Actual critical_temp (K)")
plt.ylabel("Predicted")
plt.title("Simple model (strong feature)")

plt.subplot(1, 2, 2)
plt.scatter(y_true_multi, y_pred_multi, s=6, alpha=0.5, color="#D55E00")
plt.plot([y_true_multi.min(), y_true_multi.max()],
         [y_true_multi.min(), y_true_multi.max()],
         '--', color='gray')
plt.xlabel("Actual critical_temp (K)")
plt.title("Multiple model (81 features)")

plt.tight_layout()
plt.show()

# ---
# Plot 4 – Residuals (check linear‑regression assumptions)
res_simple = np.concatenate(residuals["simple"])
res_multi   = np.concatenate(residuals["multi"])

plt.figure(figsize=(9, 4))

plt.subplot(1, 2, 1)
plt.hist(res_simple, bins=40, color="#4C72B0", edgecolor='black')
plt.title("Residuals – Simple model")
plt.xlabel("Residual (K)")

plt.subplot(1, 2, 2)
plt.hist(res_multi, bins=40, color="#D55E00", edgecolor='black')
plt.title("Residuals – Multiple model")
plt.xlabel("Residual (K)")

plt.tight_layout()
plt.show()