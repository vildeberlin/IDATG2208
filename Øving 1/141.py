import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import csv

# ---
# først skal vi laste ned og unzip datasettet
url = "https://archive.ics.uci.edu/dataset/464/superconductivty+data"
zip_path = "superconduct.zip"
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

# ---
# Finne sterkeste predictor (largest absolute ρ with target)
features = df.columns.drop("critical_temp")
corr = df[features].corrwith(df["critical_temp"])
strong_feat = corr.abs().idxmax()
strong_corr = corr[strong_feat]

print("\n=== Strong predictor (used for all folds) ===")
print(f"Feature : {strong_feat}")
print(f"Pearson ρ = {strong_corr:+.4f}")

# ---
# Gradient‑descent routine - samme som tidligere
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent(x, y, lr=2e-2, n_iter=5000, tol=1e-7):
    """Fit ŷ = w·x + b on *standardised* x using plain GD."""
    n = len(x)
    rng = np.random.default_rng(seed=42)
    w = rng.normal(0, 0.01)
    b = 0.0
    losses = []

    for it in range(1, n_iter + 1):
        y_hat = w * x + b
        loss = mse(y, y_hat)
        losses.append(loss)

        dw = (-2.0 / n) * np.sum(x * (y - y_hat))
        db = (-2.0 / n) * np.sum(y - y_hat)

        w -= lr * dw
        b -= lr * db

        if it > 1 and abs(losses[-2] - loss) < tol:
            break
    return w, b, losses


# ---
# 5‑fold random split (80 % train – 20 % test)
rng = np.random.default_rng(seed=2024)   # reproducible splits
indices = np.arange(len(df))
rng.shuffle(indices)

n = len(df)
fold_size = int(0.2 * n)                 # 20 % test per fold
metrics = []                             # list of dicts, one per fold

for fold in range(5):
    # ----- define test and train indices ---------------------------------
    start = fold * fold_size
    end   = start + fold_size
    test_idx   = indices[start:end]
    train_idx  = np.setdiff1d(indices, test_idx)

    train = df.iloc[train_idx]
    test  = df.iloc[test_idx]

    # ----- standardise the strong feature on the training set only ----------
    x_train_raw = train[strong_feat].values.astype(np.float64)
    y_train     = train["critical_temp"].values.astype(np.float64)

    μ = x_train_raw.mean()
    σ = x_train_raw.std(ddof=0)
    x_train_std = (x_train_raw - μ) / σ

    # ----- fit by GD -------------------------------------------------------
    w_std, b_std, _ = gradient_descent(x_train_std, y_train,
                                       lr=0.02, n_iter=10000)

    # ----- transform parameters back to the original scale -----------------
    w_orig = w_std / σ
    b_orig = b_std - (w_std * μ) / σ

    # ----- evaluate on the test set ----------------------------------------
    x_test_raw = test[strong_feat].values.astype(np.float64)
    y_test     = test["critical_temp"].values.astype(np.float64)

    y_pred = w_orig * x_test_raw + b_orig

    fold_mse  = mse(y_test, y_pred)
    fold_rmse = np.sqrt(fold_mse)
    # R² = 1 - SSE / SST
    sst = np.sum((y_test - y_test.mean()) ** 2)
    sse = np.sum((y_test - y_pred) ** 2)
    fold_r2 = 1 - sse / sst

    metrics.append({
        "fold":   fold + 1,
        "MSE":    fold_mse,
        "RMSE":   fold_rmse,
        "R2":     fold_r2
    })

# ---
# Print
print("\n=== 5‑fold evaluation (strong predictor only) ===")
header = f"{'Fold':<5} {'MSE':>12} {'RMSE':>12} {'R²':>10}"
print(header)
print("-" * len(header))
for m in metrics:
    print(f"{m['fold']:<5d} {m['MSE']:12.2f} {m['RMSE']:12.2f} {m['R2']:10.4f}")
