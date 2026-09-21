#   Vi skal:
#     1. Download / load the Superconductivity data set.
#     2. Identify the weak and the strong predictor (smallest /
#        largest |Pearson ρ| with the target).
#     3. Standardise each predictor, run vanilla gradient descent,
#        obtain (w,b) in the *standardised* space, then transform
#        the parameters back to the original (Kelvin) scale.
#     4. Compute training‑MSE and R² for each model.
#     5. Print a concise table with coefficient, intercept,
#        MSE and R² for the two models.

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

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
# Finne sterke og svake predictors (based on Pearson ρ)
features = df.columns.drop("critical_temp")
corr = df[features].corrwith(df["critical_temp"])

weak_feat   = corr.abs().idxmin()   # smallest absolute correlation
strong_feat = corr.abs().idxmax()   # largest absolute correlation

print("\n=== Predictor selection ===")
print(f"Weak predictor   : {weak_feat:30s}  (ρ = {corr[weak_feat]:+.4f})")
print(f"Strong predictor : {strong_feat:30s}  (ρ = {corr[strong_feat]:+.4f})")

# ---
# Gradient‑descent routine (same for both predictors)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent(x, y, lr=2e-2, n_iter=5000, tol=1e-7, verbose=False):
    """Fit ŷ = w·x + b by vanilla GD (MSE loss)."""
    n = len(x)
    rng = np.random.default_rng(seed=42)  # deterministic start
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
        if verbose and it % 500 == 0:
            print(f"Iter {it:5d} | MSE = {loss:.4f}")

    return w, b, losses


def fit_one_feature(feature_name: str):
    """Standardise, run GD, return parameters in original scale."""
    x_raw = df[feature_name].values.astype(np.float64)
    y_raw = df["critical_temp"].values.astype(np.float64)

    μ = x_raw.mean()
    σ = x_raw.std(ddof=0)
    x_std = (x_raw - μ) / σ

    w_std, b_std, loss_hist = gradient_descent(
        x_std, y_raw, lr=0.02, n_iter=10000, verbose=False
    )

    # Transform back to original units
    w_orig = w_std / σ
    b_orig = b_std - (w_std * μ) / σ

    # Performance metrics on the *original* (un‑scaled) data
    y_pred = w_orig * x_raw + b_orig
    mse_val = mse(y_raw, y_pred)
    # R² = 1 - SSE/SST
    r2_val = 1 - np.sum((y_raw - y_pred) ** 2) / np.sum((y_raw - y_raw.mean()) ** 2)

    return {
        "feature":   feature_name,
        "weight":    w_orig,
        "bias":      b_orig,
        "mse":       mse_val,
        "r2":        r2_val,
        "loss_hist": loss_hist,
    }


# ---
# Fit begge modellene
weak_res   = fit_one_feature(weak_feat)
strong_res = fit_one_feature(strong_feat)

# ---
# Printe concise comparison table
print("\n=== Comparison of the two single‑feature linear models ===")
header = f"{'Model':<10} {'Feature':<30} {'β (weight)':>12} {'Intercept (b)':>15} {'MSE':>12} {'R²':>8}"
print(header)
print("-" * len(header))

print(f"{'Weak':<10} {weak_res['feature']:<30} "
      f"{weak_res['weight']:12.6f} {weak_res['bias']:15.4f} "
      f"{weak_res['mse']:12.2f} {weak_res['r2']:8.4f}")

print(f"{'Strong':<10} {strong_res['feature']:<30} "
      f"{strong_res['weight']:12.6f} {strong_res['bias']:15.4f} "
      f"{strong_res['mse']:12.2f} {strong_res['r2']:8.4f}")
