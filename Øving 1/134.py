# Vi skal:
#   1. Download / load the Superconductivity data set.
#   2. Compute Pearson correlations with the target and pick:
#        • the weakest predictor  (smallest |ρ|)
#        • the strongest predictor (largest |ρ|)
#   3. Standardise each predictor, run a few gradient‑descent steps
#      to obtain the optimal (w,b) in the standardized space,
#      then transform the parameters back to the original scale.
#   4. Create a single figure with two panels:
#        left  – weak predictor + regression line
#        right – strong predictor + regression line

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# Finne sterk og svak predictors (based on |ρ| with the target)
features = df.columns.drop("critical_temp")
corr_with_target = df[features].corrwith(df["critical_temp"])

weak_feat   = corr_with_target.abs().idxmin()   # smallest |ρ|
strong_feat = corr_with_target.abs().idxmax()   # largest  |ρ|

print("\n=== Predictor selection ===")
print(f"Weak predictor   : {weak_feat:30s}  (ρ = {corr_with_target[weak_feat]:+.4f})")
print(f"Strong predictor : {strong_feat:30s}  (ρ = {corr_with_target[strong_feat]:+.4f})")


# ---
# Gradient‑descent routine - som tidligere
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent(x, y, lr=2e-2, n_iter=5000, tol=1e-7):
    """Return w,b (standardised space) and the loss history."""
    n = len(x)
    rng = np.random.default_rng(seed=42)
    w = rng.normal(0, 0.01)
    b = 0.0
    loss_hist = []

    for it in range(1, n_iter + 1):
        y_hat = w * x + b
        loss = mse(y, y_hat)
        loss_hist.append(loss)

        dw = (-2.0 / n) * np.sum(x * (y - y_hat))
        db = (-2.0 / n) * np.sum(y - y_hat)

        w -= lr * dw
        b -= lr * db

        if it > 1 and abs(loss_hist[-2] - loss) < tol:
            break
    return w, b, loss_hist


def fit_feature(feature_name: str):
    """Standardise, run GD, and transform coefficients back to original scale."""
    x_raw = df[feature_name].values.astype(np.float64)
    y_raw = df["critical_temp"].values.astype(np.float64)

    μ = x_raw.mean()
    σ = x_raw.std(ddof=0)                     # population std
    x_std = (x_raw - μ) / σ

    w_std, b_std, loss_hist = gradient_descent(x_std, y_raw,
                                               lr=0.02, n_iter=10000)

    # Back‑transform to original (un‑scaled) space
    w_orig = w_std / σ
    b_orig = b_std - (w_std * μ) / σ

    # Performance metrics (original scale)
    y_pred = w_orig * x_raw + b_orig
    mse_val = mse(y_raw, y_pred)
    r2_val = 1 - np.sum((y_raw - y_pred) ** 2) / np.sum((y_raw - y_raw.mean()) ** 2)

    return {
        "feature": feature_name,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "w": w_orig,
        "b": b_orig,
        "mse": mse_val,
        "r2": r2_val,
        "loss": loss_hist,
    }


# ---
# Som tidligere
weak_res   = fit_feature(weak_feat)
strong_res = fit_feature(strong_feat)


# ---
# Plotte
plt.figure(figsize=(13, 5))

# ----- left panel – weak predictor -----
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(weak_res["x_raw"], weak_res["y_raw"],
            s=8, alpha=0.5, color="#4C72B0", label="data")
x_line = np.linspace(weak_res["x_raw"].min(), weak_res["x_raw"].max(), 300)
y_line = weak_res["w"] * x_line + weak_res["b"]
ax1.plot(x_line, y_line, color="#D55E00", linewidth=2,
         label=f"fit: ŷ = {weak_res['w']:.3e}·x + {weak_res['b']:.2f}")
ax1.set_xlabel(weak_feat)
ax1.set_ylabel("critical_temp (K)")
ax1.set_title("Weak predictor")
ax1.legend()
ax1.grid(alpha=0.3)

# ----- right panel – strong predictor -----
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(strong_res["x_raw"], strong_res["y_raw"],
            s=8, alpha=0.5, color="#4C72B0", label="data")
x_line = np.linspace(strong_res["x_raw"].min(), strong_res["x_raw"].max(), 300)
y_line = strong_res["w"] * x_line + strong_res["b"]
ax2.plot(x_line, y_line, color="#D55E00", linewidth=2,
         label=f"fit: ŷ = {strong_res['w']:.3e}·x + {strong_res['b']:.2f}")
ax2.set_xlabel(strong_feat)
ax2.set_ylabel("critical_temp (K)")
ax2.set_title("Strong predictor")
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle("Linear regression lines for the weak and strong predictors",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
