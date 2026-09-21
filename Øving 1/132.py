#   1. Download / load the Superconductivity data set.
#   2. Compute Pearson correlations with the target and pick the
#      descriptor that has the largest absolute correlation → strong predictor.
#   3. Standardise that predictor (zero mean, unit variance).
#   4. Fit ŷ = w·x_std + b by vanilla gradient descent (MSE loss).
#   5. Print the learned parameters (both in the standardized space and
#      transformed back to the original scale), final training MSE,
#      and produce two figures:
#        • gd_loss_curve.png – loss vs. iteration
#        • strong_predictor_scatter.png – scatter plot with fitted line

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
# Finne sterk predictor (largest |ρ| with the target)
features = df.columns.drop("critical_temp")
corr_with_target = df[features].corrwith(df["critical_temp"])

strong_feat = corr_with_target.abs().idxmax()
strong_corr = corr_with_target[strong_feat]

print("\n=== Strong predictor (selected for GD) ===")
print(f"Feature : {strong_feat}")
print(f"Pearson ρ = {strong_corr:+.4f}")

# ---
# Standarisere
x_raw = df[strong_feat].values.astype(np.float64)   # shape (n,)
y_raw = df["critical_temp"].values.astype(np.float64)

x_mean = x_raw.mean()
x_std  = x_raw.std(ddof=0)          # population std
x_stdzd = (x_raw - x_mean) / x_std   # now mean=0, std=1

# ---
# Gradient‑descent routine 
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent(x, y, lr=2e-2, n_iter=5000, tol=1e-7, verbose=False):
    """Fit ŷ = w·x + b by plain gradient descent."""
    n = len(x)
    rng = np.random.default_rng(seed=0)
    w = rng.normal(0, 0.01)
    b = 0.0
    losses = []

    for it in range(1, n_iter + 1):
        y_hat = w * x + b
        loss = mse(y, y_hat)
        losses.append(loss)

        # gradients of MSE
        dw = (-2.0 / n) * np.sum(x * (y - y_hat))
        db = (-2.0 / n) * np.sum(y - y_hat)

        w -= lr * dw
        b -= lr * db

        if it > 1 and abs(losses[-2] - loss) < tol:
            if verbose:
                print(f"Early stop at {it} (Δloss < {tol})")
            break
        if verbose and it % 500 == 0:
            print(f"Iter {it:5d} | MSE = {loss:.4f}")

    return w, b, losses


# ---
# Kjøre GD på standardized predictor
w_opt, b_opt, loss_hist = gradient_descent(
    x_stdzd, y_raw,
    lr=0.02,
    n_iter=10000,
    verbose=False
)

print("\n=== GD result (standardised space) ===")
print(f"w (std) = {w_opt:.6f}")
print(f"b (std) = {b_opt:.6f}")
print(f"Final training MSE = {loss_hist[-1]:.4f}")

# ---
# Plotte
plt.figure(figsize=(8, 4))
plt.plot(loss_hist, color="#4C72B0")
plt.title("Training MSE vs. Gradient‑Descent Iteration", fontsize=14)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---
# Gjøre om parametere tilbake til original (un‑scaled) units
# ŷ = w_std * ((x - μ)/σ) + b_std
# ⇒ ŷ = (w_std/σ) * x + (b_std - w_std * μ/σ)
w_orig = w_opt / x_std
b_orig = b_opt - (w_opt * x_mean) / x_std

# ---
# Plotte
plt.figure(figsize=(8, 5))
plt.scatter(x_raw, y_raw, s=10, alpha=0.5, label="data", color="#4C72B0")
x_line = np.linspace(x_raw.min(), x_raw.max(), 300)
y_line = w_orig * x_line + b_orig
plt.plot(x_line, y_line, color="#D55E00", linewidth=2,
         label=f"fit: ŷ = {w_orig:.3f}·x + {b_orig:.3f}")

plt.title(f"Linear regression on strong predictor `{strong_feat}`", fontsize=14)
plt.xlabel(strong_feat)
plt.ylabel("critical_temp (K)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
