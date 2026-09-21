#  Steps:
#   1. Download & load the Superconductivity data set.
#   2. Compute correlation of each descriptor with `critical_temp`.
#   3. Pick the descriptor with the smallest |ρ| → weak predictor.
#   4. Standardise that predictor (zero mean, unit variance).
#   5. Run gradient descent to minimise MSE:
#          ŷ = w * x_std + b
#   6. Plot loss curve and the fitted line (in original units).

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
# Identifisere svak predictor 
features = df.columns.drop("critical_temp")
corr_with_target = df[features].corrwith(df["critical_temp"])
weak_feature = corr_with_target.abs().idxmin()
weak_corr   = corr_with_target[weak_feature]

print("\n=== Weak predictor (selected for GD) ===")
print(f"Feature : {weak_feature}")
print(f"Pearson ρ = {weak_corr:+.4f}")

# ---
# Standarisere den
x_raw = df[weak_feature].values.astype(np.float64)          # shape (n,)
y_raw = df["critical_temp"].values.astype(np.float64)       # shape (n,)

x_mean = x_raw.mean()
x_std  = x_raw.std(ddof=0)          # population std (ddof=0)
x_stdzd = (x_raw - x_mean) / x_std   # now mean=0, std=1

# ---
# Gradient‑descent implementation
def mse(y_true, y_pred):
    #Mean‑squared error
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(x, y, lr=1e-2, n_iter=5000, tol=1e-7, verbose=True):
    #Simple GD for a model  ŷ = w * x + b
    #Returns: w, b, list_of_losses
    n = len(x)
    # initialise with small random numbers (helps if learning‑rate is high)
    rng = np.random.default_rng(seed=42)
    w = rng.normal(loc=0.0, scale=0.01)
    b = 0.0

    losses = []
    for it in range(1, n_iter + 1):
        y_pred = w * x + b
        loss = mse(y, y_pred)
        losses.append(loss)

        # gradients
        dw = (-2.0 / n) * np.sum(x * (y - y_pred))
        db = (-2.0 / n) * np.sum(y - y_pred)

        # update
        w -= lr * dw
        b -= lr * db

        # optional early stop
        if it > 1 and abs(losses[-2] - loss) < tol:
            if verbose:
                print(f"Early stop at iteration {it} (Δloss < {tol:.1e})")
            break

        if verbose and it % 500 == 0:
            print(f"Iter {it:5d} | MSE = {loss:.4f}")

    return w, b, losses

# ---
# Kjøre GD
learning_rate = 0.02          # a modest LR works well for a single feature
max_iter      = 10000
w_opt, b_opt, loss_history = gradient_descent(
    x_stdzd, y_raw,
    lr=learning_rate,
    n_iter=max_iter,
    verbose=False
)

print("\n=== Gradient‑descent result (standardised space) ===")
print(f"Optimal weight (w) : {w_opt:.6f}")
print(f"Optimal bias  (b) : {b_opt:.6f}")
print(f"Final training MSE : {loss_history[-1]:.4f}")

# ---
# Plotte loss curve
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color="#4C72B0")
plt.title("Training MSE vs. Gradient‑Descent Iteration", fontsize=14)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------
# Plotte strek i original (un‑scaled) space
# -----------------------------------------------------------------
# Transform the learned parameters back to the original scale:
#   ŷ = w_stdzd * ((x - μ)/σ) + b
# ⇒ ŷ = (w_stdzd / σ) * x + (b - w_stdzd * μ / σ)
w_original = w_opt / x_std
b_original = b_opt - (w_opt * x_mean) / x_std

# Make a scatter + regression line plot
plt.figure(figsize=(8, 5))
plt.scatter(x_raw, y_raw, s=10, alpha=0.5, label="data", color="#4C72B0")
x_line = np.linspace(x_raw.min(), x_raw.max(), 200)
y_line = w_original * x_line + b_original
plt.plot(x_line, y_line, color="#D55E00", linewidth=2,
         label=f"fit:  ŷ = {w_original:.3f}·x + {b_original:.3f}")

plt.title(f"Linear regression on weak predictor `{weak_feature}`", fontsize=14)
plt.xlabel(weak_feature)
plt.ylabel("critical_temp (K)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()