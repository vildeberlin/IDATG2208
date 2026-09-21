#   Nå skal vi:
#   1. Download the UCI Superconductivity data (if it is not already
#      present locally).
#   2. Compute the full 81 × 81 Pearson‑correlation matrix of the
#      descriptor columns.
#   3. Pick the 15 descriptors whose |ρ| with `critical_temp` is largest.
#   4. Plot a heat‑map for those 15 descriptors together with the
#      target (16 × 16 matrix) and store the figure as PNG.

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120   # tydelige figurer på skjerm


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

# The dataset contains 81 descriptor columns + the target column.
features = df.columns.drop("critical_temp")        # 81 descriptors
target   = "critical_temp"

print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Number of descriptor columns: {len(features)}")

# ----
# Korrelert matrise
corr_matrix = df[features].corr(method="pearson")   # 81 × 81

# ---
# Nå skal vi finne de 15 features-ene som er mest korrelert med target-en
target_corr = df[features].corrwith(df[target])    # Series: 81 correlations
top15_features = (
    target_corr.abs()
              .sort_values(ascending=False)
              .head(15)
              .index
              .tolist()
)

print("\n15 features with highest |ρ| to critical_temp:")
for i, f in enumerate(top15_features, 1):
    print(f"{i:2d}. {f:30s}  ρ = {target_corr[f]: .4f}")

# ----
# Så skal vi lage en sub‑matrise som også inneholder target kolonne
# We want a (15 + 1) × (15 + 1) matrix so that the target appears in the
# heat‑map as well (makes the plot easier to read).
sub_cols = top15_features + [target]               # list of 16 column names
sub_corr = df[sub_cols].corr()                     # 16 × 16 matrix

# ---
# Plotte heat‑map
plt.figure(figsize=(12, 9))
sns.heatmap(
    sub_corr,
    annot=True,                 # write the correlation numbers in the cells
    fmt=".2f",
    cmap="RdYlBu_r",
    linewidths=0.5,
    cbar_kws={"shrink": 0.7},
)
plt.title(
    "Heat map of the 15 descriptors most correlated with `critical_temp`\n"
    "(plus the target itself)",
    fontsize=14,
    pad=20,
)
plt.tight_layout()
plt.show()
