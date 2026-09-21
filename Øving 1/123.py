#   Vi skal:
#   1. Download the Superconductivity data set (if not already present).
#   2. Compute the full 81 × 81 Pearson‑correlation matrix of the descriptors.
#   3. Extract all unordered pairs (i < j) whose absolute correlation exceeds 0.9.
#   4. Print at least five such pairs (the script prints all of them, capped
#      at the first 10 for readability) together with the exact ρ‑value.

import os
import urllib.request
import zipfile
import pandas as pd

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
# All descriptor columns (exclude the target)
features = df.columns.drop("critical_temp")

# ---
# Korrelasjon matrise for the descriptors
corr_matrix = df[features].corr(method="pearson")

# ---
# Nå skal vi finne unordered pairs med |ρ| > 0.9
high_corr_pairs = []                     # list of (feat_i, feat_j, ρ)

# Iterere over kun "the upper triangular part (i < j)"" for å unngå duplikater
for i, feat_i in enumerate(features):
    for j in range(i + 1, len(features)):
        feat_j = features[j]
        rho = corr_matrix.at[feat_i, feat_j]
        if abs(rho) > 0.9:
            high_corr_pairs.append((feat_i, feat_j, rho))

# ---
# Printe resultatet (5 par)
print("\nFeature pairs with |ρ| > 0.9")
if not high_corr_pairs:
    print("No pairs exceed the 0.9 threshold - unexpected for this data set.")
else:
    # Show up to the first 5 pairs for brevity
    for idx, (f1, f2, r) in enumerate(high_corr_pairs[:10], start=1):
        print(f"{idx:2d}. {f1:30s} – {f2:30s}  →  ρ = {r:+.3f}")

    print(f"\nTotal number of highly correlated pairs: {len(high_corr_pairs)}")
    if len(high_corr_pairs) < 5:
        print("Warning: fewer than 5 pairs found – consider lowering the threshold.")
    else:
        print("At least five highly correlated pairs have been identified.")