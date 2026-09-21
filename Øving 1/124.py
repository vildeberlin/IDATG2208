#   1. Download the Superconductivity data set (if not already present).
#   2. Compute the correlation of every descriptor with the target.
#   3. Identify:
#        • the feature with the **largest absolute** correlation  → strong predictor
#        • the feature with the **smallest absolute** correlation  → weak predictor
#   4. Print the two selected features together with their correlation
#      coefficients


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

# ---
# Korrelasjon for hver feature 
features = df.columns.drop("critical_temp")          # 81 descriptor columns
corr_with_target = df[features].corrwith(df["critical_temp"])

# ---
# Sterkeste og svakeste 
# Strong predictor – largest |ρ|
strong_feat = corr_with_target.abs().idxmax()
strong_rho = corr_with_target[strong_feat]

# Weak predictor – smallest |ρ|
weak_feat = corr_with_target.abs().idxmin()
weak_rho = corr_with_target[weak_feat]

# ---
# Printe resultatet 
print("\n=== Feature‑selection based on correlation with `critical_temp` ===")
print(f"Strong predictor  : {strong_feat:30s}  (ρ = {strong_rho:+.4f})")
print(f"Weak predictor    : {weak_feat:30s}  (ρ = {weak_rho:+.4f})")
