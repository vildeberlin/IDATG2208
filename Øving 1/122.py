#   Vi skal:
#   1. Download the Superconductivity data set (if not already present).
#   2. Compute the correlation of every descriptor with `critical_temp`.
#   3. Print the name of the most positively‑correlated feature
#      and the most negatively‑correlated feature together with their
#      correlation coefficients.

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
# All descriptor columns (everything except the target)
features = df.columns.drop("critical_temp")

# ----
# Korrelasjon mellom hver descriptor med target
corr_with_target = df[features].corrwith(df["critical_temp"])

# ----
# Sterkeste positive og negative korrelasjon
# Positiv
strongest_pos_feature = corr_with_target.idxmax()
strongest_pos_value   = corr_with_target.max()

# Negativ
strongest_neg_feature = corr_with_target.idxmin()
strongest_neg_value   = corr_with_target.min()

# ----
# Printe resultatene
# -----------------------------------------------------------------
print("\n Strongest correlations with `critical_temp`")
print(f"Strongest *positive* correlation:")
print(f"   Feature : {strongest_pos_feature}")
print(f"   ρ       : {strongest_pos_value:.4f}")

print("\nStrongest *negative* correlation:")
print(f"   Feature : {strongest_neg_feature}")
print(f"   ρ       : {strongest_neg_value:.4f}")
