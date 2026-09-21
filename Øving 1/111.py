#
#  Superconductivity dataset – raw vs. log‑transformed target
# 

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")          # for fin default style

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

# sanity check
print(f"Dataset shape: {df.shape}")          # should be (21263, 82)
print(df.head().iloc[:, :5])                 # first 5 columns of first rows

# ---
# Nå skal vi plotte den rå fordelinge til critical_temp
plt.figure(figsize=(8, 4))
sns.histplot(df["critical_temp"], bins=80, kde=True, color="steelblue")
plt.title("Critical Temperature – raw scale")
plt.xlabel("Critical temperature (K)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ---
# Log‑transform the target and plot again
# Datasettet inneholder bare postitve verdier, så vi kan bruke log.
df["log_critical_temp"] = np.log(df["critical_temp"])

plt.figure(figsize=(8, 4))
sns.histplot(df["log_critical_temp"], bins=80, kde=True, color="darkorange")
plt.title("Critical Temperature – log scale (ln)")
plt.xlabel("log(critical temperature)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
