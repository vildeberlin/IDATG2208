#   Nå skal vi:
#   - Laster Superconductivity‑datasettet fra UCI‑repoet
#   - Plotter fordelingen av kritisk temperatur (raw vs. log)

import urllib.request
import zipfile
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---
# først skal vi laste ned og unzip datasettet
# (kopriert fra 111.py)
url = "https://archive.ics.uci.edu/dataset/464/superconductivty+data"
zip_path = "superconduct.zip"
csv_name = "train.csv"               # navnet inni zip filen

# Hvis vi allerede har fila, dropper vi det
if not os.path.isfile(zip_path):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download finished.")
else:
    print("Zip file already present - skipping download.")

# Extract the CSV if it does not exist yet
if not os.path.isfile(csv_name):
    print("Extracting CSV from zip …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(csv_name)
    print("Extraction done.")
else:
    print("CSV already extracted - skipping.")

# ---
# Så skal vi load-e dataen
df = pd.read_csv(csv_name)

# sanity check
print(f"Dataset shape: {df.shape}")          # should be (21263, 82)
print(df.head().iloc[:, :5])                 # first 5 columns of first rows

# ---
# Nå skal vi sjekke at målvariabelen er positiv 
# Vi kan kun bruke log når det er positivt
if (df["critical_temp"] <= 0).any():
    sys.exit("ERROR: Det finnes kritisk temperatur ≤ 0 - log-transform kan "
             "ikke brukes på slike verdier.")
else:
    print("Alle kritiske temperaturer er positive - klar for log-transform.")

# ---
# Nå skal vi lage to nye pandas‑Series
raw_temp = df["critical_temp"]          # en for rå temp
log_temp = np.log(raw_temp)             # og en for naturlig log 

# ----
# Plot‑funksjon 
def plot_distribution(series: pd.Series, title: str,
                      xlabel: str) -> None:
    """
    Lager et histogram med KDE-kurve.
    - `series`  : pandas Series som skal visualiseres
    - `title`   : figur-tittel
    - `xlabel`  : aksen-etikett (x-aksen)
    - `outfile` : filnavn som PNG lagres til (brukes senere i rapporten)
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(series, bins=80, kde=True, color="#4C72B0")
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Antall forekomster", fontsize=12)
    plt.tight_layout()
    plt.show()          # viser figuren i interaktivt vindu (hvis du har GUI)


# ---
# Plotte raw‑fordeling
plot_distribution(
    series   = raw_temp,
    title    = "Kritisk temperatur - rå skala",
    xlabel   = "Kritisk temperatur (Kelvin)",
)

# ---
# Plotte log‑transformert fordeling
plot_distribution(
    series   = log_temp,
    title    = "Kritisk temperatur - log-skala (ln)",
    xlabel   = "ln(kritisk temperatur)",
)
