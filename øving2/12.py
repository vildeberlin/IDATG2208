import zipfile, os
import pandas as pd
import numpy as np

# Pakk ut og les inn datasettet (samme kode som i oppgave 1.1,
#    men med korrekt NaN‑håndtering)
zip_path   = "adult.zip"
extract_dir = "adult_data"

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

col_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

train_path = os.path.join(extract_dir, "adult.data")
df = pd.read_csv(
    train_path,
    header=None,
    names=col_names,
    na_values=["?", " ?"],       # både med og uten mellomrom = NaN
    skipinitialspace=True
)

df = df.replace('?', np.nan)   # sikkerhetsnett

# Velge de numeriske kolonnene vi skal inspisere
num_cols = ["age", "hours-per-week", "capital-gain", "capital-loss"]
num_df   = df[num_cols].copy()

# Funksjoner for IQR‑basert og Z‑score‑basert outlier‑deteksjon
def iqr_outliers(series, factor=1.5):
    """Returner bool‑serie hvor True betyr outlier (under/over 1.5·IQR)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper), lower, upper

def zscore_outliers(series, thresh=3.0):
    """Returner bool‑serie hvor |z| > thresh."""
    mean = series.mean()
    std  = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    z = (series - mean) / std
    return z.abs() > thresh

# Kjør begge metodene på hver numerisk variabel og lag en rapport
report = []

for col in num_cols:
    ser = num_df[col]
    # IQR‑metode
    iqr_mask, iqr_low, iqr_up = iqr_outliers(ser, factor=1.5)
    n_iqr = iqr_mask.sum()
    # Z‑score‑metode
    z_mask = zscore_outliers(ser, thresh=3.0)
    n_z = z_mask.sum()
    report.append({
        "feature": col,
        "IQR_low": iqr_low,
        "IQR_up":  iqr_up,
        "IQR_outliers": n_iqr,
        "Z_thresh": 3.0,
        "Z_outliers": n_z
    })

report_df = pd.DataFrame(report)
print("\n--- Outlier‑rapport (IQR vs. Z‑score)")
print(report_df[["feature", "IQR_outliers", "Z_outliers"]])

# Visualisering
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font_scale=1.1)

for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=num_df[col], color="#87CEEB")
    plt.title(f"Boxplot av {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Beslutning – hvordan vi skal behandle outliers
# -----------------------------------------------------------------
# Bruker en capping‑strategi:
#   lower  =  Q1 - 1.5*IQR   
#   upper  =  Q3 + 1.5*IQR
#   for capital‑gain / loss bruker vi i tillegg log(x+1) før capping
# -----------------------------------------------------------------
def cap_series(series, lower, upper):
    """Klipp (capp) verdier utenfor [lower, upper]."""
    return series.clip(lower=lower, upper=upper)

capped_df = df.copy()

for col in num_cols:
    ser = capped_df[col]
    iqr_mask, low, up = iqr_outliers(ser, factor=1.5)

    # Log‑transform for svært skjeve variabler
    if col in ["capital-gain", "capital-loss"]:
        # Log‑transform *før* capping (for å dempe ekstreme toppverdier)
        ser = np.log1p(ser)          # log(x+1) – håndterer 0 på en fin måte
        # Etter log‑transform, finn nye IQR‑grenser (ikke helt nødvendig,
        # men gjør capping konsistent)
        low, up = ser.quantile([0.25, 0.75])
        iqr = up - low
        low  = low - 1.5 * iqr
        up   = up + 1.5 * iqr
        ser = cap_series(ser, low, up)
        # Tilbake til original skala (valgfritt – mange modeller liker log‑skala)
        capped_df[col] = ser
    else:
        capped_df[col] = cap_series(ser, low, up)

# Sjekker antall rader som faktisk ble endret
changes = {}
for col in num_cols:
    n_changed = (capped_df[col] != df[col]).sum()
    changes[col] = n_changed

print("\n--- Hvor mange verdier ble endret ved capping ")
for col, n in changes.items():
    print(f"{col:15}: {n:,} av {len(df)} verdier ({n/len(df)*100:.2f} %)")

# -----------------------------------------------------------------
# 8) Lag en kort tekst‑blokk du kan kopiere til rapporten
# -----------------------------------------------------------------
summary = """
**Outlier‑deteksjon**

| feature          | IQR‑outliers | Z‑outliers | antall verdier endret (capping) |
|------------------|--------------|------------|---------------------------------|
"""
for col in num_cols:
    n_iqr = report_df.loc[report_df["feature"] == col, "IQR_outliers"].values[0]
    n_z   = report_df.loc[report_df["feature"] == col, "Z_outliers"].values[0]
    n_chg = changes[col]
    summary += f"| {col:15} | {n_iqr:12,} | {n_z:10,} | {n_chg:29,} |\n"

print(summary)
