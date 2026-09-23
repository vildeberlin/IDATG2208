# 1) Pakk ut zip‑fila (antar at den ligger i samme mappe som notebooken)
# 2) Les inn data med pandas
# 3) Tell opp manglende verdier per kolonne
# 4) Undersøk om mangelen er tilfeldig eller konsentrert i noen
#    undergrupper (f.eks. occupation vs. workclass)

import zipfile, os
import pandas as pd
import numpy as np

# Pakk ut zip‑fila
zip_path   = "adult.zip"
extract_dir = "adult_data"

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

# Les inn adult.data – angi begge mulige varianter av "?"
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
    na_values=["?", " ?"],      # både med og uten mellomrom blir NaN
    skipinitialspace=True       # fjerner eventuelle ledende mellomrom
)

# I tilfelle noen "?" har blitt stående igjen (f.eks. i eldre versjoner)
df = df.replace('?', np.nan)

print(f"Datasett‑størrelse : {df.shape[0]} rader × {df.shape[1]} kolonner\n")

# 3) Funksjon for pen utskrift (virker både i terminal og notebook)
def show(obj, max_rows=5):
    try:
        from IPython.display import display
        display(obj.head(max_rows) if isinstance(obj, pd.DataFrame) else obj)
    except Exception:
        print(obj.head(max_rows) if isinstance(obj, pd.DataFrame) else obj)

#  Første fem rader (med ? erstattet av NaN)
print("Første fem rader (med ? erstattet av NaN):")
show(df)

# Hvor mange mangler per kolonne?
missing_counts  = df.isna().sum()
missing_percent = (missing_counts / len(df) * 100).round(2)

miss_table = pd.DataFrame({
    "Missing values": missing_counts,
    "Missing %":      missing_percent
}).sort_values("Missing values", ascending=False)

print("\nAntall og % manglende per variabel:")
show(miss_table)

# Er mangelen tilfeldig? – Occupation vs. Workclass
df["occ_missing"] = df["occupation"].isna().astype(int)

crosstab = pd.crosstab(df["workclass"], df["occ_missing"],
                       margins=True, normalize="index")

print("\nAndel av rader med manglende occupation for hver workclass:")
show(crosstab.round(3))

# (valgfritt) sjekk om workclass‑mangel henger sammen med occupation‑mangel
df["wc_missing"] = df["workclass"].isna().astype(int)
crosstab_wc = pd.crosstab(df["occupation"], df["wc_missing"],
                          margins=True, normalize="index")

print("\nAndel av rader med manglende workclass for hver occupation:")
show(crosstab_wc.round(3))

# Oppsummering 
total_missing = missing_counts.sum()
pct_total = (total_missing / (df.shape[0] * df.shape[1]) * 100).round(2)

print(f"Totalt {total_missing} manglende verdier -> {pct_total}% av alle celler.")
