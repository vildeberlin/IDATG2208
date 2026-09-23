
import zipfile, os
import pandas as pd
import numpy as np

# Leser inn datasettet
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
    na_values=["?", " ?"],           # både med og uten mellomrom → NaN
    skipinitialspace=True
)

df = df.replace('?', np.nan)       # ekstra sikkerhet

# Hent kun de kategoriske variablene (string‑objekter)
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
# ekskluder målvariabelen `income` – den er også kategorisk, men vi behandler den separat
cat_cols.remove("income")
print("\nKategoriske variabler (uten target):")
print(cat_cols)

# Funksjon for å vise kardinalitet før og etter grouping
def cardinality_report(frame, cols):
    """Returner en DataFrame med antall unike verdier for hver kolonne."""
    data = {"feature": [], "unique_before": [], "unique_after": []}
    for c in cols:
        data["feature"].append(c)
        data["unique_before"].append(frame[c].nunique(dropna=True))
        data["unique_after"].append(frame[c].nunique(dropna=True))
    return pd.DataFrame(data)

THRESHOLD = 0.01                     # 1 % av hele datasettet

n_rows = len(df)

def replace_rare_categories(series, thresh=THRESHOLD, other_label="Other"):
    """
    Returnerer en kopi av `series` der alle verdier med frekvens < thresh*N
    blir erstattet med `other_label`.
    """
    # frekvens per kategori (dropna=True ignorerer NaN‑rader)
    freq = series.value_counts(dropna=True) / n_rows
    rare_cats = freq[freq < thresh].index
    # erstatt (men behold NaN – de håndteres separat som “missing‑category” senere)
    new_series = series.where(~series.isin(rare_cats), other_label)
    return new_series

# Lag en kopi av df som vi skal modifisere
df_grouped = df.copy()

# Anvend på hver kategorisk kolonne (unntatt target)
for col in cat_cols:
    df_grouped[col] = replace_rare_categories(df_grouped[col], thresh=THRESHOLD, other_label="Other")

# Vis kardinalitets‑rapport – før og etter grouping
before = {c: df[c].nunique(dropna=True)          for c in cat_cols}
after  = {c: df_grouped[c].nunique(dropna=True) for c in cat_cols}

card_report = pd.DataFrame({
    "feature": cat_cols,
    "unique_before": [before[c] for c in cat_cols],
    "unique_after" : [after[c]  for c in cat_cols]
})

print("\nKardinalitets‑rapport (før → etter \"Other\"‑grouping)")
print(card_report)

# Eksempel på de mest sjeldne kategoriene som ble gruppert
print("\nEksempel: sjeldne kategorier som nå er merket som \"Other\"")
for col in cat_cols:
    # hent de opprinnelige sjeldne verdiene
    freq = df[col].value_counts(dropna=True) / n_rows
    rare = freq[freq < THRESHOLD].index.tolist()
    if rare:
        print(f"\n{col}: {len(rare)} rare kategorier → samles til \"Other\"")
        print("  (noen eksempler):", rare[:5])   # vis kun de første 5 for lesbarhet
    else:
        print(f"\n{col}: ingen kategori under {THRESHOLD*100:.0f}%‑threshold")

# oppsumering
summary = """
**Kardinalitet før og etter grouping (threshold = 1 % av totale rader ≈ 326 forekomster)**  

| feature          | unike før | unike etter |
|------------------|----------:|------------:|
"""
for i, row in card_report.iterrows():
    summary += f"| {row['feature']:15} | {row['unique_before']:9,} | {row['unique_after']:11,} |\n"

print(summary)
