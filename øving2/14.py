
import zipfile, os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import sklearn

# Hjelpefunksjon: opprett OneHotEncoder med riktig argument
def make_one_hot_encoder():
    """
    Returnerer en OneHotEncoder‑instans som fungerer både i eldre
    (sparse) og i nyere (sparse_output) scikit‑learn‑versjoner.
    """
    try:
        # scikit‑learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        # scikit‑learn >= 1.2 bruker *sparse_output* i stedet for *sparse*
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Les inn datasettet (samme kode som i oppgave 1.1)
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
    na_values=["?", " ?"],          # både med og uten mellomrom → NaN
    skipinitialspace=True
)
df = df.replace('?', np.nan)      # ekstra sikkerhet

# Legg til en egen “Missing”‑kategori (fra oppgave 1.1)
for col in ["workclass", "occupation", "native-country"]:
    df[col] = df[col].astype("category")
    if "Missing" not in df[col].cat.categories:
        df[col] = df[col].cat.add_categories(["Missing"])
    df[col] = df[col].fillna("Missing")

# High‑cardinality grouping (1 % threshold)
THRESHOLD = 0.01          # 1 % av hele datasettet (ca 326 forekomster)

def group_rare(series, thresh=THRESHOLD, other_label="Other"):
    """Erstatt sjeldne verdier med `other_label`.  Fungerer på både
    objekt‑ og Categorical‑serier."""
    # a) legg til "Other" i kategori‑listen dersom serien er Categorical
    if isinstance(series.dtype, pd.CategoricalDtype):
        if other_label not in series.cat.categories:
            series = series.cat.add_categories([other_label])

    # b) beregn frekvens (inkluderer "Missing")
    freq = series.value_counts(dropna=False) / len(series)
    rare = freq[freq < thresh].index
    # vi vil **ikke** gruppere “Missing”
    rare = [r for r in rare if r != "Missing"]

    # c) erstatt
    return series.where(~series.isin(rare), other_label)

# Påfør på de tre kolonnene
for col in ["workclass", "occupation", "native-country"]:
    df[col] = group_rare(df[col])

# Log‑transform + IQR‑capping for kapital‑variablene (fra oppgave 1.2)
def iqr_cap(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)

for col in ["capital-gain", "capital-loss"]:
    df[col] = np.log1p(df[col])     # log(x+1) fordi mange 0‑verdier
    df[col] = iqr_cap(df[col])

# capping for hours‑per‑week (ingen log‑transform)
df["hours-per-week"] = iqr_cap(df["hours-per-week"])

# Definer numeriske vs. kategoriske kolonner
numeric_features = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week"
]

categorical_features = [
    "workclass", "marital-status", "occupation",
    "relationship", "race", "sex",
    "education", "native-country"
]

# One‑Hot‑encoding 
ohe = make_one_hot_encoder()          # automatisk håndterer riktig argument
scaler = StandardScaler()

preprocess = ColumnTransformer(
    transformers=[
        ("num", scaler, numeric_features),
        ("cat", ohe, categorical_features)
    ],
    remainder="drop"
)

pipeline = Pipeline(steps=[("preprocess", preprocess)])

# Konverter målvariabelen til 0/1
df["income"] = df["income"].map({">50K": 1, "<=50K": 0})

X = df.drop(columns=["income"])
y = df["income"]

# Pass pipelineen (får en N × D matrise)
X_enc = pipeline.fit_transform(X)

print("\n--- Dimensjonalitet etter One‑Hot‑encoding")
print(f"Antall rader : {X_enc.shape[0]}")
print(f"Antall kolonner (features) : {X_enc.shape[1]}   (≈ 52)")

# Vis noen feature‑navn 
# OneHotEncoder gir oss navn på dummy‑variablene:
ohe_feature_names = pipeline.named_steps["preprocess"]\
                               .named_transformers_["cat"]\
                               .get_feature_names_out(categorical_features)

all_feature_names = np.concatenate([numeric_features, ohe_feature_names])
print("\nFørste 20 feature‑navn (etter encoding):")
print(all_feature_names[:20])

