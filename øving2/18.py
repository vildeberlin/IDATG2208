# Biblioteker
import zipfile, os, warnings
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

# Les inn Adult‑datasettet
ZIP_PATH   = "adult.zip"
EXTRACTDIR = "adult_data"

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACTDIR)

COLS = [
    "age","workclass","fnlwgt","education","education-num",
    "marital-status","occupation","relationship","race","sex",
    "capital-gain","capital-loss","hours-per-week","native-country",
    "income"
]

df = pd.read_csv(
    os.path.join(EXTRACTDIR, "adult.data"),
    header=None,
    names=COLS,
    na_values=["?"," ?"],
    skipinitialspace=True
)

# Hjelpe‑funksjoner
def log_iqr_cap(X):
    X = pd.DataFrame(X)

    for col in X.columns:
        s = np.log1p(X[col])

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        X[col] = s.clip(lower, upper)

    return X

log_tf = FunctionTransformer(log_iqr_cap, validate=False)

def make_one_hot():
    """OneHotEncoder som fungerer både på <1.2 og ≥1.2 scikit‑learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:               # nyere versjon
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Kolonnelister
NUMERIC = ["age","fnlwgt","education-num",
           "capital-gain","capital-loss","hours-per-week"]
CATEG   = ["workclass","education","marital-status","occupation",
           "relationship","race","sex","native-country"]
TARGET  = "income"

# Leak‑free preprocessing‑pipeline
# - numerisk del:
numeric_transformer = Pipeline(steps=[
    # Log‑+ IQR‑capping kun på de tre kolonnene med ekstreme outliers
    ("log_iqr", ColumnTransformer(
        transformers=[
            ("gain",  log_tf, ["capital-gain"]),
            ("loss",  log_tf, ["capital-loss"]),
            ("hours", log_tf, ["hours-per-week"])
        ],
        remainder="passthrough"                 # resten av numeriske kolonner holdes uendret
    )),
    ("scaler", StandardScaler())                # null‑mean, unit‑var
])

# – kategorisk del:
categorical_transformer = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),  # erstatter NaN med 'Missing'
    ("ohe",    make_one_hot())                                            # one‑hot‑encoding
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC),
        ("cat", categorical_transformer, CATEG)
    ]
)

# Stratified split → 60 % train, 20 % val, 20 % test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_val_idx, test_idx = next(sss.split(df, df[TARGET]))
train_val = df.iloc[train_val_idx]
test_set  = df.iloc[test_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)  # 0.25 × 0.8 = 0.20
train_idx, val_idx = next(sss2.split(train_val, train_val[TARGET]))
train_set = train_val.iloc[train_idx]
val_set   = train_val.iloc[val_idx]

print(f"\nStørrelser – train:{len(train_set)}  val:{len(val_set)}  test:{len(test_set)}")
print("Klassefordeling i train‑setet")
print(train_set[TARGET].value_counts(normalize=True).round(3))

# FIT kun på trenings‑setet  (ingen lekkasje)
X_tr = train_set.drop(columns=TARGET)
y_tr = train_set[TARGET].map({">50K": 1, "<=50K": 0})

preprocess.fit(X_tr, y_tr)          # alle statistikker beregnes kun fra trenings‑data

# Transformér val‑ og test‑settene med samme pipeline
def transform_and_label(df_subset, name):
    X = df_subset.drop(columns=TARGET)
    y = df_subset[TARGET].map({">50K": 1, "<=50K": 0})
    X_t = preprocess.transform(X)
    print(f"{name:10s} → shape {X_t.shape}")
    return X_t, y

X_train, y_train = transform_and_label(train_set, "TRAIN")
X_val,   y_val   = transform_and_label(val_set,   "VALID")
X_test,  y_test  = transform_and_label(test_set,  "TEST")
