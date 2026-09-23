
import matplotlib.pyplot as plt

import os, zipfile, warnings
import numpy as np, pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# Load data
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
    skipinitialspace=True,
)

# Helper – log + IQR capping (same as before)
def log_iqr_cap(X):
    X = pd.DataFrame(X)
    for col in X.columns:
        s = np.log1p(X[col])
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        X[col] = s.clip(lower, upper)
    return X

log_tf = FunctionTransformer(log_iqr_cap, validate=False)

def make_one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:           # newer sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Column groups
NUMERIC = ["age","fnlwgt","education-num",
           "capital-gain","capital-loss","hours-per-week"]
CATEG   = ["workclass","education","marital-status","occupation",
           "relationship","race","sex","native-country"]
TARGET  = "income"

# Pre‑processing pipeline (leak‑free)
numeric_transformer = Pipeline(steps=[
    ("log_iqr", ColumnTransformer(
        transformers=[
            ("gain",  log_tf, ["capital-gain"]),
            ("loss",  log_tf, ["capital-loss"]),
            ("hours", log_tf, ["hours-per-week"])
        ],
        remainder="passthrough"
    )),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("ohe",    make_one_hot())
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC),
        ("cat", categorical_transformer, CATEG)
    ]
)

# Train / validation split (stratified)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_val_idx, _ = next(sss.split(df, df[TARGET]))
train_val = df.iloc[train_val_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(sss2.split(train_val, train_val[TARGET]))
train_set = train_val.iloc[train_idx]
val_set   = train_val.iloc[val_idx]

# Fit preprocessing on the train part only
X_tr = train_set.drop(columns=TARGET)
y_tr = train_set[TARGET].map({">50K": 1, "<=50K": 0})
preprocess.fit(X_tr, y_tr)

def transform(df_subset, name):
    X = df_subset.drop(columns=TARGET)
    y = df_subset[TARGET].map({">50K": 1, "<=50K": 0})
    X_t = preprocess.transform(X)
    print(f"{name:10s} → shape {X_t.shape}")
    return X_t, y

X_train, y_train = transform(train_set, "TRAIN")
X_val,   y_val   = transform(val_set,   "VALID")

# Subsample (10 k) 
SUBSIZE = 10_000
rng = np.random.default_rng(seed=42)

unique, counts = np.unique(y_train, return_counts=True)
prop = counts / counts.sum()
n_per_class = (prop * SUBSIZE).astype(int)

sub_idx = []
for cls, n in zip(unique, n_per_class):
    cls_idx = np.where(y_train == cls)[0]
    sub_idx.extend(rng.choice(cls_idx, size=n, replace=False))

sub_idx = np.array(sub_idx)
X_sub = X_train[sub_idx]
y_sub = y_train.iloc[sub_idx]

print("\nSubsample class distribution (should resemble train set):")
print(y_sub.value_counts(normalize=True).round(3))

# γ‑grid (log‑spaced) -> evaluate on validation set
gamma_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]   # six values, three decades

acc_lst = []
f1_lst  = []
sv_cnt  = []               # total number of support vectors per model

for γ in gamma_vals:
    rbf = SVC(C=1.0,               # keep C fixed so we isolate γ‑effects
              kernel="rbf",
              gamma=γ,
              probability=False,  # we only need decision_function for ROC later
              max_iter=2000,      # safe bound, prevents endless runs
              random_state=42)

    rbf.fit(X_sub, y_sub)               # train on subsample

    pred = rbf.predict(X_val)           # evaluate on validation set
    acc_lst.append(accuracy_score(y_val, pred))
    f1_lst.append(f1_score(y_val, pred))

    # total number of support vectors (sum over classes)
    sv_cnt.append(rbf.n_support_.sum())

# Choose best γ (max validation F1)
best_idx = int(np.argmax(f1_lst))
best_gamma = gamma_vals[best_idx]

print("\nBest γ (max F1) →", best_gamma)
print(f"  Accuracy @ best γ : {acc_lst[best_idx]:.4f}")
print(f"  F1‑score @ best γ : {f1_lst[best_idx]:.4f}")
print(f"  #‑SV @ best γ    : {sv_cnt[best_idx]}")

# Plot metrics vs. γ
plt.figure(figsize=(8, 5))

# left y‑axis – accuracy & F1
ax1 = plt.gca()
ax1.plot(gamma_vals, acc_lst, marker='o', label='Accuracy', color='C0')
ax1.plot(gamma_vals, f1_lst,  marker='s', label='F1‑score', color='C1')
ax1.set_xscale('log')
ax1.set_xlabel('γ (log‑scale)')
ax1.set_ylabel('Metric')
ax1.set_ylim(0, 1)
ax1.grid(alpha=0.3, which='both')

# right y‑axis – number of support vectors
ax2 = ax1.twinx()
ax2.plot(gamma_vals, sv_cnt, marker='^', label='#‑SV', color='C2')
ax2.set_ylabel('Total # of support vectors')
ax2.set_ylim(0, max(sv_cnt) * 1.1)

# combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='lower right')

plt.title('Effect of γ on RBF‑SVM (validation set)')
plt.tight_layout()
plt.show()
