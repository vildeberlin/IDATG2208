#!/usr/bin/env python3
# --------------------------------------------------------------
#  RBF‑SVM γ‑sweep (visualise under‑/over‑fitting)
# --------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

import os, zipfile, warnings
import numpy as np, pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 1) Load Adult data (same zip you already have)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 2) Helper – log + IQR capping (identical to previous notebooks)
# ------------------------------------------------------------------
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
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# ------------------------------------------------------------------
# 3) Column groups
# ------------------------------------------------------------------
NUMERIC = ["age","fnlwgt","education-num",
           "capital-gain","capital-loss","hours-per-week"]
CATEG   = ["workclass","education","marital-status","occupation",
           "relationship","race","sex","native-country"]
TARGET  = "income"

# ------------------------------------------------------------------
# 4) Pre‑processing pipeline (leak‑free)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 5) Train / validation split (stratified)
# ------------------------------------------------------------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_val_idx, _ = next(sss.split(df, df[TARGET]))
train_val = df.iloc[train_val_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(sss2.split(train_val, train_val[TARGET]))
train_set = train_val.iloc[train_idx]
val_set   = train_val.iloc[val_idx]

# ------------------------------------------------------------------
# 6) Fit preprocessing on the *train* part only
# ------------------------------------------------------------------
X_tr_raw = train_set.drop(columns=TARGET)
y_tr_raw = train_set[TARGET].map({">50K": 1, "<=50K": 0})
preprocess.fit(X_tr_raw, y_tr_raw)

def transform(df_subset, name):
    X = df_subset.drop(columns=TARGET)
    y = df_subset[TARGET].map({">50K": 1, "<=50K": 0})
    X_t = preprocess.transform(X)
    print(f"{name:10s} → shape {X_t.shape}")
    return X_t, y

X_train, y_train = transform(train_set, "TRAIN")
X_val,   y_val   = transform(val_set,   "VALID")

# ------------------------------------------------------------------
# 7) Subsample 10 k (keep class proportion)
# ------------------------------------------------------------------
SUBSIZE = 10_000
rng = np.random.default_rng(seed=42)

uniq, cnt = np.unique(y_train, return_counts=True)
prop = cnt / cnt.sum()
n_per_class = (prop * SUBSIZE).astype(int)

sub_idx = []
for cls, n in zip(uniq, n_per_class):
    idx = np.where(y_train == cls)[0]
    sub_idx.extend(rng.choice(idx, size=n, replace=False))

sub_idx = np.array(sub_idx)
X_sub = X_train[sub_idx]
y_sub = y_train.iloc[sub_idx]

# ------------------------------------------------------------------
# 8) Reduce to 2 D with PCA (keep the same projection for every model)
# ------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_sub_2d = pca.fit_transform(X_sub)
X_val_2d = pca.transform(X_val)

# ------------------------------------------------------------------
# 9) γ‑settings to sweep (four values from very smooth → very wiggly)
# ------------------------------------------------------------------
gamma_vals = [0.001, 0.01, 0.1, 1.0]      # you can add more if you wish
C_fixed    = 10                         # the C that worked best earlier

sv_counts = []     # will store # of support vectors for each γ

def plot_boundary(ax, svm, gamma):
    # grid for contour
    x_min, x_max = X_sub_2d[:,0].min() - 1, X_sub_2d[:,0].max() + 1
    y_min, y_max = X_sub_2d[:,1].min() - 1, X_sub_2d[:,1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.predict(grid).reshape(xx.shape)

    # decision surface
    cmap_bg = plt.cm.Pastel2
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)

    # validation points (true label)
    colors = ["steelblue", "darkorange"]
    markers = ["o", "s"]
    for cls, col, mk in zip([0,1], colors, markers):
        idx = (y_val == cls)
        ax.scatter(X_val_2d[idx,0], X_val_2d[idx,1],
                   c=col, marker=mk, edgecolor="k",
                   s=25, label=f"{'<=50K' if cls==0 else '>50K'} (val)",
                   alpha=0.8)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(r"$\gamma={}$".format(gamma))
    ax.legend(loc="upper right", fontsize="small")

# ------------------------------------------------------------------
# 10) Train 4 RBF‑SVMs and plot them
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, len(gamma_vals), figsize=(4*len(gamma_vals), 4),
                         sharex=True, sharey=True)

for ax, γ in zip(axes, gamma_vals):
    rbf = SVC(C=C_fixed, kernel="rbf", gamma=γ, probability=False,
              random_state=42)
    rbf.fit(X_sub_2d, y_sub)

    sv_counts.append(rbf.n_support_.sum())   # total support vectors
    plot_boundary(ax, rbf, γ)

plt.suptitle("RBF‑SVM decision boundaries for increasing γ (fixed C=10)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ------------------------------------------------------------------
# 11) Print support‑vector counts (helps with bias/variance discussion)
# ------------------------------------------------------------------
print("\nSupport‑vector counts for each γ:")
for γ, cnt in zip(gamma_vals, sv_counts):
    print(f"γ = {γ:<5} → {cnt} support vectors")
