#!/usr/bin/env python3
# --------------------------------------------------------------
# 2‑D PCA projection → retrain best Decision Tree & SVMs
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 1) Load Adult data (same zip as before)
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
    except TypeError:                # newer scikit‑learn
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
# 4) Pre‑processing pipeline (same as in all previous steps)
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
# 6) Fit the preprocessing on the *training* part only
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
# 7) Subsample 10 k rows (exactly the same as in earlier notebooks)
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

print("\nSubsample class distribution (should resemble train set):")
print(y_sub.value_counts(normalize=True).round(3))

# ------------------------------------------------------------------
# 8) PCA → keep first 2 components
# ------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_sub_2d = pca.fit_transform(X_sub)
X_val_2d = pca.transform(X_val)          # validation points for plotting

explained = pca.explained_variance_ratio_
print("\nExplained variance by the first two PCs:")
print(f"PC1 : {explained[0]:.4f}  ({explained[0]*100:.2f} %)")
print(f"PC2 : {explained[1]:.4f}  ({explained[1]*100:.2f} %)")
print(f"Combined : {(explained[:2].sum()*100):.2f} %")

# ------------------------------------------------------------------
# 9) Re‑train the three best models on the 2‑D data
# ------------------------------------------------------------------
# Decision Tree – the hyper‑parameters that were best in the earlier grid
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=10,
                              min_samples_leaf=20,
                              random_state=42)
tree.fit(X_sub_2d, y_sub)

# Linear SVM – best C from the earlier linear‑SVM search (C=1.0)
lin_svm = LinearSVC(C=1.0, loss="hinge", dual=True, random_state=42)
lin_svm.fit(X_sub_2d, y_sub)

# RBF SVM – best C & gamma from the earlier RBF search (C=10, gamma=0.1)
rbf_svm = SVC(C=10,
              kernel='rbf',
              gamma=0.1,
              probability=False,
              random_state=42)
rbf_svm.fit(X_sub_2d, y_sub)

# ------------------------------------------------------------------
# 10) Helper – plot decision surface for a fitted classifier
# ------------------------------------------------------------------
def plot_boundary(ax, clf, title):
    # Define grid
    x_min, x_max = X_sub_2d[:, 0].min() - 1, X_sub_2d[:, 0].max() + 1
    y_min, y_max = X_sub_2d[:, 1].min() - 1, X_sub_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    # contourf for the decision regions
    cmap_bg = plt.cm.Pastel2
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)

    # validation points (true label)
    colors = ["steelblue", "darkorange"]
    markers = ["o", "s"]
    for cls, col, mk in zip([0, 1], colors, markers):
        idx = (y_val == cls)
        ax.scatter(X_val_2d[idx, 0],
                   X_val_2d[idx, 1],
                   c=col,
                   marker=mk,
                   edgecolor="k",
                   label=f"{'<=50K' if cls==0 else '>50K'} (val)",
                   s=30,
                   alpha=0.8)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")

# ------------------------------------------------------------------
# 11) Plot all three boundaries side‑by‑side
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)

plot_boundary(axes[0], tree,    "Decision Tree")
plot_boundary(axes[1], lin_svm, "Linear SVM")
plot_boundary(axes[2], rbf_svm, "RBF SVM")

plt.suptitle("Decision boundaries on the 2‑D PCA projection (validation points)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
