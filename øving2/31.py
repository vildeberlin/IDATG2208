
import os
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_curve, auc

warnings.filterwarnings("ignore")

# Load Adult dataset
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

# Hjelpefunksjon – log + IQR capping
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
    """OneHotEncoder that works for both <1.2 and ≥1.2 sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
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

# Trene/validere/test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_val_idx, test_idx = next(sss.split(df, df[TARGET]))
train_val = df.iloc[train_val_idx]
test_set  = df.iloc[test_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(sss2.split(train_val, train_val[TARGET]))
train_set = train_val.iloc[train_idx]
val_set   = train_val.iloc[val_idx]

# Fit preprocessing only on the training set
X_tr = train_set.drop(columns=TARGET)
y_tr = train_set[TARGET].map({">50K": 1, "<=50K": 0})

preprocess.fit(X_tr, y_tr)   # statistics from training data only

def transform_and_label(df_subset, name):
    X = df_subset.drop(columns=TARGET)
    y = df_subset[TARGET].map({">50K": 1, "<=50K": 0})
    X_t = preprocess.transform(X)
    print(f"{name:10s} → shape {X_t.shape}")
    return X_t, y

X_train, y_train = transform_and_label(train_set, "TRAIN")
X_val,   y_val   = transform_and_label(val_set,   "VALID")
X_test,  y_test  = transform_and_label(test_set,  "TEST")

# Stratified subsample (10 000) 
SUBSAMPLE_SIZE = 10_000
rng = np.random.default_rng(seed=42)

unique, counts = np.unique(y_train, return_counts=True)
prop = counts / counts.sum()
n_per_class = (prop * SUBSAMPLE_SIZE).astype(int)

sub_idx = []
for cls, n in zip(unique, n_per_class):
    cls_idx = np.where(y_train == cls)[0]
    sub_idx.extend(rng.choice(cls_idx, size=n, replace=False))

sub_idx = np.array(sub_idx)
X_sub = X_train[sub_idx]
y_sub = y_train.iloc[sub_idx]

print("\nSubsample class distribution (should resemble training set):")
print(y_sub.value_counts(normalize=True).round(3))

# 5‑fold stratified CV (samme split som tidligere)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accuracies = []
fold_aucs       = []

# Prepare a figure for the ROC curves
plt.figure(figsize=(8, 6))

print("\n=== Per‑fold results ===")
for fold, (train_idx, _) in enumerate(skf.split(X_sub, y_sub), start=1):
    X_fold_train = X_sub[train_idx]
    y_fold_train = y_sub.iloc[train_idx]

    svm = LinearSVC(
        C=1.0,
        loss="hinge",
        dual=True,
        random_state=42,
    )
    svm.fit(X_fold_train, y_fold_train)

    # ---- Validation performance ----
    y_pred  = svm.predict(X_val)
    acc     = accuracy_score(y_val, y_pred)
    scores  = svm.decision_function(X_val)          # continuous scores for ROC
    fpr, tpr, _ = roc_curve(y_val, scores)
    roc_auc = auc(fpr, tpr)

    fold_accuracies.append(acc)
    fold_aucs.append(roc_auc)

    print(f"Fold {fold:1d} – Accuracy: {acc:.4f} – AUC: {roc_auc:.4f}")

    # Plot ROC curve for this fold
    plt.plot(fpr, tpr, lw=1.5,
             label=f"Fold {fold} (AUC = {roc_auc:.3f})")

# Summary statistics (mean ± std)
mean_acc = np.mean(fold_accuracies)
std_acc  = np.std(fold_accuracies, ddof=1)

mean_auc = np.mean(fold_aucs)
std_auc  = np.std(fold_aucs, ddof=1)

print("\n=== Summary ===")
print(f"Accuracy  – mean ± std : {mean_acc:.4f} ± {std_acc:.4f}")
print(f"AUC       – mean ± std : {mean_auc:.4f} ± {std_auc:.4f}")

# ROC plot
plt.plot([0, 1], [0, 1], "k--", lw=1, label="No‑skill")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves (Linear SVM, 5‑fold CV) on validation set")
plt.legend(loc="lower right", fontsize="small")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
