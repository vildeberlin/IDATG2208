
import matplotlib.pyplot as plt

import os
import zipfile
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# Load dataset
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

# Hjelpefunksjon – log + IQR capping (same as before)
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
    except TypeError:            # newer scikit‑learn
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

# Trene/validering/teste split 
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_val_idx, test_idx = next(sss.split(df, df[TARGET]))
train_val = df.iloc[train_val_idx]
test_set  = df.iloc[test_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(sss2.split(train_val, train_val[TARGET]))
train_set = train_val.iloc[train_idx]
val_set   = train_val.iloc[val_idx]

# Fit preprocessing on the training set only
X_tr = train_set.drop(columns=TARGET)
y_tr = train_set[TARGET].map({">50K": 1, "<=50K": 0})
preprocess.fit(X_tr, y_tr)

def transform_and_label(df_subset, name):
    X = df_subset.drop(columns=TARGET)
    y = df_subset[TARGET].map({">50K": 1, "<=50K": 0})
    X_t = preprocess.transform(X)
    print(f"{name:10s} → shape {X_t.shape}")
    return X_t, y

X_train, y_train = transform_and_label(train_set, "TRAIN")
X_val,   y_val   = transform_and_label(val_set,   "VALID")
X_test,  y_test  = transform_and_label(test_set,  "TEST")

# Subsample – 5 000 rows 
SUBSAMPLE_SIZE = 5_000         
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

# 5‑fold CV – same folds as before
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# metric containers
lin_acc, lin_f1, lin_auc = [], [], []
rbf_acc, rbf_f1, rbf_auc = [], [], []

# ROC figure (both models together)
plt.figure(figsize=(8, 6))

fold_no = 1
for train_idx, _ in skf.split(X_sub, y_sub):
    X_fold_train = X_sub[train_idx]
    y_fold_train = y_sub.iloc[train_idx]

    # ------------------ Linear SVM ------------------
    lin = LinearSVC(C=1.0, loss="hinge", dual=True, random_state=42)
    lin.fit(X_fold_train, y_fold_train)

    lin_pred = lin.predict(X_val)
    lin_acc.append(accuracy_score(y_val, lin_pred))
    lin_f1.append(f1_score(y_val, lin_pred))
    lin_score = lin.decision_function(X_val)
    fpr, tpr, _ = roc_curve(y_val, lin_score)
    lin_auc.append(auc(fpr, tpr))
    plt.plot(fpr, tpr, lw=1.2, color="C0", alpha=0.6,
             label=f"Linear  Fold {fold_no}" if fold_no == 1 else None)

    # ------------------ RBF SVM --------------------
    # Note: probability=False → we use decision_function for ROC
    rbf = SVC(C=1.0,
              kernel="rbf",
              gamma="scale",
              probability=False,      # saves a lot of time
              max_iter=2000,          # stops early if not converged
              random_state=42)
    rbf.fit(X_fold_train, y_fold_train)

    rbf_pred = rbf.predict(X_val)
    rbf_acc.append(accuracy_score(y_val, rbf_pred))
    rbf_f1.append(f1_score(y_val, rbf_pred))
    rbf_score = rbf.decision_function(X_val)
    fpr, tpr, _ = roc_curve(y_val, rbf_score)
    rbf_auc.append(auc(fpr, tpr))
    plt.plot(fpr, tpr, lw=1.2, color="C1", ls="--", alpha=0.6,
             label=f"RBF  Fold {fold_no}" if fold_no == 1 else None)

    fold_no += 1

# Summarise metrics
def mean_std(arr):
    return np.mean(arr), np.std(arr, ddof=1)

lin_acc_mu, lin_acc_sd = mean_std(lin_acc)
lin_f1_mu,  lin_f1_sd  = mean_std(lin_f1)
lin_auc_mu, lin_auc_sd  = mean_std(lin_auc)

rbf_acc_mu, rbf_acc_sd = mean_std(rbf_acc)
rbf_f1_mu,  rbf_f1_sd  = mean_std(rbf_f1)
rbf_auc_mu, rbf_auc_sd  = mean_std(rbf_auc)

print("\n=== Linear SVM (kernel='linear') ===")
print(f"Accuracy : {lin_acc_mu:.4f} ± {lin_acc_sd:.4f}")
print(f"F1‑score : {lin_f1_mu:.4f}  ± {lin_f1_sd:.4f}")
print(f"ROC‑AUC  : {lin_auc_mu:.4f} ± {lin_auc_sd:.4f}")

print("\n=== RBF SVM (kernel='rbf') ===")
print(f"Accuracy : {rbf_acc_mu:.4f} ± {rbf_acc_sd:.4f}")
print(f"F1‑score : {rbf_f1_mu:.4f}  ± {rbf_f1_sd:.4f}")
print(f"ROC‑AUC  : {rbf_auc_mu:.4f} ± {rbf_auc_sd:.4f}")

# ROC plot
plt.plot([0, 1], [0, 1], "k--", lw=1, label="No‑skill")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC – Linear vs. RBF SVM (validation set, 5‑fold CV)")
plt.legend(loc="lower right", fontsize="small")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
