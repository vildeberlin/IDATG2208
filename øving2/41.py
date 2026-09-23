
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
)

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
    except TypeError:                # newer sklearn
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

# Fit preprocessing on training part only
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

# Subsample (10 k) – keep class balance
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

# Models to compare
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Linear SVM":    LinearSVC(C=1.0, loss="hinge", dual=True, random_state=42),
    "RBF SVM":       SVC(C=1.0,
                        kernel="rbf",
                        gamma="scale",
                        probability=True,      # needed for ROC on validation set
                        random_state=42)
}

# 5‑fold CV on the subsample – collect metrics
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_metrics = {}   # model → dict of mean/std values

for name, clf in models.items():
    accs, f1s, aucs = [], [], []
    for tr_idx, te_idx in skf.split(X_sub, y_sub):
        X_tr_cv, y_tr_cv = X_sub[tr_idx], y_sub.iloc[tr_idx]
        X_te_cv, y_te_cv = X_sub[te_idx], y_sub.iloc[te_idx]

        clf.fit(X_tr_cv, y_tr_cv)

        # predictions / scores
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_te_cv)[:, 1]
        else:                                   # LinearSVC → decision_function
            scores = clf.decision_function(X_te_cv)

        pred = (scores >= 0.5).astype(int) if scores.ndim == 1 else clf.predict(X_te_cv)

        accs.append(accuracy_score(y_te_cv, pred))
        f1s.append(f1_score(y_te_cv, pred))
        fpr, tpr, _ = roc_curve(y_te_cv, scores)
        aucs.append(auc(fpr, tpr))

    cv_metrics[name] = {
        "acc_mean": np.mean(accs), "acc_std": np.std(accs, ddof=1),
        "f1_mean" : np.mean(f1s),  "f1_std" : np.std(f1s, ddof=1),
        "auc_mean": np.mean(aucs), "auc_std": np.std(aucs, ddof=1)
    }

# Validation‑set evaluation
val_metrics = {}

for name, clf in models.items():
    clf.fit(X_sub, y_sub)                # train on the whole subsample
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_val)[:, 1]
    else:
        scores = clf.decision_function(X_val)

    pred = (scores >= 0.5).astype(int) if scores.ndim == 1 else clf.predict(X_val)

    fpr, tpr, _ = roc_curve(y_val, scores)
    val_metrics[name] = {
        "accuracy": accuracy_score(y_val, pred),
        "f1"      : f1_score(y_val, pred),
        "auc"     : auc(fpr, tpr),
        "fpr"     : fpr,
        "tpr"     : tpr
    }

# Print concise numeric summary 
print("\n=== CROSS‑VALIDATION (training subsample) ===")
for name, m in cv_metrics.items():
    print(f"\n{name}")
    print(f"  Accuracy : {m['acc_mean']:.4f} ± {m['acc_std']:.4f}")
    print(f"  F1‑score : {m['f1_mean']:.4f}  ± {m['f1_std']:.4f}")
    print(f"  ROC‑AUC  : {m['auc_mean']:.4f} ± {m['auc_std']:.4f}")

print("\n=== VALIDATION SET ===")
for name, m in val_metrics.items():
    print(f"\n{name}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  F1‑score : {m['f1']:.4f}")
    print(f"  ROC‑AUC  : {m['auc']:.4f}")

# Grouped bar chart – one subplot per metric 
metrics_display = ["Accuracy", "F1‑score", "ROC‑AUC"]
# mapping from display name → key used in cv_metrics / val_metrics
cv_key_map   = {"Accuracy": "acc_mean", "F1‑score": "f1_mean", "ROC‑AUC": "auc_mean"}
val_key_map  = {"Accuracy": "accuracy", "F1‑score": "f1",      "ROC‑AUC": "auc"}

model_names = list(models.keys())
n_models    = len(model_names)
indices     = np.arange(n_models)          # x‑positions for the models
bar_width   = 0.35

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, metric in zip(axs, metrics_display):
    # CV‑mean values
    cv_vals = [cv_metrics[m][cv_key_map[metric]] for m in model_names]
    # Validation‑set single values
    val_vals = [val_metrics[m][val_key_map[metric]] for m in model_names]

    # draw bars
    ax.bar(indices - bar_width/2, cv_vals,
           width=bar_width, label="CV‑mean", color="C0", edgecolor="black")
    ax.bar(indices + bar_width/2, val_vals,
           width=bar_width, label="Validation", color="C1", edgecolor="black")

    ax.set_xticks(indices)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_title(metric)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis='y')
    if ax is axs[0]:
        ax.set_ylabel("Score")
    ax.legend()

plt.tight_layout()
plt.show()

# ROC curves on the validation set 
plt.figure(figsize=(7, 6))
for name, m in val_metrics.items():
    plt.plot(m["fpr"], m["tpr"],
             lw=2,
             label=f"{name} (AUC = {m['auc']:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="No‑skill")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves (validation set)")
plt.legend(loc="lower right", fontsize="small")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
