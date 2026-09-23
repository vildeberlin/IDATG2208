
import matplotlib.pyplot as plt
import seaborn as sns

import os, zipfile, warnings
import numpy as np, pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    average_precision_score,
    make_scorer,
)

warnings.filterwarnings("ignore")

# Load the data
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

# Helper – log + IQR capping 
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

# Column groups
NUMERIC = ["age","fnlwgt","education-num",
           "capital-gain","capital-loss","hours-per-week"]
CATEG   = ["workclass","education","marital-status","occupation",
           "relationship","race","sex","native-country"]
TARGET  = "income"

# Pre‑processing pipeline (no data leakage)
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

# Train/validation/test split 
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_val_idx, test_idx = next(sss.split(df, df[TARGET]))
train_val = df.iloc[train_val_idx]
test_set  = df.iloc[test_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(sss2.split(train_val, train_val[TARGET]))
train_set = train_val.iloc[train_idx]
val_set   = train_val.iloc[val_idx]

# Fit preprocessing on the training part only
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
X_test,  y_test  = transform(test_set,  "TEST")

# 10 k stratified subsample 
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

# Scoring for the search – PR‑AUC (average_precision)
scorer = make_scorer(average_precision_score, needs_proba=False)

# Decision‑Tree grid search
tree_clf = DecisionTreeClassifier(random_state=42)

tree_grid = {
    "max_depth":          [None, 5, 10, 15],
    "min_samples_leaf":  [1, 5, 10, 20],
    "criterion":         ["gini", "entropy"]
}

tree_search = GridSearchCV(
    estimator=tree_clf,
    param_grid=tree_grid,
    scoring=scorer,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # 3‑fold → faster
    n_jobs=1,                                 # single core → easy to watch
    verbose=0,
    refit=True
)

tree_search.fit(X_sub, y_sub)

print("\n=== Decision Tree (grid) ===")
print("Best hyper‑parameters :", tree_search.best_params_)
print(f"Best CV avg‑precision : {tree_search.best_score_:.4f}")

best_tree = tree_search.best_estimator_

# SVM – randomized search 
svm_clf = SVC(probability=True, random_state=42)   # probability needed for AP

svm_param_dist = {
    "kernel": ["linear", "rbf"],
    "C":      [0.1, 1, 10],
    # `gamma` is ignored for the linear kernel – fine to pass it anyway
    "gamma":  ["scale", 0.01, 0.1, 1]
}

svm_search = RandomizedSearchCV(
    estimator=svm_clf,
    param_distributions=svm_param_dist,
    n_iter=15,                                 # 15 random combos
    scoring=scorer,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=1,
    verbose=0,
    random_state=42,
    refit=True
)

svm_search.fit(X_sub, y_sub)

print("\n=== SVM (randomized) ===")
print("Best hyper‑parameters :", svm_search.best_params_)
print(f"Best CV avg‑precision : {svm_search.best_score_:.4f}")

best_svm = svm_search.best_estimator_

# Evaluate both tuned models on the test set
def evaluate(model, name):
    prob = model.predict_proba(X_test)[:, 1]            # probability of class 1
    pred = (prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, pred)
    f1   = f1_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec  = recall_score(y_test, pred)
    roc  = auc(*roc_curve(y_test, prob)[:2])
    ap   = average_precision_score(y_test, prob)

    print(f"\n=== {name} – TEST SET ===")
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1‑score   : {f1:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"ROC‑AUC    : {roc:.4f}")
    print(f"Avg‑Precision (PR‑AUC) : {ap:.4f}")

    return {"fpr": roc_curve(y_test, prob)[0],
            "tpr": roc_curve(y_test, prob)[1],
            "auc": roc,
            "ap" : ap,
            "name": name}

tree_test = evaluate(best_tree, "Decision Tree")
svm_test  = evaluate(best_svm , "SVM")

# ROC overlay
plt.figure(figsize=(7, 6))
plt.plot(tree_test["fpr"], tree_test["tpr"],
         lw=2, label=f"Decision Tree (AUC = {tree_test['auc']:.3f})")
plt.plot(svm_test["fpr"], svm_test["tpr"],
         lw=2, label=f"SVM ({best_svm.kernel}) (AUC = {svm_test['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="No‑skill")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Test‑set ROC – tuned Decision Tree vs. tuned SVM")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
