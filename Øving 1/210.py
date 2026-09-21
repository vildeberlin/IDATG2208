import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import csv

csv_name = "train.csv"
# ---
# Load-e dataen
df = pd.read_csv(csv_name)

X = df.drop(columns=["critical_temp"]).values.astype(np.float64)   # (n,81)
y = df["critical_temp"].values.astype(np.float64)                # (n,)

feature_names = list(df.columns.drop("critical_temp"))

# ---
# 5‑fold split
rng = np.random.default_rng(seed=2024)
indices = np.arange(len(df))
rng.shuffle(indices)

n = len(df)
fold_size = int(0.2 * n)   # 20 % test per fold

# Containers for importance vectors
importances_per_fold = []   # list of (81,) arrays
# For optional permutation importance we will stack all test sets
X_test_all = []
y_test_all = []

for fold in range(5):
    start = fold * fold_size
    end   = start + fold_size
    test_idx  = indices[start:end]
    train_idx = np.setdiff1d(indices, test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test , y_test  = X[test_idx ], y[test_idx ]

    # ---
    # Random Forest – fit on the training data
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Built‑in impurity‑based importance (mean decrease in impurity)
    importances_per_fold.append(rf.feature_importances_)

    # Store test data for the optional permutation‑importance step
    X_test_all.append(X_test)
    y_test_all.append(y_test)

# ---
# Aggregere built‑in importances (mean ± std across folds)
importances_arr = np.stack(importances_per_fold)          # shape (5,81)
mean_imp   = importances_arr.mean(axis=0)
std_imp    = importances_arr.std(axis=0, ddof=1)

# Rank features by mean importance
rank_idx = np.argsort(mean_imp)[::-1]    # descending order

print("\n=== Top‑20 features (Random‑Forest impurity importance) ===")
print(f"{'Rank':<5} {'Feature':<30} {'MeanImp':>10} {'StdImp':>10}")
print("-" * 60)
for r, idx in enumerate(rank_idx[:20], start=1):
    print(f"{r:<5} {feature_names[idx]:<30} {mean_imp[idx]:10.5f} {std_imp[idx]:10.5f}")
