import pandas as pd
import matplotlib.pyplot as plt
import zipfile, os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report
)

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

# Remove missing values

df = df.dropna()

# Split features and target

X = df.drop("income", axis=1)

y = (df["income"] == ">50K").astype(int)

# Train / Validation split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocessing

numeric_features = X.select_dtypes(include=["int64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Decision Tree GridSearchCV

tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

tree_params = {
    "classifier__max_depth": [3, 5, 7, 10, 15],
    "classifier__min_samples_leaf": [1, 5, 10, 20],
    "classifier__ccp_alpha": [0.0, 0.001, 0.01]
}

tree_grid = GridSearchCV(
    tree_pipeline,
    tree_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

tree_grid.fit(X_train, y_train)

print("\n===== BEST DECISION TREE =====")
print(tree_grid.best_params_)
print("CV F1-score:", tree_grid.best_score_)

# SVM GridSearchCV

svm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(class_weight="balanced"))
])

svm_params = {
    "classifier__kernel": ["linear", "rbf"],
    "classifier__C": [0.1, 1, 10]
}

svm_grid = GridSearchCV(
    svm_pipeline,
    svm_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("\n===== BEST SVM =====")
print(svm_grid.best_params_)
print("CV F1-score:", svm_grid.best_score_)

# Validation evaluation

best_tree = tree_grid.best_estimator_
best_svm = svm_grid.best_estimator_

tree_pred = best_tree.predict(X_val)
svm_pred = best_svm.predict(X_val)

tree_f1 = f1_score(y_val, tree_pred)
svm_f1 = f1_score(y_val, svm_pred)

tree_bal_acc = balanced_accuracy_score(y_val, tree_pred)
svm_bal_acc = balanced_accuracy_score(y_val, svm_pred)

print("\n===== VALIDATION RESULTS =====")

print("\nDecision Tree")
print("F1-score:", round(tree_f1, 4))
print("Balanced Accuracy:", round(tree_bal_acc, 4))
print(classification_report(y_val, tree_pred))

print("\nSVM")
print("F1-score:", round(svm_f1, 4))
print("Balanced Accuracy:", round(svm_bal_acc, 4))
print(classification_report(y_val, svm_pred))

# Performance plot

plt.figure(figsize=(6,4))

models = ["Decision Tree", "SVM"]
scores = [tree_f1, svm_f1]

plt.bar(models, scores)

plt.ylabel("Validation F1-score")
plt.title("Best Tuned Models")

plt.tight_layout()
plt.show()

# Final choice

if svm_f1 > tree_f1:
    print("\nFINAL MODEL: SVM")
else:
    print("\nFINAL MODEL: Decision Tree")