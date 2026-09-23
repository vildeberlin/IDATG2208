# Oppgave 1.7 : Analyse av standardisering av numeriske features
import zipfile, os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")    # slå av unødvendige warnings

# Last inn datasettet (samme preprocessing som i tidligere oppgaver)
zip_path   = "adult.zip"
extract_dir = "adult_data"

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

col_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

train_path = os.path.join(extract_dir, "adult.data")
df = pd.read_csv(
    train_path,
    header=None,
    names=col_names,
    na_values=["?", " ?"],          # både med og uten mellomrom → NaN
    skipinitialspace=True
)
df = df.replace('?', np.nan)       # ekstra sikkerhet

# Numeriske kolonner (vi behandler kun disse)
num_cols = ["age", "fnlwgt", "education-num",
            "capital-gain", "capital-loss", "hours-per-week"]
num_df = df[num_cols].copy()

# Log‑transform + IQR‑capping for kapital‑variablene
def iqr_cap(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)

for col in ["capital-gain", "capital-loss"]:
    # log1p for å håndtere mange 0‑verdier
    num_df[col] = np.log1p(num_df[col])
    num_df[col] = iqr_cap(num_df[col])

# Standardiser (zero‑mean, unit‑variance) – brukes av SVM‑modellene
scaler = StandardScaler()
num_std = pd.DataFrame(
    scaler.fit_transform(num_df),
    columns=[f"{c}_std" for c in num_cols],
    index=num_df.index
)

# Plotter – rå (log‑transformert + capping) vs. standardisert
sns.set(style="whitegrid", font_scale=1.2)

n_feat = len(num_cols)
fig, axes = plt.subplots(
    nrows=n_feat,
    ncols=2,
    figsize=(12, 3 * n_feat),
    sharex=False,
    sharey=False
)

for i, col in enumerate(num_cols):
    # ---- rå distribusjon ----
    ax_raw = axes[i, 0]
    sns.histplot(num_df[col], kde=True, stat="density",
                 bins=30, edgecolor="white", color="#4C72B0", ax=ax_raw)
    ax_raw.set_title(f"{col} (log‑transform + capping)")
    ax_raw.set_xlabel("")
    ax_raw.set_ylabel("Density")

    # ---- standardisert distribusjon ----
    ax_std = axes[i, 1]
    sns.histplot(num_std[f"{col}_std"], kde=True, stat="density",
                 bins=30, edgecolor="white", color="#DD8452", ax=ax_std)
    ax_std.set_title(f"{col} – standardisert")
    ax_std.set_xlabel("")
    ax_std.set_ylabel("")

fig.text(0.5, 0.04, "Verdi", ha="center", fontsize=14)
fig.text(0.04, 0.5, "Tetthet (density)", va="center", rotation="vertical", fontsize=14)
fig.suptitle("Fordeling av numeriske features – før og etter standardisering",
             fontsize=16, y=1.03)
plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
plt.show()

# Oppsummer kort statistikk (raw vs. standardisert)
# Bygg en dataframe av de fire serie‑objektene og reset indeksen
summary = pd.DataFrame({
    "mean_raw": num_df.mean(),
    "std_raw" : num_df.std(),
    "mean_std": num_std.mean(),
    "std_std" : num_std.std()
}).reset_index().rename(columns={"index": "feature"})

print("\n--- Sammenligning av summary‑statistikk (raw vs. standardisert) ---")
print(summary.round(3))