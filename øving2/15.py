
import zipfile, os
import pandas as pd
import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt

# Les inn Adult‑datasettet (samme som i tildigere oppgaver)
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
df = df.replace('?', np.nan)      # ekstra sikkerhet

# Lager lister med numeriske og kategoriske kolonner
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "income"]   # fjern målvariabelen

print("\nNumeriske kolonner :", num_cols)
print("Kategoriske kolonner :", cat_cols)

# Cramér’s V (kategorisk ↔ kategorisk)
def cramers_v(confusion_matrix):
    """Returnerer Cramér’s V fra en χ²‑krysstabell."""
    chi2 = st.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    # bias‑adjusted versjon (vanlig i litteraturen)
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def cramers_v_pair(col1, col2):
    sub = df[[col1, col2]].dropna()
    cm = pd.crosstab(sub[col1], sub[col2])
    if cm.shape[0] == 1 or cm.shape[1] == 1:
        return 0.0
    return cramers_v(cm.values)

# ANOVA‑F‑verdi for numerisk ↔ kategorisk (normert til [0,1])
def anova_f_pair(num_col, cat_col):
    sub = df[[num_col, cat_col]].dropna()
    groups = [g[1].values for g in sub.groupby(cat_col)[num_col]]
    if len(groups) < 2:
        return 0.0
    f, _ = st.f_oneway(*groups)          # kan gi ConstantInputWarning → ignorér
    # normaliser slik at 0 ≤ f_norm ≤ 1
    return f / (f + 1.0)

# Bygg full association‑matrix (15 × 15)
all_features = df.columns.tolist()
assoc = pd.DataFrame(np.nan, index=all_features, columns=all_features)

# diagonal = 1.0 (perfekt korrelasjon med seg selv)
np.fill_diagonal(assoc.values, 1.0)

# numerisk ↔ numerisk → Pearson
for i, c1 in enumerate(num_cols):
    for j, c2 in enumerate(num_cols):
        if i >= j:
            continue
        r = df[[c1, c2]].corr().iloc[0, 1]
        assoc.loc[c1, c2] = r
        assoc.loc[c2, c1] = r

# kategorisk ↔ kategorisk → Cramér’s V
for i, c1 in enumerate(cat_cols):
    for j, c2 in enumerate(cat_cols):
        if i >= j:
            continue
        v = cramers_v_pair(c1, c2)
        assoc.loc[c1, c2] = v
        assoc.loc[c2, c1] = v

# numerisk ↔ kategorisk → ANOVA‑F (normert)
for ncol in num_cols:
    for ccol in cat_cols:
        f_norm = anova_f_pair(ncol, ccol)
        assoc.loc[ncol, ccol] = f_norm
        assoc.loc[ccol, ncol] = f_norm

# Plot heat‑map
plt.figure(figsize=(14, 12))
sns.heatmap(
    assoc.astype(float),
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Association (Pearson / Cramér’s V / normert F)"},
    vmin=-1, vmax=1
)
plt.title(
    "Feature‑association matrix (mixed types)\n"
    "(Pearson for numeric‑numeric, Cramér’s V for cat‑cat, "
    "normert ANOVA‑F for numeric‑cat)",
    fontsize=14
)
plt.tight_layout()
plt.show()

# Finn par med sterk sammenheng (|assoc| ≥ 0.7)
threshold = 0.7
strong_pairs = []

for i, row in assoc.iterrows():
    for j, val in row.items():          # ← ***Rettet fra .iteritems() til .items()***
        if i >= j:                      # kun én retning (symmetri)
            continue
        if abs(val) >= threshold:
            strong_pairs.append((i, j, val))

print("\nPar med sterk multikollinearitet (|association| ≥ 0.7):")
for a, b, v in strong_pairs:
    print(f"{a:15} ↔ {b:15} : {v:.2f}")

# -----------------------------------------------------------------
# 8) Tekst som kan kopieres direkte inn i rapporten
# -----------------------------------------------------------------
explanation = f"""
**Multikollinearitet i Adult‑datasettet**

Heat‑map‑diagrammet bruker tre assosiasjons‑mål:
* **Pearson‑r** for numerisk‑numerisk,
* **Cramér’s V** for kategorisk‑kategorisk,
* **Normert ANOVA‑F** for numerisk‑kategorisk (skalert til [0,1]).

Paret som overskrider grensen **|association| ≥ {threshold:.1f}** er:

| Feature 1            | Feature 2            | Association |
|---------------------|---------------------|-------------|
"""
for a, b, v in strong_pairs:
    explanation += f"| {a:20} | {b:20} | {v: .2f} |\n"

explanation += """
**Hvorfor er dette viktig for lineær SVM?**  
En lineær SVM lærer en vekt‑vektor **w** som maksimerer marginen. Når to
funksjoner er nesten lineært avhengige, blir Gram‑matrisen (X·Xᵀ) nesten
singulær → numerisk ustabilitet og store koeffisienter som kan kansellere
hverandre. Modellen kan dermed bli svært sensitiv for støy og kan
over‑tilpasse på redundante variabler.

**Hvorfor er multikollinearitet mindre kritisk for Decision Tree?**  
Et tre splitter på *én* variabel per node; den vurderer ikke en lineær kombinasjon
av flere funksjoner. Selv om to variabler gir samme informasjon, vil treet bare
velge den ene (eller bruke dem i ulike under‑trær) – ingen regresjons‑koeffisienter
må estimeres, så lineær avhengighet skaper ingen numerisk problematikk. Treet
kan dermed håndtere høy korrelasjon uten å miste stabilitet, selv om det kan
resultere i litt større tre‑størrelse.

Derfor bør man for **lineær SVM** vurdere å fjerne eller kombinere de
sterkt korrelerte variablene (f.eks. `education‑num` vs. `education`,
`age` vs. `hours‑per‑week`). For **Decision Tree** er dette kun en
pragmatisk optimalisering (redusere antall split‑kandidater), men det er
ikke nødvendig for å oppnå stabil modell‑adferd.
"""
print("\n--- Tekst du kan kopiere til rapporten -----------------------------------")
print(explanation)
