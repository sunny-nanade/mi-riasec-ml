"""
run_analysis.py
---------------
Full reproducible analysis pipeline for the paper:

  "Multiple Intelligences as Predictors of RIASEC Vocational Domains:
   A Machine Learning and Explainable AI Framework for
   Psychometric Career Counseling"

Requirements: see requirements.txt
Input  : data/anonymized_data.csv
Output : results/  (figures, JSON, CSV)

Usage
-----
From the repository root:
    python src/run_analysis.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, confusion_matrix, classification_report, accuracy_score
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import shap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "anonymized_data.csv")
OUT_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
MI_COLS = [
    "Verbal Linguistic Raw Score",
    "Logical-Mathematical Raw Score",
    "Spatial-Visual Raw Score",
    "Bodily-Kinesthetic Raw Score",
    "Musical Raw Score",
    "Interpersonal Raw Score",
    "Intrapersonal Raw Score",
    "Naturalist Raw Score",
]
MI_SHORT     = ["VL", "LM", "SV", "BK", "Mus", "Inter", "Intra", "Nat"]
RIASEC_COLS  = [
    "REALISTIC Score", "INVESTIGATIVE Score", "ARTISTIC Score",
    "SOCIAL Score", "ENTERPRISING Score", "CONVENTIONAL Score",
]
RIASEC_SHORT = ["R", "I", "A", "S", "E", "C"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} records, {df.shape[1]} columns.")
assert df[MI_COLS].isnull().sum().sum() == 0, "Missing MI values"
assert df[RIASEC_COLS].isnull().sum().sum() == 0, "Missing RIASEC values"

# ---------------------------------------------------------------------------
# 1. Verify RIASEC formulas
# ---------------------------------------------------------------------------
lm    = df["Logical-Mathematical Raw Score"]
nat   = df["Naturalist Raw Score"]
inter = df["Interpersonal Raw Score"]
bk    = df["Bodily-Kinesthetic Raw Score"]
sv    = df["Spatial-Visual Raw Score"]
intra = df["Intrapersonal Raw Score"]

verified = {
    "REALISTIC Score":     (bk  + nat)             / 80  * 10,
    "INVESTIGATIVE Score": (bk  + intra + lm + sv) / 160 * 10,
    "ARTISTIC Score":      (sv  + intra + nat)      / 120 * 10,
    "SOCIAL Score":        (nat + inter)             / 80  * 10,
    "ENTERPRISING Score":  (bk  + lm + inter)       / 120 * 10,
    "CONVENTIONAL Score":  (intra + lm)             / 80  * 10,
}
print("\nFormula verification (max absolute error):")
for col, formula in verified.items():
    err = (df[col] - formula).abs().max()
    status = "PASS" if err < 0.01 else "FAIL"
    print(f"  {col}: {err:.4f}  [{status}]")

# ---------------------------------------------------------------------------
# 2. Target variable
# ---------------------------------------------------------------------------
df["Top_RIASEC"] = df[RIASEC_COLS].idxmax(axis=1).str.split().str[0]
X = df[MI_COLS].values
y = df["Top_RIASEC"].values
n_classes = len(np.unique(y))

print(f"\nClass distribution:\n{df['Top_RIASEC'].value_counts().to_string()}")

# ---------------------------------------------------------------------------
# 3. Descriptive statistics figures
# ---------------------------------------------------------------------------
# MI boxplot
fig, ax = plt.subplots(figsize=(11, 6))
mi_df = df[MI_COLS].copy()
mi_df.columns = MI_SHORT
bp = ax.boxplot(
    [mi_df[c].dropna() for c in MI_SHORT],
    labels=MI_SHORT, patch_artist=True,
    medianprops=dict(color="black", linewidth=2.5),
)
palette = ["#E8F5E9","#E3F2FD","#FFF3E0","#FCE4EC",
           "#EDE7F6","#E0F7FA","#FFFDE7","#F3E5F5"]
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color); patch.set_edgecolor("#333333")
ax.set_title("Distribution of Multiple Intelligence Raw Scores (N = 107)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Raw Score (10-40 scale)"); ax.set_xlabel("Intelligence Dimension")
ax.axhline(y=25, color="gray", linestyle="--", alpha=0.4, label="Midpoint (25)")
ax.set_ylim(8, 44); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mi_distribution_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()

# RIASEC class distribution
fig, ax = plt.subplots(figsize=(8, 5))
order = ["ENTERPRISING","REALISTIC","CONVENTIONAL","SOCIAL","INVESTIGATIVE","ARTISTIC"]
counts = df["Top_RIASEC"].value_counts().reindex(order)
bars = ax.bar(counts.index, counts.values,
              color=["#1565C0","#2E7D32","#6A1B9A","#AD1457","#E65100","#00695C"],
              edgecolor="black", linewidth=0.6)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{v}\n({v/107*100:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("Distribution of Dominant RIASEC Types (N = 107)", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Students"); ax.set_xlabel("Dominant RIASEC Domain")
ax.set_ylim(0, 42)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/riasec_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# MI correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
mi_corr = df[MI_COLS].corr(); mi_corr.index = MI_SHORT; mi_corr.columns = MI_SHORT
mask = np.triu(np.ones_like(mi_corr, dtype=bool), k=1)
sns.heatmap(mi_corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
            vmin=-1, vmax=1, mask=mask, ax=ax, linewidths=0.5,
            cbar_kws={"label": "Pearson r"}, annot_kws={"size": 10})
ax.set_title("Pearson Correlation Matrix of MI Raw Scores", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mi_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# MI x RIASEC cross-correlation
fig, ax = plt.subplots(figsize=(8, 7))
xc = pd.DataFrame(
    [[df[MC].corr(df[RC]) for RC in RIASEC_COLS] for MC in MI_COLS],
    index=MI_SHORT, columns=RIASEC_SHORT, dtype=float,
)
sns.heatmap(xc, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
            vmin=-0.2, vmax=1.0, ax=ax, linewidths=0.5,
            cbar_kws={"label": "Pearson r"}, annot_kws={"size": 10})
ax.set_title("Correlation Between MI Scores and RIASEC Domains",
             fontsize=13, fontweight="bold")
ax.set_ylabel("MI Dimension"); ax.set_xlabel("RIASEC Domain")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mi_riasec_cross_correlation.png", dpi=300, bbox_inches="tight")
plt.close()

# Save descriptive stats
mi_stats = df[MI_COLS].describe().T
mi_stats.index = MI_SHORT
mi_stats["skew"]     = df[MI_COLS].skew().values
mi_stats["kurtosis"] = df[MI_COLS].kurtosis().values
mi_stats.round(2).to_csv(f"{OUT_DIR}/mi_descriptive_stats.csv")

ri_stats = df[RIASEC_COLS].describe().T
ri_stats.index = RIASEC_SHORT
ri_stats["skew"]     = df[RIASEC_COLS].skew().values
ri_stats["kurtosis"] = df[RIASEC_COLS].kurtosis().values
ri_stats.round(2).to_csv(f"{OUT_DIR}/riasec_descriptive_stats.csv")

print("\nDescriptive figures saved.")

# ---------------------------------------------------------------------------
# 4. Clustering
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow + silhouette
inertias, silhouettes = [], []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lb = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, lb))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(range(2, 8), inertias, "bo-", linewidth=2.5, markersize=8)
ax1.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="Selected k=3")
ax1.set_xlabel("Number of Clusters (k)"); ax1.set_ylabel("Inertia (WCSS)")
ax1.set_title("Elbow Method", fontsize=13, fontweight="bold"); ax1.legend()
ax2.plot(range(2, 8), silhouettes, "go-", linewidth=2.5, markersize=8)
ax2.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="Selected k=3")
ax2.set_xlabel("Number of Clusters (k)"); ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Analysis", fontsize=13, fontweight="bold"); ax2.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/elbow_silhouette.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSilhouette scores: { {k+2: round(s, 3) for k, s in enumerate(silhouettes)} }")

# K-Means k=3
km3   = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = km3.fit_predict(X_scaled)
sil_k3 = silhouette_score(X_scaled, df["Cluster"])
print(f"Silhouette k=3: {sil_k3:.3f}")

cm_mean = df.groupby("Cluster")[MI_COLS].mean()
ctot    = cm_mean.sum(axis=1).sort_values(ascending=False)
names   = {
    ctot.index[0]: "High Cognitive Engagement",
    ctot.index[1]: "Moderate Cognitive Engagement",
    ctot.index[2]: "Developing Cognitive Engagement",
}
df["Archetype"] = df["Cluster"].map(names)
for nm, n in df["Archetype"].value_counts().items():
    print(f"  {nm}: n={n}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]; df["PCA2"] = X_pca[:, 1]
pv = pca.explained_variance_ratio_ * 100
print(f"PCA variance: PC1={pv[0]:.1f}%, PC2={pv[1]:.1f}%")

clr = {
    "High Cognitive Engagement":       "#2196F3",
    "Moderate Cognitive Engagement":   "#FF9800",
    "Developing Cognitive Engagement": "#4CAF50",
}
fig, ax = plt.subplots(figsize=(9, 7))
for arch, color in clr.items():
    m = df["Archetype"] == arch
    ax.scatter(df.loc[m, "PCA1"], df.loc[m, "PCA2"],
               c=color, label=f"{arch} (n={m.sum()})",
               alpha=0.75, s=60, edgecolors="white", linewidth=0.5)
ax.set_xlabel(f"PC 1 ({pv[0]:.1f}% variance)")
ax.set_ylabel(f"PC 2 ({pv[1]:.1f}% variance)")
ax.set_title(f"Cognitive Archetypes via K-Means (k=3, Silhouette={sil_k3:.3f})",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/clustering_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# Radar chart
angles = np.linspace(0, 2 * np.pi, 8, endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for arch, color in clr.items():
    cid  = [k for k, v in names.items() if v == arch][0]
    vals = cm_mean.loc[cid].values.tolist(); vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=2, marker="o", label=arch)
    ax.fill(angles, vals, color=color, alpha=0.12)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(MI_SHORT, size=11)
ax.set_rlabel_position(30); ax.yaxis.set_tick_params(labelsize=9)
ax.set_title("Cluster Profiles: Mean MI Scores\nper Cognitive Archetype",
             fontsize=12, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.15), fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cluster_profiles_radar.png", dpi=300, bbox_inches="tight")
plt.close()

df.to_csv(f"{OUT_DIR}/clustered_data.csv", index=False)
print("Clustering figures saved.")

# ---------------------------------------------------------------------------
# 5. Supervised classification
# ---------------------------------------------------------------------------
print("\n5-fold cross-validation results:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_s = scaler.fit_transform(X)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial"),
    "SVM (RBF)":           SVC(kernel="rbf", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
}
results = {}
for name, model in models.items():
    Xu   = X_s if name in ("Logistic Regression", "SVM (RBF)") else X
    accs = cross_val_score(model, Xu, y, cv=cv, scoring="accuracy")
    f1s  = cross_val_score(model, Xu, y, cv=cv, scoring="f1_weighted")
    results[name] = {
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
        "f1_mean":       float(f1s.mean()),  "f1_std":       float(f1s.std()),
        "accuracy_folds": accs.tolist(),
    }
    print(f"  {name}: Acc={accs.mean():.3f}+/-{accs.std():.3f}  "
          f"F1={f1s.mean():.3f}+/-{f1s.std():.3f}")

best = max(results, key=lambda k: results[k]["accuracy_mean"])
print(f"\nBest model: {best} ({results[best]['accuracy_mean']:.3f})")

# LR CV predictions for confusion matrix
lr_cv_pred = cross_val_predict(
    LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial"),
    X_s, y, cv=cv,
)

# LR hold-out
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                            random_state=42, stratify=y)
lr_ho = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial")
lr_ho.fit(scaler.fit_transform(X_tr), y_tr)
y_pred_ho = lr_ho.predict(scaler.transform(X_te))
ho_acc = accuracy_score(y_te, y_pred_ho)
print(f"LR hold-out accuracy (n={len(y_te)}): {ho_acc:.3f}")

# Model comparison chart (sorted by accuracy)
sorted_m = sorted(results.items(), key=lambda x: -x[1]["accuracy_mean"])
names_m = [m[0] for m in sorted_m]
accs_m  = [m[1]["accuracy_mean"] for m in sorted_m]
stds_m  = [m[1]["accuracy_std"]  for m in sorted_m]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(names_m, accs_m, yerr=stds_m, capsize=6,
              color=["#1565C0","#2E7D32","#E65100","#6A1B9A"],
              edgecolor="black", linewidth=0.6, alpha=0.88)
ax.axhline(y=1/n_classes, color="red", linestyle="--", linewidth=1.8,
           label=f"Random Baseline ({1/n_classes:.1%})")
ax.axhline(y=33/107, color="darkorange", linestyle=":", linewidth=1.8,
           label="Majority-Class Baseline (30.8%)")
for bar, acc, std in zip(bars, accs_m, stds_m):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+0.015,
            f"{acc:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_ylabel("Accuracy (5-Fold CV)", fontsize=12)
ax.set_title("Model Comparison: Predicting RIASEC Domain from MI Scores\n"
             "(Stratified 5-Fold Cross-Validation, sorted by accuracy)",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05); ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# Confusion matrices (LR — best model)
labels = sorted(np.unique(y))
lr_cv_acc = results["Logistic Regression"]["accuracy_mean"]
cm_ho = confusion_matrix(y_te, y_pred_ho, labels=labels)
cm_cv = confusion_matrix(y, lr_cv_pred, labels=labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm_ho, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax1, linewidths=0.5)
ax1.set_title(f"Hold-out Test Set (n=27, Acc={ho_acc:.1%})", fontsize=12, fontweight="bold")
ax1.set_ylabel("Actual"); ax1.set_xlabel("Predicted")
sns.heatmap(cm_cv, annot=True, fmt="d", cmap="Greens",
            xticklabels=labels, yticklabels=labels, ax=ax2, linewidths=0.5)
ax2.set_title(f"5-Fold CV (N=107, Acc={lr_cv_acc:.1%})", fontsize=12, fontweight="bold")
ax2.set_ylabel("Actual"); ax2.set_xlabel("Predicted")
plt.suptitle("Logistic Regression: RIASEC Domain from MI Scores",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Classification report
lr_full = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial")
with open(f"{OUT_DIR}/classification_report.txt", "w") as f:
    f.write("Logistic Regression Classification Report\n")
    f.write("Predicting Dominant RIASEC Domain from 8 MI Raw Scores\n\n")
    f.write(f"5-Fold CV Accuracy: {lr_cv_acc:.3f} +/- {results['Logistic Regression']['accuracy_std']:.3f}\n")
    f.write(f"Hold-out Accuracy (25% split, n=27): {ho_acc:.3f}\n")
    f.write(f"Random Baseline: {1/n_classes:.3f}\n")
    f.write(f"Majority-Class Baseline: {33/107:.3f}\n\n")
    f.write("--- 5-Fold CV Report ---\n")
    f.write(classification_report(y, lr_cv_pred, zero_division=0))

print("Classification figures saved.")

# ---------------------------------------------------------------------------
# 6. SHAP (Random Forest)
# ---------------------------------------------------------------------------
print("\nComputing SHAP values (Random Forest)...")
rf_full = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_full.fit(X, y)
explainer   = shap.TreeExplainer(rf_full)
shap_values = explainer.shap_values(X)
class_list  = list(rf_full.classes_)

if isinstance(shap_values, list):
    shap_mean = {c: np.abs(shap_values[i]).mean(axis=0)
                 for i, c in enumerate(class_list)}
else:
    shap_mean = {c: np.abs(shap_values[:, :, i]).mean(axis=0)
                 for i, c in enumerate(class_list)}

shap_df    = pd.DataFrame(shap_mean, index=MI_SHORT)
shap_total = shap_df.sum(axis=1).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(MI_SHORT))
colors_shap = plt.cm.Set2(np.linspace(0, 1, len(class_list)))
for i, cls in enumerate(class_list):
    vals = shap_df.loc[shap_total.index, cls].values
    ax.barh(shap_total.index, vals, left=bottom,
            color=colors_shap[i], label=cls, edgecolor="white", linewidth=0.3)
    bottom += vals
ax.set_xlabel("Mean |SHAP value| (summed over RIASEC classes)", fontsize=12)
ax.set_title("SHAP Feature Importance for RIASEC Domain Prediction\n"
             "(Random Forest, TreeExplainer)", fontsize=13, fontweight="bold")
ax.legend(title="RIASEC Class", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# Feature importance (RF Gini)
fi_order = np.argsort(rf_full.feature_importances_)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh([MI_SHORT[i] for i in fi_order], rf_full.feature_importances_[fi_order],
        color="#1976D2", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
ax.set_title("Random Forest Feature Importance for RIASEC Prediction",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("SHAP figures saved.")

# ---------------------------------------------------------------------------
# 7. Save ML results JSON
# ---------------------------------------------------------------------------
ml_results = {
    "n_samples":      int(len(df)),
    "n_classes":      int(n_classes),
    "random_baseline": float(1 / n_classes),
    "majority_class_baseline": float(33 / 107),
    "model_comparison": results,
    "best_model":     best,
    "lr_cv_accuracy_mean": float(results["Logistic Regression"]["accuracy_mean"]),
    "lr_cv_accuracy_std":  float(results["Logistic Regression"]["accuracy_std"]),
    "lr_holdout_accuracy": float(ho_acc),
    "shap_total_importance": {k: float(v) for k, v in shap_total.items()},
    "formula_verification": {
        col: bool((df[col] - formula).abs().max() < 0.01)
        for col, formula in verified.items()
    },
    "clustering": {
        "silhouette_k2": float(silhouettes[0]),
        "silhouette_k3": float(sil_k3),
        "pca_var_pc1":   float(pv[0]),
        "pca_var_pc2":   float(pv[1]),
        "archetype_sizes": {
            nm: int((df["Archetype"] == nm).sum()) for nm in clr
        },
    },
}
with open(f"{OUT_DIR}/ml_results.json", "w") as f:
    json.dump(ml_results, f, indent=2)

print("\nAll results saved to:", OUT_DIR)
print("Done.")
