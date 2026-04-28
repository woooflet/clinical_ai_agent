# %% [markdown]
# # tool composition: UMAP + ICU Phenotype Clustering
#
# Discover latent patient phenotypes from LSTM embeddings:
# 1. UMAP dimensionality reduction (2D visualization)
# 2. K-Means clustering with silhouette score selection
# 3. Cluster characterization (mortality, LOS, ICD chapter)
# 4. Latent subgroup Cox: cluster membership as Cox covariate

# %%
import numpy as np
import pandas as pd
import pickle, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import umap
from lifelines import CoxPHFitter

# ── Data paths: set MIMIC_DIR / SURVIVAL_DATA_DIR in your .env 
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv as _lde
    _lde(_REPO / ".env")
except ImportError:
    pass
_SA_DATA = _Path(_os.environ.get("SURVIVAL_DATA_DIR", str(_REPO / "survival_analysis" / "data")))
_SA_RES  = _REPO / "survival_analysis" / "results"
_SA_FIG  = _REPO / "survival_analysis" / "figures"
_SA_MOD  = _REPO / "survival_analysis" / "models"
DATA_DIR = str(_SA_DATA)
RES_DIR  = str(_SA_RES)
FIG_DIR  = str(_SA_FIG)
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ## Load embeddings and metadata

# %%
print("Loading LSTM embeddings...")
embeddings   = np.load(os.path.join(RES_DIR, "lstm_embeddings.npy"))
emb_ids      = np.load(os.path.join(RES_DIR, "lstm_embedding_ids.npy"))

meta   = pd.read_csv(os.path.join(DATA_DIR, "temporal_meta.csv"))
static = pd.read_csv(os.path.join(DATA_DIR, "static_features.csv"))

# Align metadata with embeddings
emb_df = pd.DataFrame({"icustay_id": emb_ids.astype(int)})
emb_df = emb_df.merge(meta, on="icustay_id", how="left")

print(f"  Embeddings: {embeddings.shape}")
print(f"  Matched meta rows: {emb_df['died'].notna().sum()}")

# Scale embeddings
scaler     = StandardScaler()
emb_scaled = scaler.fit_transform(embeddings)

# %% [markdown]
# ## UMAP dimensionality reduction

# %%
print("\nPCA pre-reduction (128 → 20 dims) before UMAP...")
from sklearn.decomposition import PCA
pca = PCA(n_components=20, random_state=42)
emb_pca = pca.fit_transform(emb_scaled)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

print("Fitting UMAP (2D, euclidean, n_neighbors=15)...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",   # avoids angular tree in pynndescent
    random_state=42,
    n_jobs=1,
    low_memory=True,
)
umap_2d = reducer.fit_transform(emb_pca)
print(f"  UMAP output shape: {umap_2d.shape}")
np.save(os.path.join(RES_DIR, "umap_2d.npy"), umap_2d)

# %% [markdown]
# ## K-Means clustering with silhouette selection

# %%
print("\nSelecting optimal k via silhouette score...")
k_range   = range(3, 8)
sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(umap_2d)
    sil = silhouette_score(umap_2d, labels_k)
    sil_scores.append(sil)
    print(f"  k={k}: silhouette={sil:.4f}")

best_k = list(k_range)[np.argmax(sil_scores)]
print(f"\nBest k: {best_k} (silhouette={max(sil_scores):.4f})")

# Fit final KMeans
km_final     = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_ids  = km_final.fit_predict(umap_2d)
emb_df["cluster"] = cluster_ids

# Silhouette plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(k_range), sil_scores, "bo-", markersize=8)
ax.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette score")
ax.set_title("K-Means Cluster Selection via Silhouette Score")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "silhouette_scores.png"), dpi=150)
plt.close()

# %% [markdown]
# ## UMAP scatter plots

# %%
print("\nPlotting UMAP visualizations...")
died_vals = emb_df["died"].fillna(0).values

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: colored by cluster
colors_cluster = cm.tab10(np.linspace(0, 1, best_k))
for k in range(best_k):
    mask = cluster_ids == k
    axes[0].scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                    c=[colors_cluster[k]], s=4, alpha=0.5, label=f"Cluster {k}")
axes[0].set_title(f"UMAP — ICU Phenotype Clusters (k={best_k})", fontsize=12)
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
axes[0].legend(markerscale=3, fontsize=9)

# Panel 2: colored by outcome
sc = axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1],
                     c=died_vals, cmap="RdYlGn_r", s=3, alpha=0.5)
plt.colorbar(sc, ax=axes[1], label="Mortality (1=died)")
axes[1].set_title("UMAP — ICU Outcomes", fontsize=12)
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "umap_clusters.png"), dpi=150)
plt.close()
print("  Saved: umap_clusters.png")

# %% [markdown]
# ## Cluster characterization

# %%
print("\nCharacterizing clusters...")

# Merge cluster ids back to static features
clust_static = emb_df[["icustay_id","cluster","died","icu_hours"]].merge(
    static[["ICUSTAY_ID"] + [c for c in static.columns if c.startswith("diag_")]],
    left_on="icustay_id", right_on="ICUSTAY_ID", how="left"
)

# Recover dominant ICD chapter from one-hot
diag_cols = [c for c in static.columns if c.startswith("diag_")]
if diag_cols:
    clust_static["top_diag"] = clust_static[diag_cols].idxmax(axis=1).str.replace("diag_","")

summary_rows = []
for k in range(best_k):
    mask = clust_static["cluster"] == k
    sub  = clust_static[mask]
    row  = {
        "cluster":        k,
        "n":              mask.sum(),
        "mortality_pct":  round(sub["died"].mean() * 100, 1),
        "median_icu_hrs": round(sub["icu_hours"].median(), 1),
    }
    if "top_diag" in sub.columns:
        row["top_diag"] = sub["top_diag"].value_counts().index[0] if len(sub) > 0 else "N/A"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values("mortality_pct", ascending=False)
print("\nCluster Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(RES_DIR, "cluster_summary.csv"), index=False)

# Heatmap of cluster characteristics
fig, ax = plt.subplots(figsize=(8, 5))
heatmap_data = summary_df.set_index("cluster")[["n","mortality_pct","median_icu_hrs"]].T
im = ax.imshow(heatmap_data.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(best_k))
ax.set_xticklabels([f"C{k}" for k in summary_df["cluster"]])
ax.set_yticks(range(3))
ax.set_yticklabels(["Count","Mortality %","Median LOS (h)"])
ax.set_title("Cluster Characteristics Heatmap")
plt.colorbar(im, ax=ax)
for i in range(3):
    for j in range(best_k):
        ax.text(j, i, f"{heatmap_data.values[i, j]:.1f}", ha="center", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_heatmap.png"), dpi=150)
plt.close()
print("  Saved: cluster_heatmap.png")

# %% [markdown]
# ## Latent subgroup Cox analysis

# %%
print("\nFitting latent subgroup Cox model (cluster membership as covariate)...")

# Get full static features for IDs that have embeddings
# Use icu_hours and died from meta
cox_input = clust_static[["icustay_id","cluster","died","icu_hours"]].dropna()

# Add cluster as one-hot covariates
cluster_dummies = pd.get_dummies(cox_input["cluster"], prefix="clust")
cox_df = pd.concat([cox_input[["icu_hours","died"]], cluster_dummies], axis=1)
# Drop one dummy for reference category
cox_df.drop(columns=[f"clust_{best_k - 1}"], inplace=True)

cph2 = CoxPHFitter(penalizer=0.1)
cph2.fit(cox_df, duration_col="icu_hours", event_col="died")
print(cph2.summary[["coef","exp(coef)","p"]].to_string())
cluster_cox_cindex = cph2.concordance_index_
print(f"\nLatent subgroup Cox C-index: {cluster_cox_cindex:.4f}")

# HR forest plot for cluster terms
hr_clust = cph2.summary[["exp(coef)","coef lower 95%","coef upper 95%"]].copy()
hr_clust.columns = ["hr","ci_lo","ci_hi"]
fig, ax = plt.subplots(figsize=(7, 4))
y_pos = np.arange(len(hr_clust))
ax.barh(y_pos, hr_clust["hr"],
        xerr=[hr_clust["hr"]-hr_clust["ci_lo"], hr_clust["ci_hi"]-hr_clust["hr"]],
        align="center", color="#9b59b6", alpha=0.8, capsize=3, height=0.5)
ax.axvline(1.0, color="red", linestyle="--")
ax.set_yticks(y_pos)
ax.set_yticklabels(hr_clust.index, fontsize=9)
ax.set_xlabel("Hazard Ratio (95% CI)")
ax.set_title(f"Latent Subgroup Cox — Cluster Hazard Ratios\n(C-index = {cluster_cox_cindex:.3f})")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_cox_hr.png"), dpi=150)
plt.close()
print("  Saved: cluster_cox_hr.png")

# Save results
results = {
    "best_k": best_k,
    "sil_scores": dict(zip(list(k_range), [round(s, 4) for s in sil_scores])),
    "cluster_cox_cindex": round(cluster_cox_cindex, 4),
    "cluster_summary": summary_df.to_dict("records"),
}
with open(os.path.join(RES_DIR, "clustering_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Assertions
for k in range(best_k):
    assert (cluster_ids == k).sum() > 50, f"Cluster {k} too small: {(cluster_ids==k).sum()}"

print(f"\nok tool composition complete. Best k={best_k}, clusters characterized.")
