import sys, pathlib
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
_AGENT_DIR   = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_AGENT_DIR))
import config

import numpy as np
import pandas as pd
import joblib, pickle
import xgboost as xgb
import shap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

_REPO         = config._ROOT
SA_MODELS     = _REPO / "survival_analysis" / "models"
SA_DATA       = config._SURVIVAL_DATA
SA_RESULTS    = _REPO / "survival_analysis" / "results"
AGENT_MODELS  = config.MODELS_DIR

AGENT_MODELS.mkdir(parents=True, exist_ok=True)

print("[save_models] Loading XGBoost from JSON ...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(SA_MODELS / "xgboost_mortality.json"))
print(f"  n_features_in_: {xgb_model.n_features_in_}")

print("[save_models] Loading static_features.csv ...")
static_csv = SA_DATA / "static_features.csv"
df = pd.read_csv(static_csv)

ID_COLS    = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]
LABEL_COLS = ["died", "icu_hours"]
feat_cols  = [c for c in df.columns
              if c not in set(ID_COLS + LABEL_COLS) and df[c].notna().any()]
print(f"  {len(feat_cols)} feature columns: {feat_cols[:6]} ...")
assert len(feat_cols) == xgb_model.n_features_in_, (
    f"Feature count mismatch: csv={len(feat_cols)} vs model={xgb_model.n_features_in_}"
)

print("[save_models] Refitting SimpleImputer + StandardScaler ...")
X = df[feat_cols].values.astype(float)
imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X)
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_imp)
print("  Imputer/scaler fitted.")

xgb_bundle = {
    "model":     xgb_model,
    "imputer":   imputer,
    "scaler":    scaler,
    "feat_cols": feat_cols,
}
out_xgb = AGENT_MODELS / "xgb_mortality.pkl"
joblib.dump(xgb_bundle, out_xgb)
print(f"[save_models] Saved XGBoost bundle → {out_xgb}")

print("[save_models] Building SHAP TreeExplainer (may take ~30s) ...")
explainer = shap.TreeExplainer(xgb_model)
shap_sample = explainer.shap_values(X_sc[:50])
print(f"  SHAP sample shape: {np.array(shap_sample).shape}")

out_shap = AGENT_MODELS / "shap_explainer.pkl"
with open(out_shap, "wb") as f:
    pickle.dump(explainer, f)
print(f"[save_models] Saved SHAP explainer → {out_shap}")

print("[save_models] Loading UMAP embeddings ...")
umap_2d  = np.load(SA_RESULTS / "umap_2d.npy")
emb_ids  = np.load(SA_RESULTS / "lstm_embedding_ids.npy").astype(int)
meta     = pd.read_csv(SA_DATA / "temporal_meta.csv")

print(f"  umap_2d: {umap_2d.shape}, emb_ids: {emb_ids.shape}")

print("[save_models] Running KMeans(k=3) on UMAP ...")
km          = KMeans(n_clusters=3, random_state=42, n_init=20)
cluster_ids = km.fit_predict(umap_2d)
print(f"  Cluster sizes: {np.bincount(cluster_ids).tolist()}")

CLUSTER_DESCRIPTIONS = {
    0: "ICU phenotype cluster 0",
    1: "ICU phenotype cluster 1",
    2: "ICU phenotype cluster 2",
}

umap_df = pd.DataFrame({
    "icustay_id": emb_ids,
    "umap_x":     umap_2d[:, 0],
    "umap_y":     umap_2d[:, 1],
    "cluster":    cluster_ids,
})

meta_enriched = meta.merge(umap_df, on="icustay_id", how="left")

cluster_summary = []
for k in range(3):
    mask = meta_enriched["cluster"] == k
    n = mask.sum()
    mort_pct = meta_enriched.loc[mask, "died"].mean() * 100 if n > 0 else 0
    med_hrs  = meta_enriched.loc[mask, "icu_hours"].median() if n > 0 else 0
    cluster_summary.append({"cluster": k, "n": n, "mortality_pct": round(mort_pct, 1),
                             "median_icu_hrs": round(med_hrs, 1)})

sorted_clusters = sorted(cluster_summary, key=lambda x: x["mortality_pct"], reverse=True)
desc_map = {}
labels = ["High-acuity phenotype (elevated mortality)", 
          "Moderate-acuity phenotype", 
          "Low-acuity phenotype (low mortality)"]
for i, c in enumerate(sorted_clusters):
    desc_map[c["cluster"]] = labels[i]

meta_enriched["cluster_description"] = meta_enriched["cluster"].map(desc_map)

print(f"  Enriched meta shape: {meta_enriched.shape}")
print(f"  Columns: {meta_enriched.columns.tolist()}")
print(f"  UMAP coverage: {meta_enriched['umap_x'].notna().sum()} / {len(meta_enriched)} rows")
print("\n  Cluster summary:")
for row in cluster_summary:
    print(f"    cluster {row['cluster']}: n={row['n']}, "
          f"mort={row['mortality_pct']}%, median_hrs={row['median_icu_hrs']}, "
          f"desc='{desc_map[row['cluster']]}'")

meta_enriched.to_csv(SA_DATA / "temporal_meta.csv", index=False)
print(f"\n[save_models] Saved enriched temporal_meta.csv → {SA_DATA / 'temporal_meta.csv'}")

print("\n[save_models] ok All done!")
print(f"  {out_xgb}")
print(f"  {out_shap}")
print(f"  {SA_DATA / 'temporal_meta.csv'} (enriched with umap_x, umap_y, cluster)")
