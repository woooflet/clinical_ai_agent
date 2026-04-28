# %% [markdown]
# # external APIs: Cox Proportional Hazards + XGBoost Models
#
# Layer 1 of the analysis:
# - Cox PH model on static features → hazard ratios, C-index, KM curves
# - XGBoost binary mortality classifier → AUC, GridSearch tuning (bonus)
# - SHAP feature importance for XGBoost
#
# =============================================================================
# TRAINED MODEL WEIGHTS
# =============================================================================
# This script produces the XGBoost and SHAP model weights that are committed
# to git at clinical_agent/models/ (via scripts/save_models.py afterward).
#
# Output of THIS script (survival_analysis/models/ — gitignored):
#   xgboost_mortality.json  — XGBoost native model (intermediate)
#   xgb_test_probs.npy      — test set predictions
#   cox_xgb_results.json    — evaluation metrics
#
# Final committed weights (clinical_agent/models/ — produced by save_models.py):
#   xgb_mortality.pkl       — XGBoost bundle: {model, imputer, scaler, feat_cols}
#   shap_explainer.pkl      — SHAP TreeExplainer for xgb_mortality
#
# To REPRODUCE the weights from scratch:
#   1. Ensure MIMIC_DIR is set in .env and MIMIC DuckDB is available
#   2. Run survival_analysis/ scripts in order:
#        02_extract_static_features.py   (extracts features from MIMIC)
#        04_cox_model.py                 (this script — trains XGBoost)
#        python clinical_agent/scripts/save_models.py  (packages for the agent)
#
# Pre-trained weights are committed to git — retraining is NOT required to
# run the agent. The weights were trained on a MIMIC-III cohort of ~38,000
# first ICU stays (no external data needed beyond what's in the .pkl files).
# =============================================================================

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")
import os, pickle

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

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
DATA_DIR  = str(_SA_DATA)
FIG_DIR   = str(_SA_FIG)
RES_DIR   = str(_SA_RES)
MODEL_DIR = str(_SA_MOD)
for d in [FIG_DIR, RES_DIR, MODEL_DIR, DATA_DIR]:
    _os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## Load and prepare data

# %%
df = pd.read_csv(os.path.join(DATA_DIR, "static_features.csv"))
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"Mortality: {df['died'].mean():.3f} ({df['died'].sum():.0f} deaths)")
print(f"Median ICU hours: {df['icu_hours'].median():.1f}")

# Identify feature columns (exclude identifiers and labels)
ID_COLS     = ["SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME","OUTTIME"]
LABEL_COLS  = ["died","icu_hours"]
excl         = set(ID_COLS + LABEL_COLS)
feature_cols = [c for c in df.columns if c not in excl]
print(f"Feature count: {len(feature_cols)}")

X = df[feature_cols].copy()

# Drop columns that are entirely NaN (SimpleImputer would silently exclude them)
all_nan_cols = [c for c in feature_cols if X[c].isna().all()]
if all_nan_cols:
    print(f"  Dropping all-NaN columns: {all_nan_cols}")
    X.drop(columns=all_nan_cols, inplace=True)
    feature_cols = [c for c in feature_cols if c not in all_nan_cols]

y_event    = df["died"].values.astype(int)
y_duration = df["icu_hours"].values.astype(float)

# Impute and scale
imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X)
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# Save imputer/scaler for later use
pickle.dump(imputer, open(os.path.join(MODEL_DIR, "imputer.pkl"), "wb"))
pickle.dump(scaler,  open(os.path.join(MODEL_DIR, "scaler.pkl"),  "wb"))

# %% [markdown]
# ## Kaplan-Meier curves by diagnosis chapter

# %%
print("\nPlotting Kaplan-Meier curves by diagnosis chapter...")

# Identify diag columns
diag_cols = [c for c in df.columns if c.startswith("diag_")]

# Assign back for KM plotting
df_km = df[["icu_hours", "died"]].copy()
# Recover the chapter label from one-hot
df_km["chapter"] = "misc"
for dc in diag_cols:
    chapter_name = dc.replace("diag_", "")
    df_km.loc[df[dc] == 1, "chapter"] = chapter_name

# Pick top N chapters by count for readability
top_chapters = df_km["chapter"].value_counts().head(8).index.tolist()

fig, ax = plt.subplots(figsize=(11, 7))
colors = cm.tab10(np.linspace(0, 1, len(top_chapters)))
kmf = KaplanMeierFitter()

logrank_ps = []
for chap, color in zip(top_chapters, colors):
    mask = df_km["chapter"] == chap
    t = df_km.loc[mask, "icu_hours"]
    e = df_km.loc[mask, "died"]
    kmf.fit(t, e, label=f"{chap} (n={mask.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color=color)

ax.set_xlim(0, 336)   # 14 days
ax.set_xlabel("Hours in ICU", fontsize=12)
ax.set_ylabel("Survival Probability", fontsize=12)
ax.set_title("Kaplan-Meier Curves by Primary Diagnosis Chapter\n(MIMIC-III First ICU Stays)", fontsize=13)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "km_by_chapter.png"), dpi=150)
plt.close()
print(f"  Saved: km_by_chapter.png")

# %% [markdown]
# ## Cox Proportional Hazards Model

# %%
print("\nFitting Cox PH model...")

# Use a focused feature set for Cox (avoid high-cardinality dummies)
cox_features  = ([c for c in feature_cols if c.startswith("lab_")]
               + [c for c in feature_cols if c.startswith("vital_")]
               + ["age_at_icu", "gender_m", "n_prior_admissions"])
cox_features  = [c for c in cox_features if c in df.columns]

df_cox = df[cox_features + ["icu_hours","died"]].copy()
df_cox_imp = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(df_cox),
    columns=df_cox.columns
)

# Penalized Cox to handle collinearity
cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.1)
cph.fit(df_cox_imp, duration_col="icu_hours", event_col="died")

print(cph.summary[["coef","exp(coef)","p","coef lower 95%","coef upper 95%"]].head(20).to_string())

cox_cindex = cph.concordance_index_
print(f"\nCox C-index: {cox_cindex:.4f}")

# Hazard ratio forest plot — use exponentiated CI columns from lifelines
hr_df = cph.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%"]].copy()
hr_df.columns = ["hr","ci_lo","ci_hi"]
hr_df = hr_df.sort_values("hr", ascending=True).head(20)

# Clamp CIs to ensure non-negative error bars
err_lo = np.maximum(hr_df["hr"].values - hr_df["ci_lo"].values, 0)
err_hi = np.maximum(hr_df["ci_hi"].values - hr_df["hr"].values, 0)

fig, ax = plt.subplots(figsize=(9, 7))
y_pos = np.arange(len(hr_df))
ax.barh(y_pos, hr_df["hr"], xerr=[err_lo, err_hi],
        align="center", color="#4a90d9", alpha=0.8, capsize=3, height=0.6)
ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(hr_df.index, fontsize=9)
ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=11)
ax.set_title(f"Cox PH Hazard Ratios — Top 20 Features\n(C-index = {cox_cindex:.3f})", fontsize=12)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cox_hazard_ratios.png"), dpi=150)
plt.close()
print("  Saved: cox_hazard_ratios.png")


# %% [markdown]
# ## XGBoost Binary Classifier + GridSearch

# %%
print("\nTraining XGBoost classifier...")

# Full feature set
X_train, X_test, y_train, y_test, dur_train, dur_test = train_test_split(
    X_scaled, y_event, y_duration, test_size=0.2, random_state=42, stratify=y_event
)

# Scale pos_weight for class imbalance
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  pos_weight: {pos_weight:.2f}  (class imbalance correction)")

# GridSearch over key hyperparameters
param_grid = {
    "max_depth":        [3, 5],
    "learning_rate":    [0.05, 0.1],
    "n_estimators":     [200, 400],
    "subsample":        [0.8],
    "colsample_bytree": [0.8],
}
base_xgb = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=pos_weight,
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(base_xgb, param_grid, cv=cv, scoring="roc_auc",
                            n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
print(f"\n  Best params: {grid_search.best_params_}")
print(f"  Best CV AUC: {grid_search.best_score_:.4f}")

# Test set evaluation
y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
xgb_auc    = roc_auc_score(y_test, y_prob_xgb)
xgb_brier  = brier_score_loss(y_test, y_prob_xgb)
# C-index
xgb_cindex = concordance_index(dur_test, -y_prob_xgb, y_test)

print(f"  XGBoost AUC:     {xgb_auc:.4f}")
print(f"  XGBoost Brier:   {xgb_brier:.4f}")
print(f"  XGBoost C-index: {xgb_cindex:.4f}")

# Save model
best_xgb.save_model(os.path.join(MODEL_DIR, "xgboost_mortality.json"))
np.save(os.path.join(RES_DIR, "xgb_test_probs.npy"), y_prob_xgb)
np.save(os.path.join(RES_DIR, "xgb_test_labels.npy"), y_test)
np.save(os.path.join(RES_DIR, "xgb_test_durations.npy"), dur_test)

# %% [markdown]
# ## SHAP Feature Importance

# %%
print("\nComputing SHAP values (may take 1-2 min)...")
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test[:500])  # sample for speed

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test[:500], feature_names=feature_cols,
                  max_display=20, show=False)
plt.title("SHAP Feature Importance — XGBoost Mortality Model", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_summary.png")

# %% [markdown]
# ## Calibration Curve

# %%
print("\nPlotting calibration curve...")
frac_pos, mean_pred = calibration_curve(y_test, y_prob_xgb, n_bins=10)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(mean_pred, frac_pos, "bo-", label=f"XGBoost (AUC={xgb_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve — Mortality Prediction")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "calibration_xgb.png"), dpi=150)
plt.close()
print("  Saved: calibration_xgb.png")

# %% [markdown]
# ## Save results for RAG literature evaluation

# %%
results = {
    "cox_cindex": cox_cindex,
    "xgb_auc": xgb_auc,
    "xgb_brier": xgb_brier,
    "xgb_cindex": xgb_cindex,
    "xgb_best_params": grid_search.best_params_,
    "xgb_cv_auc": grid_search.best_score_,
}
import json
with open(os.path.join(RES_DIR, "cox_xgb_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Assertions
assert cox_cindex > 0.60, f"Cox C-index too low: {cox_cindex:.4f}"
assert xgb_auc    > 0.65, f"XGBoost AUC too low: {xgb_auc:.4f}"
print(f"\nok external APIs complete. Cox C-index={cox_cindex:.3f}, XGB AUC={xgb_auc:.3f}")
