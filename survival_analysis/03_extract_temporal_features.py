# %% [markdown]
# # clinical scoring: Temporal Feature Extraction
#
# Build 6-hour windowed time series for each ICU stay (capped at 72h = 12 steps).
# Key fix: uses SQL files (not inline SQL) to avoid shell "Argument list too long".

# %%
import subprocess
import pandas as pd
import numpy as np
import pickle
import os
import tempfile
from pathlib import Path
from io import StringIO

# ── MIMIC path: resolution order 
# 1. MIMIC_DB env var  (explicit path to the DuckDB file)
# 2. MIMIC_CSV_DIR/mimic_duck.db  (DuckDB alongside the raw CSVs)
# 3. Legacy MIMIC_DIR/mimic_duck.db  (old layout — kept for compatibility)
#
# Build the DuckDB from CSVs with:
#   python clinical_agent/scripts/build_mimic_duckdb.py
def _resolve_mimic_db():
    import os
    from pathlib import Path as _P
    if v := os.environ.get("MIMIC_DB"):
        return str(_P(v))
    for key in ("MIMIC_CSV_DIR", "MIMIC_DIR"):
        d = os.environ.get(key)
        if d and _P(d, "mimic_duck.db").exists():
            return str(_P(d) / "mimic_duck.db")
    return None

_resolved_db = _resolve_mimic_db()
DB_PATH = _resolved_db
DATA_DIR = str(Path(os.environ.get("SURVIVAL_DATA_DIR",
    str(_REPO_ROOT / "survival_analysis" / "data"))))
os.makedirs(DATA_DIR, exist_ok=True)

N_TIMESTEPS = 12
FEATURE_NAMES = ["creatinine","bilirubin","platelets","lactate","wbc",
                 "bicarbonate","inr","map","hr","spo2"]
N_FEATURES = len(FEATURE_NAMES)


def query_duckdb_file(sql: str) -> pd.DataFrame:
    """Write SQL to a temp file and run duckdb with -f to avoid arg length limits."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write(sql)
        sql_path = f.name
    try:
        result = subprocess.run(
            ["duckdb", DB_PATH, "-csv", "-f", sql_path],
            capture_output=True, text=True, check=True, timeout=1800  # 30 min timeout
        )
    finally:
        os.unlink(sql_path)
    return pd.read_csv(StringIO(result.stdout))


# %% [markdown]
# ## Step 1: Load cohort

# %%
cohort = pd.read_csv(os.path.join(DATA_DIR, "static_features.csv"),
                     usecols=["SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME","died","icu_hours"])
cohort["INTIME"] = pd.to_datetime(cohort["INTIME"])
print(f"Cohort: {len(cohort):,} ICU stays")

# %% [markdown]
# ## Step 2: Pull lab values (72h window)
# Uses subquery into icustays table — no giant IN clause

# %%
print("\nStep 2: Pulling temporal lab values (72h window)...")

LAB_ITEMIDS = {
    50912: "creatinine",
    50885: "bilirubin",
    51265: "platelets",
    50813: "lactate",
    51301: "wbc",
    50882: "bicarbonate",
    51237: "inr",
}
lab_itemid_list = ",".join(str(k) for k in LAB_ITEMIDS.keys())

lab_sql = f"""
SELECT
    l.HADM_ID,
    l.ITEMID,
    l.CHARTTIME,
    l.VALUENUM,
    i.INTIME,
    i.ICUSTAY_ID
FROM labevents l
JOIN icustays i ON l.HADM_ID = i.HADM_ID
-- Only first ICU stay per patient (no external ID list needed)
WHERE i.ICUSTAY_ID IN (
    SELECT FIRST(ICUSTAY_ID ORDER BY INTIME)
    FROM icustays
    GROUP BY SUBJECT_ID
)
  AND l.ITEMID IN ({lab_itemid_list})
  AND l.VALUENUM IS NOT NULL
  AND l.VALUENUM > 0
  AND l.CHARTTIME >= i.INTIME
  AND l.CHARTTIME <= i.INTIME + INTERVAL 72 HOUR
"""
print("  Querying labevents (72h, first ICU stays)...")
lab_raw = query_duckdb_file(lab_sql)
lab_raw["CHARTTIME"] = pd.to_datetime(lab_raw["CHARTTIME"])
lab_raw["INTIME"]    = pd.to_datetime(lab_raw["INTIME"])
print(f"  Lab rows fetched: {len(lab_raw):,}")

# %% [markdown]
# ## Step 3: Pull vital signs (72h window)

# %%
print("\nStep 3: Pulling temporal vital signs (72h window)...")

MAP_ITEMIDS  = [220052, 220181, 51, 455]
HR_ITEMIDS   = [220045, 211]
SPO2_ITEMIDS = [220277, 646]

vital_groups = {"map": MAP_ITEMIDS, "hr": HR_ITEMIDS, "spo2": SPO2_ITEMIDS}
all_vital_iids   = [i for ids in vital_groups.values() for i in ids]
vital_itemid_list = ",".join(str(k) for k in all_vital_iids)
iid_to_group     = {i: g for g, ids in vital_groups.items() for i in ids}

vital_sql = f"""
SELECT
    c.ICUSTAY_ID,
    c.ITEMID,
    c.CHARTTIME,
    c.VALUENUM,
    i.INTIME
FROM chartevents c
JOIN icustays i ON c.ICUSTAY_ID = i.ICUSTAY_ID
WHERE i.ICUSTAY_ID IN (
    SELECT FIRST(ICUSTAY_ID ORDER BY INTIME)
    FROM icustays
    GROUP BY SUBJECT_ID
)
  AND c.ITEMID IN ({vital_itemid_list})
  AND c.VALUENUM IS NOT NULL
  AND c.VALUENUM > 0
  AND c.CHARTTIME >= i.INTIME
  AND c.CHARTTIME <= i.INTIME + INTERVAL 72 HOUR
  AND c.ERROR IS NULL
"""
print("  Querying chartevents (72h window, may take 10-25 min)...")
vital_raw = query_duckdb_file(vital_sql)
vital_raw["CHARTTIME"] = pd.to_datetime(vital_raw["CHARTTIME"])
vital_raw["INTIME"]    = pd.to_datetime(vital_raw["INTIME"])
print(f"  Vital rows fetched: {len(vital_raw):,}")

# %% [markdown]
# ## Step 4: Bin into 6h windows

# %%
print("\nStep 4: Building windowed sequences...")

# Lab features
lab_raw["hours_in"] = (lab_raw["CHARTTIME"] - lab_raw["INTIME"]).dt.total_seconds() / 3600
lab_raw["time_bin"] = (lab_raw["hours_in"] / 6).astype(int).clip(0, N_TIMESTEPS - 1)
lab_raw["feature"]  = lab_raw["ITEMID"].map(LAB_ITEMIDS)
lab_agg = (lab_raw.groupby(["ICUSTAY_ID","time_bin","feature"])["VALUENUM"]
           .mean().reset_index())

# Vital features
vital_raw["hours_in"] = (vital_raw["CHARTTIME"] - vital_raw["INTIME"]).dt.total_seconds() / 3600
vital_raw = vital_raw[vital_raw["hours_in"].between(0, 72)]
vital_raw["time_bin"] = (vital_raw["hours_in"] / 6).astype(int).clip(0, N_TIMESTEPS - 1)
vital_raw["feature"]  = vital_raw["ITEMID"].map(iid_to_group)
vital_agg = (vital_raw.groupby(["ICUSTAY_ID","time_bin","feature"])["VALUENUM"]
             .mean().reset_index())

all_agg   = pd.concat([lab_agg, vital_agg], ignore_index=True)
fmap      = {f: i for i, f in enumerate(FEATURE_NAMES)}

# Compute per-feature global median for imputation of missing timesteps
feature_medians = {}
for feat in FEATURE_NAMES:
    vals = all_agg[all_agg["feature"] == feat]["VALUENUM"].dropna()
    feature_medians[feat] = vals.median() if len(vals) > 0 else 0.0

print(f"  Aggregated rows: {len(all_agg):,}")
print(f"  Unique ICU stays with data: {all_agg['ICUSTAY_ID'].nunique():,}")

# %% [markdown]
# ## Step 5: Build numpy arrays

# %%
print("\nBuilding per-patient arrays...")
sequences = {}

for icustay_id, group in all_agg.groupby("ICUSTAY_ID"):
    arr = np.full((N_TIMESTEPS, N_FEATURES), np.nan)
    for _, row in group.iterrows():
        t = int(row["time_bin"])
        fi = fmap.get(row["feature"], -1)
        if fi >= 0:
            arr[t, fi] = row["VALUENUM"]
    # Forward-fill then back-fill each feature
    for fi in range(N_FEATURES):
        col = arr[:, fi]
        mask = np.isnan(col)
        if mask.all():
            arr[:, fi] = feature_medians.get(FEATURE_NAMES[fi], 0.0)
        else:
            valid_idx = np.where(~mask)[0]
            for j in range(len(valid_idx) - 1):
                arr[valid_idx[j]:valid_idx[j+1], fi] = col[valid_idx[j]]
            arr[valid_idx[-1]:, fi] = col[valid_idx[-1]]
            if valid_idx[0] > 0:
                arr[:valid_idx[0], fi] = col[valid_idx[0]]
    sequences[int(icustay_id)] = arr

print(f"  Built {len(sequences):,} sequences")

# %% [markdown]
# ## Step 6: Normalize and save

# %%
# Z-score normalization across all patients
all_arrays = np.stack(list(sequences.values()))
all_clean  = np.nan_to_num(all_arrays, nan=0.0)
feat_mean  = all_clean.reshape(-1, N_FEATURES).mean(axis=0)
feat_std   = all_clean.reshape(-1, N_FEATURES).std(axis=0)
feat_std[feat_std == 0] = 1.0

sequences_norm = {
    sid: (np.nan_to_num(arr, nan=0.0) - feat_mean) / feat_std
    for sid, arr in sequences.items()
}

np.save(os.path.join(DATA_DIR, "feat_mean.npy"), feat_mean)
np.save(os.path.join(DATA_DIR, "feat_std.npy"),  feat_std)

# Metadata
meta_rows = []
for icustay_id in sequences_norm:
    row = cohort[cohort["ICUSTAY_ID"] == icustay_id]
    if len(row) > 0:
        r = row.iloc[0]
        meta_rows.append({
            "icustay_id": icustay_id,
            "subject_id": int(r["SUBJECT_ID"]),
            "hadm_id":    int(r["HADM_ID"]),
            "died":       int(r["died"]),
            "icu_hours":  float(r["icu_hours"]),
        })
meta_df = pd.DataFrame(meta_rows)

seq_path  = os.path.join(DATA_DIR, "temporal_sequences.pkl")
meta_path = os.path.join(DATA_DIR, "temporal_meta.csv")

with open(seq_path, "wb") as f:
    pickle.dump(sequences_norm, f, protocol=4)
meta_df.to_csv(meta_path, index=False)

print(f"Saved: {seq_path}")
print(f"Saved: {meta_path}")
print(f"  {len(sequences_norm):,} sequences, mortality={meta_df['died'].mean():.3f}")

assert len(sequences_norm) > 3000, f"Too few sequences: {len(sequences_norm)}"
sample   = list(sequences_norm.values())[0]
assert sample.shape == (N_TIMESTEPS, N_FEATURES), f"Bad shape: {sample.shape}"
print("\nok clinical scoring complete.")
