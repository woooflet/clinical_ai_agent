# %% [markdown]
# # patient data: Static Feature Extraction
#
# Build a per-ICU-stay feature matrix from the first 6 hours of admission.
# Joins ICUSTAYS, ADMISSIONS, PATIENTS, DIAGNOSES_ICD, LABEVENTS, CHARTEVENTS.
# Output: data/static_features.csv

# %%
import subprocess
import pandas as pd
import numpy as np
import os
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
OUT_DIR = str(Path(os.environ.get("SURVIVAL_DATA_DIR",
    str(_REPO_ROOT / "survival_analysis" / "data"))))
os.makedirs(OUT_DIR, exist_ok=True)


def query_duckdb(sql: str) -> pd.DataFrame:
    result = subprocess.run(
        ["duckdb", DB_PATH, "-csv", "-c", sql],
        capture_output=True, text=True, check=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr)
    return pd.read_csv(StringIO(result.stdout))


# %% [markdown]
# ## Step 1: Build the ICU cohort (one row per first ICU stay per patient)

# %%
print("Step 1: Building ICU cohort...")
cohort_sql = """
SELECT
    i.SUBJECT_ID,
    i.HADM_ID,
    i.ICUSTAY_ID,
    i.INTIME,
    i.OUTTIME,
    DATEDIFF('hour', i.INTIME, i.OUTTIME)      AS icu_hours,
    p.GENDER,
    DATEDIFF('year', p.DOB, i.INTIME)           AS age_at_icu,
    a.HOSPITAL_EXPIRE_FLAG                       AS died,
    a.ADMISSION_TYPE,
    a.INSURANCE
FROM icustays i
JOIN admissions a  ON i.HADM_ID    = a.HADM_ID
JOIN patients   p  ON i.SUBJECT_ID = p.SUBJECT_ID
-- First ICU stay per patient only (avoid data leakage from subsequent stays)
WHERE i.ICUSTAY_ID IN (
    SELECT FIRST(ICUSTAY_ID ORDER BY INTIME)
    FROM icustays
    GROUP BY SUBJECT_ID
)
  AND DATEDIFF('hour', i.INTIME, i.OUTTIME) > 0
  AND i.INTIME IS NOT NULL
  AND i.OUTTIME IS NOT NULL
ORDER BY i.SUBJECT_ID
"""
cohort = query_duckdb(cohort_sql)

# Cap ages > 200 (MIMIC obfuscates very old patients to 300)
cohort["age_at_icu"] = cohort["age_at_icu"].clip(upper=91)

print(f"  Cohort size: {len(cohort):,} ICU stays")
print(f"  Mortality rate: {cohort['died'].mean():.3f}")
print(f"  Median ICU hours: {cohort['icu_hours'].median():.1f}")
print(cohort.dtypes)

# %% [markdown]
# ## Step 2: Primary diagnosis ICD-9 chapter

# %%
print("\nStep 2: Adding ICD-9 diagnosis chapter...")
icd_sql = """
SELECT d.HADM_ID, d.ICD9_CODE
FROM diagnoses_icd d
WHERE d.SEQ_NUM = 1
  AND d.ICD9_CODE IS NOT NULL
  AND d.ICD9_CODE NOT LIKE 'E%'
  AND d.ICD9_CODE NOT LIKE 'V%'
"""
icd_df = query_duckdb(icd_sql)

def icd9_to_chapter(code):
    """Map ICD-9 code to chapter number (0-17)."""
    try:
        num = int(str(code)[:3])
    except (ValueError, TypeError):
        return 15  # misc
    ranges = [(1,140),(140,240),(240,280),(280,290),(290,320),(320,390),
              (390,460),(460,520),(520,580),(580,630),(630,680),(680,710),
              (710,740),(740,760),(760,779),(780,800),(800,1000)]
    chapter_names = ["infectious","neoplasms","endocrine","blood","mental","nervous",
                     "circulatory","respiratory","digestive","genitourinary","pregnancy",
                     "skin","musculoskeletal","congenital","perinatal","misc","injury"]
    for i, (lo, hi) in enumerate(ranges):
        if lo <= num < hi:
            return chapter_names[i] if i < len(chapter_names) else "misc"
    return "misc"

icd_df["diag_chapter"] = icd_df["ICD9_CODE"].apply(icd9_to_chapter)
icd_df = icd_df[["HADM_ID", "diag_chapter"]].drop_duplicates("HADM_ID")
cohort = cohort.merge(icd_df, on="HADM_ID", how="left")
cohort["diag_chapter"] = cohort["diag_chapter"].fillna("misc")
print(f"  Diagnosis chapter distribution:\n{cohort['diag_chapter'].value_counts().head(8)}")

# %% [markdown]
# ## Step 3: Prior admissions count

# %%
print("\nStep 3: Prior admissions count...")
prior_sql = """
SELECT
    i.SUBJECT_ID,
    i.ICUSTAY_ID,
    COUNT(a2.HADM_ID) AS n_prior_admissions
FROM icustays i
JOIN admissions a1 ON i.HADM_ID = a1.HADM_ID
LEFT JOIN admissions a2
    ON a2.SUBJECT_ID = i.SUBJECT_ID
   AND a2.ADMITTIME < a1.ADMITTIME
WHERE i.ICUSTAY_ID IN (
    SELECT FIRST(ICUSTAY_ID ORDER BY INTIME)
    FROM icustays
    GROUP BY SUBJECT_ID
)
GROUP BY i.SUBJECT_ID, i.ICUSTAY_ID
"""
prior_df = query_duckdb(prior_sql)
cohort = cohort.merge(prior_df[["ICUSTAY_ID","n_prior_admissions"]], on="ICUSTAY_ID", how="left")
cohort["n_prior_admissions"] = cohort["n_prior_admissions"].fillna(0).astype(int)
print(f"  Mean prior admissions: {cohort['n_prior_admissions'].mean():.2f}")

# %% [markdown]
# ## Step 4: First-6h lab values (LABEVENTS)

# %%
print("\nStep 4: First-6h laboratory values...")

# Lab ITEMIDs
LAB_ITEMIDS = {
    50912: "lab_creatinine",
    50885: "lab_bilirubin",
    51265: "lab_platelets",
    50813: "lab_lactate",
    51301: "lab_wbc",
    50821: "lab_pao2",
    50882: "lab_bicarbonate",
    50971: "lab_potassium",
    50983: "lab_sodium",
    51006: "lab_bun",
    51237: "lab_inr",
}
lab_itemid_list = ",".join(str(k) for k in LAB_ITEMIDS.keys())

lab_sql = f"""
SELECT
    l.HADM_ID,
    l.ITEMID,
    l.VALUENUM
FROM labevents l
JOIN icustays i ON l.HADM_ID = i.HADM_ID
WHERE l.ITEMID IN ({lab_itemid_list})
  AND l.VALUENUM IS NOT NULL
  AND l.VALUENUM > 0
  AND l.CHARTTIME >= i.INTIME
  AND l.CHARTTIME <= i.INTIME + INTERVAL 6 HOUR
  AND i.ICUSTAY_ID IN (
    SELECT FIRST(ICUSTAY_ID ORDER BY INTIME)
    FROM icustays
    GROUP BY SUBJECT_ID
  )
"""
lab_raw = query_duckdb(lab_sql)
print(f"  Raw lab rows (first 6h): {len(lab_raw):,}")

# Pivot: mean value per HADM_ID per ITEMID
lab_pivot = lab_raw.groupby(["HADM_ID", "ITEMID"])["VALUENUM"].mean().unstack(fill_value=np.nan)
lab_pivot.columns = [LAB_ITEMIDS.get(int(c), f"lab_{c}") for c in lab_pivot.columns if int(c) in LAB_ITEMIDS]
lab_pivot = lab_pivot.reset_index()
cohort = cohort.merge(lab_pivot, on="HADM_ID", how="left")
print(f"  Lab features added: {[c for c in cohort.columns if c.startswith('lab_')]}")

# %% [markdown]
# ## Step 5: First-6h vital signs (CHARTEVENTS)

# %%
print("\nStep 5: First-6h vital signs...")

# Consolidated MAP: prefer invasive, fall back to non-invasive, then CareVue
MAP_ITEMIDS    = [220052, 220181, 51, 455, 442]
HR_ITEMIDS     = [220045, 211]
RR_ITEMIDS     = [220210, 224690, 618]
SPO2_ITEMIDS   = [220277, 646]
GCS_V_ITEMIDS  = [223900]
GCS_M_ITEMIDS  = [223901]
TEMP_ITEMIDS   = [223762]
FIO2_ITEMIDS   = [223835]
SBP_ITEMIDS    = [220179]

vital_groups = {
    "vital_map":   MAP_ITEMIDS,
    "vital_hr":    HR_ITEMIDS,
    "vital_rr":    RR_ITEMIDS,
    "vital_spo2":  SPO2_ITEMIDS,
    "vital_gcs_v": GCS_V_ITEMIDS,
    "vital_gcs_m": GCS_M_ITEMIDS,
    "vital_temp":  TEMP_ITEMIDS,
    "vital_fio2":  FIO2_ITEMIDS,
    "vital_sbp":   SBP_ITEMIDS,
}

all_vital_itemids = [iid for ids in vital_groups.values() for iid in ids]
vital_itemid_list = ",".join(str(k) for k in all_vital_itemids)

# itemid → group name
itemid_to_group = {}
for group, iids in vital_groups.items():
    for iid in iids:
        itemid_to_group[iid] = group

vital_sql = f"""
SELECT
    c.ICUSTAY_ID,
    c.ITEMID,
    c.VALUENUM
FROM chartevents c
JOIN icustays i ON c.ICUSTAY_ID = i.ICUSTAY_ID
WHERE c.ITEMID IN ({vital_itemid_list})
  AND c.VALUENUM IS NOT NULL
  AND c.VALUENUM > 0
  AND c.CHARTTIME >= i.INTIME
  AND c.CHARTTIME <= i.INTIME + INTERVAL 6 HOUR
  AND i.ICUSTAY_ID IN (
    SELECT FIRST(ICUSTAY_ID ORDER BY INTIME)
    FROM icustays
    GROUP BY SUBJECT_ID
  )
  AND c.ERROR IS NULL
"""
print("  Querying CHARTEVENTS (this will take 1-3 min)...")
vital_raw = query_duckdb(vital_sql)
print(f"  Raw vital rows (first 6h): {len(vital_raw):,}")

# Map itemid → group and pivot
vital_raw["group"] = vital_raw["ITEMID"].map(itemid_to_group)
vital_pivot = vital_raw.groupby(["ICUSTAY_ID", "group"])["VALUENUM"].mean().unstack(fill_value=np.nan).reset_index()
vital_pivot.columns.name = None
cohort = cohort.merge(vital_pivot, on="ICUSTAY_ID", how="left")
print(f"  Vital features added: {[c for c in cohort.columns if c.startswith('vital_')]}")

# %% [markdown]
# ## Step 6: Derived features and encoding

# %%
print("\nStep 6: Derived features...")

# GCS total (sum of verbal + motor; eye opening not reliably coded in one ITEMID)
if "vital_gcs_v" in cohort.columns and "vital_gcs_m" in cohort.columns:
    cohort["vital_gcs_total"] = cohort["vital_gcs_v"].fillna(3) + cohort["vital_gcs_m"].fillna(3)
else:
    cohort["vital_gcs_total"] = np.nan

# Encode gender
cohort["gender_m"] = (cohort["GENDER"] == "M").astype(int)

# One-hot encode diagnosis chapter
diag_dummies = pd.get_dummies(cohort["diag_chapter"], prefix="diag")
cohort = pd.concat([cohort, diag_dummies], axis=1)

# One-hot encode insurance
ins_dummies = pd.get_dummies(cohort["INSURANCE"], prefix="ins")
cohort = pd.concat([cohort, ins_dummies], axis=1)

# One-hot encode admission type
adm_dummies = pd.get_dummies(cohort["ADMISSION_TYPE"], prefix="adm")
cohort = pd.concat([cohort, adm_dummies], axis=1)

# Drop raw string columns
cohort.drop(columns=["GENDER","diag_chapter","INSURANCE","ADMISSION_TYPE"], inplace=True)

print(f"  Total features: {cohort.shape[1]}")
print(f"  Cohort shape: {cohort.shape}")

# %% [markdown]
# ## Step 7: Save and validate

# %%
# Validate key constraints
assert len(cohort) > 5000, f"Expected >5000 stays, got {len(cohort)}"
assert cohort["died"].mean() < 0.40, f"Implausible mortality rate: {cohort['died'].mean():.3f}"
assert cohort["icu_hours"].min() > 0

# Report missingness
miss = cohort.isnull().mean().sort_values(ascending=False)
print("\nTop-10 most missing features:")
print(miss.head(10).to_string())

# Save
out_path = os.path.join(OUT_DIR, "static_features.csv")
cohort.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {cohort.shape}")
print(f"Mortality: {cohort['died'].mean():.3f} ({cohort['died'].sum():.0f} deaths)")
print("\nok patient data complete.")
