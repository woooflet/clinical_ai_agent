# %% [markdown]
# # itemid discovery: ITEMID Discovery
#
# Identify and verify the MIMIC-III ITEMIDs for our target clinical variables.
# Uses literature-validated ITEMIDs (Harutyunyan et al. 2019, Johnson et al. 2017)
# and verifies each against actual row counts in the DuckDB database.

# %%
import subprocess
import pandas as pd
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
    """Run a SQL query against the DuckDB file and return a DataFrame."""
    result = subprocess.run(
        ["duckdb", DB_PATH, "-csv", "-c", sql],
        capture_output=True, text=True, check=True
    )
    return pd.read_csv(StringIO(result.stdout))


# %% [markdown]
# ## Browse d_items and d_labitems for our target variables

# %%
# Check CHARTEVENTS item schema
schema_chart = query_duckdb("SELECT * FROM d_items LIMIT 3")
print("d_items columns:", list(schema_chart.columns))

schema_lab = query_duckdb("SELECT * FROM d_labitems LIMIT 3")
print("d_labitems columns:", list(schema_lab.columns))

# %%
# Search d_items for MAP, GCS, SpO2, HR
chart_hits = query_duckdb("""
    SELECT ITEMID, LABEL, DBSOURCE, CATEGORY, LINKSTO
    FROM d_items
    WHERE LOWER(LABEL) LIKE '%arterial%mean%'
       OR LOWER(LABEL) LIKE '%non invasive blood pressure mean%'
       OR LOWER(LABEL) LIKE '%heart rate%'
       OR LOWER(LABEL) LIKE '%spo2%'
       OR LOWER(LABEL) LIKE '%o2 saturation%'
       OR LOWER(LABEL) LIKE '%gcs%total%'
       OR LOWER(LABEL) LIKE '%glasgow%total%'
       OR LOWER(LABEL) LIKE '%respiratory rate%'
       OR LOWER(LABEL) LIKE '%temperature celsius%'
       OR LOWER(LABEL) LIKE '%fraction of inspired o2%'
    ORDER BY LABEL
""")
print(f"\nCHARTEVENTS candidates ({len(chart_hits)} rows):")
print(chart_hits.to_string(index=False))

# %%
# Search d_labitems
lab_hits = query_duckdb("""
    SELECT ITEMID, LABEL, FLUID, CATEGORY
    FROM d_labitems
    WHERE LOWER(LABEL) LIKE '%creatinine%'
       OR LOWER(LABEL) LIKE '%bilirubin%'
       OR LOWER(LABEL) LIKE '%platelet%'
       OR LOWER(LABEL) LIKE '%lactate%'
       OR LOWER(LABEL) LIKE '%white blood cell%'
       OR LOWER(LABEL) LIKE '%pao2%'
       OR LOWER(LABEL) LIKE '%bicarbonate%'
       OR LOWER(LABEL) LIKE '%potassium%'
       OR LOWER(LABEL) LIKE '%sodium%'
       OR LOWER(LABEL) LIKE '%urea nitrogen%'
    ORDER BY LABEL
""")
print(f"\nLABEVENTS candidates ({len(lab_hits)} rows):")
print(lab_hits.to_string(index=False))

# %% [markdown]
# ## Define final ITEMID set (literature-validated)

# %%
# Literature-validated ITEMIDs for MIMIC-III
# Sources: Harutyunyan et al. 2019 (Benchmark), Johnson et al. 2017
FINAL_CHARTEVENTS_ITEMIDS = {
    220052: "MAP_invasive",    # Arterial Blood Pressure mean (MetaVision)
    220181: "MAP_noninv",      # Non Invasive Blood Pressure mean (MetaVision)
    220210: "RR",              # Respiratory Rate
    220045: "HR",              # Heart Rate
    220179: "SBP",             # Non Invasive Blood Pressure systolic
    223900: "GCS_verbal",      # GCS - Verbal Response (MetaVision)
    223901: "GCS_motor",       # GCS - Motor Response (MetaVision)
    220277: "SpO2",            # O2 saturation pulseoxymetry (MetaVision)
    223762: "Temp_C",          # Temperature Celsius (MetaVision)
    224690: "RR_total",        # Respiratory Rate Total (MetaVision)
    # CareVue equivalents for older patients
    51:    "MAP_cv",           # Arterial BP Mean (CareVue)
    455:   "MAP_cv_noninv",    # NBP Mean (CareVue)
    442:   "MAP_cv_art",       # Manual Blood Pressure Mean(L) (CareVue)
    211:   "HR_cv",            # Heart Rate (CareVue)
    618:   "RR_cv",            # Respiratory Rate (CareVue)
    646:   "SpO2_cv",          # SpO2 (CareVue)
    223835:"FiO2",             # Fraction of Inspired O2 (MetaVision)
}

FINAL_LABEVENTS_ITEMIDS = {
    50912: "Creatinine",
    50885: "Bilirubin_total",
    51265: "Platelets",
    50813: "Lactate",
    51301: "WBC",
    50821: "PaO2",
    50882: "Bicarbonate",
    50971: "Potassium",
    50983: "Sodium",
    51006: "BUN",
    51003: "Troponin_T",
    51237: "INR",
}

# %%
# Build lookup dataframe
rows = []
for itemid, var_name in FINAL_CHARTEVENTS_ITEMIDS.items():
    rows.append({"itemid": itemid, "variable_name": var_name, "source_table": "chartevents"})
for itemid, var_name in FINAL_LABEVENTS_ITEMIDS.items():
    rows.append({"itemid": itemid, "variable_name": var_name, "source_table": "labevents"})

itemid_df = pd.DataFrame(rows)
print(f"\nTotal ITEMIDs in lookup: {len(itemid_df)}")
print(itemid_df.to_string(index=False))

# %% [markdown]
# ## Verify row counts in the actual tables

# %%
print("\nVerifying row counts in database...")
counts = []
for _, row in itemid_df.iterrows():
    table = row["source_table"]
    itemid = row["itemid"]
    sql = f"SELECT COUNT(*) AS cnt FROM {table} WHERE ITEMID = {itemid}"
    try:
        cnt_df = query_duckdb(sql)
        cnt = int(cnt_df["cnt"].iloc[0])
        counts.append(cnt)
        print(f"  {row['variable_name']:20s} itemid={itemid:6d}: {cnt:>10,} rows")
    except Exception as e:
        print(f"  {row['variable_name']:20s}: ERROR — {e}")
        counts.append(0)

itemid_df["row_count"] = counts

# %%
# Filter out any ITEMIDs with 0 rows (not in this database version)
valid_itemids = itemid_df[itemid_df["row_count"] > 0].copy()
print(f"\n{len(valid_itemids)}/{len(itemid_df)} ITEMIDs have data in this DB")

# Save the verified lookup
output_path = os.path.join(OUT_DIR, "itemid_lookup.csv")
valid_itemids.to_csv(output_path, index=False)
print(f"Saved: {output_path}")

# Assertions
assert len(valid_itemids) >= 15, f"Expected >=15 valid ITEMIDs, got {len(valid_itemids)}"
print("\nok itemid discovery complete.")
