"""
test_phase8.py — ML models test suite: CPU ML model tools.

Tests: predict_mortality, explain_prediction, get_phenotype, find_similar_patients.

Run from the repo root:
    ./run_tests.sh
    ./run_tests.sh --module test_phase8

Target: ≥ 20/20 tests PASS.
"""

import sys, warnings, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
warnings.filterwarnings("ignore")

import config
from tools import ml_models

# --- Load models (mirrors server.py startup) 
ml_models.load_models(
    xgb_path   = config.XGB_MODEL_PATH,
    shap_path  = config.SHAP_EXPLAINER_PATH,
    static_csv = config.STATIC_FEATURES_CSV,
    meta_csv   = config.TEMPORAL_META_CSV,
)

# Register tools into a test FastMCP instance so we can call them
from fastmcp import FastMCP
mcp = FastMCP("test")
ml_models.register(mcp)

import asyncio

def call(tool_name: str, **kwargs):
    """Synchronously invoke an MCP tool and return the result dict."""
    import json
    result = asyncio.run(mcp.call_tool(tool_name, kwargs))
    # FastMCP newer versions return a ToolResult with .structured_content
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    # Fall back: parse from first content item's text
    if hasattr(result, "content") and result.content:
        first = result.content[0]
        if hasattr(first, "text"):
            return json.loads(first.text)
    # Old style: list of content objects
    if isinstance(result, list):
        if hasattr(result[0], "text"):
            return json.loads(result[0].text)
        return result
    return result

# --- Test helpers 
PASS_COUNT = 0
FAIL_COUNT = 0

def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    suffix = f"  ← {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")

# --- Pick a test patient that exists in the static cohort 
# Use subject_id=10006 (confirmed MIMIC patient from patient data tests)
# and a second patient from the temporal meta
import pandas as pd
static_df = pd.read_csv(config.STATIC_FEATURES_CSV)
static_df.columns = [c.lower() for c in static_df.columns]
meta_df   = pd.read_csv(config.TEMPORAL_META_CSV)

# Patient that must be in static features
TEST_PID_STATIC = int(static_df["subject_id"].iloc[0])
# Patient in both static and temporal meta
shared = set(static_df["subject_id"].astype(int)) & set(meta_df["subject_id"].astype(int))
TEST_PID_BOTH = next(iter(shared))

print(f"\n[ML models Tests] Using static patient: {TEST_PID_STATIC}, "
      f"shared patient: {TEST_PID_BOTH}")
print("=" * 60)

# 
# 1. predict_mortality
# 
print("\n▶ predict_mortality")
t0  = time.perf_counter()
r   = call("predict_mortality", patient_id=TEST_PID_STATIC)
lat = (time.perf_counter() - t0) * 1000
print(f"  result: {r}")

check("No error key",              "error" not in r)
check("patient_id in result",      r.get("patient_id") == TEST_PID_STATIC)
check("mortality_probability float",
      isinstance(r.get("mortality_probability"), float),
      str(r.get("mortality_probability")))
check("probability in [0, 1]",
      0.0 <= r.get("mortality_probability", -1) <= 1.0)
check("mortality_label is string", isinstance(r.get("mortality_label"), str))
check("model_type is string",      isinstance(r.get("model_type"), str))
check("latency < 500ms",           lat < 500, f"{lat:.1f}ms")

# --- Missing patient 
r_miss = call("predict_mortality", patient_id=999999999)
check("Missing patient returns error key", "error" in r_miss)

# 
# 2. explain_prediction
# 
print("\n▶ explain_prediction")
t0  = time.perf_counter()
r   = call("explain_prediction", patient_id=TEST_PID_STATIC)
lat = (time.perf_counter() - t0) * 1000
print(f"  result keys: {list(r.keys())}")

check("No error key",              "error" not in r)
check("patient_id present",        r.get("patient_id") == TEST_PID_STATIC)
check("base_value is float",       isinstance(r.get("base_value"), float))
check("prediction is float",       isinstance(r.get("prediction"), float))
check("top_features is list",      isinstance(r.get("top_features"), list))
check("≥ 1 feature in top_features",
      len(r.get("top_features", [])) >= 1,
      str(len(r.get("top_features", []))))
check("top_features ≤ 10",         len(r.get("top_features", [])) <= 10)

feats = r.get("top_features", [])
if feats:
    f0 = feats[0]
    check("feature has 'feature' key",    "feature" in f0)
    check("feature has 'shap_value' key", "shap_value" in f0)
    check("feature has 'direction' key",  "direction" in f0)
    check("direction is valid string",
          f0.get("direction") in ("increases risk", "decreases risk"),
          str(f0.get("direction")))
else:
    for label in ["feature has 'feature' key", "feature has 'shap_value' key",
                  "feature has 'direction' key", "direction is valid string"]:
        check(label, False, "empty top_features")

check("latency < 500ms", lat < 500, f"{lat:.1f}ms")

# --- Missing patient 
r_miss = call("explain_prediction", patient_id=999999999)
check("Missing patient returns error", "error" in r_miss)

# 
# 3. get_phenotype
# 
print("\n▶ get_phenotype")
t0  = time.perf_counter()
r   = call("get_phenotype", patient_id=TEST_PID_BOTH)
lat = (time.perf_counter() - t0) * 1000
print(f"  result: {r}")

check("No error key",                 "error" not in r)
check("patient_id present",           r.get("patient_id") == TEST_PID_BOTH)
check("cluster_id is int 0/1/2",
      r.get("cluster_id") in (0, 1, 2),
      str(r.get("cluster_id")))
check("umap_x is float",              isinstance(r.get("umap_x"), float))
check("umap_y is float",              isinstance(r.get("umap_y"), float))
check("cluster_description non-empty",
      isinstance(r.get("cluster_description"), str) and len(r.get("cluster_description", "")) > 3)
check("cluster_size > 0",            r.get("cluster_size", 0) > 0)
check("cluster_mortality_pct float",  isinstance(r.get("cluster_mortality_pct"), float))
check("latency < 500ms",              lat < 500, f"{lat:.1f}ms")

# --- Missing patient 
r_miss = call("get_phenotype", patient_id=999999999)
check("Missing patient returns error", "error" in r_miss)

# 
# 4. find_similar_patients
# 
print("\n▶ find_similar_patients")
t0  = time.perf_counter()
r   = call("find_similar_patients", patient_id=TEST_PID_STATIC, k=5)
lat = (time.perf_counter() - t0) * 1000
print(f"  result keys: {list(r.keys())}, neighbours: {len(r.get('neighbours', []))}")

check("No error key",               "error" not in r)
check("patient_id present",         r.get("patient_id") == TEST_PID_STATIC)
check("neighbours is list",         isinstance(r.get("neighbours"), list))
check("returned ≤ k neighbours",    len(r.get("neighbours", [])) <= 5)
check("returned ≥ 1 neighbour",     len(r.get("neighbours", [])) >= 1)

nbrs = r.get("neighbours", [])
if nbrs:
    n0 = nbrs[0]
    check("neighbour has subject_id",      "subject_id" in n0)
    check("neighbour has distance",        "distance" in n0)
    check("distance is non-negative float",
          isinstance(n0.get("distance"), float) and n0["distance"] >= 0,
          str(n0.get("distance")))
    check("query patient not in neighbours",
          all(n["subject_id"] != TEST_PID_STATIC for n in nbrs))
else:
    for label in ["neighbour has subject_id", "neighbour has distance",
                  "distance is non-negative float", "query patient not in neighbours"]:
        check(label, False, "empty neighbours list")

# Test default k=10
r10 = call("find_similar_patients", patient_id=TEST_PID_STATIC)
check("default k=10 returns ≤ 10 neighbours",
      len(r10.get("neighbours", [])) <= 10)
check("default k=10 returns ≥ 5 neighbours",
      len(r10.get("neighbours", [])) >= 5)

check("latency < 500ms", lat < 500, f"{lat:.1f}ms")

# --- Missing patient 
r_miss = call("find_similar_patients", patient_id=999999999)
check("Missing patient returns error", "error" in r_miss)

# 
# Summary
# 
total = PASS_COUNT + FAIL_COUNT
print(f"\n{'='*60}")
print(f"ML models Results: {PASS_COUNT}/{total} tests PASSED")
if FAIL_COUNT == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAIL: {FAIL_COUNT} test(s) FAILED")
print("=" * 60)
