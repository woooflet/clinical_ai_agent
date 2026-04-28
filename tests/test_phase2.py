"""
test_phase2.py — Quick smoke test for all 9 patient data patient_data tools.

Run from the repo root:
    ./run_tests.sh
    ./run_tests.sh --module test_phase2

Uses subject_id=10006 (confirmed in MIMIC-III) for all tests.
"""

import sys, warnings, json, traceback
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "clinical_agent"))
warnings.filterwarnings("ignore")

import asyncio
import duckdb
import config
from tools import patient_data
from fastmcp import FastMCP

# ── Inject connection 
print("Connecting to DuckDB …")
con = duckdb.connect(str(config.MIMIC_DB), read_only=True)
patient_data.set_db_connection(con)
print("connected to DuckDB\n")

# ── Register tools on a throwaway FastMCP instance 
mcp = FastMCP("patient data-test")
patient_data.register(mcp)

async def get_tool_names():
    tools = await mcp.list_tools()
    return sorted(t.name for t in tools)

tool_names = asyncio.run(get_tool_names())
print(f"Tools registered ({len(tool_names)}): {tool_names}\n")

PATIENT_ID = 10006  # Confirmed MIMIC patient

PASSES = 0
FAILS  = 0

async def main():
    global PASSES, FAILS

    async def call(name, **kwargs):
        global PASSES, FAILS
        label = f"{name}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"
        print('-' * 60)
        print(f"  {label}")
        print('-' * 60)
        try:
            result = await mcp.call_tool(name, kwargs)
            # FastMCP 3.x returns ToolResult with .content list of TextContent
            raw   = result.content[0].text
            parsed = json.loads(raw)
            snippet = json.dumps(parsed, default=str, indent=2)
            if len(snippet) > 2000:
                snippet = snippet[:2000] + "\n  … [truncated]"
            print(snippet)
            print(f"PASSED\n")
            PASSES += 1
            return parsed
        except Exception:
            traceback.print_exc()
            print(f"FAILED\n")
            FAILS += 1
            return None

    # ── 1. get_patient_info 
    r = await call("get_patient_info", patient_id=PATIENT_ID)
    assert r and "error" not in r, "get_patient_info returned error"
    assert r.get("gender") in ("M", "F"), "gender should be M or F"
    assert "age_at_admission" in r, "age_at_admission missing"

    # ── 2. get_patient_labs 
    r = await call("get_patient_labs", patient_id=PATIENT_ID, hours=48)
    assert r and "labs" in r, "get_patient_labs missing 'labs' key"
    print(f"    → {r['total_records']} lab records, {len(r['labs'])} distinct tests")

    # ── 3. get_patient_vitals 
    r = await call("get_patient_vitals", patient_id=PATIENT_ID, hours=72)
    assert r and "vitals" in r, "get_patient_vitals missing 'vitals' key"
    print(f"    → {r['total_records']} vital records, types: {list(r['vitals'].keys())}")

    # ── 4. get_patient_medications 
    r = await call("get_patient_medications", patient_id=PATIENT_ID)
    assert isinstance(r, list), "get_patient_medications should return a list"
    print(f"    → {len(r)} prescription records")

    # ── 5. get_patient_notes (all categories) 
    r = await call("get_patient_notes", patient_id=PATIENT_ID)
    assert isinstance(r, list), "get_patient_notes should return a list"
    cats = [n.get("category") for n in r if "category" in n]
    print(f"    → {len(r)} notes, categories: {cats}")

    # ── 6. get_patient_notes (filtered by category) 
    r = await call("get_patient_notes", patient_id=PATIENT_ID, category="Discharge summary")
    assert isinstance(r, list), "get_patient_notes (filtered) should return a list"

    # ── 7. get_patient_diagnoses 
    r = await call("get_patient_diagnoses", patient_id=PATIENT_ID)
    assert isinstance(r, list), "get_patient_diagnoses should return a list"
    print(f"    → {len(r)} diagnosis codes")

    # ── 8. get_patient_microbiology 
    r = await call("get_patient_microbiology", patient_id=PATIENT_ID)
    assert isinstance(r, list), "get_patient_microbiology should return a list"
    print(f"    → {len(r)} microbiology records")

    # ── 9. query_cohort_sql — valid query 
    r = await call(
        "query_cohort_sql",
        sql=f"SELECT subject_id, hadm_id, admission_type FROM admissions "
            f"WHERE subject_id={PATIENT_ID} LIMIT 3",
    )
    assert isinstance(r, list) and len(r) > 0, "query_cohort_sql should return rows"
    assert r[0].get("subject_id") == PATIENT_ID, "subject_id mismatch"

    # ── 9b. query_cohort_sql — auto-LIMIT injection (no LIMIT in query) 
    r = await call(
        "query_cohort_sql",
        sql="SELECT subject_id FROM patients",   # no LIMIT — should be injected
    )
    assert isinstance(r, list) and len(r) <= 500, "auto-LIMIT should cap at 500"

    # ── 9c. query_cohort_sql — mutation blocked 
    r = await call("query_cohort_sql", sql="DROP TABLE admissions")
    assert r and "error" in r[0], "query_cohort_sql should block mutations"

    # ── 10. get_icd9_description — exact code (no dot) 
    r = await call("get_icd9_description", code="41401")
    assert r and "short_title" in r, "get_icd9_description missing short_title"
    print(f"    → {r.get('short_title')} | {r.get('category')}")

    # ── 10b. get_icd9_description — dotted code 
    r = await call("get_icd9_description", code="414.01")
    assert r and "short_title" in r, "get_icd9_description (dotted) missing short_title"

    print()
    print("=" * 64)
    print(f"  PATIENT DATA COMPLETE — {PASSES} PASSED  |  {FAILS} FAILED")
    print("=" * 64)

asyncio.run(main())
