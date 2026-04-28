"""
test_phase6.py — tool composition Tool Enrichment & Composition tests.

Run from the clinical_agent/ directory:
    cd /home/dog/VS/clinical-ai-agent/clinical_agent
    /home/dog/VS/clinical-ai-agent/clinicalAgentEnv/bin/python test_phase6.py

Target: >= 20 tests PASSING.
"""


import sys
import warnings
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "clinical_agent"))
warnings.filterwarnings("ignore")

# ── Bootstrap 
import duckdb
import config
from tools import patient_data, nlp_tools, clinical_scores

# Shared DuckDB connection
con = duckdb.connect(str(config.MIMIC_DB), read_only=True)
patient_data.set_db_connection(con)
nlp_tools.set_db_connection(con)
clinical_scores.set_db_connection(con)

# Register with a throw-away FastMCP so @mcp.tool() decoration works
from fastmcp import FastMCP
mcp = FastMCP("test_phase6")
patient_data.register(mcp)
nlp_tools.load_nlp_models(config.SCISPACY_MODEL)
nlp_tools.register(mcp)
clinical_scores.register(mcp)

import asyncio

# ── Helpers 
_PASS = 0
_FAIL = 0


def run(name: str, tool_name: str, **kwargs):
    import json
    try:
        result = asyncio.run(mcp.call_tool(tool_name, kwargs))
        # FastMCP 3.x returns ToolResult with a list of TextContent
        # Try to extract the inner value
        if hasattr(result, "content") and result.content:
            first = result.content[0]
            if hasattr(first, "text"):
                try:
                    return json.loads(first.text)
                except Exception:
                    return {"__raw__": first.text}
        # Fallback: if it's already dict-like (older fastmcp)
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            try:
                return json.loads(str(result))
            except Exception:
                pass
        return result
    except Exception as exc:
        return {"__exception__": str(exc)}



def check(condition: bool, label: str, detail: str = ""):
    global _PASS, _FAIL
    symbol = "ok" if condition else "FAIL:"
    print(f"  {symbol}  {label}" + (f"  [{detail}]" if detail else ""))
    if condition:
        _PASS += 1
    else:
        _FAIL += 1


# 
# 6a — get_patient_summary
# 
print("\n" + "=" * 60)
print("6a  get_patient_summary (patient_id=10006)")
print("=" * 60)

t0 = time.time()
summary = run("get_patient_summary", "get_patient_summary", patient_id=10006)
elapsed = time.time() - t0

# ── structure checks ──
check("error" not in summary, "No error returned")
check("patient_info" in summary, "Has patient_info key")
check("vitals_summary" in summary, "Has vitals_summary key")
check("recent_labs" in summary, "Has recent_labs key")
check("diagnoses" in summary, "Has diagnoses key")

# ── patient_info sub-checks ──
pi = summary.get("patient_info", {})
check(isinstance(pi.get("subject_id"), int), "patient_info.subject_id is int",
      str(pi.get("subject_id")))
check("age_at_admission" in pi, "patient_info has age_at_admission")
check("gender" in pi, "patient_info has gender")

# ── vitals_summary sub-checks ──
vs = summary.get("vitals_summary", {})
check(len(vs) > 0, f"vitals_summary has entries ({len(vs)} vitals)")
if vs:
    first_vital = next(iter(vs.values()))
    check(all(k in first_vital for k in ("mean", "min", "max")),
          "Each vital has mean/min/max keys",
          str(first_vital))

# ── recent_labs sub-checks ──
labs = summary.get("recent_labs", [])
check(len(labs) > 0, f"recent_labs has entries ({len(labs)} records)")
if labs:
    check(all(k in labs[0] for k in ("name", "value")),
          "Lab record has name and value keys")

# ── diagnoses sub-checks ──
dx = summary.get("diagnoses", [])
check(len(dx) > 0, f"diagnoses has entries ({len(dx)} codes)")

# ── latency ──
check(elapsed < 5.0, f"Response time < 5s", f"{elapsed:.2f}s")

# ── unknown patient graceful error ──
bad = run("bad patient summary", "get_patient_summary", patient_id=999999999)
check("error" in bad, "Returns error dict for unknown patient_id")


# 
# 6b — analyze_patient_notes
# 
print("\n" + "=" * 60)
print("6b  analyze_patient_notes (patient_id=10006)")
print("=" * 60)

t0 = time.time()
note_result = run("analyze_patient_notes", "analyze_patient_notes",
                  patient_id=10006, deidentify=True, max_notes=2)
elapsed = time.time() - t0

check("error" not in note_result, "No error returned")
check("notes_analyzed" in note_result, "Has notes_analyzed key")
n_analyzed = note_result.get("notes_analyzed", 0)
check(n_analyzed >= 1, f"notes_analyzed >= 1  ({n_analyzed})")
check("diseases" in note_result, "Has diseases key")
check("chemicals" in note_result, "Has chemicals key")
check("lab_values" in note_result, "Has lab_values key")
check("vitals" in note_result, "Has vitals key")
check("notes_preview" in note_result, "Has notes_preview key")

preview = note_result.get("notes_preview", [])
check(len(preview) >= 1, f"notes_preview has >= 1 entry ({len(preview)})")
if preview:
    p0 = preview[0]
    check(all(k in p0 for k in ("category", "chartdate", "text_snippet")),
          "Preview has category, chartdate, text_snippet keys")

check(elapsed < 60.0, f"Completes within 60s (NLP can be slow)", f"{elapsed:.1f}s")


# 
# 6b  PHI de-identification check
# 
print("\n" + "=" * 60)
print("6b  PHI de-id: patient name must NOT appear in entities")
print("=" * 60)

# Synthesise a note with a fake PHI name and run through de-id + NER
phi_note = (
    "Patient: John Doe  DOB: 01/01/1950  MRN: 12345678\n"
    "Diagnosis: acute myocardial infarction\n"
    "Medications: aspirin 81 mg daily, metoprolol 25 mg\n"
    "Creatinine 1.4 mg/dL  Sodium: 138 mEq/L"
)

# Call deidentify_note directly then NER
deid     = run("phi_deid", "deidentify_note", text=phi_note)
anon_text = deid.get("anonymized_text", phi_note)
ner      = run("phi_ner", "extract_clinical_entities", text=anon_text)

diseases  = [d.lower() for d in ner.get("diseases", [])]
chemicals = [c.lower() for c in ner.get("chemicals", [])]
all_entities = diseases + chemicals

check("john doe" not in all_entities,
      "PHI name 'John Doe' not in NER entities after de-id",
      f"entities={all_entities[:5]}")
check(len(diseases) > 0 or len(chemicals) > 0,
      "Clinical entities (diseases/chemicals) still extracted after de-id",
      f"diseases={diseases[:3]}, chemicals={chemicals[:3]}")


# 
# 6c — calculate_sofa_from_patient
# 
print("\n" + "=" * 60)
print("6c  calculate_sofa_from_patient (patient_id=10006)")
print("=" * 60)

t0 = time.time()
sofa_result = run("sofa_from_patient", "calculate_sofa_from_patient",
                  patient_id=10006, hours=24)
elapsed = time.time() - t0

check("error" not in sofa_result, "No error returned")
check("score" in sofa_result, "Has score key")

if "score" in sofa_result:
    score = sofa_result["score"]
    check(isinstance(score, int), f"score is int ({score})")
    check(0 <= score <= 24, f"score in valid range 0–24 ({score})")

check("risk_level" in sofa_result, "Has risk_level")
check("component_breakdown" in sofa_result, "Has component_breakdown")
check("values_used" in sofa_result, "Has values_used dict")

if "values_used" in sofa_result:
    vu = sofa_result["values_used"]
    check("hours_window" in vu, "values_used has hours_window")
    check("creatinine" in vu, "values_used has creatinine")

check(elapsed < 10.0, f"Completes within 10s", f"{elapsed:.2f}s")


# 
# 6c — calculate_qsofa_from_patient
# 
print("\n" + "=" * 60)
print("6c  calculate_qsofa_from_patient (patient_id=10006)")
print("=" * 60)

t0 = time.time()
qsofa_result = run("qsofa_from_patient", "calculate_qsofa_from_patient",
                   patient_id=10006, hours=24)
elapsed = time.time() - t0

check("error" not in qsofa_result, "No error returned")
check("score" in qsofa_result, "Has score key")

if "score" in qsofa_result:
    score = qsofa_result["score"]
    check(isinstance(score, int), f"score is int ({score})")
    check(0 <= score <= 3, f"score in valid range 0–3 ({score})")

check("values_used" in qsofa_result, "Has values_used dict")

if "values_used" in qsofa_result:
    vu = qsofa_result["values_used"]
    check("respiratory_rate" in vu, "values_used has respiratory_rate")
    check("sbp" in vu, "values_used has sbp")
    check("altered_mentation_inferred" in vu, "values_used has altered_mentation_inferred")

check(elapsed < 5.0, f"Completes within 5s", f"{elapsed:.2f}s")


# 
# 6d — MIMIC whitespace-padded lab table regex
# 
print("\n" + "=" * 60)
print("6d  _LAB_TABLE_PATTERN — MIMIC whitespace table format")
print("=" * 60)

table_note = """\
Pertinent Results:
Sodium                 138
Creatinine             1.4
Platelets              245
WBC                    12.3
Bilirubin              0.9
Glucose                142
"""

ner_table = run("table_ner", "extract_clinical_entities", text=table_note)
table_labs = ner_table.get("lab_values", [])
found_names = {lab["name"].strip().lower() for lab in table_labs}

check("sodium" in found_names,     f"Sodium extracted from table format  ({found_names})")
check("creatinine" in found_names, f"Creatinine extracted from table format")
check("platelets" in found_names,  f"Platelets extracted from table format")
check("wbc" in found_names,        f"WBC extracted from table format")


# 
# Summary
# 
print("\n" + "=" * 60)
total = _PASS + _FAIL
print(f"tool composition Results: {_PASS}/{total} PASSED   {_FAIL} FAILED")
if _FAIL == 0:
    print("🎉  ALL TESTS PASSED")
elif _PASS >= 20:
    print("ok  Target of ≥20 tests met!")
else:
    print(f"FAIL:  Only {_PASS} tests passed — target is ≥20")
print("=" * 60)

sys.exit(0 if _FAIL == 0 else 1)
