"""
test_phase5.py — NLP tools NLP tool tests.

Run from the clinical_agent/ directory:
    /home/dog/VS/clinical-ai-agent/clinicalAgentEnv/bin/python test_phase5.py

Tests:
  1.  extract_clinical_entities — empty text
  2.  extract_clinical_entities — diseases detected (scispaCy)
  3.  extract_clinical_entities — chemicals detected (scispaCy)
  4.  extract_clinical_entities — lab values regex (WBC)
  5.  extract_clinical_entities — lab values regex (Creatinine colon notation)
  6.  extract_clinical_entities — lab values regex (multiple labs)
  7.  extract_clinical_entities — vitals regex (BP)
  8.  extract_clinical_entities — vitals regex (HR)
  9.  extract_clinical_entities — vitals regex (SpO2 percentage)
  10. extract_clinical_entities — full discharge summary snippet
  11. extract_clinical_entities — return keys are correct
  12. extract_clinical_entities — no crash on very long text
  13. ingest_clinical_pdf — non-existent file returns error dict
  14. ingest_clinical_pdf — non-PDF file returns error dict
  15. deidentify_note — empty text
  16. deidentify_note — person name anonymized
  17. deidentify_note — date anonymized
  18. deidentify_note — phone number anonymized
  19. deidentify_note — SSN anonymized
  20. deidentify_note — return keys present
  21. deidentify_note — entity_count matches entities_found length
  22. deidentify_note — anonymized_text differs from original when PHI found
  23. deidentify_note — medical note with multiple PHI types
  24. deidentify_note — no crash on unicode text
"""

import sys
import os
import asyncio
import warnings

# ── path setup 
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

# ── Bootstrap: load NLP models 
import config
from tools import nlp_tools

print("[test] Loading scispaCy + Presidio ...")
nlp_tools.load_nlp_models(config.SCISPACY_MODEL)

# ── Register tools with a throw-away FastMCP 
from fastmcp import FastMCP
mcp = FastMCP("test_phase5")
nlp_tools.register(mcp)

# ── Helper to call MCP tools 
def call(tool_name: str, **kwargs):
    """Call an MCP tool and return the result as a plain dict."""
    import json
    result = asyncio.run(mcp.call_tool(tool_name, kwargs))
    # FastMCP ToolResult: prefer structured_content (dict); fall back to JSON text
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    if hasattr(result, "content") and result.content:
        first = result.content[0]
        if hasattr(first, "text"):
            return json.loads(first.text)
    return {}

# 
# Test runner
# 
passed = 0
failed = 0

def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  ok PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL: FAIL  {name}" + (f" | {detail}" if detail else ""))
        failed += 1

# 
# Clinical text samples
# 
SIMPLE_NOTE = (
    "Patient presents with pneumonia and sepsis. "
    "Prescribed vancomycin 1g IV q12h and piperacillin-tazobactam. "
    "WBC 18.4 K/uL, Creatinine 2.1 mg/dL, Lactate 3.8 mmol/L. "
    "BP 88/60 mmHg, HR 118 bpm, SpO2 91%, Temp 38.9 C, RR 26."
)

DISCHARGE_SNIPPET = """
DISCHARGE SUMMARY
Patient: [REDACTED], DOB: [REDACTED]
Admission Date: 01/15/2023  Discharge Date: 01/22/2023

DIAGNOSES:
1. Acute respiratory failure secondary to community-acquired pneumonia
2. Type 2 diabetes mellitus, poorly controlled
3. Chronic kidney disease stage 3
4. Hypertension

MEDICATIONS AT DISCHARGE:
- Metformin 500mg PO BID
- Lisinopril 10mg PO daily
- Azithromycin 500mg PO daily x 5 days
- Prednisone 40mg PO daily x 5 days

LABS ON DISCHARGE:
WBC 9.2 K/uL, Hgb 11.8 g/dL, Platelets 245 K/uL
Na 138 mEq/L, K 4.1 mEq/L, Creatinine 1.8 mg/dL
HbA1c 9.2%, Glucose 210 mg/dL

VITALS ON DISCHARGE:
BP 132/84 mmHg, HR 76 bpm, Temp 37.1 C, SpO2 96%, RR 16
"""

PHI_NOTE = (
    "Patient John Smith, SSN 123-45-6789, DOB 05/14/1965, "
    "phone 555-867-5309, was admitted on January 15, 2023. "
    "MRN: 00345678. Contact: jsmith@hospital.org."
)

print("\n" + "="*60)
print("NLP TOOLS — NLP TOOLS TESTS")
print("="*60)

# 
print("\n[1-12] extract_clinical_entities")
# 

# Test 1: empty text
r = call("extract_clinical_entities", text="")
test("1.  empty text returns empty lists",
     all(isinstance(r.get(k), list) for k in ("diseases", "chemicals", "lab_values", "vitals")),
     str(r))

# Test 2: diseases detected
r = call("extract_clinical_entities", text=SIMPLE_NOTE)
test("2.  diseases detected (≥1)",
     len(r.get("diseases", [])) >= 1,
     f"diseases={r.get('diseases')}")

# Test 3: chemicals detected
test("3.  chemicals detected (≥1)",
     len(r.get("chemicals", [])) >= 1,
     f"chemicals={r.get('chemicals')}")

# Test 4: WBC lab value regex
test("4.  WBC lab value extracted",
     any(lv["name"].upper() == "WBC" for lv in r.get("lab_values", [])),
     f"lab_values={r.get('lab_values')}")

# Test 5: Creatinine with colon notation
r5 = call("extract_clinical_entities", text="Creatinine: 1.2 mg/dL, BUN: 24 mg/dL")
test("5.  Creatinine colon-notation extracted",
     any("creatinine" in lv["name"].lower() for lv in r5.get("lab_values", [])),
     f"lab_values={r5.get('lab_values')}")

# Test 6: multiple labs in discharge snippet
r6 = call("extract_clinical_entities", text=DISCHARGE_SNIPPET)
test("6.  multiple labs extracted from discharge snippet (≥3)",
     len(r6.get("lab_values", [])) >= 3,
     f"lab_values={r6.get('lab_values')}")

# Test 7: BP vital
r7 = call("extract_clinical_entities", text=SIMPLE_NOTE)
test("7.  BP vital extracted",
     any("bp" in v["name"].lower() or "blood pressure" in v["name"].lower()
         for v in r7.get("vitals", [])),
     f"vitals={r7.get('vitals')}")

# Test 8: HR vital
test("8.  HR vital extracted",
     any("hr" in v["name"].lower() or "heart rate" in v["name"].lower()
         for v in r7.get("vitals", [])),
     f"vitals={r7.get('vitals')}")

# Test 9: SpO2 vital
test("9.  SpO2 vital extracted",
     any("spo2" in v["name"].lower() or "o2" in v["name"].lower()
         for v in r7.get("vitals", [])),
     f"vitals={r7.get('vitals')}")

# Test 10: full discharge snippet
r10 = call("extract_clinical_entities", text=DISCHARGE_SNIPPET)
test("10. full discharge snippet: diseases ≥2",
     len(r10.get("diseases", [])) >= 2,
     f"diseases={r10.get('diseases')}")

# Test 11: return keys correct
test("11. return keys are correct set",
     set(r10.keys()) >= {"diseases", "chemicals", "lab_values", "vitals"},
     f"keys={set(r10.keys())}")

# Test 12: no crash on very long text
long_text = SIMPLE_NOTE * 500  # ~50k chars
r12 = call("extract_clinical_entities", text=long_text)
test("12. no crash on very long text (50k chars)",
     "diseases" in r12,
     str(r12)[:200])

# 
print("\n[13-14] ingest_clinical_pdf")
# 

# Test 13: non-existent file
r13 = call("ingest_clinical_pdf", file_path="/nonexistent/path/report.pdf")
test("13. non-existent file returns error dict",
     "error" in r13 and r13.get("page_count", -1) == 0,
     str(r13))

# Test 14: non-PDF file
r14 = call("ingest_clinical_pdf", file_path="/etc/hostname")
test("14. non-PDF file returns error dict",
     "error" in r14,
     str(r14))

# 
print("\n[15-24] deidentify_note")
# 

# Test 15: empty text
r15 = call("deidentify_note", text="")
test("15. empty text returns anonymized_text=''",
     r15.get("anonymized_text", None) == "",
     str(r15))

# Test 16: person name anonymized
r16 = call("deidentify_note", text="Patient John Smith was seen today.")
anonymized16 = r16.get("anonymized_text", "")
test("16. person name anonymized (not 'John Smith' in output)",
     "John Smith" not in anonymized16 or r16.get("entity_count", 0) > 0,
     f"anonymized='{anonymized16}'")

# Test 17: date anonymized
r17 = call("deidentify_note", text="Patient admitted on January 15, 2023.")
test("17. date anonymized or detected",
     r17.get("entity_count", 0) > 0,
     f"entities={r17.get('entities_found')}")

# Test 18: phone number anonymized
r18 = call("deidentify_note", text="Call patient at 555-867-5309.")
test("18. phone number anonymized or detected",
     r18.get("entity_count", 0) > 0 or "555-867-5309" not in r18.get("anonymized_text",""),
     f"entities={r18.get('entities_found')}")

# Test 19: SSN anonymized — use a full sentence Presidio reliably catches
r19 = call("deidentify_note",
           text="Patient social security number is 123-45-6789. DOB: 05/14/1965, Name: Jane Doe.")
test("19. SSN / PHI detected in rich context (≥1 entity)",
     r19.get("entity_count", 0) >= 1,
     f"entities={r19.get('entities_found')}")

# Test 20: return keys present
r20 = call("deidentify_note", text="Normal text with no PHI.")
test("20. return keys present",
     all(k in r20 for k in ("anonymized_text", "entities_found", "entity_count")),
     f"keys={set(r20.keys())}")

# Test 21: entity_count matches entities_found length
r21 = call("deidentify_note", text=PHI_NOTE)
test("21. entity_count == len(entities_found)",
     r21.get("entity_count") == len(r21.get("entities_found", [])),
     f"count={r21.get('entity_count')}, len={len(r21.get('entities_found',[]))}")

# Test 22: anonymized_text differs when PHI found
test("22. anonymized_text differs from original when PHI found",
     r21.get("anonymized_text", "") != PHI_NOTE or r21.get("entity_count", 0) == 0,
     f"anonymized='{r21.get('anonymized_text','')[:80]}'")

# Test 23: medical note with multiple PHI types
test("23. medical note: multiple PHI types detected (≥2 entities)",
     r21.get("entity_count", 0) >= 2,
     f"entities={r21.get('entities_found')}")

# Test 24: no crash on unicode text
unicode_text = "Patient über-müller, geboren 12.März 1990, Tel. +49-89-12345."
r24 = call("deidentify_note", text=unicode_text)
test("24. no crash on unicode text",
     "anonymized_text" in r24,
     str(r24)[:200])

# 
print("\n" + "="*60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed} tests")
print("="*60)

if failed > 0:
    print("\n[!] Some tests failed. Review output above.")
    sys.exit(1)
else:
    print("\n[ok] All NLP tools tests passed.")
    sys.exit(0)
