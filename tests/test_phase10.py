"""
test_phase10.py — agentic loop: Agentic Loop Integration Tests

Runs 6 end-to-end clinical scenarios through the full Nemotron ↔ MCP tool
agentic loop.  For each scenario we assert:
  • The final response is a non-empty string.
  • At least one MCP tool was called.
  • No unhandled Python exceptions.

We do NOT assert which specific tools are called — Nemotron decides that.

Demo patient: subject_id=27513 (sepsis survivor, 54 notes, high-acuity cluster,
              confirmed present in MIMIC, static_features.csv, temporal_meta.csv,
              and temporal_sequences.pkl).

Run from the repo root:
    ./run_tests.sh
    ./run_tests.sh --module test_phase10

If the LLM endpoint (LLM server) is unreachable the stub path is exercised and
all connectivity-dependent checks are skipped gracefully.
"""

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure local imports work regardless of CWD
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# GPU 0 for LSTM / ClinicalBERT
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# --- Bootstrap 
print("=" * 65)
print("agentic loop — Agentic Loop Integration Tests")
print("=" * 65)

import agent_client

print("\n[Bootstrap] Starting server assets …")
t0 = time.time()
agent_client.build_agent()
boot_secs = time.time() - t0
print(f"[Bootstrap] Done in {boot_secs:.1f}s")

# --- Connectivity probe 
try:
    agent_client._get_llm_client().models.list()
    LLM_REACHABLE = True
    print(f"[Bootstrap] LLM endpoint reachable ok")
except Exception as e:
    LLM_REACHABLE = False
    print(f"[Bootstrap] LLM endpoint UNREACHABLE: {e}")
    print("           Tests will exercise stub path — tool-call count checks skipped.")

# --- Tool count sanity check 
import asyncio
tools_registered = asyncio.run(agent_client.mcp.list_tools())
print(f"[Bootstrap] {len(tools_registered)} tools registered")

# --- Test harness 
passed = 0
failed = 0
errors: list[str] = []

DEMO_PID = 27513  # Primary demo patient (sepsis survivor)


def ok(name: str) -> None:
    global passed
    passed += 1
    print(f"  ok {name}")


def fail(name: str, reason: str = "") -> None:
    global failed
    failed += 1
    msg = f"{name}: {reason}" if reason else name
    errors.append(msg)
    print(f"  FAIL {msg}")


def check(name: str, cond: bool, reason: str = "") -> None:
    if cond:
        ok(name)
    else:
        fail(name, reason or "assertion failed")


# --- Scenario runner 

SCENARIOS: list[dict] = [
    {
        "id": "S1",
        "label": "Patient overview",
        "query": f"Give me a complete overview of patient {DEMO_PID}.",
        "min_tools": 1,
    },
    {
        "id": "S2",
        "label": "Mortality risk + explanation",
        "query": (
            f"Analyze patient {DEMO_PID}'s mortality risk and explain "
            "the key clinical drivers behind the prediction."
        ),
        "min_tools": 1,
    },
    {
        "id": "S3",
        "label": "Sepsis scoring",
        "query": (
            f"Is patient {DEMO_PID} at risk for sepsis? "
            "Calculate the relevant sepsis scores and interpret them."
        ),
        "min_tools": 1,
    },
    {
        "id": "S4",
        "label": "Clinical notes NLP",
        "query": (
            f"Summarize the clinical notes for patient {DEMO_PID}. "
            "What diseases and medications are documented?"
        ),
        "min_tools": 1,
    },
    {
        "id": "S5",
        "label": "Phenotype cluster + similar patients",
        "query": (
            f"What phenotype cluster is patient {DEMO_PID} in? "
            "Find similar historical patients and describe the cluster's prognosis."
        ),
        "min_tools": 1,
    },
    {
        "id": "S6",
        "label": "Literature: sepsis management",
        "query": (
            "What does recent clinical literature say about sepsis management "
            "in ICU patients? Summarize key evidence-based recommendations."
        ),
        "min_tools": 1,
    },
]


def run_scenario(scenario: dict) -> dict:
    """Run one scenario and return results dict."""
    sid   = scenario["id"]
    label = scenario["label"]
    query = scenario["query"]

    print('')
    print(f"[{sid}] {label}")
    print(f"Query: {query}")
    print('-' * 60)

    t_start = time.time()
    try:
        response, trace = agent_client.run_agent(
            user_query=query,
            max_turns=15,
            verbose=True,
        )
        elapsed = time.time() - t_start
        return {
            "ok": True,
            "response": response,
            "trace": trace,
            "elapsed": elapsed,
        }
    except Exception as exc:
        elapsed = time.time() - t_start
        import traceback
        tb = traceback.format_exc(limit=5)
        print(f"  [EXCEPTION] {exc}\n{tb}")
        return {
            "ok": False,
            "response": "",
            "trace": [],
            "elapsed": elapsed,
            "exception": str(exc),
        }


# --- Run all scenarios 
print("\n\n[A] Tool registry checks")
check("≥ 40 tools registered", len(tools_registered) >= 40,
      f"got {len(tools_registered)}")
check("get_patient_summary registered",
      any(t.name == "get_patient_summary" for t in tools_registered))
check("predict_mortality registered",
      any(t.name == "predict_mortality" for t in tools_registered))
check("calculate_sofa_from_patient registered",
      any(t.name == "calculate_sofa_from_patient" for t in tools_registered))
check("analyze_patient_notes registered",
      any(t.name == "analyze_patient_notes" for t in tools_registered))
check("get_phenotype registered",
      any(t.name == "get_phenotype" for t in tools_registered))
check("search_clinical_literature_rag registered",
      any(t.name == "search_clinical_literature_rag" for t in tools_registered))

# --- OpenAI tool schema checks 
print("\n[B] OpenAI tool schema conversion")
try:
    oai_tools = agent_client.get_openai_tools()
    check("get_openai_tools returns list", isinstance(oai_tools, list))
    check("same count as registered tools",
          len(oai_tools) == len(tools_registered),
          f"openai={len(oai_tools)} vs registered={len(tools_registered)}")
    if oai_tools:
        t0_schema = oai_tools[0]
        check("each entry has type='function'",
              all(t.get("type") == "function" for t in oai_tools))
        check("each entry has function.name",
              all(t.get("function", {}).get("name") for t in oai_tools))
        check("each entry has function.parameters",
              all("parameters" in t.get("function", {}) for t in oai_tools))
except Exception as exc:
    fail("B1", str(exc)); fail("B2",""); fail("B3",""); fail("B4",""); fail("B5","")

# --- Per-scenario integration tests 
print("\n[C] Scenario integration tests")

scenario_results: list[dict] = []
for scenario in SCENARIOS:
    result = run_scenario(scenario)
    scenario_results.append(result)
    sid   = scenario["id"]
    label = scenario["label"]
    min_t = scenario["min_tools"]

    check(f"C-{sid}: no exception",
          result["ok"],
          result.get("exception", ""))

    check(f"C-{sid}: response is non-empty string",
          isinstance(result["response"], str) and len(result["response"]) > 0,
          f"got empty response (is_stub={'STUB' in result['response']})")

    if LLM_REACHABLE:
        check(f"C-{sid}: ≥ {min_t} tool(s) called",
              len(result["trace"]) >= min_t,
              f"tool_calls={len(result['trace'])}")
    else:
        # Stub path: no tool calls made, but the loop still worked
        ok(f"C-{sid}: ≥ {min_t} tool(s) called [SKIPPED — LLM unreachable, stub used]")

    print(f"\n  [{sid}] Summary: "
          f"tools_called={len(result['trace'])}, "
          f"response_len={len(result['response'])} chars, "
          f"elapsed={result['elapsed']:.1f}s")

# --- Tool trace quality checks 
print("\n[D] Tool trace quality (first LLM-driven scenario with tools)")
if LLM_REACHABLE:
    # Find first scenario that actually called tools
    traced = next((r for r in scenario_results if r["ok"] and r["trace"]), None)
    if traced:
        trace = traced["trace"]
        check("trace entries have 'tool' key",
              all("tool" in t for t in trace))
        check("trace entries have 'args' key",
              all("args" in t for t in trace))
        check("trace entries have 'result_snippet' key",
              all("result_snippet" in t for t in trace))
        check("trace entries have 'latency_ms' key",
              all("latency_ms" in t for t in trace))
        check("all latencies are positive numbers",
              all(isinstance(t["latency_ms"], (int, float)) and t["latency_ms"] > 0
                  for t in trace))
    else:
        for i in range(1, 6):
            ok(f"D{i}: skip (no LLM tool calls in any scenario)")
else:
    for i in range(1, 6):
        ok(f"D{i}: skip (LLM unreachable)")

# --- Demo patient reachability spot-check 
print(f"\n[E] Demo patient spot-check (subject_id={DEMO_PID})")
try:
    import config, duckdb
    _con = duckdb.connect(str(config.MIMIC_DB), read_only=True)
    row = _con.execute(
        f"SELECT subject_id FROM admissions WHERE subject_id={DEMO_PID} LIMIT 1"
    ).fetchone()
    check("demo patient exists in MIMIC admissions", row is not None,
          f"subject_id={DEMO_PID} not found")
    _con.close()
except Exception as exc:
    fail("E1", str(exc))

# Check static features
try:
    import pandas as pd
    sf = pd.read_csv(config.STATIC_FEATURES_CSV, usecols=["SUBJECT_ID"])
    check("demo patient in static_features.csv",
          DEMO_PID in sf["SUBJECT_ID"].values,
          f"subject_id={DEMO_PID} missing")
except Exception as exc:
    fail("E2", str(exc))

# Check temporal meta
try:
    tm = pd.read_csv(config.TEMPORAL_META_CSV)
    col = "subject_id" if "subject_id" in tm.columns else "SUBJECT_ID"
    check("demo patient in temporal_meta.csv",
          DEMO_PID in tm[col].values,
          f"subject_id={DEMO_PID} missing")
except Exception as exc:
    fail("E3", str(exc))

# --- Summary 
total = passed + failed
print(f"\n{'=' * 65}")
print(f"agentic loop Results: {passed}/{total} tests PASSED")
if not LLM_REACHABLE:
    print("NOTE: LLM endpoint was UNREACHABLE — stub path was exercised.")
    print("      Re-run when LLM server is online for full live validation.")

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  FAIL {e}")

print("\nScenario summary:")
for scenario, result in zip(SCENARIOS, scenario_results):
    status = "ok" if result["ok"] and (
        not LLM_REACHABLE or len(result["trace"]) >= scenario["min_tools"]
    ) else "FAIL"
    tools_n = len(result["trace"])
    resp_n  = len(result["response"])
    print(
        f"  {status} [{scenario['id']}] {scenario['label']:<35} "
        f"tools={tools_n} resp={resp_n}chars {result['elapsed']:.1f}s"
    )

print("=" * 65)
sys.exit(0 if failed == 0 else 1)
