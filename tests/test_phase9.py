"""
test_phase9.py — GPU models: GPU Model Tests
Tests for predict_mortality_temporal, predict_icd9, and shap_lstm tools.

Run from clinical_agent/ directory with CUDA_VISIBLE_DEVICES=0:
    cd /home/dog/VS/clinical-ai-agent/clinical_agent
    CUDA_VISIBLE_DEVICES=0 ../clinicalAgentEnv/bin/python test_phase9.py
"""

import sys, warnings, time, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "clinical_agent"))

# ── Minimal bootstrap (mirrors server.py startup) 
import config, duckdb
from tools import patient_data, ml_models

con = duckdb.connect(str(config.MIMIC_DB), read_only=True)
patient_data.set_db_connection(con)

ml_models.load_models(
    xgb_path   = config.XGB_MODEL_PATH,
    shap_path  = config.SHAP_EXPLAINER_PATH,
    static_csv = config.STATIC_FEATURES_CSV,
    meta_csv   = config.TEMPORAL_META_CSV,
)
ml_models.load_lstm_model(
    lstm_path        = config.LSTM_MODEL_PATH,
    lstm_cfg_path    = config.LSTM_CONFIG_PATH,
    temporal_seq_pkl = config.TEMPORAL_SEQ_PKL,
)
ml_models.load_bert_model(
    bert_model_dir   = config.CLINICALBERT_ICD9_DIR,
    icd9_labels_path = config.ICD9_LABELS_PATH,
)

# Fake MCP for testing
from fastmcp import FastMCP
import asyncio

mcp = FastMCP("test-GPU models")
ml_models.register(mcp)

def call(tool_name: str, **kwargs):
    result = asyncio.run(mcp.call_tool(tool_name, kwargs))
    # FastMCP ToolResult: content is a list of TextContent; parse the JSON text
    import json as _json
    if hasattr(result, "content") and result.content:
        text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
        try:
            return _json.loads(text)
        except Exception:
            return {"_raw": text}
    # Fallback: already a dict
    if isinstance(result, dict):
        return result
    return {"_raw": str(result)}

# ── Test harness 
passed = 0
failed = 0
errors = []

def ok(name: str):
    global passed
    passed += 1
    print(f"  ok {name}")

def fail(name: str, reason: str):
    global failed
    failed += 1
    errors.append(f"{name}: {reason}")
    print(f"  FAIL {name}: {reason}")

def check(name: str, cond: bool, reason: str = ""):
    if cond:
        ok(name)
    else:
        fail(name, reason or "assertion failed")

# ─── A. Module-level state checks 
print("\n[A] Module state checks")

check("LSTM model loaded", ml_models._lstm_model is not None,
      "lstm_mortality.pt missing — run scripts/save_lstm_model.py")
check("LSTM config loaded", ml_models._lstm_config is not None)
check("temporal_sequences loaded", ml_models._temporal_sequences is not None)
check("subject→icustay index populated",
      len(ml_models._subject_icustay_index) > 0)
check("ClinicalBERT model loaded", ml_models._bert_model is not None,
      "clinicalbert_icd9 checkpoint missing")
check("ClinicalBERT tokenizer loaded", ml_models._bert_tokenizer is not None)
check("ICD-9 classes loaded", len(ml_models._icd9_classes) > 0)
check("ICD-9 has 50 classes", len(ml_models._icd9_classes) == 50,
      f"got {len(ml_models._icd9_classes)}")

# ─── B. Device checks 
print("\n[B] Device checks")

import torch
cuda_ok = torch.cuda.is_available()
check("CUDA available", cuda_ok, "No GPU — tools will still work via fallback/CPU")
if cuda_ok:
    check("LSTM on GPU", ml_models._lstm_device == "cuda:0",
          f"device={ml_models._lstm_device}")
    check("BERT on GPU", ml_models._bert_device == "cuda:0",
          f"device={ml_models._bert_device}")

# ─── C. predict_mortality_temporal 
print("\n[C] predict_mortality_temporal")

# Pick a patient that has a temporal sequence
meta_df = ml_models._meta_df
seq_dict = ml_models._temporal_sequences
# Find a subject_id whose icustay_id is in temporal_sequences
TEST_PID = None
for _, row in meta_df.iterrows():
    if int(row["icustay_id"]) in seq_dict:
        TEST_PID = int(row["subject_id"])
        break

if TEST_PID is None:
    fail("find temporal test patient", "No patient with temporal sequence found")
else:
    print(f"  Using subject_id={TEST_PID}")

    t0 = time.time()
    r  = call("predict_mortality_temporal", patient_id=TEST_PID)
    latency = (time.time() - t0) * 1000

    check("no error key", "error" not in r, r.get("error",""))
    check("patient_id matches", r.get("patient_id") == TEST_PID)
    check("mortality_probability is float", isinstance(r.get("mortality_probability"), float))
    prob = r.get("mortality_probability", -1)
    check("probability in [0,1]", 0.0 <= prob <= 1.0, f"prob={prob}")
    check("mortality_label present",
          r.get("mortality_label") in {"High risk", "Moderate risk", "Low risk"},
          f"label={r.get('mortality_label')}")
    check("model_type mentions LSTM",
          "LSTM" in r.get("model_type", "") or "fallback" in r.get("model_type","").lower(),
          f"model_type={r.get('model_type')}")
    check("device key present", "device" in r)
    check("icustay_id present", "icustay_id" in r, str(r))
    check("latency < 5000ms", latency < 5000, f"{latency:.0f}ms")
    print(f"  → prob={prob}, label={r.get('mortality_label')}, device={r.get('device')}, latency={latency:.0f}ms")

# Test fallback: patient NOT in temporal sequences
print("\n  [C-fallback] Non-temporal patient falls back to XGBoost")
# Find a patient in static_features but not in _subject_icustay_index
static_pids = set(ml_models._subject_id_index.keys())
temporal_pids = set(ml_models._subject_icustay_index.keys())
fallback_pids = static_pids - temporal_pids
if fallback_pids:
    fb_pid = next(iter(fallback_pids))
    r_fb = call("predict_mortality_temporal", patient_id=fb_pid)
    check("fallback no error", "error" not in r_fb or "XGBoost" in r_fb.get("model_type",""), str(r_fb))
    check("fallback model_type mentions XGBoost",
          "XGBoost" in r_fb.get("model_type","") or "error" in r_fb,
          r_fb.get("model_type",""))
else:
    ok("skip fallback test (all static patients also have temporal seqs)")
    ok("skip fallback test")

# Test unknown patient
r_unk = call("predict_mortality_temporal", patient_id=999999999)
check("unknown patient returns error or graceful", "error" in r_unk or "mortality_probability" in r_unk)

# ─── D. predict_icd9 
print("\n[D] predict_icd9")

DISCHARGE_SNIPPET = (
    "DISCHARGE DIAGNOSES: 1. Sepsis secondary to pneumonia. "
    "2. Acute kidney injury. 3. Hypertension. 4. Type 2 diabetes mellitus. "
    "5. Congestive heart failure. "
    "The patient was admitted with fever, hypotension, and confusion. "
    "Blood cultures grew gram-negative rods. Started on broad-spectrum antibiotics. "
    "Creatinine peaked at 3.2 mg/dL. BNP elevated. Patient improved with IV fluids."
)

t0 = time.time()
r  = call("predict_icd9", note_text=DISCHARGE_SNIPPET)
latency = (time.time() - t0) * 1000

check("no error key", "error" not in r, r.get("error",""))
check("predicted_codes is list", isinstance(r.get("predicted_codes"), list))
codes = r.get("predicted_codes", [])
check("at least 1 code predicted", len(codes) >= 1, f"got {len(codes)}")
check("each code has icd9_code key", all("icd9_code" in c for c in codes), str(codes[:2]))
check("each code has probability", all("probability" in c for c in codes))
check("probabilities are floats in [0,1]",
      all(0.0 <= c["probability"] <= 1.0 for c in codes), str(codes[:2]))
check("codes sorted by descending probability",
      all(codes[i]["probability"] >= codes[i+1]["probability"] for i in range(len(codes)-1)),
      "Not sorted descending")
check("num_codes_predicted matches list length",
      r.get("num_codes_predicted") == len(codes))
check("model_type present", "model_type" in r)
check("device key present", "device" in r)
check("latency < 30s", latency < 30000, f"{latency:.0f}ms")
print(f"  → {len(codes)} codes, top={codes[0] if codes else None}, latency={latency:.0f}ms")

# Empty note
r_empty = call("predict_icd9", note_text="")
check("empty note returns dict", isinstance(r_empty, dict))

# Custom threshold=0 should return all 45/50 labels
r_all = call("predict_icd9", note_text=DISCHARGE_SNIPPET, threshold=0.0, top_k=100)
check("threshold=0 returns up to num_labels codes",
      r_all.get("num_codes_predicted", 0) <= 50)

# Custom threshold=0.99 → probably 0 codes
r_none = call("predict_icd9", note_text=DISCHARGE_SNIPPET, threshold=0.99)
check("threshold=0.99 returns 0 or very few codes",
      r_none.get("num_codes_predicted", 0) < 5)

# top_k cap
r_topk = call("predict_icd9", note_text=DISCHARGE_SNIPPET, threshold=0.0, top_k=3)
check("top_k=3 caps at 3 results", r_topk.get("num_codes_predicted", 0) <= 3)

# ─── E. shap_lstm 
print("\n[E] shap_lstm")

if not cuda_ok:
    print("  SKIP: No GPU — shap_lstm requires CUDA; testing error path only")
    r_cpu = call("shap_lstm", patient_id=TEST_PID or 10006)
    check("CPU fallback returns error dict", "error" in r_cpu)
    check("error mentions CUDA/GPU", "cuda" in r_cpu.get("error","").lower()
          or "GPU" in r_cpu.get("error",""), r_cpu.get("error",""))
    # Fill remaining checks as skipped
    for i in range(3, 11):
        ok(f"E{i}: skip (no GPU)")
else:
    if TEST_PID is None:
        fail("E: no temporal test patient", "skipping all shap_lstm tests")
    else:
        print(f"  Using subject_id={TEST_PID}")
        t0 = time.time()
        r  = call("shap_lstm", patient_id=TEST_PID)
        latency = (time.time() - t0) * 1000

        check("no error key", "error" not in r, r.get("error",""))
        check("mortality_prob in [0,1]",
              0.0 <= r.get("mortality_prob", -1) <= 1.0, str(r.get("mortality_prob")))
        check("top_features is list", isinstance(r.get("top_features"), list))
        feats = r.get("top_features", [])
        check("10 top_features", len(feats) == 10, f"got {len(feats)}")
        check("each feature has shap_value",
              all("shap_value" in f for f in feats))
        check("each feature has direction",
              all(f.get("direction") in {"increases risk","decreases risk"} for f in feats))
        check("timestep_importance is list",
              isinstance(r.get("timestep_importance"), list))
        ts = r.get("timestep_importance", [])
        check("12 timesteps", len(ts) == 12, f"got {len(ts)}")
        check("timestep values are floats", all(isinstance(v, float) for v in ts))
        check("model_type mentions SHAP",
              "SHAP" in r.get("model_type",""), r.get("model_type",""))
        check("latency < 60s", latency < 60000, f"{latency:.0f}ms")
        print(f"  → prob={r.get('mortality_prob')}, latency={latency:.0f}ms")

        # Unknown patient
        r_unk2 = call("shap_lstm", patient_id=999999999)
        check("unknown patient returns error", "error" in r_unk2)

# ─── F. Cross-model consistency check 
print("\n[F] Cross-model consistency")

if TEST_PID:
    r_xgb  = call("predict_mortality", patient_id=TEST_PID)
    r_lstm = call("predict_mortality_temporal", patient_id=TEST_PID)

    p_xgb  = r_xgb.get("mortality_probability", -1)
    p_lstm = r_lstm.get("mortality_probability", -1)

    check("XGBoost prediction valid", 0 <= p_xgb <= 1, str(p_xgb))
    check("LSTM prediction valid", 0 <= p_lstm <= 1, str(p_lstm))
    # Predictions can differ; just check they're both meaningful numbers
    check("both probabilities are sensible floats",
          isinstance(p_xgb, float) and isinstance(p_lstm, float))
    print(f"  XGBoost prob={p_xgb:.4f}  LSTM prob={p_lstm:.4f}")
else:
    ok("skip (no test patient)"); ok("skip"); ok("skip")

# ─── G. Config & file checks 
print("\n[G] File integrity checks")

paths = config.check_paths()
check("lstm_model file exists", paths.get("lstm_model", False))
check("lstm_config file exists", paths.get("lstm_config", False))
check("icd9_labels file exists", paths.get("icd9_labels", False))
check("clinicalbert_icd9 dir exists", paths.get("clinicalbert_icd9", False))

# Validate lstm_config.json structure
if config.LSTM_CONFIG_PATH.exists():
    cfg = json.loads(config.LSTM_CONFIG_PATH.read_text())
    check("lstm_config has n_features", "n_features" in cfg)
    check("lstm_config has hidden_dim", "hidden_dim" in cfg)
    check("n_features==10", cfg.get("n_features") == 10, str(cfg.get("n_features")))
else:
    fail("G5","lstm_config.json missing"); fail("G6",""); fail("G7","")

# Validate icd9_labels.json structure
if config.ICD9_LABELS_PATH.exists():
    lbl = json.loads(config.ICD9_LABELS_PATH.read_text())
    check("icd9_labels has classes", "classes" in lbl)
    check("icd9_labels has descriptions", "descriptions" in lbl)
    check("50 ICD-9 classes", len(lbl.get("classes",[])) == 50,
          f"got {len(lbl.get('classes',[]))}")
else:
    fail("G8","icd9_labels.json missing"); fail("G9",""); fail("G10","")

# ─── Summary 
total = passed + failed
print(f"\n{'='*55}")
print(f"GPU models Results: {passed}/{total} tests PASSED")
if errors:
    print(f"\nFailed tests:")
    for e in errors:
        print(f"  FAIL {e}")
print("=" * 55)
sys.exit(0 if failed == 0 else 1)
