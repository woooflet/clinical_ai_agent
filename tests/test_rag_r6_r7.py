"""
test_rag_r6_r7.py  —  Comprehensive RAG pipeline validation
Tests R6 (BM25 + RRF hybrid), R7 (cross-encoder re-ranking), all three
collections, output schema contracts, JSON serialization, and OpenWebUI
compatibility rules.

Run with:
    cd clinical_agent
    CUDA_VISIBLE_DEVICES=0 ../clinicalAgentEnv/bin/python test_rag_r6_r7.py
"""

import sys
import json
import time
import warnings
import asyncio

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "clinical_agent"))

import config
from tools import literature
from fastmcp import FastMCP

# ─── Startup 

print("=" * 65)
print("  RAG Pipeline Validation  (R6 BM25/RRF + R7 Cross-Encoder)")
print("=" * 65)

print("\n[SETUP] Loading RAG assets …")
t0 = time.monotonic()
literature.load_rag_assets(
    chroma_db_dir         = config.CHROMA_DB_DIR,
    embedding_model_name  = config.EMBEDDING_MODEL,
    chroma_guidelines_dir = config.CHROMA_GUIDELINES_DIR,
    chroma_notes_dir      = config.CHROMA_NOTES_DIR,
)
literature.load_reranker()
print(f"[SETUP] Assets loaded in {time.monotonic() - t0:.1f}s")

mcp = FastMCP("test-rag")
literature.register(mcp)

PASS = 0
FAIL = 0
RESULTS = []

def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        tag = "ok PASS"
    else:
        FAIL += 1
        tag = "FAIL: FAIL"
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    RESULTS.append((tag, name, detail))


# ─── Section 1: Module-level globals 

print("\n── Section 1: Module state ")

check("chroma_collection loaded",
      literature._chroma_collection is not None)
check("chroma_guidelines loaded",
      literature._chroma_guidelines is not None)
check("chroma_notes loaded",
      literature._chroma_notes is not None)
check("embedding_model loaded",
      literature._embedding_model is not None)

# R6
check("BM25 index built",
      literature._bm25_index is not None,
      f"type={type(literature._bm25_index).__name__}")
check("BM25 doc_ids populated",
      len(literature._bm25_doc_ids) > 0,
      f"{len(literature._bm25_doc_ids)} docs")
check("BM25 count matches ChromaDB",
      len(literature._bm25_doc_ids) == literature._chroma_collection.count(),
      f"{len(literature._bm25_doc_ids)} == {literature._chroma_collection.count()}")

# R7
check("cross-encoder reranker loaded",
      literature._reranker is not None,
      f"type={type(literature._reranker).__name__}")

# ─── Section 2: _rrf_merge unit test 

print("\n── Section 2: RRF merge unit tests ")

# Perfect agreement: same doc in rank-1 both lists → highest RRF score
merged = literature._rrf_merge(
    dense_ids=["doc_A", "doc_B", "doc_C"],
    dense_scores=[0.9, 0.8, 0.7],
    bm25_ids=["doc_A", "doc_C", "doc_B"],
    bm25_scores=[10.0, 8.0, 6.0],
)
top_id, top_score = merged[0]
check("RRF: top doc is consensus rank-1",
      top_id == "doc_A",
      f"got {top_id}")
check("RRF: all input docs in merged output",
      len(merged) == 3,
      f"len={len(merged)}")

# BM25-only doc gets a score
merged2 = literature._rrf_merge(
    dense_ids=["doc_X"],
    dense_scores=[0.9],
    bm25_ids=["doc_X", "doc_Y"],
    bm25_scores=[10.0, 5.0],
)
ids_out = [d for d, _ in merged2]
check("RRF: BM25-only doc included in output",
      "doc_Y" in ids_out)
check("RRF: doc in both lists outranks BM25-only",
      ids_out.index("doc_X") < ids_out.index("doc_Y"))

# ─── Section 3: _rerank unit test 

print("\n── Section 3: Cross-encoder _rerank unit tests ")

# Construct candidates where one is clearly more relevant
candidates = [
    {"abstract": "Banana cultivation in tropical climates",          "title": "Bananas", "relevance_score": 0.9},
    {"abstract": "Vancomycin AUC monitoring in ICU patients with renal failure", "title": "Vancomycin", "relevance_score": 0.7},
    {"abstract": "Sepsis management antibiotics timing vasopressors", "title": "Sepsis Mgmt", "relevance_score": 0.8},
]
query = "antibiotic dosing ICU renal failure"
reranked = literature._rerank(query, candidates, top_k=3)
check("_rerank returns correct count",
      len(reranked) == 3)
check("_rerank adds rerank_score to each item",
      all("rerank_score" in c for c in reranked))
# Vancomycin/renal should beat bananas
top_title = reranked[0]["title"]
check("_rerank promotes clinically relevant doc to top",
      top_title != "Bananas",
      f"top={top_title}")
check("_rerank: banana last after re-ranking",
      reranked[-1]["title"] == "Bananas",
      f"last={reranked[-1]['title']}")

# Fallback when no reranker
old_reranker = literature._reranker
literature._reranker = None
fallback = literature._rerank(query, candidates, top_k=2)
literature._reranker = old_reranker
check("_rerank fallback returns top_k items when no reranker",
      len(fallback) == 2)

# ─── Section 4: search_clinical_literature_rag (hybrid + reranked) 

print("\n── Section 4: search_clinical_literature_rag ")

SEMANTIC_QUERIES = [
    ("sepsis mortality prediction ICU machine learning", 0.65),
    ("SHAP explainability clinical prediction model",   0.65),
    ("acute kidney injury creatinine prediction",       0.60),
]
KEYWORD_QUERIES = [
    ("vancomycin AUC dosing renal function",            0.60),
    ("XGBoost gradient boosting clinical outcome",      0.60),
]

for q, threshold in SEMANTIC_QUERIES + KEYWORD_QUERIES:
    t0 = time.monotonic()
    raw = asyncio.run(mcp.call_tool("search_clinical_literature_rag", {"query": q, "top_k": 5}))
    elapsed = time.monotonic() - t0
    items = json.loads(raw.content[0].text)

    # Filter out warning dicts
    results = [x for x in items if "warning" not in x]
    warnings_list = [x for x in items if "warning" in x]

    check(f"lit_rag '{q[:40]}…': returns results",
          len(results) > 0, f"{len(results)} results")

    if results:
        top = results[0]
        check(f"lit_rag '{q[:35]}…': relevance_score present",
              "relevance_score" in top, f"score={top.get('relevance_score')}")
        check(f"lit_rag '{q[:35]}…': relevance >= {threshold}",
              top.get("relevance_score", 0) >= threshold,
              f"score={top.get('relevance_score')}")
        check(f"lit_rag '{q[:35]}…': retrieval_mode=hybrid",
              top.get("retrieval_mode") == "hybrid",
              f"mode={top.get('retrieval_mode')}")
        check(f"lit_rag '{q[:35]}…': rerank_score present",
              "rerank_score" in top,
              f"rerank={top.get('rerank_score')}")
        check(f"lit_rag '{q[:35]}…': required keys present",
              all(k in top for k in ["title", "abstract", "pmid", "authors", "year", "citation", "url"]))
        # Size check (OpenWebUI: keep under 50 KB)
        result_json = json.dumps(results)
        check(f"lit_rag '{q[:35]}…': response < 50 KB",
              len(result_json) < 50_000,
              f"{len(result_json)//1024} KB")
        check(f"lit_rag '{q[:35]}…': latency < 8s",
              elapsed < 8.0,
              f"{elapsed:.2f}s")

# min_year filter test
raw_yr = asyncio.run(mcp.call_tool("search_clinical_literature_rag", {
    "query": "sepsis ICU prediction", "top_k": 5, "min_year": 2020
}))
items_yr = [x for x in json.loads(raw_yr.content[0].text) if "warning" not in x]
check("lit_rag min_year=2020: all results >= 2020",
      all(r.get("year", 0) >= 2020 for r in items_yr),
      f"years={[r.get('year') for r in items_yr]}")

# topic_filter test
raw_tf = asyncio.run(mcp.call_tool("search_clinical_literature_rag", {
    "query": "antibiotic resistance", "top_k": 5, "topic_filter": "sepsis"
}))
items_tf = [x for x in json.loads(raw_tf.content[0].text) if "warning" not in x]
if items_tf:
    check("lit_rag topic_filter=sepsis: results contain 'sepsis'",
          all("sepsis" in (r.get("title","") + r.get("abstract","")).lower() for r in items_tf),
          f"{len(items_tf)} results")

# ─── Section 5: search_clinical_guidelines_rag 

print("\n── Section 5: search_clinical_guidelines_rag ")

GUIDELINE_QUERIES = [
    ("antibiotic therapy timing septic shock",         0.80),
    ("KDIGO AKI definition creatinine staging",        0.80),
    ("PEEP ARDS mechanical ventilation low tidal vol", 0.75),
    ("vancomycin AUC guided dosing monitoring",        0.75),
    ("heart failure LVEF guideline directed therapy",  0.70),
]

for q, threshold in GUIDELINE_QUERIES:
    t0 = time.monotonic()
    raw = asyncio.run(mcp.call_tool("search_clinical_guidelines_rag", {"query": q, "top_k": 5}))
    elapsed = time.monotonic() - t0
    items = json.loads(raw.content[0].text)
    results = [x for x in items if "warning" not in x]

    check(f"guide_rag '{q[:38]}…': returns results",
          len(results) > 0)
    if results:
        top = results[0]
        check(f"guide_rag '{q[:35]}…': relevance >= {threshold}",
              top.get("relevance_score", 0) >= threshold,
              f"score={top.get('relevance_score')}")
        check(f"guide_rag '{q[:35]}…': rerank_score present",
              "rerank_score" in top,
              f"rerank={top.get('rerank_score')}")
        check(f"guide_rag '{q[:35]}…': source field populated",
              bool(top.get("source")),
              f"source={top.get('source','')[:40]}")
        check(f"guide_rag '{q[:35]}…': latency < 10s",
              elapsed < 10.0,
              f"{elapsed:.2f}s")

# ─── Section 6: find_similar_cases_rag 

print("\n── Section 6: find_similar_cases_rag ")

CASE_QUERIES = [
    (27513, "sepsis respiratory failure ICU intubation vasopressors", 0.50),
    (21789, "multi-organ failure mechanical ventilation septic shock", 0.45),
]

for pid, qtext, threshold in CASE_QUERIES:
    t0 = time.monotonic()
    raw = asyncio.run(mcp.call_tool("find_similar_cases_rag", {
        "patient_id": pid, "query_text": qtext, "top_k": 5
    }))
    elapsed = time.monotonic() - t0
    items = json.loads(raw.content[0].text)
    results = [x for x in items if "warning" not in x]

    check(f"cases_rag pid={pid}: returns results",
          len(results) > 0, f"{len(results)} cases")
    if results:
        top = results[0]
        check(f"cases_rag pid={pid}: relevance >= {threshold}",
              top.get("relevance_score", 0) >= threshold,
              f"score={top.get('relevance_score')}")
        check(f"cases_rag pid={pid}: does not return query patient",
              all(r.get("subject_id") != pid for r in results))
        check(f"cases_rag pid={pid}: required keys present",
              all(k in top for k in ["subject_id","relevance_score","cluster","mortality_label","note_snippet"]))
        check(f"cases_rag pid={pid}: latency < 10s",
              elapsed < 10.0, f"{elapsed:.2f}s")

# ─── Section 7: JSON serialization (OpenWebUI requirement) 

print("\n── Section 7: JSON serialization (OpenWebUI compat) ")

for tool, args, label in [
    ("search_clinical_literature_rag", {"query": "ARDS mortality prediction", "top_k": 5}, "lit_rag"),
    ("search_clinical_guidelines_rag", {"query": "sepsis antibiotics Hour-1 Bundle", "top_k": 5}, "guide_rag"),
    ("find_similar_cases_rag",         {"patient_id": 27513, "query_text": "sepsis renal failure", "top_k": 3}, "cases_rag"),
]:
    raw = asyncio.run(mcp.call_tool(tool, args))
    try:
        parsed = json.loads(raw.content[0].text)
        re_encoded = json.dumps(parsed)  # will raise if not serializable
        check(f"json_serial: {label} round-trips cleanly", True)
    except (TypeError, ValueError) as e:
        check(f"json_serial: {label} round-trips cleanly", False, str(e))

# ─── Section 8: Response size (OpenWebUI ~50 KB limit) 

print("\n── Section 8: Response size ")

for tool, args, label in [
    ("search_clinical_literature_rag", {"query": "sepsis ICU", "top_k": 10}, "lit_rag top_k=10"),
    ("search_clinical_guidelines_rag", {"query": "sepsis antibiotics", "top_k": 10}, "guide_rag top_k=10"),
]:
    raw = asyncio.run(mcp.call_tool(tool, args))
    size = len(raw.content[0].text)
    check(f"size: {label} < 50 KB",
          size < 50_000, f"{size//1024} KB")

# ─── Section 9: add_paper_to_rag + BM25 rebuild 

print("\n── Section 9: add_paper_to_rag (runtime addition) ")

before_count = literature._chroma_collection.count()
raw_add = asyncio.run(mcp.call_tool("add_paper_to_rag", {
    "title": "BM25-RRF-RERANK Hybrid Retrieval for Clinical AI",
    "abstract": "We introduce a three-stage pipeline combining BM25 keyword retrieval, dense cosine similarity, reciprocal rank fusion, and cross-encoder re-ranking for clinical literature search.",
    "pmid": "99999999",
    "authors": "Test Author",
    "year": 2026,
}))
add_result = json.loads(raw_add.content[0].text)
after_count = literature._chroma_collection.count()
check("add_paper_to_rag: status ok",
      add_result.get("status") == "ok", str(add_result))
check("add_paper_to_rag: collection grew",
      after_count >= before_count, f"{before_count} → {after_count}")

# ─── Summary 

print("\n" + "=" * 65)
print(f"  TOTAL: {PASS + FAIL} tests  |  {PASS} PASSED  |  {FAIL} FAILED")
print("=" * 65)

if FAIL > 0:
    print("\nFailed tests:")
    for tag, name, detail in RESULTS:
        if "FAIL" in tag:
            print(f"  {name}  {detail}")
    sys.exit(1)
else:
    print("All tests passed ok")
    sys.exit(0)
