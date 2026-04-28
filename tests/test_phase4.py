"""
test_phase4.py — Validation tests for external APIs: External APIs & Web Search.

Run from the clinical_agent/ directory:
    python test_phase4.py

Tests:
  - PubMed search (real network call to NCBI)
  - Semantic Scholar search (real network call)
  - SearXNG web search (may fail if not running)
  - Firecrawl scrape URL (may fail if not running, falls back)
  - check_drug_interactions: metoprolol + amiodarone (known interaction)
  - get_drug_info: warfarin
  - check_drug_interactions: vancomycin + gentamicin (nephrotoxicity)
  - check_drug_interactions: fewer than 2 drugs (error case)
  - search_clinical_literature_rag (expects empty collection warning)
  - add_paper_to_rag then search_clinical_literature_rag
"""

import sys
import asyncio
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "clinical_agent"))

import config
from fastmcp import FastMCP

# ── Bootstrap 
from tools import literature, web_search, drug_tools

mcp = FastMCP("test-external APIs")

# Load assets
literature.load_rag_assets(config.CHROMA_DB_DIR, config.EMBEDDING_MODEL)
drug_tools.load_drugbank(config.DRUGBANK_CSV)  # csv_path ignored; SQLite used directly

# Register tools
literature.register(mcp)
web_search.register(mcp)
drug_tools.register(mcp)


async def call(tool_name: str, args: dict):
    """Call an MCP tool and return the parsed Python object."""
    import json
    result = await mcp.call_tool(tool_name, args)
    # FastMCP 3.x returns a ToolResult with .content list of TextContent items
    if hasattr(result, 'content'):
        content = result.content
        if content and hasattr(content[0], 'text'):
            try:
                return json.loads(content[0].text)
            except (json.JSONDecodeError, TypeError):
                return content[0].text
    # Fallback: bare iterable of TextContent
    if hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
        items = list(result)
        if items and hasattr(items[0], 'text'):
            try:
                return json.loads(items[0].text)
            except Exception:
                return items[0].text
    return result


def run(coro):
    return asyncio.run(coro)


PASS = 0
FAIL = 0


def check(test_name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {test_name}" + (f": {detail}" if detail else ""))
    if condition:
        PASS += 1
    else:
        FAIL += 1


# ── 1. PubMed 
print("\n=== PubMed ===")
try:
    results = run(call("search_pubmed", {"query": "SOFA score sepsis mortality", "max_results": 3}))
    check("Returns a list", isinstance(results, list))
    if isinstance(results, list) and results and "error" not in results[0]:
        first = results[0]
        check("Has pmid", bool(first.get("pmid")))
        check("Has title", bool(first.get("title")))
        check("Has abstract", bool(first.get("abstract")))
        check("Has url", "pubmed.ncbi" in first.get("url", ""))
        check("Year is int", isinstance(first.get("year"), int))
        print(f"     Title: {first['title'][:80]}...")
    else:
        check("No error in results", False, str(results))
except Exception as e:
    check("PubMed call succeeded", False, str(e))

# ── 2. Semantic Scholar 
print("\n=== Semantic Scholar ===")
try:
    results = run(call("search_semantic_scholar", {"query": "MIMIC-III clinical database", "max_results": 3}))
    check("Returns a list", isinstance(results, list))
    if isinstance(results, list) and results and "error" not in results[0]:
        first = results[0]
        check("Has paper_id", bool(first.get("paper_id")))
        check("Has title", bool(first.get("title")))
        check("Has citations", isinstance(first.get("citations"), int))
        check("Has url", "semanticscholar" in first.get("url", ""))
        print(f"     Title: {first['title'][:80]}  (citations: {first.get('citations')})")
    else:
        check("No error in results", False, str(results)[:200])
except Exception as e:
    check("Semantic Scholar call succeeded", False, str(e))

# ── 3. SearXNG 
print("\n=== SearXNG Web Search ===")
try:
    results = run(call("web_search", {"query": "sepsis treatment guidelines 2024", "num_results": 3}))
    check("Returns a list", isinstance(results, list))
    if isinstance(results, list) and results:
        first = results[0]
        if "error" in first:
            check("SearXNG running (or graceful error)", "not running" in first["error"] or
                  "SearXNG" in first["error"], f"Error: {first['error']}")
        else:
            check("Has url", bool(first.get("url")))
            check("Has title", bool(first.get("title")))
            print(f"     First result: {first.get('title', '')[:60]}")
except Exception as e:
    check("web_search call succeeded", False, str(e))

# ── 4. scrape_url (fallback mode) 
print("\n=== scrape_url (PubMed abstract page) ===")
try:
    result = run(call("scrape_url", {"url": "https://pubmed.ncbi.nlm.nih.gov/28009595/"}))
    check("Returns a dict", isinstance(result, dict))
    if isinstance(result, dict) and "error" not in result:
        check("Has url", bool(result.get("url")))
        check("Has markdown_content", bool(result.get("markdown_content")))
        check("Has source", bool(result.get("source")))
        print(f"     Source: {result.get('source')} | Content length: {len(result.get('markdown_content',''))}")
    else:
        check("No error", False, str(result)[:200])
except Exception as e:
    check("scrape_url call succeeded", False, str(e))

# ── 5. Drug Interactions: metoprolol + amiodarone 
print("\n=== Drug Interactions: metoprolol + amiodarone ===")
try:
    result = run(call("check_drug_interactions", {"drug_list": ["metoprolol", "amiodarone"]}))
    check("Returns a dict", isinstance(result, dict))
    if isinstance(result, dict) and "error" not in result:
        check("drugs_checked has 2 entries", len(result.get("drugs_checked", [])) == 2)
        check("Both drugs found", all(d["found"] for d in result.get("drugs_checked", [])))
        check("Has interactions", result.get("total_interactions", 0) > 0)
        check("total_interactions is int", isinstance(result.get("total_interactions"), int))
        check("high_severity_count is int", isinstance(result.get("high_severity_count"), int))
        if result.get("interactions"):
            first_ix = result["interactions"][0]
            check("Interaction has drug_a", bool(first_ix.get("drug_a")))
            check("Interaction has drug_b", bool(first_ix.get("drug_b")))
            check("Interaction has severity", first_ix.get("severity") in ("high", "moderate", "minor"))
            check("Interaction has description", bool(first_ix.get("description")))
            print(f"     Found {result['total_interactions']} interactions")
            print(f"     High: {result['high_severity_count']} | Moderate: {result['moderate_severity_count']} | Minor: {result['minor_severity_count']}")
            print(f"     Top: [{first_ix['severity']}] {first_ix['description'][:100]}...")
    else:
        check("No error", False, str(result)[:300])
except Exception as e:
    check("check_drug_interactions call succeeded", False, str(e))

# ── 6. Drug Interactions: vancomycin + gentamicin 
print("\n=== Drug Interactions: vancomycin + gentamicin (nephrotoxicity) ===")
try:
    result = run(call("check_drug_interactions", {"drug_list": ["vancomycin", "gentamicin"]}))
    check("Returns a dict", isinstance(result, dict))
    if isinstance(result, dict) and "error" not in result:
        check("Has interactions", result.get("total_interactions", 0) > 0)
        print(f"     Found {result.get('total_interactions', 0)} interactions")
    else:
        check("No error", False, str(result)[:200])
except Exception as e:
    check("vancomycin+gentamicin call succeeded", False, str(e))

# ── 7. get_drug_info: warfarin 
print("\n=== get_drug_info: warfarin ===")
try:
    result = run(call("get_drug_info", {"drug_name": "warfarin"}))
    check("Returns a dict", isinstance(result, dict))
    if isinstance(result, dict) and "error" not in result:
        check("Has name", bool(result.get("name")))
        check("Has drugbank_id", bool(result.get("drugbank_id")))
        check("Has indication", bool(result.get("indication")))
        check("Has mechanism_of_action", bool(result.get("mechanism_of_action")))
        print(f"     Name: {result.get('name')} | ID: {result.get('drugbank_id')}")
    else:
        check("No error", False, str(result)[:200])
except Exception as e:
    check("get_drug_info call succeeded", False, str(e))

# ── 8. Drug list < 2 (error case) 
print("\n=== check_drug_interactions: single drug (error case) ===")
try:
    result = run(call("check_drug_interactions", {"drug_list": ["aspirin"]}))
    check("Returns a dict", isinstance(result, dict))
    check("Has error message", bool(result.get("error")))
except Exception as e:
    check("Single drug error case succeeded", False, str(e))

# ── 9. RAG: empty collection 
print("\n=== RAG: search on empty/seeded collection ===")
try:
    # Add a paper first
    add_result = run(call("add_paper_to_rag", {
        "title": "MIMIC-III, a freely accessible critical care database",
        "abstract": "We present MIMIC-III, a large, freely-available database comprising "
                    "deidentified health-related data associated with over 40,000 patients "
                    "who stayed in critical care units of the Beth Israel Deaconess Medical Center.",
        "pmid": "27219127",
        "authors": "Johnson AE, Pollard TJ, Shen L, et al.",
        "year": 2016,
    }))
    check("add_paper_to_rag returns status ok", add_result.get("status") == "ok")
    check("add_paper_to_rag returns collection_size >= 1",
          add_result.get("collection_size", 0) >= 1)

    # Search
    rag_results = run(call("search_clinical_literature_rag",
                           {"query": "MIMIC critical care database", "top_k": 3}))
    check("RAG search returns a list", isinstance(rag_results, list))
    if isinstance(rag_results, list) and rag_results:
        first = rag_results[0]
        if "error" not in first:
            check("RAG result has title", bool(first.get("title")))
            check("RAG result has abstract", bool(first.get("abstract")))
            check("RAG result has distance", first.get("distance") is not None)
            print(f"     Top RAG result: {first['title'][:70]} (dist={first.get('distance')})")
        else:
            check("No RAG error", False, first["error"])
except Exception as e:
    check("RAG workflow succeeded", False, str(e))


# ── Summary 
print(f"\n{'='*50}")
total = PASS + FAIL
print(f"external APIs Tests: {PASS}/{total} PASSED")
if FAIL > 0:
    print(f"  *** {FAIL} tests FAILED — review output above ***")
else:
    print("  All tests passed!")
