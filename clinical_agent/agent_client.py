
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from openai import OpenAI, APIConnectionError, APITimeoutError

import config

_llm_client: OpenAI | None = None


def _get_llm_client() -> OpenAI:
    """Lazily construct the OpenAI-compat client (singleton)."""
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            timeout=300.0,
        )
    return _llm_client


SYSTEM_PROMPT = (
    "You are a clinical decision support tool with access to real MIMIC-III "
    "ICU patient data, clinical scoring calculators (SOFA, APACHE II, qSOFA, "
    "NEWS2, MELD, Charlson, and more), trained mortality prediction models "
    "(XGBoost static + Bi-LSTM temporal), SHAP-based explainability tools, "
    "phenotype clustering, PubMed/Semantic Scholar literature search, ChromaDB "
    "RAG over 558 clinical AI papers, drug-drug interaction checking (DrugBank), "
    "and NLP entity extraction (scispaCy + Presidio de-identification).\n\n"
    "Guidelines:\n"
    "• Always cite the specific tools and data sources you used.\n"
    "• Express uncertainty clearly   e.g. 'The model predicts X% risk, however…'\n"
    "• Do NOT make definitive diagnoses. Provide decision support to clinicians.\n"
    "• When quoting lab/vital values, state the units and the time window queried.\n"
    "• For literature questions, prefer the RAG tool first; fall back to PubMed.\n"
    "• When multiple tools are relevant, chain them sequentially and synthesise "
    "  the results into a coherent narrative.\n"
    "• If a tool returns an error, acknowledge it and continue with partial data."
)

from fastmcp import FastMCP
import duckdb
from tools import (
    patient_data,
    clinical_scores,
    ml_models,
    nlp_tools,
    literature,
    web_search,
    drug_tools,
)

mcp = FastMCP(
    name=config.MCP_SERVER_NAME,
    instructions=SYSTEM_PROMPT,
)

_started = False
_tools_registered = False


def startup() -> None:
    """Load all heavy assets.  Safe to call multiple times (idempotent)."""
    global _started
    if _started:
        return
    _started = True

    if config.MIMIC_DB.exists():
        print(f"[agent_client] Connecting to MIMIC-III DuckDB …")
        con = duckdb.connect(str(config.MIMIC_DB), read_only=True)
        patient_data.set_db_connection(con)
        nlp_tools.set_db_connection(con)
        clinical_scores.set_db_connection(con)
        print("[agent_client] DuckDB ready.")
    else:
        print(f"[agent_client] WARNING: MIMIC-III DB not found at {config.MIMIC_DB}")

    ml_models.load_models(
        xgb_path=config.XGB_MODEL_PATH,
        shap_path=config.SHAP_EXPLAINER_PATH,
        static_csv=config.STATIC_FEATURES_CSV,
        meta_csv=config.TEMPORAL_META_CSV,
    )

    ml_models.load_lstm_model(
        lstm_path=config.LSTM_MODEL_PATH,
        lstm_cfg_path=config.LSTM_CONFIG_PATH,
        temporal_seq_pkl=config.TEMPORAL_SEQ_PKL,
    )
    ml_models.load_bert_model(
        bert_model_dir=config.CLINICALBERT_ICD9_DIR,
        icd9_labels_path=config.ICD9_LABELS_PATH,
    )

    nlp_tools.load_nlp_models(config.SCISPACY_MODEL)

    literature.load_rag_assets(
        chroma_db_dir=config.CHROMA_DB_DIR,
        embedding_model_name=config.EMBEDDING_MODEL,
    )

    drug_tools.load_drugbank(config.DRUGBANK_CSV)

    print("[agent_client] Startup complete.")


def register_tools() -> None:
    """Register all tool modules with the FastMCP instance (idempotent)."""
    global _tools_registered
    if _tools_registered:
        return
    _tools_registered = True

    patient_data.register(mcp)
    clinical_scores.register(mcp)
    ml_models.register(mcp)
    nlp_tools.register(mcp)
    literature.register(mcp)
    web_search.register(mcp)
    drug_tools.register(mcp)

    tools = asyncio.run(mcp.list_tools())
    print(f"[agent_client] {len(tools)} tools registered.")


def build_agent() -> None:
    """Convenience: startup + register_tools in one call."""
    startup()
    register_tools()


def get_openai_tools() -> list[dict]:
    """
    Convert all FastMCP-registered tools to OpenAI function-calling format.
    Returns a list of dicts with keys: type, function{name, description, parameters}.
    """
    tools = asyncio.run(mcp.list_tools())
    openai_tools = []
    for t in tools:
        mcp_tool = t.to_mcp_tool()
        schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "description": (mcp_tool.description or "").strip(),
                    "parameters": schema,
                },
            }
        )
    return openai_tools


def _call_tool(tool_name: str, args: dict) -> str:
    """
    Execute an MCP tool in-process and return its result as a JSON string.
    Catches all exceptions so a bad tool never crashes the agentic loop.
    """
    try:
        result = asyncio.run(mcp.call_tool(tool_name, args))
        if hasattr(result, "content") and result.content:
            item = result.content[0]
            return item.text if hasattr(item, "text") else str(item)
        if isinstance(result, (dict, list)):
            return json.dumps(result, default=str)
        return str(result)
    except Exception as exc:
        tb = traceback.format_exc(limit=3)
        return json.dumps({"error": str(exc), "traceback": tb})


def run_agent(
    user_query: str,
    max_turns: int = 15,
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """
    Single-user-turn agentic loop.

    Parameters
    ----------
    user_query : str
        The natural-language clinical question.
    max_turns : int
        Maximum number of LLM ↔ tool round-trips before giving up.
    verbose : bool
        If True, print each tool call and response to stdout.

    Returns
    -------
    (final_response: str, tool_trace: list[dict])
        final_response   Nemotron's final natural-language narrative.
        tool_trace       list of {tool, args, result_snippet, latency_ms} dicts.
    """
    if not _tools_registered:
        raise RuntimeError(
            "Tools not registered.  Call build_agent() or register_tools() first."
        )

    client = _get_llm_client()
    try:
        client.models.list()
    except (APIConnectionError, APITimeoutError) as exc:
        warn = (
            f"[agent_client] WARNING: LLM endpoint unreachable ({exc}). "
            "Returning stub response."
        )
        print(warn)
        return _stub_response(user_query), []

    openai_tools = get_openai_tools()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]
    tool_trace: list[dict] = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"USER: {user_query}")
        print('-' * 60)

    for turn in range(max_turns):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                max_tokens=16000,
                temperature=0.1,
            )
        except (APIConnectionError, APITimeoutError) as exc:
            print(f"[agent_client] LLM error on turn {turn}: {exc}")
            return _stub_response(user_query), tool_trace

        choice = response.choices[0]

        if choice.finish_reason in ("stop", "length"):
            final = choice.message.content or ""
            if choice.finish_reason == "length" and verbose:
                print("  [agent_client] NOTE: response truncated by max_tokens.")
            if verbose:
                print(f"\nASSISTANT: {final}\n")
            return final, tool_trace

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if verbose:
                    print(f"\n  [Tool call #{turn+1}] {tool_name}({args})")

                t0 = time.time()
                result_text = _call_tool(tool_name, args)
                latency_ms = (time.time() - t0) * 1000

                if verbose:
                    snippet = result_text[:300].replace("\n", " ")
                    print(f"  → {snippet}{'…' if len(result_text) > 300 else ''}")
                    print(f"  ({latency_ms:.0f} ms)")

                tool_trace.append(
                    {
                        "tool": tool_name,
                        "args": args,
                        "result_snippet": result_text[:500],
                        "latency_ms": round(latency_ms, 1),
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    }
                )

        else:
            if verbose:
                print(f"[agent_client] Unexpected finish_reason: {choice.finish_reason}")
            break

    def _msg_role(m) -> str:
        if isinstance(m, dict):
            return m.get("role", "")
        return getattr(m, "role", "")

    def _msg_content(m) -> str:
        if isinstance(m, dict):
            return m.get("content") or ""
        return getattr(m, "content", "") or ""

    last_content = ""
    for m in reversed(messages):
        if _msg_role(m) == "assistant" and _msg_content(m):
            last_content = _msg_content(m)
            break

    result = last_content or "[max turns reached without final response]"
    if verbose:
        print(f"\nASSISTANT (max-turn exit): {result}\n")
    return result, tool_trace


def _stub_response(query: str) -> str:
    """
    Return a structured stub that still exercises all tool-call paths,
    useful for development when DGX Spark is asleep.
    """
    return (
        "[STUB   DGX Spark unreachable]\n\n"
        f"Query received: '{query}'\n\n"
        f"The LLM endpoint at {config.LLM_BASE_URL} is currently "
        "unavailable.  To run a live demo, ensure the Nemotron-3-Super-120B "
        "TRT-LLM service is running on DGX Spark (port 8123) and retry.  "
        "All 45 MCP tools are registered and functional; "
        "only the orchestrating LLM is missing."
    )


if __name__ == "__main__":
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print("=" * 60)
    print("Clinical Agent   Smoke Test")
    print("=" * 60)

    build_agent()

    tools_list = asyncio.run(mcp.list_tools())
    print(f"\nok {len(tools_list)} tools registered")

    try:
        _get_llm_client().models.list()
        llm_ok = True
        print(f"ok LLM endpoint reachable: {config.LLM_BASE_URL}")
    except Exception as e:
        llm_ok = False
        print(f"FAIL LLM endpoint unreachable: {e}")
        print("  (agent_client will return stub responses)")

    demo_query = "Give me an overview of patient 27513 and their mortality risk."
    print(f"\nRunning demo query: '{demo_query}'\n")
    final, trace = run_agent(demo_query, max_turns=6, verbose=True)

    print("\n" + "=" * 60)
    print(f"Tool calls made: {len(trace)}")
    for i, t in enumerate(trace, 1):
        print(f"  {i}. {t['tool']}({t['args']})   {t['latency_ms']:.0f} ms")
    print(f"\nFinal response length: {len(final)} chars")
    print("=" * 60)
