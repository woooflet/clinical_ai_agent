import sys
import argparse

from fastmcp import FastMCP

import config
from tools import patient_data, clinical_scores, ml_models, nlp_tools, literature, web_search, drug_tools

mcp = FastMCP(
    name=config.MCP_SERVER_NAME,
    instructions=(
        "You are a clinical decision support tool with access to MIMIC-III patient data, "
        "clinical scoring calculators, trained mortality prediction models, PubMed literature, "
        "and drug interaction checking. Always cite your sources and express uncertainty "
        "when clinical evidence is limited. Do not make definitive diagnoses. "
        "Provide decision support to assist qualified clinicians."
    ),
)

_MIMIC_OK       = False
_ML_OK          = False
_LSTM_OK        = False
_BERT_OK        = False
_NLP_OK         = False
_RAG_OK         = False
_DRUG_OK        = False

def startup() -> None:
    global _MIMIC_OK, _ML_OK, _LSTM_OK, _BERT_OK, _NLP_OK, _RAG_OK, _DRUG_OK

    print("\n[server] Starting Clinical Agent Server")

    try:
        if config.MIMIC_DB is None:
            raise FileNotFoundError(
                "MIMIC_DB not configured.\n"
                "        Set MIMIC_CSV_DIR (CSV source directory) and optionally\n"
                "        MIMIC_DB (DuckDB path) in your .env, then run:\n"
                "          python clinical_agent/scripts/build_mimic_duckdb.py\n"
                "        MIMIC patient data tools will be DISABLED."
            )
        if not config.MIMIC_DB.exists():
            raise FileNotFoundError(
                f"mimic_duck.db not found at {config.MIMIC_DB}\n"
                f"        Build it from your CSVs with:\n"
                f"          python clinical_agent/scripts/build_mimic_duckdb.py\n"
                f"        Or re-run ./deploy.sh to be guided through setup.\n"
                f"        MIMIC patient data tools will be DISABLED."
            )
        import duckdb
        con = duckdb.connect(str(config.MIMIC_DB), read_only=True)
        patient_data.set_db_connection(con)
        nlp_tools.set_db_connection(con)
        clinical_scores.set_db_connection(con)
        literature.set_db_connection(con)
        _MIMIC_OK = True
        print(f"[server]  ok  MIMIC-III DuckDB  →  {config.MIMIC_DB}")
    except Exception as exc:
        print(f"[server]  FAIL  MIMIC-III DuckDB  DISABLED  ({exc})")

    try:
        ml_models.load_models(
            xgb_path   = config.XGB_MODEL_PATH,
            shap_path  = config.SHAP_EXPLAINER_PATH,
            static_csv = config.STATIC_FEATURES_CSV,
            meta_csv   = config.TEMPORAL_META_CSV,
        )
        _ML_OK = True
        print(f"[server]  ok  XGBoost + SHAP models loaded")
    except Exception as exc:
        print(f"[server]  FAIL  XGBoost/SHAP  DISABLED  ({exc})")

    try:
        ml_models.load_lstm_model(
            lstm_path        = config.LSTM_MODEL_PATH,
            lstm_cfg_path    = config.LSTM_CONFIG_PATH,
            temporal_seq_pkl = config.TEMPORAL_SEQ_PKL,
        )
        _LSTM_OK = True
        print(f"[server]  ok  Bi-LSTM temporal model loaded")
    except Exception as exc:
        print(f"[server]  FAIL  Bi-LSTM  DISABLED  ({exc})")

    try:
        ml_models.load_bert_model(
            bert_model_dir   = config.CLINICALBERT_ICD9_DIR,
            icd9_labels_path = config.ICD9_LABELS_PATH,
        )
        _BERT_OK = True
        print(f"[server]  ok  ClinicalBERT ICD-9 classifier loaded")
    except Exception as exc:
        print(f"[server]  FAIL  ClinicalBERT  DISABLED  ({exc})")

    try:
        nlp_tools.load_nlp_models(config.SCISPACY_MODEL)
        _NLP_OK = True
        print(f"[server]  ok  scispaCy NER model loaded  ({config.SCISPACY_MODEL})")
    except Exception as exc:
        print(f"[server]  FAIL  scispaCy  DISABLED  ({exc})")

    try:
        literature.load_rag_assets(
            chroma_db_dir         = config.CHROMA_DB_DIR,
            embedding_model_name  = config.EMBEDDING_MODEL,
            chroma_guidelines_dir = config.CHROMA_GUIDELINES_DIR,
            chroma_notes_dir      = config.CHROMA_NOTES_DIR,
        )
        literature.load_reranker()
        _RAG_OK = True
        print(f"[server]  ok  ChromaDB RAG + cross-encoder loaded")
    except Exception as exc:
        print(f"[server]  FAIL  ChromaDB/RAG  DISABLED  ({exc})")

    try:
        drug_tools.load_drugbank(config.DRUGBANK_CSV)
        _DRUG_OK = True
        print(f"[server]  ok  DrugBank SQLite loaded")
    except Exception as exc:
        print(f"[server]  FAIL  DrugBank  DISABLED  ({exc})")

    print("\n[server] Subsystem Summary")
    statuses = {
        "MIMIC DuckDB":      _MIMIC_OK,
        "XGBoost/SHAP":      _ML_OK,
        "Bi-LSTM":           _LSTM_OK,
        "ClinicalBERT":      _BERT_OK,
        "scispaCy NER":      _NLP_OK,
        "ChromaDB RAG":      _RAG_OK,
        "DrugBank":          _DRUG_OK,
    }
    for name, ok in statuses.items():
        icon = "ok" if ok else "FAIL"
        print(f"[server]   {icon}  {name}")
    n_ok = sum(statuses.values())
    print(f"[server]  {n_ok}/{len(statuses)} subsystems active\n")

def register_tools() -> None:
    patient_data.register(mcp)
    clinical_scores.register(mcp)
    ml_models.register(mcp)
    nlp_tools.register(mcp)
    literature.register(mcp)
    web_search.register(mcp)
    drug_tools.register(mcp)

    import asyncio
    tools = asyncio.run(mcp.list_tools())
    tool_names = sorted([t.name for t in tools])
    print(f"[server] {len(tool_names)} tools registered: {tool_names}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical AI MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http", "streamable-http"],
        default="stdio",
        help=(
            "MCP transport: "
            "'stdio' for in-process MCP clients (default), "
            "'streamable-http' for OpenWebUI (endpoint /mcp), "
            "'sse' for legacy SSE debugging"
        ),
    )
    parser.add_argument("--host", default=config.MCP_HOST)
    parser.add_argument("--port", type=int, default=config.MCP_PORT)
    args = parser.parse_args()

    startup()
    register_tools()

    if args.transport in ("streamable-http",):
        print(f"[server] Starting Streamable HTTP MCP server at http://{args.host}:{args.port}/mcp")
        print(f"[server] Configure OpenWebUI: Admin > Settings > Tools > Add Connection")
        print(f"[server]   URL: http://localhost:{args.port}/mcp   Type: MCP (Streamable HTTP)")
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport in ("http", "sse"):
        print(f"[server] Starting SSE/HTTP MCP server at http://{args.host}:{args.port}/sse")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        print("[server] Starting stdio MCP server (ready for MCP client connections)")
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
