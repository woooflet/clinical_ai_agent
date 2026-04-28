Clinical MCP Agent

A 48-tool MCP server for ICU clinical decision support built on MIMIC-III, FastMCP, and Open WebUI.

Setup requires deploy.sh (Linux/systemd). For macOS or Windows run the script to see manual instructions.


Prerequisites

- Python 3.11+
- Docker and Docker Compose (for Open WebUI and SearXNG)
- MIMIC-III access: credentialed download from physionet.org/content/mimiciii/1.4
- DrugBank SQLite: drugbank.db at drugbank-mcp-server/data/drugbank.db
- Any OpenAI-compatible LLM endpoint


Quick start

    git clone https://github.com/YOUR_USERNAME/clinical-mcp-agent.git
    cd clinical-mcp-agent
    ./deploy.sh

The script prompts for your data paths and LLM endpoint, installs systemd services, optionally starts Open WebUI via Docker, and optionally seeds the RAG databases. It prints the next steps when done.


deploy.sh flags

    ./deploy.sh               Full install (first-time setup)
    ./deploy.sh --start       Start both services (must already be installed)
    ./deploy.sh --stop        Stop both services
    ./deploy.sh --restart     Restart both services after any code change
    ./deploy.sh --status      Show live service status and endpoint URLs
    ./deploy.sh --uninstall   Remove systemd units and stop Docker containers
    ./deploy.sh --help        Show usage


What the script does (install mode)

Step 1  Creates the Python virtualenv at clinicalAgentEnv/ and installs requirements.txt.
        Installs scispaCy and en_ner_bc5cdr_md separately with --no-deps.

Step 2  Prompts for MIMIC_CSV_DIR, MIMIC_DB path, DrugBank path, LLM base URL,
        LLM model name, and NCBI API key. Writes all settings to .env.

Step 3  Optionally builds the MIMIC-III DuckDB from your CSV directory.
        This takes 10 to 40 minutes depending on hardware (CHARTEVENTS alone is ~33 GB).
        The build is resumable if interrupted.

Step 4  Installs and enables systemd units clinical-mcp-server and open-webui,
        stamping the correct venv and repo paths into the service files.

Step 5  Optionally starts Open WebUI and SearXNG via Docker Compose.

Step 6  Optionally seeds the three ChromaDB collections:
          a) clinical_literature  (~600 PubMed papers, needs NCBI API key)
          b) clinical_guidelines  (PDF files in clinical_agent/data/guidelines/)
          c) mimic_notes          (~5000 discharge summaries, needs MIMIC DuckDB)

Step 7  Starts the MCP server service.


Connecting Open WebUI to the MCP server

1. Open http://localhost:8080 and sign in as admin.
2. Go to Admin Panel, Settings, Tools.
3. Click Add Tool Server and enter:
     URL:  http://YOUR_HOST_IP:8001/mcp
     Type: MCP (Streamable HTTP)
4. Click Save. You should see 45 tools listed.

Enable native function calling on your model (required):

    Admin Panel, Models, select your model, Advanced, Function Calling: Native


Manual run without systemd

    source clinicalAgentEnv/bin/activate
    cd clinical_agent
    python server.py --transport streamable-http --host 0.0.0.0 --port 8001


Seeding RAG databases manually

    cd clinical_agent

    # Clinical literature (needs NCBI API key)
    NCBI_API_KEY=your_key python scripts/batch_seed_chroma.py

    # Clinical guidelines (place PDFs in clinical_agent/data/guidelines/ first)
    python scripts/seed_guidelines.py

    # MIMIC note index
    python scripts/seed_mimic_notes.py


Preparing ML models

The trained model files are not in the repo. Generate them from the survival analysis notebooks.

    cd clinical_agent

    # XGBoost and SHAP explainer
    python scripts/save_models.py

    # LSTM temporal model and ClinicalBERT ICD-9 classifier (GPU recommended)
    CUDA_VISIBLE_DEVICES=0 python scripts/save_lstm_model.py

CPU-only deployment: skip save_lstm_model.py. Three of the 45 tools require a GPU; all others work on CPU.


Configuration

All settings can be overridden with environment variables or via .env:

    MIMIC_CSV_DIR     Directory containing raw MIMIC-III CSV files
    MIMIC_DB          Path to mimic_duck.db (defaults to MIMIC_CSV_DIR/mimic_duck.db)
    DRUGBANK_DB       Path to drugbank.db
    LLM_BASE_URL      OpenAI-compatible LLM endpoint (default http://localhost:8123/v1)
    LLM_MODEL         Model name string
    NCBI_API_KEY      Free from ncbi.nlm.nih.gov/account (raises PubMed rate limit)
    EMBEDDING_MODEL   HuggingFace embedding model for RAG (default ncbi/MedCPT-Article-Encoder)
    SEARXNG_URL       SearXNG API endpoint (default http://localhost:8082)


Troubleshooting

Tools not appearing in Open WebUI
  Check the server is reachable: curl http://YOUR_HOST_IP:8001/mcp
  Verify function_calling is set to native on the model.
  Check logs: journalctl -u clinical-mcp-server -f

SSL errors during pip install
  pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org PACKAGE

scispaCy fails to load
  Install with --no-deps. The nlp_tools module auto-patches the spaCy 3.8 config on startup.

Open WebUI cannot reach MCP server from Docker
  Use the host machine LAN IP in the tool server URL, not localhost or 127.0.0.1.

Embedding model mismatch after changing EMBEDDING_MODEL
  Wipe and re-seed all three ChromaDB collections. Mixing embedding models produces wrong results.
