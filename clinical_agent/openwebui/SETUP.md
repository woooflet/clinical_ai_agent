# OpenWebUI + Clinical AI MCP Server — Setup Guide

## Overview

Replaces `gradio_demo.py` with **Open WebUI** (port **8080**) wired to the
existing 45-tool FastMCP server (port **8001**) and the Nemotron 120B LLM on
DGX Spark (port **8123**).

```
Browser  →  Open WebUI :8080
             │
             ├── LLM calls  →  Nemotron 120B @ 192.168.12.110:8123
             └── MCP tools  →  Clinical MCP Server :8001/mcp
                                 (45 tools: DuckDB, XGBoost, SHAP,
                                  LSTM, ClinicalBERT, ChromaDB RAG,
                                  PubMed, DrugBank, scispaCy, …)
```

---

## Step 1 — Install `uv` (if not already installed)

`uv` is a fast Python package manager that can pin a specific Python version
for a tool, which is exactly what we need since Open WebUI officially requires
Python 3.11 and the system has Python 3.13.

```bash
# Check if uv is already installed
which uv && uv --version

# If not installed:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Then reload your shell:
source ~/.local/bin/env   # or restart the terminal
```

---

## Step 2 — Pre-download Open WebUI (recommended)

Running `uvx` in the systemd service will download Open WebUI on first launch
which can take a while. Do it once interactively so the cache is warm:

```bash
# This downloads open-webui with Python 3.11 and puts it in uv's tool cache.
# First run takes ~5–10 min depending on connection speed.
mkdir -p /home/dog/open-webui/data
~/.local/bin/uvx --python 3.11 open-webui@latest serve --port 8080
# (Ctrl-C once you see "Application startup complete" — just warming the cache)
```

---

## Step 3 — Generate a Real Secret Key

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Open `open-webui.service` and replace `CHANGE_ME_use_a_32byte_random_hex_string`
with the output before installing the service.

```bash
nano /home/dog/VS/AIinHealthcare/clinical_agent/openwebui/open-webui.service
# Edit the WEBUI_SECRET_KEY line
```

---

## Step 4 — Install the Systemd Services

```bash
# Copy service files to systemd
sudo cp /home/dog/VS/AIinHealthcare/clinical_agent/openwebui/clinical-mcp-server.service \
        /etc/systemd/system/

sudo cp /home/dog/VS/AIinHealthcare/clinical_agent/openwebui/open-webui.service \
        /etc/systemd/system/

# Reload systemd and enable both services at boot
sudo systemctl daemon-reload
sudo systemctl enable clinical-mcp-server open-webui

# Start the MCP server first (models take ~2–3 min to load)
sudo systemctl start clinical-mcp-server

# Monitor startup — wait until you see "X tools registered"
journalctl -u clinical-mcp-server -f
# (Ctrl-C when ready)

# Then start Open WebUI
sudo systemctl start open-webui
journalctl -u open-webui -f
# (Ctrl-C when you see "Application startup complete")
```

---

## Step 5 — Configure Open WebUI (first-time setup)

Open a browser: **http://localhost:8080**

1. Create your admin account (first login creates the admin).

### 5a. Verify the Nemotron 120B LLM connection

The service file pre-sets `OPENAI_API_BASE_URL` and `OPENAI_API_KEY`, so Open
WebUI should auto-discover the model. To confirm:

- **Admin Panel → Settings → Connections → OpenAI API**
- URL: `http://192.168.12.110:8123/v1` — click **Verify**
- The Nemotron model should appear in the list.

### 5b. Add the Clinical MCP Server as a Tool

1. **Admin Panel → Settings → Tools**
2. Click the **"+"** button (Add Connection)
3. Fill in:
   - **Type:** `MCP (Streamable HTTP)`
   - **URL:** `http://localhost:8001/mcp`
   - **Name:** `Clinical AI MCP (45 tools)`
4. Click **Save** — all 45 clinical tools will appear.

### 5c. Enable tools for the Nemotron model

1. **Workspace → Models → find Nemotron**
2. Edit → **Tools** tab → enable `Clinical AI MCP (45 tools)`
3. **Advanced Parameters** → **Function Calling** → set to `Native`
4. Save.

---

## Step 6 — Test It

In the Open WebUI chat, select Nemotron and send:

```
Give me a complete overview of patient 27513 — demographics, recent vitals, labs, and diagnoses.
```

You should see tool calls in the sidebar as Nemotron chains through the MCP tools.

**Full demo scenario set (same as the old Gradio demo):**

| Scenario | Query |
|----------|-------|
| S1 — Patient Overview | `Give me a complete overview of patient 27513 — demographics, recent vitals, labs, and diagnoses.` |
| S2 — Mortality + SHAP | `Analyze patient 27513's mortality risk in detail and explain the top contributing factors using SHAP.` |
| S3 — Sepsis Scoring | `Is patient 27513 at risk for sepsis? Calculate SOFA and qSOFA scores from their MIMIC data.` |
| S4 — Clinical Notes NLP | `Summarize the clinical notes for patient 27513 and extract key diseases, medications, and abnormal labs.` |
| S5 — Phenotype | `What phenotype cluster does patient 27513 belong to, and who are the 5 most similar historical patients?` |
| S6 — Literature RAG | `What does recent literature say about sepsis management and early warning strategies in ICU patients?` |

**Demo patients:**
| ID | Outcome | Notes | Use case |
|----|---------|-------|---------|
| 27513 | Survived | 54 | Primary — sepsis survivor |
| 21789 | Died | 22 | High mortality contrast |
| 20707 | Survived | 14 | AMI + afib |
| 11228 | Died | 47 | Sepsis + encephalopathy |

---

## Service Management Cheat Sheet

```bash
# Status
sudo systemctl status clinical-mcp-server
sudo systemctl status open-webui

# Logs (live)
journalctl -u clinical-mcp-server -f
journalctl -u open-webui -f

# Restart after code changes to server.py / tools/
sudo systemctl restart clinical-mcp-server

# Stop everything
sudo systemctl stop open-webui clinical-mcp-server

# Disable from auto-start at boot
sudo systemctl disable open-webui clinical-mcp-server
```

---

## Ports Summary

| Service | Port | URL |
|---------|------|-----|
| **Open WebUI** | **8080** | http://localhost:8080 |
| Clinical MCP Server | 8001 | http://localhost:8001/mcp |
| Nemotron 120B (DGX Spark) | 8123 | http://192.168.12.110:8123/v1 |
| SearXNG (web search, optional) | 8080 | ⚠️ **Port conflict!** |

> **⚠️ SearXNG port conflict:** SearXNG currently runs on port 8080.
> Change it before starting Open WebUI:
> ```bash
> docker stop searxng && docker rm searxng
> docker run -d -p 8082:8080 --name searxng searxng/searxng
> ```
> Then update `config.py`:
> ```python
> SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8082")
> ```
> And restart the MCP server: `sudo systemctl restart clinical-mcp-server`

---

## Troubleshooting

**MCP server fails to start (models not found)**
```bash
ls /home/dog/VS/AIinHealthcare/clinical_agent/models/
# Expected: xgb_mortality.pkl  shap_explainer.pkl  lstm_mortality.pt
#           lstm_config.json   icd9_labels.json    clinicalbert_icd9/
```

**OpenWebUI can't connect to MCP server**
```bash
# Confirm the MCP endpoint is live
curl -s http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" | head -50
```

**OpenWebUI can't reach Nemotron**
```bash
curl http://192.168.12.110:8123/v1/models
# Should return: {"data": [{"id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", ...}]}
```

**Tool calls not showing up in chat**
- Ensure the model has **Function Calling → Native** set (not "Default").
- Verify the MCP tool connection shows a green status in Admin → Settings → Tools.
