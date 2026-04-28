#!/usr/bin/env bash

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}=== $* ===${RESET}\n"; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="${REPO_DIR}/clinical_agent"
SERVICE_DIR="${AGENT_DIR}/openwebui"
DOCKER_DIR="${REPO_DIR}/docker"
VENV_DIR="${REPO_DIR}/clinicalAgentEnv"
CONFIG_FILE="${AGENT_DIR}/config.py"
ENV_FILE="${REPO_DIR}/.env"

MCP_SERVICE="clinical-mcp-server"
WEBUI_SERVICE="open-webui"
MCP_UNIT_SRC="${SERVICE_DIR}/clinical-mcp-server.service"
WEBUI_UNIT_SRC="${SERVICE_DIR}/open-webui.service"
MCP_UNIT_DST="/etc/systemd/system/clinical-mcp-server.service"
WEBUI_UNIT_DST="/etc/systemd/system/open-webui.service"

need_sudo() {
    if [[ $EUID -ne 0 ]]; then
        if ! sudo -n true 2>/dev/null; then
            warn "Some steps require sudo. You may be prompted for your password."
        fi
    fi
}

service_active()  { command -v systemctl &>/dev/null && systemctl is-active  --quiet "$1" 2>/dev/null || return 1; }
service_enabled() { command -v systemctl &>/dev/null && systemctl is-enabled --quiet "$1" 2>/dev/null || return 1; }

case "$(uname -s 2>/dev/null)" in
    Linux*)   OS_TYPE="linux"   ;;
    Darwin*)  OS_TYPE="macos"   ;;
    CYGWIN*|MINGW*|MSYS*) OS_TYPE="windows" ;;
    *)        OS_TYPE="unknown" ;;
esac

if [[ "$OS_TYPE" == "linux" ]] && grep -qi microsoft /proc/version 2>/dev/null; then
    OS_TYPE="wsl"
fi

info "Detected platform: ${OS_TYPE}"

print_status() {
    header "Service Status"
    if ! command -v systemctl &>/dev/null; then
        warn "systemctl not available on this platform (${OS_TYPE})."
        warn "Start the MCP server manually:"
        warn "  cd ${AGENT_DIR} && python server.py --transport streamable-http"
        return
    fi
    for svc in "$MCP_SERVICE" "$WEBUI_SERVICE"; do
        if service_active "$svc"; then
            echo -e "  ${GREEN}●${RESET} ${BOLD}${svc}${RESET}  active (running)"
            journalctl -u "$svc" --no-pager -n 1 2>/dev/null \
                | tail -1 | sed 's/^/      /'
        else
            local state
            state=$(systemctl is-active "$svc" 2>/dev/null || true)
            echo -e "  ${RED}●${RESET} ${BOLD}${svc}${RESET}  ${state}"
        fi
    done

    echo ""
    echo -e "  ${BOLD}Endpoints:${RESET}"
    echo -e "    MCP Server  →  http://localhost:8001/mcp"
    echo -e "    Open WebUI  →  http://localhost:8080"
    echo ""
}

MODE="install"
case "${1:-}" in
    --start)     MODE="start"     ;;
    --stop)      MODE="stop"      ;;
    --restart)   MODE="restart"   ;;
    --status)    MODE="status"    ;;
    --uninstall) MODE="uninstall" ;;
    --help|-h)   MODE="help"      ;;
    "")          MODE="install"   ;;
    *)  error "Unknown flag: $1"; echo "Run ./deploy.sh --help"; exit 1 ;;
esac

if [[ "$MODE" == "help" ]]; then
    cat <<EOF

${BOLD}Clinical MCP Agent   deploy.sh${RESET}

USAGE
  ./deploy.sh [FLAG]

FLAGS
  (none)          Full install: create venv, configure paths, install systemd
                  units, optionally build MIMIC DuckDB, seed RAG, start services.
  --start         Start MCP server + Open WebUI (units must already be installed).
  --stop          Gracefully stop both services.
  --restart       Restart both services (e.g. after a code change).
  --status        Show live status of both services + endpoint URLs.
  --uninstall     Stop services, disable + remove systemd units, stop Docker
                  containers. Does NOT delete data or model files.
  --help          Show this help message.

EXAMPLES
  ./deploy.sh
  ./deploy.sh --restart
  ./deploy.sh --stop
  ./deploy.sh --uninstall

EOF
    exit 0
fi

do_stop() {
    header "Stopping Services"
    for svc in "$MCP_SERVICE" "$WEBUI_SERVICE"; do
        if service_active "$svc"; then
            sudo systemctl stop "$svc"
            success "Stopped ${svc}"
        else
            info "${svc} was not running"
        fi
    done

    if command -v docker &>/dev/null && [[ -f "${DOCKER_DIR}/docker-compose.yml" ]]; then
        if docker compose -f "${DOCKER_DIR}/docker-compose.yml" ps -q 2>/dev/null | grep -q .; then
            info "Stopping Docker containers..."
            docker compose -f "${DOCKER_DIR}/docker-compose.yml" stop
            success "Docker containers stopped"
        fi
    fi
}

if [[ "$MODE" == "stop" ]]; then
    need_sudo; do_stop; echo ""; print_status; exit 0
fi

do_start() {
    header "Starting Services"

    if ! [[ -f "$MCP_UNIT_DST" ]]; then
        error "MCP service unit not installed at ${MCP_UNIT_DST}"
        error "Run ./deploy.sh (without flags) to do a full install first."
        exit 1
    fi

    sudo systemctl start "$MCP_SERVICE"
    success "Started ${MCP_SERVICE}"

    sudo systemctl start "$WEBUI_SERVICE"
    success "Started ${WEBUI_SERVICE}"
}

if [[ "$MODE" == "start" ]]; then
    need_sudo; do_start; echo ""; print_status; exit 0
fi

if [[ "$MODE" == "restart" ]]; then
    need_sudo
    header "Restarting Services"
    for svc in "$MCP_SERVICE" "$WEBUI_SERVICE"; do
        sudo systemctl restart "$svc"
        success "Restarted ${svc}"
    done
    echo ""
    info "Waiting 5s for startup..."
    sleep 5
    print_status
    exit 0
fi

if [[ "$MODE" == "status" ]]; then
    print_status; exit 0
fi

if [[ "$MODE" == "uninstall" ]]; then
    need_sudo
    header "Uninstalling Clinical Agent Services"

    warn "This will stop and remove the systemd services."
    warn "Your data, models, and ChromaDB files will NOT be deleted."
    echo ""
    read -rp "Continue? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }

    for svc in "$MCP_SERVICE" "$WEBUI_SERVICE"; do
        if service_active "$svc"; then
            sudo systemctl stop "$svc"
            success "Stopped ${svc}"
        fi
        if service_enabled "$svc"; then
            sudo systemctl disable "$svc"
            success "Disabled ${svc}"
        fi
    done

    for unit in "$MCP_UNIT_DST" "$WEBUI_UNIT_DST"; do
        if [[ -f "$unit" ]]; then
            sudo rm "$unit"
            success "Removed ${unit}"
        fi
    done

    sudo systemctl daemon-reload
    success "systemd reloaded"

    if command -v docker &>/dev/null && [[ -f "${DOCKER_DIR}/docker-compose.yml" ]]; then
        if docker compose -f "${DOCKER_DIR}/docker-compose.yml" ps -q 2>/dev/null | grep -q .; then
            info "Stopping Docker containers..."
            docker compose -f "${DOCKER_DIR}/docker-compose.yml" down
            success "Docker containers removed"
        fi
    fi

    echo ""
    success "Uninstall complete."
    echo -e "  ${YELLOW}Note:${RESET} venv, models, and data are still at ${REPO_DIR}"
    echo -e "  To fully clean up: rm -rf ${REPO_DIR}"
    exit 0
fi


if [[ "$OS_TYPE" == "macos" ]]; then
    echo ""
    echo -e "${BOLD}${CYAN}+==============================================================+${RESET}"
    echo -e "${BOLD}${CYAN}|  macOS Detected   Manual Setup Guide                         |${RESET}"
    echo -e "${BOLD}${CYAN}+==============================================================+${RESET}"
    echo ""
    echo -e "  ${BOLD}deploy.sh uses Linux systemd for process management.${RESET}"
    echo -e "  On macOS, run the agent manually instead. Steps:"
    echo ""
    echo -e "  ${BOLD}1. Install Python 3.11+ (if needed)${RESET}"
    echo -e "     brew install python@3.11"
    echo ""
    echo -e "  ${BOLD}2. Create virtual environment${RESET}"
    echo -e "     python3 -m venv ${VENV_DIR}"
    echo -e "     source ${VENV_DIR}/bin/activate"
    echo -e "     pip install -r requirements.txt"
    echo -e "     pip install --no-deps scispacy"
    echo -e "     pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
    echo ""
    echo -e "  ${BOLD}3. Configure paths${RESET}"
    echo -e "     cp .env.example .env"
    echo -e "     # Edit .env   set MIMIC_CSV_DIR, MIMIC_DB, LLM_BASE_URL"
    echo ""
    echo -e "  ${BOLD}4. Build MIMIC DuckDB (if you have the CSVs)${RESET}"
    echo -e "     python clinical_agent/scripts/build_mimic_duckdb.py"
    echo ""
    echo -e "  ${BOLD}5. Install OpenWebUI${RESET}"
    echo -e "     pip install open-webui"
    echo -e "     open-webui serve   # opens http://localhost:8080"
    echo ""
    echo -e "  ${BOLD}6. Start the MCP server${RESET}"
    echo -e "     cd ${AGENT_DIR}"
    echo -e "     python server.py --transport streamable-http"
    echo -e "     # Runs on http://localhost:8001/mcp"
    echo ""
    echo -e "  ${BOLD}7. Connect OpenWebUI to the MCP server${RESET}"
    echo -e "     Admin Panel → Settings → Tools → Add Tool Server"
    echo -e "     URL: http://localhost:8001/mcp   Type: MCP (Streamable HTTP)"
    echo ""
    echo -e "  ${YELLOW}Tip: to auto-restart on macOS, create a launchd plist or use:${RESET}"
    echo -e "     brew install supervisor && supervisord  (community approach)"
    echo ""
    exit 0
fi

if [[ "$OS_TYPE" == "windows" ]]; then
    echo ""
    echo -e "${BOLD}${RED}+==============================================================+${RESET}"
    echo -e "${BOLD}${RED}|  Windows Detected   WSL2 Required                            |${RESET}"
    echo -e "${BOLD}${RED}+==============================================================+${RESET}"
    echo ""
    echo -e "  deploy.sh requires a Linux shell. On Windows, use ${BOLD}WSL2${RESET}:"
    echo ""
    echo -e "  ${BOLD}Option A: WSL2 (recommended)${RESET}"
    echo -e "    1. Open PowerShell as Administrator and run:"
    echo -e "         wsl --install"
    echo -e "    2. Restart, then open the Ubuntu app"
    echo -e "    3. In WSL, clone this repo and run:"
    echo -e "         ./deploy.sh"
    echo ""
    echo -e "  ${BOLD}Option B: Docker Desktop${RESET}"
    echo -e "    1. Install Docker Desktop for Windows"
    echo -e "    2. Clone this repo, then:"
    echo -e "         docker compose -f docker/docker-compose.yml up -d"
    echo -e "    3. Then start the MCP server:"
    echo -e "         docker exec -it clinical-agent python server.py --transport streamable-http"
    echo ""
    exit 0
fi

if [[ "$OS_TYPE" == "unknown" ]]; then
    warn "Unrecognised OS   proceeding, but systemd steps may fail."
    warn "If this fails, follow the macOS manual guide above (same steps)."
fi

if [[ "$OS_TYPE" == "wsl" ]]; then
    warn "Running inside WSL. systemd may not be active."
    warn "If 'systemctl' commands fail, use the manual run approach:"
    warn "  cd ${AGENT_DIR} && python server.py --transport streamable-http"
    echo ""
fi

header "Clinical Agent   Full Deployment"
echo -e "  Repo: ${BOLD}${REPO_DIR}${RESET}"
echo -e "  OS:   ${BOLD}${OS_TYPE}${RESET}"
echo ""

need_sudo

header "Step 1/6: Python Environment"

if [[ -d "$VENV_DIR" ]]; then
    success "virtualenv already exists at ${VENV_DIR}"
else
    info "Creating virtualenv at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
    success "virtualenv created"
fi

PY="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

info "Installing Python dependencies from requirements.txt..."
"$PIP" install --quiet --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r "${REPO_DIR}/requirements.txt"
success "Python dependencies installed"

if ! "$PY" -c "import scispacy" 2>/dev/null; then
    info "Installing scispaCy (--no-deps)..."
    "$PIP" install --quiet --no-deps scispacy
    "$PIP" install --quiet --no-deps \
        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
    success "scispaCy + en_ner_bc5cdr_md installed"
else
    success "scispaCy already installed"
fi

header "Step 2/7: Configuration"

if [[ -f "$ENV_FILE" ]]; then
    info "Loading existing ${ENV_FILE}"
    set -o allexport; source "$ENV_FILE"; set +o allexport
fi

echo ""
echo -e "${BOLD}Configuration (press Enter to accept the current value):${RESET}"
echo ""

echo -e "  ${CYAN}MIMIC-III data${RESET}"
echo -e "  MIMIC is distributed as CSV files from PhysioNet."
echo -e "  If you have them, enter the directory path here."
echo -e "  Leave blank to skip MIMIC tools (all other tools still work)."
echo ""
read -rp "  Path to MIMIC-III CSV directory  [${MIMIC_CSV_DIR:-skip}]: " input_csv
MIMIC_CSV_DIR="${input_csv:-${MIMIC_CSV_DIR:-}}"
if [[ -n "$MIMIC_CSV_DIR" ]] && ! [[ -d "$MIMIC_CSV_DIR" ]]; then
    warn "Directory not found: ${MIMIC_CSV_DIR}   double-check the path"
fi

if [[ -n "$MIMIC_CSV_DIR" ]]; then
    _default_db="${MIMIC_CSV_DIR}/mimic_duck.db"
    read -rp "  Path for mimic_duck.db (will be created if needed)  [${MIMIC_DB:-${_default_db}}]: " input_db
    MIMIC_DB="${input_db:-${MIMIC_DB:-${_default_db}}}"
else
    MIMIC_DB="${MIMIC_DB:-}"
fi

echo ""
echo -e "  ${CYAN}DrugBank (optional)${RESET}    drug interaction tool. Press Enter to skip."
read -rp "  Full path to drugbank.db  [${DRUGBANK_DB:-skip}]: " input_drugbank
DRUGBANK_DB="${input_drugbank:-${DRUGBANK_DB:-}}"
if [[ -n "$DRUGBANK_DB" ]] && ! [[ -f "$DRUGBANK_DB" ]]; then
    warn "drugbank.db not found at ${DRUGBANK_DB}   drug interaction tool will be limited"
fi

echo ""
echo -e "  ${CYAN}LLM endpoint${RESET}"
read -rp "  LLM API base URL  [${LLM_BASE_URL:-http://localhost:8123/v1}]: " input_llm
LLM_BASE_URL="${input_llm:-${LLM_BASE_URL:-http://localhost:8123/v1}}"

read -rp "  LLM model name   [${LLM_MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}]: " input_model
LLM_MODEL="${input_model:-${LLM_MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}}"

read -rp "  NCBI API key (optional, raises PubMed rate limit) [${NCBI_API_KEY:-}]: " input_ncbi
NCBI_API_KEY="${input_ncbi:-${NCBI_API_KEY:-}}"

cat > "$ENV_FILE" <<EOF

MIMIC_CSV_DIR=${MIMIC_CSV_DIR}

MIMIC_DB=${MIMIC_DB}

DRUGBANK_DB=${DRUGBANK_DB}

LLM_BASE_URL=${LLM_BASE_URL}
LLM_MODEL=${LLM_MODEL}

NCBI_API_KEY=${NCBI_API_KEY}
EOF
success "Settings saved to ${ENV_FILE}"

header "Step 3/7: MIMIC-III DuckDB Setup"

BUILD_DB="n"
if [[ -n "$MIMIC_CSV_DIR" ]] && [[ -d "$MIMIC_CSV_DIR" ]]; then
    if [[ -n "$MIMIC_DB" ]] && [[ -f "$MIMIC_DB" ]]; then
        success "DuckDB already exists at ${MIMIC_DB}   skipping build"
        BUILD_DB="skip"
    else
        echo ""
        echo -e "  MIMIC CSVs found at: ${BOLD}${MIMIC_CSV_DIR}${RESET}"
        echo -e "  DuckDB target:       ${BOLD}${MIMIC_DB:-${MIMIC_CSV_DIR}/mimic_duck.db}${RESET}"
        echo ""
        echo -e "  ${YELLOW}Building the DuckDB takes 10-40 minutes depending on hardware.${RESET}"
        echo -e "  CHARTEVENTS.csv alone is ~33 GB uncompressed."
        echo -e "  The build is resumable   re-run if interrupted."
        echo ""
        read -rp "  Build DuckDB now? [y/N]: " BUILD_DB
    fi
else
    info "No MIMIC CSV directory configured   skipping DuckDB build"
    info "MIMIC patient data tools will be DISABLED until you configure MIMIC_CSV_DIR"
    info "Re-run ./deploy.sh at any time to add MIMIC support"
fi

if [[ "$BUILD_DB" =~ ^[Yy]$ ]]; then
    info "Building MIMIC DuckDB (this may take 10-40 min)..."
    MIMIC_CSV_DIR="${MIMIC_CSV_DIR}" \
    MIMIC_DB="${MIMIC_DB:-${MIMIC_CSV_DIR}/mimic_duck.db}" \
    "${VENV_DIR}/bin/python" "${AGENT_DIR}/scripts/build_mimic_duckdb.py"
    if [[ $? -eq 0 ]]; then
        success "DuckDB build complete"
        if [[ -z "${MIMIC_DB:-}" ]]; then
            MIMIC_DB="${MIMIC_CSV_DIR}/mimic_duck.db"
            sed -i "s|^MIMIC_DB=.*|MIMIC_DB=${MIMIC_DB}|" "$ENV_FILE"
            info "Updated MIMIC_DB in ${ENV_FILE}"
        fi
    else
        warn "DuckDB build encountered errors   check output above"
        warn "You can retry later: MIMIC_CSV_DIR=${MIMIC_CSV_DIR} python clinical_agent/scripts/build_mimic_duckdb.py"
    fi
fi

header "Step 4/7: Installing systemd Service Units"

TMP_MCP_UNIT="/tmp/clinical-mcp-server.service"
sed \
    -e "s|WorkingDirectory=.*|WorkingDirectory=${AGENT_DIR}|g" \
    -e "s|ExecStart=.*python \\\\|ExecStart=${VENV_DIR}/bin/python \\\\|g" \
    -e "s|/home/dog/VS/AIinHealthcare/clinical_agent/server\.py|${AGENT_DIR}/server.py|g" \
    -e "s|/home/dog/VS/clinical-ai-agent/clinical_agent/server\.py|${AGENT_DIR}/server.py|g" \
    "$MCP_UNIT_SRC" > "$TMP_MCP_UNIT"

sudo cp "$TMP_MCP_UNIT" "$MCP_UNIT_DST"
sudo cp "$WEBUI_UNIT_SRC" "$WEBUI_UNIT_DST"

sudo systemctl daemon-reload
sudo systemctl enable "$MCP_SERVICE" "$WEBUI_SERVICE"

success "Installed and enabled ${MCP_SERVICE}"
success "Installed and enabled ${WEBUI_SERVICE}"

header "Step 5/7: Docker (Open WebUI + SearXNG)"

USE_DOCKER="n"
if command -v docker &>/dev/null; then
    echo ""
    read -rp "  Start Open WebUI via Docker Compose? [y/N]: " USE_DOCKER
fi

if [[ "$USE_DOCKER" =~ ^[Yy]$ ]]; then
    if [[ -f "${DOCKER_DIR}/.env" ]]; then
        info "Using existing ${DOCKER_DIR}/.env"
    else
        cp "${DOCKER_DIR}/.env.example" "${DOCKER_DIR}/.env"
        sed -i "s|LLM_BASE_URL=.*|LLM_BASE_URL=${LLM_BASE_URL}|g" "${DOCKER_DIR}/.env"
        info "Created ${DOCKER_DIR}/.env from template"
    fi
    docker compose -f "${DOCKER_DIR}/docker-compose.yml" up -d
    success "Docker: open-webui + searxng containers started"
else
    info "Skipping Docker   using systemd open-webui.service instead"
fi

header "Step 6/7: RAG Database Seeding (Optional)"

echo ""
echo -e "  ${BOLD}ChromaDB collections to seed:${RESET}"
echo -e "    a) Clinical literature  (603 PubMed papers, ~3 min, needs NCBI key)"
echo -e "    b) Clinical guidelines  (10 PDFs   place in ${AGENT_DIR}/data/guidelines/ first)"
echo -e "    c) MIMIC note index     (5,000 discharge summaries, ~20 min)"
echo -e "    s) Skip seeding         (if you have existing ChromaDB data)"
echo ""
read -rp "  Choose [a/b/c/s, or 'abc' to run all]: " seed_choice

cd "$AGENT_DIR"

if [[ "$seed_choice" == *"a"* ]]; then
    info "Seeding clinical literature (background)..."
    NCBI_API_KEY="${NCBI_API_KEY}" \
    nohup "${VENV_DIR}/bin/python" scripts/batch_seed_chroma.py \
        > logs/seed_literature.log 2>&1 &
    success "Literature seeder running in background   tail logs/seed_literature.log"
fi

if [[ "$seed_choice" == *"b"* ]]; then
    if ls data/guidelines/*.pdf &>/dev/null 2>&1; then
        info "Seeding clinical guidelines..."
        "${VENV_DIR}/bin/python" scripts/seed_guidelines.py
        success "Guidelines seeded"
    else
        warn "No PDFs found in ${AGENT_DIR}/data/guidelines/   skipping guideline seed"
        warn "Place PDFs there and run: cd ${AGENT_DIR} && ../clinicalAgentEnv/bin/python scripts/seed_guidelines.py"
    fi
fi

if [[ "$seed_choice" == *"c"* ]]; then
    if [[ -n "${MIMIC_DB:-}" ]] && [[ -f "${MIMIC_DB}" ]]; then
        info "Seeding MIMIC note index (background, ~20 min)..."
        MIMIC_DB="${MIMIC_DB}" \
        nohup "${VENV_DIR}/bin/python" scripts/seed_mimic_notes.py \
            > logs/seed_mimic_notes.log 2>&1 &
        success "MIMIC note seeder running in background   tail logs/seed_mimic_notes.log"
    else
        warn "MIMIC DuckDB not found   skipping MIMIC note seed"
        warn "Build the DuckDB first: python clinical_agent/scripts/build_mimic_duckdb.py"
    fi
fi

cd "$REPO_DIR"

header "Step 7/7: Starting Services"

sudo systemctl start "$MCP_SERVICE"
success "Started ${MCP_SERVICE}"

if [[ ! "$USE_DOCKER" =~ ^[Yy]$ ]]; then
    sudo systemctl start "$WEBUI_SERVICE"
    success "Started ${WEBUI_SERVICE}"
fi

sleep 5
print_status

echo -e "${BOLD}Next steps:${RESET}"
echo ""
echo -e "  1. Open WebUI:  ${CYAN}http://localhost:8080${RESET}"
echo -e "     Create your admin account on first visit."
echo ""
echo -e "  2. Add the MCP tool server in Open WebUI:"
echo -e "     Admin Panel → Settings → Tools → + Add Tool Server"
echo -e "     URL: ${CYAN}http://$(hostname -I | awk '{print $1}'):8001/mcp${RESET}"
echo -e "     Type: ${BOLD}MCP (Streamable HTTP)${RESET}"
echo ""
echo -e "  3. Enable native function calling on your model:"
echo -e "     Admin Panel → Models → your model → Advanced → Function Calling: ${BOLD}Native${RESET}"
echo ""
echo -e "  4. After any code change:"
echo -e "     ${CYAN}./deploy.sh --restart${RESET}"
echo ""
