#!/usr/bin/env bash
# run_tests.sh
#
# Run the unit/integration test suite from the repo root.
# Each test is a standalone script that must execute inside clinical_agent/.
# Do NOT run test_phase10 here — that suite calls the live LLM endpoint.
#
# Usage:
#   ./run_tests.sh           Run all unit tests (unit tests + RAG)
#   ./run_tests.sh --module  Run a single test, e.g.: ./run_tests.sh test_phase3
#   ./run_tests.sh --e2e     Run the end-to-end agentic scenario test (slow, needs LLM)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="${REPO_DIR}/clinical_agent"
VENV_PY="${REPO_DIR}/clinicalAgentEnv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: virtualenv not found at ${VENV_PY}"
    echo "Run ./deploy.sh first to create the environment."
    exit 1
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

UNIT_TESTS=(
    test_phase2
    test_phase3
    test_phase4
    test_phase5
    test_phase6
    test_phase7
    test_phase8
    test_phase9
    test_rag_r6_r7
)

E2E_TEST="test_phase10"

run_test() {
    local name="$1"
    local src="${REPO_DIR}/tests/${name}.py"
    if [[ ! -f "$src" ]]; then
        echo -e "${RED}[SKIP]${RESET} ${name}.py not found"
        return
    fi
    echo -e "\n${CYAN}[RUN]${RESET} ${name}"
    if CUDA_VISIBLE_DEVICES=0 \
       "$VENV_PY" "$src"; then
        echo -e "${GREEN}[PASS]${RESET} ${name}"
    else
        echo -e "${RED}[FAIL]${RESET} ${name}"
        FAILED+=("$name")
    fi
}

FAILED=()

if [[ "${1:-}" == "--e2e" ]]; then
    echo -e "${YELLOW}Running end-to-end agentic test (needs live LLM endpoint).${RESET}"
    echo -e "${YELLOW}This calls the Nemotron inference server and may be slow.${RESET}"
    echo ""
    run_test "$E2E_TEST"

elif [[ "${1:-}" == "--module" && -n "${2:-}" ]]; then
    run_test "$2"

else
    echo -e "${BOLD}Clinical MCP Agent - Unit Test Suite${RESET}"
    echo -e "Running ${#UNIT_TESTS[@]} test modules from ${REPO_DIR}/tests/"
    echo -e "${YELLOW}Skipping ${E2E_TEST} (pass --e2e to run it).${RESET}"
    echo ""

    for t in "${UNIT_TESTS[@]}"; do
        run_test "$t"
    done
fi

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo -e "${GREEN}All tests passed.${RESET}"
    exit 0
else
    echo -e "${RED}Failed: ${FAILED[*]}${RESET}"
    exit 1
fi
