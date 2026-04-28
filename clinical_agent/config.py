import os
from pathlib import Path
from dotenv import load_dotenv

_HERE = Path(__file__).parent
_ROOT = _HERE.parent

load_dotenv(_HERE / ".env")
load_dotenv(_ROOT / ".env")

_mimic_csv_env = os.getenv("MIMIC_CSV_DIR", "")
MIMIC_CSV_DIR: Path | None = Path(_mimic_csv_env) if _mimic_csv_env else None

def _resolve_mimic_db() -> Path | None:
    explicit = os.getenv("MIMIC_DB", "")
    if explicit:
        return Path(explicit)
    if MIMIC_CSV_DIR:
        return MIMIC_CSV_DIR / "mimic_duck.db"
    return None

MIMIC_DB: Path | None = _resolve_mimic_db()

_SURVIVAL_DATA = Path(
    os.getenv(
        "SURVIVAL_DATA_DIR",
        str(_ROOT / "survival_analysis" / "data"),
    )
)
STATIC_FEATURES_CSV = _SURVIVAL_DATA / "static_features.csv"
TEMPORAL_SEQ_PKL    = _SURVIVAL_DATA / "temporal_sequences.pkl"
TEMPORAL_META_CSV   = _SURVIVAL_DATA / "temporal_meta.csv"

RADIOLOGY_NOTES_CSV = Path(
    os.getenv(
        "RADIOLOGY_NOTES_CSV",
        str(_ROOT / "df_430_radiology_notes.csv"),
    )
)

MODELS_DIR            = _HERE / "models"
XGB_MODEL_PATH        = MODELS_DIR / "xgb_mortality.pkl"
SHAP_EXPLAINER_PATH   = MODELS_DIR / "shap_explainer.pkl"
LSTM_MODEL_PATH       = MODELS_DIR / "lstm_mortality.pt"
LSTM_CONFIG_PATH      = MODELS_DIR / "lstm_config.json"
ICD9_LABELS_PATH      = MODELS_DIR / "icd9_labels.json"
CLINICALBERT_ICD9_DIR = MODELS_DIR / "clinicalbert_icd9"

DATA_DIR              = _HERE / "data"
CHROMA_DB_DIR         = DATA_DIR / "chroma_db"
CHROMA_GUIDELINES_DIR = DATA_DIR / "chroma_guidelines"
CHROMA_NOTES_DIR      = DATA_DIR / "chroma_mimic_notes"
GUIDELINES_DIR        = DATA_DIR / "guidelines"
ICD9_CSV              = DATA_DIR / "icd9_codes.csv"
DRUGBANK_CSV          = DATA_DIR / "drugbank_interactions.csv"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.12.110:8123/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "not-needed")
LLM_MODEL    = os.getenv("LLM_MODEL",    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")

PUBMED_BASE_URL           = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARXNG_URL               = os.getenv("SEARXNG_URL",   "http://localhost:8082")
FIRECRAWL_URL             = os.getenv("FIRECRAWL_URL", "http://localhost:3002")

def _load_ncbi_key() -> str:
    if v := os.getenv("NCBI_API_KEY"):
        return v
    for candidate in (_ROOT / "NCBI_API_KEY", _HERE / "NCBI_API_KEY"):
        if candidate.exists():
            return candidate.read_text().strip()
    return ""

NCBI_API_KEY = _load_ncbi_key()

SCISPACY_MODEL  = "en_ner_bc5cdr_md"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ncbi/MedCPT-Article-Encoder")

MCP_SERVER_NAME = "clinical-agent"
MCP_HOST        = "0.0.0.0"
MCP_PORT        = 8001

DEFAULT_LAB_HOURS   = 48
DEFAULT_VITAL_HOURS = 72
DEFAULT_NOTE_LIMIT  = 5
DEFAULT_SQL_LIMIT   = 500


def check_paths() -> dict:
    def _ex(p) -> bool:
        return p.exists() if p is not None else False

    return {
        "mimic_csv_dir":      MIMIC_CSV_DIR.is_dir() if MIMIC_CSV_DIR else False,
        "mimic_db":           _ex(MIMIC_DB),
        "static_features":    STATIC_FEATURES_CSV.exists(),
        "temporal_meta":      TEMPORAL_META_CSV.exists(),
        "temporal_seq":       TEMPORAL_SEQ_PKL.exists(),
        "xgb_model":          XGB_MODEL_PATH.exists(),
        "shap_explainer":     SHAP_EXPLAINER_PATH.exists(),
        "lstm_model":         LSTM_MODEL_PATH.exists(),
        "lstm_config":        LSTM_CONFIG_PATH.exists(),
        "icd9_labels":        ICD9_LABELS_PATH.exists(),
        "clinicalbert_icd9":  CLINICALBERT_ICD9_DIR.exists(),
        "chroma_literature":  CHROMA_DB_DIR.exists(),
        "chroma_guidelines":  CHROMA_GUIDELINES_DIR.exists(),
        "chroma_mimic_notes": CHROMA_NOTES_DIR.exists(),
    }


if __name__ == "__main__":
    import json
    print(f"MIMIC_CSV_DIR    : {MIMIC_CSV_DIR}")
    print(f"MIMIC_DB         : {MIMIC_DB}")
    print(f"MIMIC_DB exists  : {MIMIC_DB.exists() if MIMIC_DB else False}")
    print(f"SURVIVAL_DATA_DIR: {_SURVIVAL_DATA}")
    print()
    print(json.dumps(check_paths(), indent=2))
