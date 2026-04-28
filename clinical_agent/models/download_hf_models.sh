#!/usr/bin/env bash
# =============================================================================
# clinical_agent/models/download_hf_models.sh
#
# Downloads HuggingFace model dependencies needed by the Clinical AI Agent.
# Run this once after cloning the repo:
#
#   cd clinical-ai-agent
#   bash clinical_agent/models/download_hf_models.sh
#
# Requirements: huggingface-cli  (pip install huggingface_hub[cli])
# =============================================================================
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_hf_cli() {
    if ! command -v huggingface-cli &>/dev/null; then
        echo "[ERROR] huggingface-cli not found."
        echo "  Install with: pip install 'huggingface_hub[cli]'"
        exit 1
    fi
}

# 
# 1. MedCPT-Article-Encoder  (RAG embedding model)
#    Used by: tools/literature.py, scripts/batch_seed_chroma.py,
#             scripts/seed_guidelines.py, scripts/seed_mimic_notes.py
#    Source:  https://huggingface.co/ncbi/MedCPT-Article-Encoder
#    Note:    Downloaded automatically by sentence-transformers at first use.
#             Listed here for documentation / offline pre-cache purposes.
# 
download_medcpt() {
    echo "[1/2] Caching MedCPT-Article-Encoder..."
    huggingface-cli download ncbi/MedCPT-Article-Encoder \
        --local-dir "${MODELS_DIR}/medcpt_cache" \
        --local-dir-use-symlinks False \
        --quiet
    echo "      → ${MODELS_DIR}/medcpt_cache"
    echo "      (sentence-transformers also caches this automatically on first run)"
}

# 
# 2. Bio_ClinicalBERT base model  (NLP tools + ICD-9 classifier fallback)
#    Used by: tools/nlp_tools.py, tools/ml_models.py (predict_icd9 fallback)
#    Source:  https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
#
#    Fine-tuned checkpoint: The ICD-9 fine-tuned weights (clinicalbert_icd9/)
#    are committed to this repo via Git LFS (model.safetensors, 414MB).
#    If Git LFS is not set up, ml_models.py automatically falls back to this
#    base model for zero-shot ICD-9 prediction.
#
#    Training provenance — how the fine-tuned checkpoint was created:
#      Base model : emilyalsentzer/Bio_ClinicalBERT
#      Task       : Multi-label ICD-9 classification (top-50 codes by frequency)
#      Data       : MIMIC-III DIAGNOSES_ICD + NOTEEVENTS (discharge summaries)
#                   (requires PhysioNet credentialing — cannot be distributed)
#      Training   : TransformerTrainer, 1 epoch, lr=2e-5, batch=8, fp16
#                   See: survival_analysis/05_lstm_model.py (commented section)
#                   and the original notebook (AIinHealthcare repo, not public)
#      Checkpoint : checkpoint-25  (25 steps × batch 8 = 200 training samples)
#      Metrics    : Micro-F1 ~0.42 on held-out MIMIC split (limited by
#                   small training set — clinical notes are access-restricted)
# 
download_clinicalbert_base() {
    echo "[2/2] Caching Bio_ClinicalBERT base model..."
    huggingface-cli download emilyalsentzer/Bio_ClinicalBERT \
        --local-dir "${MODELS_DIR}/bio_clinicalbert_base" \
        --local-dir-use-symlinks False \
        --quiet
    echo "      → ${MODELS_DIR}/bio_clinicalbert_base"
}

# 
# Custom-trained models (committed directly to git — no download needed)
# 
# xgb_mortality.pkl    — XGBoost binary mortality classifier
#   Trained by: survival_analysis/04_cox_model.py
#   Data:       MIMIC-III static features (02_extract_static_features.py output)
#   Features:   ~45 lab + vital + demographic features from first 6h of ICU stay
#   AUC:        ~0.85 on held-out MIMIC test split
#
# shap_explainer.pkl   — SHAP TreeExplainer for xgb_mortality
#   Created by: clinical_agent/scripts/save_models.py
#
# lstm_mortality.pt    — Bi-LSTM + Self-Attention temporal mortality model
#   Trained by: survival_analysis/05_lstm_model.py
#   Data:       MIMIC-III 72h temporal sequences (03_extract_temporal_features.py)
#   Input:      (batch, 12 timesteps × 10 features)  [6h bins over 72h]
#   AUC:        ~0.88 on held-out MIMIC test split
#
# lstm_config.json     — Architecture spec to reconstruct model at inference time
# icd9_labels.json     — Top-50 ICD-9 codes for ClinicalBERT output mapping
# 

main() {
    check_hf_cli

    echo "================================================="
    echo " Clinical AI Agent — HuggingFace Model Setup"
    echo "================================================="
    echo "Models dir: ${MODELS_DIR}"
    echo ""

    download_medcpt
    echo ""
    download_clinicalbert_base
    echo ""

    echo "================================================="
    echo " Done. Custom-trained weights (XGBoost, LSTM,"
    echo " SHAP) are already in models/ — committed to git."
    echo ""
    echo " ClinicalBERT fine-tuned checkpoint (414MB) is"
    echo " in models/clinicalbert_icd9/ via Git LFS."
    echo " Run: git lfs pull   (if not already fetched)"
    echo "================================================="
}

main "$@"
