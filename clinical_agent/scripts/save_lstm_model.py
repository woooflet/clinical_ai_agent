"""
scripts/save_lstm_model.py   GPU model preparation.

Run with clinicalAgentEnv:
    cd clinical_agent
    ../clinicalAgentEnv/bin/python scripts/save_lstm_model.py

What this does:
  1. Copies survival_analysis/models/lstm_best.pt → models/lstm_mortality.pt
     and saves metadata (n_features, hidden_dim, n_layers) alongside it as
     models/lstm_config.json   so ml_models.py can reconstruct the model
     architecture without importing the training script.
  2. Reconstructs the top-50 ICD-9 label mapping from MIMIC-III
     (replicates tutorial_4_clinicalbert_icd9.py logic with IS_TEST_RUN=False)
     and saves it as models/icd9_labels.json   needed for predict_icd9 to
     map sigmoid output indices back to real ICD-9 codes.
"""

import sys, shutil, json, os
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_AGENT_DIR   = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_AGENT_DIR))
import config

SRC_LSTM    = config._ROOT / "survival_analysis" / "models" / "lstm_best.pt"
MODELS_DIR  = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DST_LSTM        = MODELS_DIR / "lstm_mortality.pt"
DST_LSTM_CFG    = MODELS_DIR / "lstm_config.json"
DST_ICD9_LABELS = MODELS_DIR / "icd9_labels.json"
DST_ICD9_MODEL  = MODELS_DIR / "clinicalbert_icd9"

MIMIC_DB        = config.MIMIC_DB
BERT_CHECKPOINT = config._ROOT / "Assignment_Tutorials" / "clinicalbert_icd9_results" / "checkpoint-25"

print("=" * 60)
print("Step 1: Copying LSTM model...")
if not SRC_LSTM.exists():
    print(f"  ERROR: Source not found: {SRC_LSTM}")
    sys.exit(1)

shutil.copy2(SRC_LSTM, DST_LSTM)
print(f"  ok Copied {SRC_LSTM.name} → {DST_LSTM}")

lstm_config = {
    "n_features": 10,
    "hidden_dim": 64,
    "n_layers": 2,
    "dropout": 0.3,
    "n_timesteps": 12,
    "description": (
        "Bi-LSTM + Scaled Self-Attention trained on 72h ICU temporal sequences. "
        "Input: (batch, 12, 10). Output: mortality logit (sigmoid for probability). "
        "Matches BiLSTMAttentionLogits in survival_analysis/05_lstm_model.py."
    ),
}
with open(DST_LSTM_CFG, "w") as f:
    json.dump(lstm_config, f, indent=2)
print(f"  ok Saved lstm_config.json: {lstm_config}")

print("\nStep 2: Building ICD-9 label mapping from MIMIC-III...")
try:
    import duckdb
    con = duckdb.connect(str(MIMIC_DB), read_only=True)

    result = con.execute("""
        SELECT ICD9_CODE, COUNT(*) AS cnt
        FROM diagnoses_icd
        WHERE ICD9_CODE IS NOT NULL
        GROUP BY ICD9_CODE
        ORDER BY cnt DESC
        LIMIT 50
    """).df()

    top_50_codes = result["ICD9_CODE"].tolist()
    print(f"  Top-50 ICD-9 codes: {top_50_codes[:10]} ...")

    descs = {}
    for code in top_50_codes:
        row = con.execute(
            "SELECT SHORT_TITLE FROM d_icd_diagnoses WHERE ICD9_CODE = ? LIMIT 1",
            [code]
        ).fetchone()
        descs[code] = row[0] if row else code

    con.close()

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit([top_50_codes])
    classes = mlb.classes_.tolist()

    icd9_labels = {
        "classes": classes,
        "descriptions": {c: descs.get(c, c) for c in classes},
        "num_labels": len(classes),
        "source": "MIMIC-III DIAGNOSES_ICD top-50 codes by frequency",
    }
    with open(DST_ICD9_LABELS, "w") as f:
        json.dump(icd9_labels, f, indent=2)
    print(f"  ok Saved icd9_labels.json: {len(classes)} classes")
    print(f"    Classes[0:5]: {classes[:5]}")

except Exception as e:
    print(f"  WARNING: Could not build ICD-9 label map: {e}")
    print("  predict_icd9 will fall back to generic LABEL_N outputs.")

print("\nStep 3: Copying ClinicalBERT ICD-9 checkpoint...")
if BERT_CHECKPOINT.exists():
    if DST_ICD9_MODEL.exists():
        shutil.rmtree(DST_ICD9_MODEL)
    shutil.copytree(BERT_CHECKPOINT, DST_ICD9_MODEL)
    print(f"  ok Copied checkpoint-25 → {DST_ICD9_MODEL}")
else:
    print(f"  WARNING: ClinicalBERT checkpoint not found at {BERT_CHECKPOINT}")
    print("  predict_icd9 will fall back to pretrained Bio_ClinicalBERT (no fine-tuning).")

print("\nStep 4: Validating LSTM load + forward pass...")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SelfAttention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.scale = hidden_dim ** 0.5
            self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        def forward(self, x):
            Q, K, V = self.q(x), self.k(x), self.v(x)
            scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
            attn   = F.softmax(scores, dim=-1)
            return torch.bmm(attn, V), attn

    class BiLSTMAttentionLogits(nn.Module):
        def __init__(self, n_features, hidden_dim=64, n_layers=2, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Linear(n_features, hidden_dim)
            self.lstm = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=n_layers, batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0,
            )
            self.attention = SelfAttention(hidden_dim * 2)
            self.dropout   = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.input_proj(x)
            lstm_out, _ = self.lstm(x)
            attn_out, attn_weights = self.attention(lstm_out)
            context = attn_out.mean(dim=1)
            context = self.dropout(context)
            logit = self.classifier[:-1](context)
            return logit.squeeze(-1), attn_weights

        def predict_proba(self, x):
            logit, attn = self.forward(x)
            return torch.sigmoid(logit), attn

        def get_embedding(self, x):
            x = self.input_proj(x)
            lstm_out, _ = self.lstm(x)
            attn_out, _ = self.attention(lstm_out)
            return attn_out.mean(dim=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttentionLogits(
        n_features=lstm_config["n_features"],
        hidden_dim=lstm_config["hidden_dim"],
        n_layers=lstm_config["n_layers"],
        dropout=lstm_config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(DST_LSTM, map_location=device))
    model.eval()

    dummy = torch.randn(2, lstm_config["n_timesteps"], lstm_config["n_features"]).to(device)
    with torch.no_grad():
        prob, attn = model.predict_proba(dummy)
    print(f"  ok LSTM forward pass OK on {device}")
    print(f"    Dummy probs: {prob.cpu().numpy()}")
    print(f"    Attn shape: {attn.shape}")

except Exception as e:
    print(f"  ERROR during LSTM validation: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("model preparation complete.")
print(f"  LSTM model:    {DST_LSTM}")
print(f"  LSTM config:   {DST_LSTM_CFG}")
print(f"  ICD-9 labels:  {DST_ICD9_LABELS}")
print(f"  ClinicalBERT:  {DST_ICD9_MODEL}")
print("=" * 60)
