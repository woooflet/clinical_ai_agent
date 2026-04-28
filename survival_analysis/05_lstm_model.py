# %% [markdown]
# # NLP tools: Bi-LSTM + Self-Attention Mortality Classifier
#
# Custom deep learning model (bonus):
# - Bidirectional LSTM with 2 layers
# - Scaled dot-product self-attention over timesteps
# - Trains on temporal sequences from clinical scoring
# - Outputs per-patient embeddings for tool composition clustering
#
# =============================================================================
# TRAINED MODEL WEIGHTS
# =============================================================================
# This script produces the LSTM weights that are committed to git at
# clinical_agent/models/ (via scripts/save_lstm_model.py afterward).
#
# Output of THIS script (survival_analysis/models/ — gitignored):
#   lstm_best.pt            — best checkpoint (by val AUC, early stopping)
#   lstm_embeddings.npy     — per-patient 128-dim embeddings (for clustering)
#   lstm_embedding_ids.npy  — ICUSTAY_IDs matching embedding rows
#   lstm_results.json       — evaluation metrics
#
# Final committed weights (clinical_agent/models/ — produced by save_lstm_model.py):
#   lstm_mortality.pt       — state_dict of best checkpoint
#   lstm_config.json        — architecture spec (n_features, hidden_dim, etc.)
#
# Architecture:
#   Input     : (batch, 12 timesteps, 10 features)  — 6-hour bins over 72h ICU
#   Features  : creatinine, bilirubin, platelets, lactate, WBC, bicarbonate,
#               INR, MAP, HR, SpO2
#   Model     : Linear projection → Bi-LSTM(64, 2 layers) → Self-Attention
#               → mean pool → Dropout → Linear(64) → ReLU → Linear(1)
#   Training  : AdamW, lr=1e-3, CosineAnnealing, BCEWithLogitsLoss,
#               pos_weight for class imbalance, early stopping (patience=8)
#   Reported  : Test AUC ~0.88, Brier ~0.11
#
# To REPRODUCE the weights from scratch:
#   1. Ensure MIMIC_DIR is set in .env and MIMIC DuckDB is available
#   2. Run (requires GPU for reasonable training time):
#        03_extract_temporal_features.py  (builds temporal sequences)
#        CUDA_VISIBLE_DEVICES=0 python survival_analysis/05_lstm_model.py
#        python clinical_agent/scripts/save_lstm_model.py
#
# Pre-trained weights are committed to git — retraining is NOT required.
# =============================================================================

# %%
import numpy as np
import pandas as pd
import pickle, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

# ── Data paths: set MIMIC_DIR / SURVIVAL_DATA_DIR in your .env 
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv as _lde
    _lde(_REPO / ".env")
except ImportError:
    pass
_SA_DATA = _Path(_os.environ.get("SURVIVAL_DATA_DIR", str(_REPO / "survival_analysis" / "data")))
_SA_RES  = _REPO / "survival_analysis" / "results"
_SA_FIG  = _REPO / "survival_analysis" / "figures"
_SA_MOD  = _REPO / "survival_analysis" / "models"
DATA_DIR  = str(_SA_DATA)
FIG_DIR   = str(_SA_FIG)
RES_DIR   = str(_SA_RES)
MODEL_DIR = str(_SA_MOD)
for d in [FIG_DIR, RES_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## Dataset

# %%
class ICUSequenceDataset(Dataset):
    def __init__(self, icustay_ids, sequences, labels):
        self.ids   = icustay_ids
        self.seqs  = sequences
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid   = self.ids[idx]
        x     = torch.tensor(self.seqs[sid], dtype=torch.float32)
        y     = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y, sid


# %% [markdown]
# ## Model: Bi-LSTM + Self-Attention

# %%
class SelfAttention(nn.Module):
    """Scaled dot-product self-attention over the time axis."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = hidden_dim ** 0.5
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        # x: (batch, time, hidden_dim)
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale    # (batch, T, T)
        attn   = F.softmax(scores, dim=-1)
        out    = torch.bmm(attn, V)                               # (batch, T, hidden_dim)
        return out, attn


class BiLSTMAttention(nn.Module):
    def __init__(self, n_features, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.attention = SelfAttention(hidden_dim * 2)
        self.dropout   = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, T, n_features)
        x = self.input_proj(x)                    # (batch, T, hidden_dim)
        lstm_out, _ = self.lstm(x)                # (batch, T, 2*hidden_dim)
        attn_out, attn_weights = self.attention(lstm_out)
        # Attention-weighted sum over time
        context = attn_out.mean(dim=1)            # (batch, 2*hidden_dim)
        context = self.dropout(context)
        logit   = self.classifier(context)        # (batch, 1)
        return logit.squeeze(-1), attn_weights    # (batch,), (batch, T, T)

    def get_embedding(self, x):
        """Return the embedding vector before the classification head."""
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out)
        context = attn_out.mean(dim=1)
        return context   # (batch, 2*hidden_dim)


# %% [markdown]
# ## Load data and split

# %%
print("Loading temporal sequences...")
with open(os.path.join(DATA_DIR, "temporal_sequences.pkl"), "rb") as f:
    sequences = pickle.load(f)
meta = pd.read_csv(os.path.join(DATA_DIR, "temporal_meta.csv"))
print(f"  Sequences: {len(sequences):,}  |  Meta: {len(meta):,}")
print(f"  Mortality: {meta['died'].mean():.3f}")

# Only keep IDs present in both sequences and meta
valid_ids  = [sid for sid in meta["icustay_id"].tolist() if sid in sequences]
meta_valid = meta[meta["icustay_id"].isin(valid_ids)].reset_index(drop=True)
labels     = meta_valid["died"].values.astype(int)
ids_arr    = meta_valid["icustay_id"].values

print(f"  Valid paired records: {len(valid_ids):,}")

# Train/val/test split (70/15/15)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
idx_train, idx_temp = next(splitter.split(ids_arr, labels))

ids_temp   = ids_arr[idx_temp]
labels_temp = labels[idx_temp]
splitter2  = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
idx_val, idx_test = next(splitter2.split(ids_temp, labels_temp))

train_ids   = ids_arr[idx_train];    train_labels   = labels[idx_train]
val_ids     = ids_temp[idx_val];     val_labels     = labels_temp[idx_val]
test_ids    = ids_temp[idx_test];    test_labels    = labels_temp[idx_test]

print(f"  Train: {len(train_ids):,}  Val: {len(val_ids):,}  Test: {len(test_ids):,}")
print(f"  Mortality — Train: {train_labels.mean():.3f}  Val: {val_labels.mean():.3f}  Test: {test_labels.mean():.3f}")

# %% [markdown]
# ## DataLoaders

# %%
BATCH_SIZE = 128
train_ds = ICUSequenceDataset(train_ids, sequences, train_labels)
val_ds   = ICUSequenceDataset(val_ids,   sequences, val_labels)
test_ds  = ICUSequenceDataset(test_ids,  sequences, test_labels)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

sample_x, sample_y, _ = next(iter(train_dl))
N_FEATURES  = sample_x.shape[2]
N_TIMESTEPS = sample_x.shape[1]
print(f"  Sequence shape per batch: {sample_x.shape}  — ({BATCH_SIZE} × {N_TIMESTEPS} timesteps × {N_FEATURES} features)")

# %% [markdown]
# ## Training

# %%
HIDDEN_DIM = 64
N_LAYERS   = 2
DROPOUT    = 0.3
LR         = 1e-3
EPOCHS     = 50
PATIENCE   = 8

model = BiLSTMAttention(N_FEATURES, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Weighted BCE loss for class imbalance
pos_weight = torch.tensor([(labels == 0).sum() / (labels == 1).sum()],
                           dtype=torch.float32).to(DEVICE)
criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Redefine model output to use logits (remove final sigmoid for loss)
class BiLSTMAttentionLogits(BiLSTMAttention):
    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out)
        context = attn_out.mean(dim=1)
        context = self.dropout(context)
        # Return logit (no sigmoid) for BCEWithLogitsLoss
        logit = self.classifier[:-1](context)  # exclude final sigmoid
        return logit.squeeze(-1), attn_weights

    def predict_proba(self, x):
        logit, attn = self.forward(x)
        return torch.sigmoid(logit), attn

model = BiLSTMAttentionLogits(N_FEATURES, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses, val_losses, val_aucs = [], [], []
best_val_auc = 0.0
patience_cnt = 0

print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    # -- Train --
    model.train()
    epoch_loss = 0.0
    for x_batch, y_batch, _ in train_dl:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(y_batch)

    avg_train_loss = epoch_loss / len(train_ds)
    train_losses.append(avg_train_loss)

    # -- Validate --
    model.eval()
    val_loss_sum = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch, _ in val_dl:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            val_loss_sum += loss.item() * len(y_batch)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_val_loss  = val_loss_sum / len(val_ds)
    val_auc       = roc_auc_score(all_labels, all_probs)
    val_losses.append(avg_val_loss)
    val_aucs.append(val_auc)

    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")

    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_cnt = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_best.pt"))
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best val AUC: {best_val_auc:.4f})")
            break

print(f"\nBest validation AUC: {best_val_auc:.4f}")

# %% [markdown]
# ## Plot training curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label="Train loss", color="#e74c3c")
ax1.plot(val_losses,   label="Val loss",   color="#3498db")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(val_aucs, color="#2ecc71", label="Val AUC")
ax2.axhline(best_val_auc, color="orange", linestyle="--", label=f"Best: {best_val_auc:.3f}")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC-ROC")
ax2.set_title("Validation AUC over Training"); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lstm_training_curves.png"), dpi=150)
plt.close()
print("Saved: lstm_training_curves.png")

# %% [markdown]
# ## Test set evaluation

# %%
print("\nEvaluating on test set...")
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "lstm_best.pt"), map_location=DEVICE))
model.eval()

test_probs, test_true = [], []
with torch.no_grad():
    for x_batch, y_batch, _ in test_dl:
        probs, _ = model.predict_proba(x_batch.to(DEVICE))
        test_probs.extend(probs.cpu().numpy())
        test_true.extend(y_batch.numpy())

test_probs = np.array(test_probs)
test_true  = np.array(test_true)

lstm_auc   = roc_auc_score(test_true, test_probs)
lstm_brier = brier_score_loss(test_true, test_probs)
print(f"  LSTM Test AUC:   {lstm_auc:.4f}")
print(f"  LSTM Brier:      {lstm_brier:.4f}")

# ROC curve
fpr, tpr, _ = roc_curve(test_true, test_probs)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="#2ecc71", lw=2, label=f"Bi-LSTM+Attn AUC={lstm_auc:.3f}")
ax.plot([0,1],[0,1],"k--", label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Bi-LSTM Mortality Model")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lstm_roc.png"), dpi=150)
plt.close()
print("Saved: lstm_roc.png")

np.save(os.path.join(RES_DIR, "lstm_test_probs.npy"), test_probs)
np.save(os.path.join(RES_DIR, "lstm_test_labels.npy"), test_true)

# %% [markdown]
# ## Extract patient embeddings for tool composition

# %%
print("\nExtracting patient embeddings (all patients)...")
all_dl = DataLoader(
    ICUSequenceDataset(ids_arr, sequences, labels),
    batch_size=256, shuffle=False, num_workers=2
)

all_embeddings = []
all_emb_ids    = []
model.eval()
with torch.no_grad():
    for x_batch, _, sid_batch in all_dl:
        emb = model.get_embedding(x_batch.to(DEVICE))
        all_embeddings.append(emb.cpu().numpy())
        all_emb_ids.extend(sid_batch.numpy())

embeddings     = np.vstack(all_embeddings)
embedding_ids  = np.array(all_emb_ids)

np.save(os.path.join(RES_DIR, "lstm_embeddings.npy"),    embeddings)
np.save(os.path.join(RES_DIR, "lstm_embedding_ids.npy"), embedding_ids)
print(f"  Embeddings shape: {embeddings.shape}")

# Save results
results = {
    "lstm_auc":       lstm_auc,
    "lstm_brier":     lstm_brier,
    "best_val_auc":   best_val_auc,
    "n_epochs_run":   len(train_losses),
    "hidden_dim":     HIDDEN_DIM,
    "n_layers":       N_LAYERS,
}
with open(os.path.join(RES_DIR, "lstm_results.json"), "w") as f:
    json.dump(results, f, indent=2)

assert lstm_auc > 0.70, f"LSTM AUC too low: {lstm_auc:.4f}"
print(f"\nok NLP tools complete. LSTM AUC={lstm_auc:.4f}")
