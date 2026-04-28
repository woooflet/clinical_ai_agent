
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

_xgb_bundle: Optional[dict]       = None
_shap_explainer                   = None
_static_df: Optional[pd.DataFrame] = None
_meta_df: Optional[pd.DataFrame]  = None
_knn_model                        = None
_subject_id_index: dict[int, int] = {}
_meta_subject_index: dict[int, list[int]] = {}
_feature_cols: list[str]          = []

_lstm_model                       = None
_lstm_config: Optional[dict]      = None
_temporal_sequences: Optional[dict] = None
_subject_icustay_index: dict[int, list[int]] = {}
_lstm_device: Optional[str]       = None   # "cuda:0" or "cpu"

_bert_model                       = None
_bert_tokenizer                   = None
_icd9_classes: list[str]          = []
_icd9_descriptions: dict[str, str] = {}
_bert_device: Optional[str]       = None

def load_models(
    xgb_path: Path,
    shap_path: Path,
    static_csv: Path,
    meta_csv: Path,
) -> None:
    """
    Called by server.py at startup to load models and data into module state.
    Gracefully skips missing files   tools will return informative errors
    rather than crashing the server.
    """
    global _xgb_bundle, _shap_explainer, _static_df, _meta_df, _knn_model
    global _feature_cols, _subject_id_index, _meta_subject_index

    if xgb_path.exists():
        import joblib
        _xgb_bundle = joblib.load(xgb_path)
        _feature_cols = _xgb_bundle["feat_cols"]
        print(f"[ml_models] XGBoost bundle loaded from {xgb_path}")
    else:
        print(f"[ml_models] WARNING: xgb_mortality.pkl not found at {xgb_path} "
              "  run scripts/save_models.py first")

    if shap_path.exists():
        import pickle
        with open(shap_path, "rb") as f:
            _shap_explainer = pickle.load(f)
        print(f"[ml_models] SHAP explainer loaded from {shap_path}")
    else:
        print(f"[ml_models] WARNING: shap_explainer.pkl not found at {shap_path}")

    if static_csv.exists():
        _static_df = pd.read_csv(static_csv)
        col_lower_map = {c: c.lower() for c in ["SUBJECT_ID", "HADM_ID",
                                                  "ICUSTAY_ID", "INTIME", "OUTTIME"]}
        _static_df.rename(columns=col_lower_map, inplace=True)

        _subject_id_index = {
            int(sid): idx
            for idx, sid in enumerate(_static_df["subject_id"].astype(int))
        }
        n_feat = len(_feature_cols) if _feature_cols else "?"
        print(f"[ml_models] static_features loaded: {len(_static_df):,} rows, "
              f"{n_feat} feature cols (from bundle)")
    else:
        print(f"[ml_models] WARNING: static_features.csv not found at {static_csv}")

    if meta_csv.exists():
        _meta_df = pd.read_csv(meta_csv)
        _meta_df.columns = [c.lower() for c in _meta_df.columns]
        for idx, sid in enumerate(_meta_df["subject_id"].astype(int)):
            _meta_subject_index.setdefault(int(sid), []).append(idx)
        print(f"[ml_models] temporal_meta loaded: {len(_meta_df):,} rows")
    else:
        print(f"[ml_models] WARNING: temporal_meta.csv not found at {meta_csv}")

    if _static_df is not None and _feature_cols:
        from sklearn.neighbors import NearestNeighbors
        X = _static_df[_feature_cols].fillna(0).values
        _knn_model = NearestNeighbors(n_neighbors=11, metric="euclidean", n_jobs=-1)
        _knn_model.fit(X)
        print(f"[ml_models] k-NN fitted on {len(_static_df):,} patients "
              f"× {len(_feature_cols)} features")

def load_lstm_model(lstm_path: Path, lstm_cfg_path: Path, temporal_seq_pkl: Path) -> None:
    """
    Load the Bi-LSTM temporal mortality model onto GPU 0 (falls back to CPU).
    Also loads temporal_sequences.pkl and builds icustay→subject_id index.
    Called by server.py at startup after load_models().
    """
    global _lstm_model, _lstm_config, _temporal_sequences
    global _subject_icustay_index, _lstm_device

    if not lstm_path.exists():
        print(f"[ml_models] WARNING: lstm_mortality.pt not found at {lstm_path} "
              "  run scripts/save_lstm_model.py first")
        return
    if not lstm_cfg_path.exists():
        print(f"[ml_models] WARNING: lstm_config.json not found at {lstm_cfg_path}")
        return

    import json, torch, torch.nn as nn, torch.nn.functional as F

    with open(lstm_cfg_path) as f:
        _lstm_config = json.load(f)

    class _SelfAttention(nn.Module):
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

    class _BiLSTMAttentionLogits(nn.Module):
        def __init__(self, n_features, hidden_dim, n_layers, dropout):
            super().__init__()
            self.input_proj = nn.Linear(n_features, hidden_dim)
            self.lstm = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=n_layers, batch_first=True, bidirectional=True,
                dropout=dropout if n_layers > 1 else 0,
            )
            self.attention   = _SelfAttention(hidden_dim * 2)
            self.dropout     = nn.Dropout(dropout)
            self.classifier  = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid(),
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

    import os
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available():
        _lstm_device = "cuda:0"
    else:
        _lstm_device = "cpu"

    cfg = _lstm_config
    model = _BiLSTMAttentionLogits(
        n_features=cfg["n_features"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    state = torch.load(lstm_path, map_location=_lstm_device, weights_only=True)
    model.load_state_dict(state)
    model.to(_lstm_device)
    model.eval()
    _lstm_model = model
    print(f"[ml_models] LSTM loaded on {_lstm_device} from {lstm_path}")

    if temporal_seq_pkl.exists():
        import pickle
        with open(temporal_seq_pkl, "rb") as f:
            _temporal_sequences = pickle.load(f)
        print(f"[ml_models] temporal_sequences loaded: {len(_temporal_sequences):,} sequences")

        if _meta_df is not None:
            for _, row in _meta_df.iterrows():
                sid   = int(row["subject_id"])
                icu_id = int(row["icustay_id"])
                _subject_icustay_index.setdefault(sid, []).append(icu_id)
        print(f"[ml_models] subject→icustay index: {len(_subject_icustay_index):,} patients")
    else:
        print(f"[ml_models] WARNING: temporal_sequences.pkl not found at {temporal_seq_pkl}")

def load_bert_model(bert_model_dir: Path, icd9_labels_path: Path) -> None:
    """
    Load ClinicalBERT ICD-9 multi-label classifier onto GPU 0 (falls back CPU).
    Also loads ICD-9 label class list saved by scripts/save_lstm_model.py.
    """
    global _bert_model, _bert_tokenizer, _icd9_classes, _icd9_descriptions, _bert_device

    if not bert_model_dir.exists():
        print(f"[ml_models] WARNING: ClinicalBERT checkpoint not found at {bert_model_dir} "
              "  run scripts/save_lstm_model.py first")
        return

    import json, torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if torch.cuda.is_available():
        _bert_device = "cuda:0"
    else:
        _bert_device = "cpu"

    print(f"[ml_models] Loading ClinicalBERT ICD-9 model on {_bert_device} ...")
    _bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    _bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_model_dir))
    _bert_model.to(_bert_device)
    _bert_model.eval()
    print(f"[ml_models] ClinicalBERT loaded: {_bert_model.config.num_labels or '?'} labels")

    if icd9_labels_path.exists():
        with open(icd9_labels_path) as f:
            data = json.load(f)
        _icd9_classes      = data.get("classes", [])
        _icd9_descriptions = data.get("descriptions", {})
        print(f"[ml_models] ICD-9 label map loaded: {len(_icd9_classes)} classes")
    else:
        print(f"[ml_models] WARNING: icd9_labels.json not found at {icd9_labels_path}")

def _lookup_patient_row(patient_id: int) -> Optional[pd.Series]:
    """Return the static_features row for a subject_id, or None if not found."""
    idx = _subject_id_index.get(int(patient_id))
    if idx is None:
        return None
    return _static_df.iloc[idx]

def _get_feature_vector(row: pd.Series) -> np.ndarray:
    """Extract and preprocess feature vector using imputer + scaler from bundle."""
    X_df  = pd.DataFrame(
        [row[_feature_cols].values.astype(float)],
        columns=_feature_cols,
    )
    X_imp = _xgb_bundle["imputer"].transform(X_df)
    X_sc  = _xgb_bundle["scaler"].transform(X_imp)
    return X_sc

def register(mcp) -> None:
    """Attach all ML model tools to the FastMCP instance."""

    @mcp.tool()
    def predict_mortality(patient_id: int) -> dict:
        """
        Predict ICU mortality probability for a patient using a trained XGBoost
        model.

        Looks up the patient's row from the pre-computed static_features.csv
        cohort (46k patients) and runs a forward pass on CPU (~1ms).

        Args:
            patient_id: MIMIC subject_id (integer, e.g. 10006).

        Returns:
            {
                patient_id:            int,
                mortality_probability: float  (0 1),
                mortality_label:       str    ("High risk" / "Moderate risk" /
                                               "Low risk"),
                actual_outcome:        int    (1 = died in hospital, 0 = survived),
                model_type:            str,
                note:                  str,
            }
        """
        if _xgb_bundle is None:
            return {"error": "XGBoost model not loaded. Run scripts/save_models.py first.",
                    "patient_id": patient_id}
        if _static_df is None:
            return {"error": "static_features.csv not loaded.", "patient_id": patient_id}

        row = _lookup_patient_row(patient_id)
        if row is None:
            return {"error": f"Patient {patient_id} not found in static cohort.",
                    "patient_id": patient_id,
                    "hint": "patient_id must be a MIMIC subject_id (range 2 99999)"}

        X_sc  = _get_feature_vector(row)
        prob  = float(_xgb_bundle["model"].predict_proba(X_sc)[0, 1])

        if prob >= 0.5:
            label = "High risk"
        elif prob >= 0.2:
            label = "Moderate risk"
        else:
            label = "Low risk"

        return {
            "patient_id":            int(patient_id),
            "mortality_probability": round(prob, 4),
            "mortality_label":       label,
            "actual_outcome":        int(row["died"]) if pd.notna(row["died"]) else None,
            "model_type":            "XGBoost (static 6h features, 44 variables)",
            "note":                  (
                "Prediction is based on the first 6 hours of the ICU stay using "
                "MIMIC-III data. Not a real-time clinical decision tool."
            ),
        }

    @mcp.tool()
    def explain_prediction(patient_id: int) -> dict:
        """
        Explain an XGBoost mortality prediction using SHAP TreeExplainer.

        Returns the top-10 features by absolute SHAP value, showing which
        clinical variables most influenced the mortality prediction and in
        which direction.

        Args:
            patient_id: MIMIC subject_id.

        Returns:
            {
                patient_id:    int,
                base_value:    float  (expected model output over training data),
                prediction:    float  (final mortality probability),
                top_features:  [
                    {
                        feature:       str,
                        shap_value:    float,
                        patient_value: float | None,
                        direction:     "increases risk" | "decreases risk",
                    }
                ],
                model_type:    str,
            }
        """
        if _shap_explainer is None:
            return {"error": "SHAP explainer not loaded. Run scripts/save_models.py first.",
                    "patient_id": patient_id}
        if _xgb_bundle is None:
            return {"error": "XGBoost bundle not loaded.", "patient_id": patient_id}
        if _static_df is None:
            return {"error": "static_features.csv not loaded.", "patient_id": patient_id}

        row = _lookup_patient_row(patient_id)
        if row is None:
            return {"error": f"Patient {patient_id} not found in static cohort.",
                    "patient_id": patient_id}

        X_sc         = _get_feature_vector(row)
        shap_values  = _shap_explainer.shap_values(X_sc)
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1]).flatten()
        else:
            sv = np.array(shap_values).flatten()

        base_value = float(_shap_explainer.expected_value
                           if not isinstance(_shap_explainer.expected_value, (list, np.ndarray))
                           else _shap_explainer.expected_value[1])
        prediction = float(base_value + sv.sum())
        prediction = max(0.0, min(1.0, prediction))

        feat_cols  = _xgb_bundle["feat_cols"]
        order      = np.argsort(np.abs(sv))[::-1][:10]
        top_feats  = []
        raw_row    = row[feat_cols].values.astype(float)
        for i in order:
            sv_val    = float(sv[i])
            raw_val   = float(raw_row[i])
            top_feats.append({
                "feature":       feat_cols[i],
                "shap_value":    round(sv_val, 5),
                "patient_value": None if np.isnan(raw_val) else round(raw_val, 4),
                "direction":     "increases risk" if sv_val > 0 else "decreases risk",
            })

        return {
            "patient_id":   int(patient_id),
            "base_value":   round(base_value, 5),
            "prediction":   round(prediction, 4),
            "top_features": top_feats,
            "model_type":   "SHAP TreeExplainer on XGBoost",
        }

    @mcp.tool()
    def get_phenotype(patient_id: int) -> dict:
        """
        Look up the UMAP-derived phenotype cluster assignment for a patient.

        Cluster assignments were computed via KMeans(k=3) on LSTM hidden-state
        UMAP projections (ICU_Survival_Analysis_v7.ipynb). Three phenotypes
        emerged from 72h temporal trajectories:
          - High-acuity (cluster with ~21% mortality, longest stays)
          - Moderate-acuity (cluster with ~2% mortality, mid-length stays)
          - Low-acuity (cluster with ~0.6% mortality, short stays)

        Args:
            patient_id: MIMIC subject_id.

        Returns:
            {
                patient_id:           int,
                icustay_id:           int,
                cluster_id:           int   (0, 1, or 2),
                umap_x:               float,
                umap_y:               float,
                cluster_description:  str,
                cluster_size:         int,
                cluster_mortality_pct: float,
            }
        """
        if _meta_df is None:
            return {"error": "temporal_meta.csv not loaded.", "patient_id": patient_id}

        row_indices = _meta_subject_index.get(int(patient_id))
        if not row_indices:
            return {"error": f"Patient {patient_id} not found in temporal_meta.",
                    "patient_id": patient_id,
                    "hint": "Only patients with LSTM temporal sequences have phenotypes."}

        row = _meta_df.iloc[row_indices[0]]

        cluster_id = int(row["cluster"]) if pd.notna(row.get("cluster")) else None
        if cluster_id is None:
            return {"error": "Cluster data not available. Run scripts/save_models.py to enrich temporal_meta.",
                    "patient_id": patient_id}

        cluster_rows = _meta_df[_meta_df["cluster"] == cluster_id]
        cluster_size = int(len(cluster_rows))
        cluster_mort = float(cluster_rows["died"].mean() * 100) if cluster_size > 0 else 0.0

        return {
            "patient_id":             int(patient_id),
            "icustay_id":             int(row["icustay_id"]),
            "cluster_id":             cluster_id,
            "umap_x":                 round(float(row["umap_x"]), 4) if pd.notna(row.get("umap_x")) else None,
            "umap_y":                 round(float(row["umap_y"]), 4) if pd.notna(row.get("umap_y")) else None,
            "cluster_description":    str(row.get("cluster_description", f"Cluster {cluster_id}")),
            "cluster_size":           cluster_size,
            "cluster_mortality_pct":  round(cluster_mort, 1),
        }

    @mcp.tool()
    def find_similar_patients(patient_id: int, k: int = 10) -> dict:
        """
        Find the k most similar historical ICU patients using Euclidean k-NN
        on 44 static clinical features (labs, vitals, demographics, diagnoses).

        The k-NN model is fit at server startup on 46k MIMIC-III ICU stays.
        Median imputation + standard scaling are applied before distance
        computation   the same preprocessing used during XGBoost training.

        Args:
            patient_id: MIMIC subject_id.
            k:          Number of neighbours to return (1 50, default 10).

        Returns:
            {
                patient_id: int,
                k:          int,
                neighbours: [
                    {
                        subject_id:        int,
                        distance:          float,
                        mortality_label:   int   (0 or 1),
                        icu_hours:         float,
                    }
                ],
                note: str,
            }
        """
        if _knn_model is None:
            return {"error": "k-NN model not built. static_features.csv must be loaded at startup.",
                    "patient_id": patient_id}
        if _static_df is None:
            return {"error": "static_features.csv not loaded.", "patient_id": patient_id}

        row = _lookup_patient_row(patient_id)
        if row is None:
            return {"error": f"Patient {patient_id} not found in static cohort.",
                    "patient_id": patient_id}

        k_actual = max(1, min(int(k), 50))
        k_query  = k_actual + 1

        X_raw = row[_feature_cols].fillna(0).values.reshape(1, -1)
        distances, indices = _knn_model.kneighbors(X_raw, n_neighbors=k_query)

        neighbours = []
        query_subject = int(patient_id)
        for dist, idx in zip(distances[0], indices[0]):
            nbr_row = _static_df.iloc[idx]
            nbr_sid = int(nbr_row["subject_id"])
            if nbr_sid == query_subject:
                continue
            died_val = nbr_row.get("died")
            hrs_val  = nbr_row.get("icu_hours")
            neighbours.append({
                "subject_id":      nbr_sid,
                "distance":        round(float(dist), 4),
                "mortality_label": int(died_val) if pd.notna(died_val) else None,
                "icu_hours":       round(float(hrs_val), 1) if pd.notna(hrs_val) else None,
            })
            if len(neighbours) >= k_actual:
                break

        return {
            "patient_id": int(patient_id),
            "k":          len(neighbours),
            "neighbours": neighbours,
            "note":       (
                "Similarity computed using Euclidean distance on 44 normalised "
                "static features (first 6h of ICU stay). Use get_patient_diagnoses "
                "or get_patient_info to retrieve full details on each neighbour."
            ),
        }

    @mcp.tool()
    def predict_mortality_temporal(patient_id: int) -> dict:
        """
        Predict ICU mortality using the Bi-LSTM + Self-Attention model trained
        on 72h temporal sequences (12 × 6h bins, 10 vital/lab features each).

        Runs on GPU 0 if available. Falls back to XGBoost static prediction if
        the patient's temporal sequence is not available.

        Args:
            patient_id: MIMIC subject_id.

        Returns:
            {
                patient_id:            int,
                mortality_probability: float  (0 1),
                mortality_label:       str,
                actual_outcome:        int | None,
                model_type:            str,
                device:                str,
                note:                  str,
            }
        """
        import torch

        icustay_ids = _subject_icustay_index.get(int(patient_id))
        if _lstm_model is None or _temporal_sequences is None or not icustay_ids:
            if _xgb_bundle is not None:
                row = _lookup_patient_row(patient_id)
                if row is not None:
                    X_sc = _get_feature_vector(row)
                    prob = float(_xgb_bundle["model"].predict_proba(X_sc)[0, 1])
                    label = "High risk" if prob >= 0.5 else ("Moderate risk" if prob >= 0.2 else "Low risk")
                    reason = ("LSTM model not loaded" if _lstm_model is None
                              else "No temporal sequence for this patient")
                    return {
                        "patient_id":            int(patient_id),
                        "mortality_probability": round(prob, 4),
                        "mortality_label":       label,
                        "actual_outcome":        int(row["died"]) if pd.notna(row.get("died")) else None,
                        "model_type":            "XGBoost fallback (LSTM unavailable)",
                        "device":                "cpu",
                        "note":                  f"Fallback used: {reason}.",
                    }
            return {"error": "Neither LSTM nor XGBoost available.", "patient_id": patient_id}

        icu_id = None
        for iid in icustay_ids:
            if iid in _temporal_sequences:
                icu_id = iid
                break
        if icu_id is None:
            return {"error": f"No temporal sequence for patient {patient_id}.",
                    "patient_id": patient_id}

        seq = _temporal_sequences[icu_id]
        x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(_lstm_device)

        with torch.no_grad():
            prob_t, _ = _lstm_model.predict_proba(x)
        prob  = float(prob_t.cpu().item())
        label = "High risk" if prob >= 0.5 else ("Moderate risk" if prob >= 0.2 else "Low risk")

        actual = None
        if _meta_df is not None:
            rows = _meta_df[_meta_df["icustay_id"] == icu_id]
            if not rows.empty:
                v = rows.iloc[0]["died"]
                actual = int(v) if pd.notna(v) else None

        return {
            "patient_id":            int(patient_id),
            "icustay_id":            int(icu_id),
            "mortality_probability": round(prob, 4),
            "mortality_label":       label,
            "actual_outcome":        actual,
            "model_type":            "Bi-LSTM + Self-Attention (72h temporal, 10 features)",
            "device":                _lstm_device,
            "note": (
                "Prediction based on 12 × 6h temporal bins of vital signs and lab "
                "values from the first 72h of the ICU stay."
            ),
        }

    @mcp.tool()
    def predict_icd9(note_text: str, threshold: float = 0.3, top_k: int = 10) -> dict:
        """
        Predict ICD-9 diagnosis codes from a clinical note using a fine-tuned
        Bio_ClinicalBERT multi-label classifier (45 most common MIMIC codes).

        Runs on GPU 0 if available; CPU fallback (~3 8s).

        Args:
            note_text:  Clinical note text (discharge summary, progress note, etc.).
                        Truncated to 512 tokens (BERT limit).
            threshold:  Sigmoid threshold for positive label (default 0.3).
            top_k:      Maximum number of codes to return (default 10).

        Returns:
            {
                predicted_codes: [
                    {
                        icd9_code:    str,
                        description:  str,
                        probability:  float,
                    }
                ],
                num_codes_predicted: int,
                model_type:          str,
                device:              str,
                note:                str,
            }
        """
        if _bert_model is None or _bert_tokenizer is None:
            return {
                "error": "ClinicalBERT ICD-9 model not loaded. Run scripts/save_lstm_model.py first.",
                "note_text_preview": note_text[:100],
            }

        import torch

        inputs = _bert_tokenizer(
            note_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(_bert_device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = _bert_model(**inputs).logits
        probs = torch.sigmoid(logits[0]).cpu().numpy()

        results = []
        for i, p in enumerate(probs):
            if p >= threshold:
                code = _icd9_classes[i] if i < len(_icd9_classes) else f"LABEL_{i}"
                desc = _icd9_descriptions.get(code, code)
                results.append({"icd9_code": code, "description": desc, "probability": round(float(p), 4)})

        results.sort(key=lambda x: x["probability"], reverse=True)
        results = results[:top_k]

        return {
            "predicted_codes":      results,
            "num_codes_predicted":  len(results),
            "model_type":           "Bio_ClinicalBERT fine-tuned on MIMIC-III top-50 ICD-9 codes",
            "device":               _bert_device,
            "note": (
                f"Multi-label classifier with sigmoid threshold={threshold}. "
                "Trained on discharge summaries from MIMIC-III. "
                "Not a substitute for professional medical coding."
            ),
        }

    @mcp.tool()
    def shap_lstm(patient_id: int) -> dict:
        """
        Explain the Bi-LSTM temporal mortality prediction using SHAP DeepExplainer.

        Computes expected SHAP values over time for the patient's 72h ICU sequence,
        returning per-timestep feature importance summed across the time axis.

        Requires GPU (CUDA). Falls back to a message if GPU is unavailable.

        Args:
            patient_id: MIMIC subject_id.

        Returns:
            {
                patient_id:       int,
                mortality_prob:   float,
                top_features:     [
                    {
                        feature_idx:  int,
                        shap_value:   float,
                        direction:    "increases risk" | "decreases risk",
                    }
                ],
                timestep_importance: [float]  (sum |SHAP| per timestep),
                model_type:          str,
                note:                str,
            }
        """
        if _lstm_model is None or _temporal_sequences is None:
            return {"error": "LSTM model not loaded.", "patient_id": patient_id}
        if _lstm_device == "cpu":
            return {
                "error": "shap_lstm requires CUDA. No GPU available.",
                "patient_id": patient_id,
                "suggestion": "Use explain_prediction for CPU-compatible XGBoost SHAP values.",
            }

        import torch, shap
        import numpy as np

        icustay_ids = _subject_icustay_index.get(int(patient_id))
        if not icustay_ids:
            return {"error": f"Patient {patient_id} not found in temporal index.",
                    "patient_id": patient_id}

        icu_id = None
        for iid in icustay_ids:
            if iid in _temporal_sequences:
                icu_id = iid
                break
        if icu_id is None:
            return {"error": f"No temporal sequence for patient {patient_id}.",
                    "patient_id": patient_id}

        seq = _temporal_sequences[icu_id]
        x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(_lstm_device)

        all_ids   = list(_temporal_sequences.keys())
        bg_ids    = all_ids[:50]
        bg_arrays = [_temporal_sequences[i] for i in bg_ids]
        background = torch.tensor(np.array(bg_arrays), dtype=torch.float32).to(_lstm_device)

        class _LSTMWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                logit, _ = self.model.forward(x)
                return logit.unsqueeze(-1)

        wrapper = _LSTMWrapper(_lstm_model)

        import torch.backends.cudnn as cudnn_backend
        prev_cudnn = cudnn_backend.enabled
        cudnn_backend.enabled = False
        _lstm_model.train()
        try:
            explainer   = shap.GradientExplainer(wrapper, background)
            shap_values = explainer.shap_values(x)
        finally:
            _lstm_model.eval()
            cudnn_backend.enabled = prev_cudnn

        if isinstance(shap_values, list):
            sv = np.array(shap_values[0]).squeeze(0)
        else:
            sv = np.array(shap_values).squeeze()

        sv = np.array(sv)
        if sv.ndim == 3:
            sv = sv.squeeze(-1)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        with torch.no_grad():
            prob_t, _ = _lstm_model.predict_proba(x)
        prob = float(prob_t.cpu().item())

        feat_importance = sv.sum(axis=0)
        feat_importance = feat_importance.flatten()
        order = np.argsort(np.abs(feat_importance))[::-1].tolist()
        top_feats = [
            {
                "feature_idx": int(order[k]),
                "shap_value":  round(float(feat_importance[order[k]]), 5),
                "direction":   "increases risk" if float(feat_importance[order[k]]) > 0 else "decreases risk",
            }
            for k in range(len(order))
        ]

        timestep_imp = [round(float(v), 5) for v in np.abs(sv).sum(axis=1).tolist()]

        return {
            "patient_id":          int(patient_id),
            "icustay_id":          int(icu_id),
            "mortality_prob":      round(prob, 4),
            "top_features":        top_feats,
            "timestep_importance": timestep_imp,
            "model_type":          "SHAP GradientExplainer on Bi-LSTM + Self-Attention",
            "note": (
                "Feature indices correspond to the 10 temporal features: "
                "[HR, SBP, DBP, SpO2, Temp, RR, WBC, creatinine, bilirubin, GCS]. "
                "Timestep 0 = first 6h of ICU stay, timestep 11 = hours 66 72."
            ),
        }
