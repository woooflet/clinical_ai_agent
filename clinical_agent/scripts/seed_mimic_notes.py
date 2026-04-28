
import sys
import os
import re
import hashlib
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

import config
import duckdb
import pandas as pd

CLUSTER_SAMPLES = {
    2: 2500,
    0: 1500,
    1: 1000,
}
CHUNK_TOKENS  = 300
CHUNK_OVERLAP = 50

COLLECTION_NAME = "mimic_notes"

_PHI_ENTITIES = [
    "PERSON", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS",
    "US_SSN", "US_DRIVER_LICENSE", "MEDICAL_LICENSE", "NRP",
    "LOCATION", "ORGANIZATION", "URL", "IP_ADDRESS",
]


_analyzer   = None
_anonymizer = None


def _init_presidio() -> None:
    """Load Presidio engines once."""
    global _analyzer, _anonymizer
    if _analyzer is not None:
        return
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        _analyzer   = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
        print("[presidio] AnalyzerEngine + AnonymizerEngine loaded")
    except Exception as e:
        print(f"[presidio] WARNING: could not load Presidio   PHI will NOT be stripped: {e}")


def _deidentify(text: str) -> str:
    """
    Strip PHI from *text* using Presidio.
    Returns the original text unchanged if Presidio failed to load.

    Critical: call this BEFORE embedding   even though MIMIC is already
    de-identified at the dataset level, residual quasi-identifiers (shifted
    ages, hospital names, geographic descriptors) can be encoded into
    embedding space.
    """
    if _analyzer is None or _anonymizer is None:
        return text
    try:
        from presidio_anonymizer.entities import OperatorConfig
        ar = _analyzer.analyze(text=text, entities=_PHI_ENTITIES, language="en")
        ops = {e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in _PHI_ENTITIES}
        return _anonymizer.anonymize(text=text, analyzer_results=ar, operators=ops).text
    except Exception:
        return text


def _chunk_note(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping chunks of ~chunk_tokens words.

    We use words as a proxy for tokens (average English word ≈ 1.3 sub-tokens
    for BERT-family models).  A 300-word chunk ≈ 390 sub-tokens, well within
    the 512 token limit of MedCPT-Article-Encoder.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_tokens, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_tokens - overlap
    return chunks


def _get_discharge_note(con: duckdb.DuckDBPyConnection, subject_id: int) -> str | None:
    """Fetch the most recent discharge summary text for *subject_id*."""
    df = con.execute(
        """
        SELECT TEXT FROM noteevents
        WHERE SUBJECT_ID = ?
          AND CATEGORY = 'Discharge summary'
          AND ISERROR IS NULL
        ORDER BY CHARTDATE DESC
        LIMIT 1
        """,
        [subject_id],
    ).df()
    if df.empty or df.iloc[0]["TEXT"] is None:
        return None
    return str(df.iloc[0]["TEXT"])


def _get_diagnoses_snippet(con: duckdb.DuckDBPyConnection, subject_id: int, max_diags: int = 5) -> str:
    """Return a short comma-joined string of ICD-9 long_titles for *subject_id*."""
    df = con.execute(
        """
        SELECT d.LONG_TITLE
        FROM diagnoses_icd di
        JOIN d_icd_diagnoses d ON di.ICD9_CODE = d.ICD9_CODE
        WHERE di.SUBJECT_ID = ?
        LIMIT ?
        """,
        [subject_id, max_diags],
    ).df()
    if df.empty:
        return ""
    return "; ".join(df["LONG_TITLE"].dropna().tolist())


def main() -> None:
    print("=" * 70)
    print("  MIMIC Note Semantic Index Seeder")
    print("=" * 70)

    print(f"\n[1/6] Loading temporal_meta from {config.TEMPORAL_META_CSV} ...")
    meta = pd.read_csv(config.TEMPORAL_META_CSV)
    if "subject_id" not in meta.columns:
        meta.columns = [c.lower() for c in meta.columns]
    if "cluster" not in meta.columns:
        raise RuntimeError(
            "temporal_meta.csv has no 'cluster' column. "
            "Run scripts/save_models.py first to enrich it with cluster labels."
        )
    print(f"  Loaded {len(meta):,} rows. Clusters: {meta['cluster'].value_counts().to_dict()}")

    print(f"\n[2/6] Sampling subject IDs per cluster ...")
    sampled_ids: list[int] = []
    for cluster_id, n_samples in CLUSTER_SAMPLES.items():
        pool = meta[meta["cluster"] == cluster_id]["subject_id"].dropna().astype(int).unique()
        n_actual = min(n_samples, len(pool))
        chosen = (
            pd.Series(pool).sample(n=n_actual, random_state=42).tolist()
        )
        print(f"  Cluster {cluster_id}: requested {n_samples:,}, pool {len(pool):,} → sampled {n_actual:,}")
        sampled_ids.extend(chosen)
    print(f"  Total sampled: {len(sampled_ids):,} subject IDs")

    print(f"\n[3/6] Connecting to MIMIC-III DuckDB at {config.MIMIC_DB} ...")
    con = duckdb.connect(str(config.MIMIC_DB), read_only=True)
    print("  Connected.")

    print("\n[4/6] Initialising Presidio de-identification ...")
    _init_presidio()

    print(f"\n[5/6] Initialising ChromaDB collection '{COLLECTION_NAME}' ...")
    config.CHROMA_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    import chromadb
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_NOTES_DIR))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Collection '{COLLECTION_NAME}'   existing docs: {collection.count():,}")

    print(f"\n[6/6] Loading embedding model '{config.EMBEDDING_MODEL}' ...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    print(f"  Model loaded.")

    print(f"\nProcessing {len(sampled_ids):,} patients ...")
    total_chunks   = 0
    skipped_no_note = 0
    upserted_new   = 0

    meta_lookup = (
        meta[["subject_id", "cluster", "died"]]
        .drop_duplicates("subject_id")
        .set_index("subject_id")
        .to_dict("index")
    )

    BATCH_SIZE = 100

    ids_batch        = []
    embeddings_batch = []
    documents_batch  = []
    metadatas_batch  = []

    def flush_batch():
        nonlocal upserted_new
        if not ids_batch:
            return
        collection.upsert(
            ids=ids_batch,
            embeddings=embeddings_batch,
            documents=documents_batch,
            metadatas=metadatas_batch,
        )
        upserted_new += len(ids_batch)
        ids_batch.clear()
        embeddings_batch.clear()
        documents_batch.clear()
        metadatas_batch.clear()

    for i, subject_id in enumerate(sampled_ids, 1):
        if i % 200 == 0 or i == 1:
            print(f"  [{i:>5}/{len(sampled_ids)}] collection size: {collection.count():,}")

        raw_text = _get_discharge_note(con, subject_id)
        if raw_text is None:
            skipped_no_note += 1
            continue

        deid_text = _deidentify(raw_text[:50_000])

        chunks = _chunk_note(deid_text)
        if not chunks:
            skipped_no_note += 1
            continue

        diag_snippet = _get_diagnoses_snippet(con, subject_id)

        patient_meta = meta_lookup.get(subject_id, {})
        cluster      = int(patient_meta.get("cluster", -1))
        mortality    = int(patient_meta.get("died", -1))

        chunk_embeddings = embed_model.encode(chunks, show_progress_bar=False).tolist()

        for chunk_idx, (chunk_text, chunk_emb) in enumerate(zip(chunks, chunk_embeddings)):
            doc_id = hashlib.md5(f"{subject_id}_{chunk_idx}".encode()).hexdigest()[:16]

            ids_batch.append(doc_id)
            embeddings_batch.append(chunk_emb)
            documents_batch.append(chunk_text)
            metadatas_batch.append({
                "subject_id":        subject_id,
                "cluster":           cluster,
                "mortality_label":   mortality,
                "diagnoses_snippet": diag_snippet[:500],
                "chunk_index":       chunk_idx,
            })
            total_chunks += 1

        if len(ids_batch) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    con.close()

    print("\n" + "=" * 70)
    print("Seeding complete!")
    print(f"  Patients processed :  {len(sampled_ids) - skipped_no_note:,}")
    print(f"  Patients skipped   :  {skipped_no_note:,}  (no discharge note found)")
    print(f"  Chunks embedded    :  {total_chunks:,}")
    print(f"  Final collection size: {collection.count():,} documents")
    print("=" * 70)


if __name__ == "__main__":
    main()
