"""
scripts/seed_guidelines.py    Ingest clinical guideline PDFs into
a separate ChromaDB collection ('clinical_guidelines').

Reads PDFs from clinical_agent/data/guidelines/, extracts text with pdfplumber,
applies section-aware chunking, embeds with ncbi/MedCPT-Article-Encoder, and
upserts into a dedicated ChromaDB collection.

Run from clinical_agent/ directory:
    nohup python scripts/seed_guidelines.py > logs/seed_guidelines.log 2>&1 &

Or interactively:
    python scripts/seed_guidelines.py

Expected runtime: ~5-10 minutes for 10 PDFs on CPU.
"""


import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_DIR  = SCRIPT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

import config

GUIDELINES_DIR    = config.GUIDELINES_DIR
CHROMA_DIR        = config.CHROMA_GUIDELINES_DIR
MANIFEST_PATH     = GUIDELINES_DIR / "download_manifest.json"

GUIDELINE_CATALOG = {
    "surviving_sepsis_2021.pdf": {
        "source": "Surviving Sepsis Campaign Guidelines 2021",
        "tags":   "sepsis,antibiotics,vasopressors,critical_care,ICU,fluid_resuscitation",
        "url":    "https://link.springer.com/article/10.1007/s00134-021-06506-y",
    },
    "kdigo_aki_2012.pdf": {
        "source": "KDIGO Clinical Practice Guideline for Acute Kidney Injury 2012",
        "tags":   "acute_kidney_injury,renal,creatinine,ICU,dialysis,urine_output",
        "url":    "https://kdigo.org/guidelines/acute-kidney-injury/",
    },
    "kdigo_ckd_2024.pdf": {
        "source": "KDIGO Clinical Practice Guideline for CKD 2024",
        "tags":   "chronic_kidney_disease,renal,egfr,proteinuria,CKD",
        "url":    "https://kdigo.org/guidelines/ckd-mbd/",
    },
    "ardsnet_ventilation_protocol.pdf": {
        "source": "ARDSNet Low Tidal Volume Ventilation Protocol (ARMA Trial)",
        "tags":   "ARDS,mechanical_ventilation,PEEP,lung_protective,tidal_volume,FiO2",
        "url":    "https://www.ardsnet.org/tools.shtml",
    },
    "aha_heart_failure_2022.pdf": {
        "source": "AHA/ACC/HFSA Guideline for the Management of Heart Failure 2022",
        "tags":   "heart_failure,cardiology,ejection_fraction,diuretics,ACE_inhibitor,LVEF",
        "url":    "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063",
    },
    "aha_atrial_fibrillation_2023.pdf": {
        "source": "AHA/ACC/HRS Guideline for Diagnosis and Management of Atrial Fibrillation 2023",
        "tags":   "atrial_fibrillation,cardiology,anticoagulation,rate_control,rhythm_control",
        "url":    "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001193",
    },
    "ashp_vancomycin_dosing_2020.pdf": {
        "source": "ASHP/IDSA/SIDP Vancomycin AUC-Guided Dosing Consensus Guidelines 2020",
        "tags":   "vancomycin,pharmacokinetics,dosing,antibiotics,AUC,MRSA,renal,AKI",
        "url":    "https://academic.oup.com/ajhp/article/77/11/835/5810521",
    },
    "idsa_community_pneumonia_2019.pdf": {
        "source": "IDSA/ATS Community-Acquired Pneumonia Guidelines 2019",
        "tags":   "pneumonia,CAP,antibiotics,infectious_disease,respiratory,azithromycin",
        "url":    "https://academic.oup.com/cid/article/71/7/e45/5838088",
    },
    "nice_sepsis_ng253.pdf": {
        "source": "NICE Guideline NG253: Suspected Sepsis (Adults 16+) 2024",
        "tags":   "sepsis,NEWS2,antibiotics,NICE,UK,diagnosis,lactate,blood_cultures",
        "url":    "https://www.nice.org.uk/guidance/ng253",
    },
    "who_sepsis_2017.pdf": {
        "source": "WHO: Improving the Prevention, Diagnosis and Management of Sepsis 2017",
        "tags":   "sepsis,WHO,global_health,prevention,diagnosis,management",
        "url":    "https://iris.who.int/handle/10665/274742",
    },
}


_SECTION_BOUNDARY = re.compile(
    r'\n(?='
    r'(?:\d+\.\d*\s+[A-Z])'           # e.g. "3.2 Antibiotic Selection"
    r'|(?:[A-Z][A-Z\s]{8,}[:\n])'
    r'|(?:\bRECOMMENDATION\b)'
    r'|(?:\bStrong recommendation\b)'
    r'|(?:\bWeak recommendation\b)'
    r'|(?:\bEvidence\b.*\blevel\b)'
    r'|(?:We recommend\b)'
    r'|(?:We suggest\b)'
    r')',
    re.IGNORECASE,
)

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80
MIN_CHUNK_LEN = 60


def extract_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split guideline text at section boundaries, then further split long sections
    with overlap. Each chunk carries source attribution.
    """
    sections = _SECTION_BOUNDARY.split(text)
    chunks   = []

    for section in sections:
        section = section.strip()
        if len(section) < MIN_CHUNK_LEN:
            continue

        words = section.split()
        if len(words) <= CHUNK_SIZE:
            chunks.append({"text": section, "source": source})
        else:
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_words = words[i : i + CHUNK_SIZE]
                chunk_text  = " ".join(chunk_words)
                if len(chunk_text) >= MIN_CHUNK_LEN:
                    chunks.append({"text": chunk_text, "source": source})

    return chunks


def make_chunk_id(filename: str, chunk_index: int, text: str) -> str:
    """Stable, unique ID for each chunk."""
    h = hashlib.md5(f"{filename}:{chunk_index}:{text[:80]}".encode()).hexdigest()[:12]
    return f"guideline_{filename.replace('.pdf','').replace('-','_')}_{chunk_index:04d}_{h}"


def main():
    print("=" * 70)
    print("  Clinical Guidelines ChromaDB Seeder   ")
    print(f"  PDFs source:   {GUIDELINES_DIR}")
    print(f"  ChromaDB dest: {CHROMA_DIR}")
    print("=" * 70)

    available = sorted(GUIDELINES_DIR.glob("*.pdf"))
    if not available:
        print(f"\nERROR: No PDFs found in {GUIDELINES_DIR}")
        print("Run scripts/download_guidelines.py first.")
        sys.exit(1)
    print(f"\nFound {len(available)} PDF(s): {[p.name for p in available]}")

    print(f"\n[1/3] Connecting to ChromaDB at {CHROMA_DIR} …")
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="clinical_guidelines",
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Existing chunks: {collection.count()}")

    print(f"\n[2/3] Loading embedding model '{config.EMBEDDING_MODEL}' …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    dim   = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)()
    print(f"  Model loaded. Embedding dim: {dim}")

    print(f"\n[3/3] Processing PDFs …\n")
    total_chunks    = 0
    total_upserted  = 0
    skipped_pdfs    = []

    for pdf_path in available:
        filename = pdf_path.name
        meta     = GUIDELINE_CATALOG.get(filename)

        if meta is None:
            print(f"─── {filename}: NOT IN CATALOG   skipping")
            print(f"    Add an entry to GUIDELINE_CATALOG in seed_guidelines.py to include it.")
            skipped_pdfs.append(filename)
            continue

        source = meta["source"]
        print(f"─── {filename}")
        print(f"    Source: {source}")
        print(f"    Size:   {pdf_path.stat().st_size / 1024:.0f} KB")

        try:
            raw_text = extract_text(pdf_path)
        except Exception as e:
            print(f"    ERROR extracting text: {e}")
            skipped_pdfs.append(filename)
            continue

        if not raw_text.strip():
            print(f"    WARNING: No text extracted (scanned PDF?). Skipping.")
            skipped_pdfs.append(filename)
            continue

        print(f"    Extracted {len(raw_text):,} characters")

        chunks = chunk_text(raw_text, source)
        print(f"    Chunked into {len(chunks)} segments")
        total_chunks += len(chunks)

        BATCH_SIZE  = 32
        file_upsert = 0

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start : batch_start + BATCH_SIZE]

            texts     = [c["text"] for c in batch]
            ids       = [
                make_chunk_id(filename, batch_start + i, c["text"])
                for i, c in enumerate(batch)
            ]
            metadatas = [
                {
                    "source":       source,
                    "filename":     filename,
                    "tags":         meta["tags"],
                    "url":          meta["url"],
                    "is_guideline": True,
                    "chunk_index":  batch_start + i,
                }
                for i, c in enumerate(batch)
            ]

            embeddings = model.encode(
                texts,
                batch_size=16,
                show_progress_bar=False,
            ).tolist()

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            file_upsert  += len(batch)
            total_upserted += len(batch)

        print(f"    ok Upserted {file_upsert} chunks. Collection total: {collection.count()}")
        print()

    print("=" * 70)
    print(f"  DONE.")
    print(f"  PDFs processed:  {len(available) - len(skipped_pdfs)}/{len(available)}")
    print(f"  Total chunks:    {total_chunks}")
    print(f"  Total upserted:  {total_upserted}")
    print(f"  Collection size: {collection.count()}")
    if skipped_pdfs:
        print(f"  Skipped:         {skipped_pdfs}")
    print("=" * 70)

    print("\n─── Smoke test queries ───")
    test_queries = [
        "antibiotic treatment for septic shock within one hour",
        "vancomycin AUC dosing MRSA target 400 600",
        "PEEP tidal volume ARDS lung protective ventilation",
        "heart failure ejection fraction treatment guideline",
        "AKI creatinine definition staging KDIGO",
    ]
    for q in test_queries:
        emb = model.encode(q).tolist()
        res = collection.query(query_embeddings=[emb], n_results=2)
        print(f"\nQuery: '{q}'")
        for i, (doc, meta_r, dist) in enumerate(zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )):
            relevance = round(1.0 - dist, 3)
            print(f"  [{i+1}] relevance={relevance:.3f}  source={meta_r.get('source','?')[:50]}")
            print(f"       text: {doc[:100].strip()}…")

    print("\nGuideline seeding complete.")


if __name__ == "__main__":
    main()
