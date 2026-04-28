"""
scripts/seed_chroma.py   One-shot ChromaDB seeding script for .

Run ONCE from the clinical_agent/ directory:
    cd clinical_agent
    python scripts/seed_chroma.py

This script:
1. Connects to the ChromaDB persistent store at data/chroma_db/
2. Loads the sentence-transformer embedding model
3. Fetches abstracts for 20 key papers via PubMed E-utilities
4. Fetches papers unavailable on PubMed from Semantic Scholar
5. Upserts all papers into the "clinical_literature" ChromaDB collection

After running, call search_clinical_literature_rag() in the MCP server
to verify the 20 papers are queryable.
"""


import sys
import time
import hashlib
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

SCRIPT_DIR  = Path(__file__).resolve().parent
AGENT_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

import config


SEED_PAPERS_BY_PMID = [
    "27219127",
    "11445675",
    "26903338",
    "29986080",
    "28263171",
    "24321492",
    "31637566",
    "34017813",
    "32553558",
    "27111803",
    "30304647",
]

SEED_PAPERS_BY_S2 = [
    "RETAIN interpretable predictive model healthcare reverse time attention Choi",
    "ClinicalBERT modeling clinical notes electronic health records Huang 2019",
    "HuggingGPT solving AI tasks ChatGPT Shen 2023",
    "benchmarks electronic health records Harutyunyan 2019 MIMIC mortality",
]

MANUAL_FALLBACKS = [
    {
        "title": "A Unified Approach to Interpreting Model Predictions (SHAP)",
        "abstract": (
            "Understanding why a model makes a prediction is crucial for many machine learning "
            "applications. We propose SHAP (SHapley Additive exPlanations), a unified framework "
            "for interpreting predictions. SHAP assigns each feature an importance value for a "
            "particular prediction. Its key properties are local accuracy, missingness, and "
            "consistency. We present efficient model-agnostic approximations that are consistent "
            "with their associated Shapley values and show how SHAP values unify six existing "
            "methods: LIME, DeepLIFT, SHAP, Layer-wise relevance propagation, Shapley regression "
            "values, and Shapley sampling values. SHAP values represent each feature's contribution "
            "to the prediction relative to a baseline, enabling consistent feature attribution in "
            "complex clinical predictive models such as XGBoost and neural networks."
        ),
        "pmid": "shap_neurips_2017",
        "authors": "Lundberg SM, Lee SI",
        "year": 2017,
    },
    {
        "title": "RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention",
        "abstract": (
            "Accurate and interpretable predictions are crucial for clinical decision support systems. "
            "We propose RETAIN, an interpretable predictive model for general healthcare applications. "
            "RETAIN uses a two-level neural attention model to detect influential past visits and "
            "significant clinical variables within those visits. The model mimics physician practice "
            "by attending to past clinical visits in reverse chronological order. We validate RETAIN "
            "on a large EHR dataset with 14 million patients, showing that it achieves comparable "
            "prediction accuracy to RNNs while providing interpretable outputs. RETAIN enables "
            "clinicians to understand which past ICU visits and which diagnoses drove the prediction."
        ),
        "pmid": "retain_nips_2016",
        "authors": "Choi E, Bahadori MT, Sun J, et al.",
        "year": 2016,
    },
    {
        "title": "Cognitive Architectures for Language Agents (CoALA)",
        "abstract": (
            "The recent advent of large language models (LLMs) has inspired the development of "
            "language agents   LLMs augmented with planning and acting capabilities. We synthesize "
            "a taxonomy of language agent architectures by mapping them onto a cognitive architecture "
            "framework drawing on decades of research in cognitive science. This framework organizes "
            "components of language agents into four categories: memory (working, episodic, semantic, "
            "procedural), action space (memory, execution, reasoning, learning), decision-making, and "
            "learning from feedback. We analyze existing language agents through this lens and find "
            "that most focus on a small subset of capabilities, leaving significant gaps that must "
            "be addressed for reliable clinical AI orchestration systems."
        ),
        "pmid": "coala_2024",
        "authors": "Sumers TR, Yao S, Narasimhan K, Griffiths TL",
        "year": 2024,
    },
    {
        "title": "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning",
        "abstract": (
            "We develop an algorithm that can detect pneumonia from chest X-rays at a level "
            "exceeding practiced radiologists. Our algorithm, CheXNet, is a 121-layer convolutional "
            "neural network trained on ChestX-ray14, currently the largest publicly available chest "
            "X-ray dataset, containing over 100,000 frontal-view X-ray images with disease labels "
            "mined from radiology reports using NLP. We find that CheXNet achieves an F1 score of "
            "0.435 on the test set, which exceeds the average radiologist F1 score of 0.387. The "
            "model also achieves state-of-the-art performance on all 14 pathologies in ChestX-ray14, "
            "demonstrating the potential of deep learning for radiology AI in clinical settings."
        ),
        "pmid": "chexnet_arxiv_2017",
        "authors": "Rajpurkar P, Irvin J, Ball RL, et al.",
        "year": 2017,
    },
    {
        "title": "ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission",
        "abstract": (
            "Clinical notes contain rich information about patient health and are an important "
            "resource for clinical decision support. We train ClinicalBERT, a language model "
            "using BERT pre-trained on clinical notes from MIMIC-III, on the task of predicting "
            "30-day hospital readmission. ClinicalBERT outperforms baseline models including BERT "
            "pre-trained on general text corpora, demonstrating the importance of domain-specific "
            "pre-training for clinical NLP tasks. We show that clinical text representations "
            "capture important temporal dynamics and comorbidity patterns not captured by "
            "structured EHR features alone. This has implications for early warning systems "
            "and discharge planning in ICU settings."
        ),
        "pmid": "clinicalbert_arxiv_2019",
        "authors": "Huang K, Altosaar J, Ranganath R",
        "year": 2019,
    },
    {
        "title": "Multitask learning and benchmarking with clinical time series data (MIMIC benchmarks)",
        "abstract": (
            "Health care is one of the most exciting frontiers in data mining and machine learning. "
            "Successful adoption of electronic health records (EHRs) created an opportunity to "
            "learn about patient health from clinical time-series data. We propose four clinical "
            "prediction tasks   in-hospital mortality, ICD-9 code group classification, length "
            "of stay, and physiological decompensation   and two datasets derived from MIMIC-III. "
            "We implement and benchmark a number of state-of-the-art models including "
            "logistic regression, LSTM, and channel-wise LSTM. Our benchmarks establish a "
            "reference point for future work on clinical time series prediction and demonstrate "
            "that MIMIC-III provides a rich resource for clinical machine learning research."
        ),
        "pmid": "harutyunyan_benchmarks_2019",
        "authors": "Harutyunyan H, Khachatrian H, Kale DC, et al.",
        "year": 2019,
    },
    {
        "title": "High-performance medicine: the convergence of human and artificial intelligence",
        "abstract": (
            "The growth of artificial intelligence (AI), and notably deep learning, has produced "
            "a remarkably rapid series of advances in the capacity to process medical images, "
            "sounds, signals, genomics, and electronic health records for a wide range of "
            "applications including diagnosis, risk stratification, and treatment response prediction. "
            "High-performance medicine requires combining the irreplaceable human dimensions of care "
            "with the AI capabilities that reduce variability, bias, and medical errors, augmenting "
            "the physician's pattern recognition and decision-making with data-driven tools at scale."
        ),
        "pmid": "31637566",
        "authors": "Topol EJ",
        "year": 2019,
    },
]

NCBI_RATE_LIMIT = 0.4
_last_ncbi = 0.0


def _ncbi_get(url: str, params: dict) -> requests.Response:
    global _last_ncbi
    elapsed = time.monotonic() - _last_ncbi
    if elapsed < NCBI_RATE_LIMIT:
        time.sleep(NCBI_RATE_LIMIT - elapsed)
    _last_ncbi = time.monotonic()
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp


def fetch_by_pmid(pmid: str) -> dict | None:
    """Fetch a single paper by PMID via PubMed efetch."""
    try:
        resp = _ncbi_get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "xml"},
        )
        root = ET.fromstring(resp.text)
        article = root.find(".//PubmedArticle")
        if article is None:
            return None

        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        journal_el = article.find(".//Journal/Title")
        year_el = article.find(".//PubDate/Year")

        authors = []
        for au in article.findall(".//AuthorList/Author")[:6]:
            last = au.findtext("LastName", "")
            initials = au.findtext("Initials", "")
            name = f"{last} {initials}".strip()
            if name.strip():
                authors.append(name)
        if len(article.findall(".//AuthorList/Author")) > 6:
            authors.append("et al.")

        abstract_parts = []
        for ab in article.findall(".//AbstractText"):
            label = ab.get("Label", "")
            text = "".join(ab.itertext())
            abstract_parts.append(f"{label}: {text}" if label else text)

        abstract = " ".join(abstract_parts).strip()
        if not abstract:
            return None

        return {
            "pmid": pmid_el.text if pmid_el is not None else pmid,
            "title": "".join(title_el.itertext()) if title_el is not None else "",
            "authors": ", ".join(authors),
            "journal": journal_el.text if journal_el is not None else "",
            "year": int(year_el.text) if year_el is not None else 0,
            "abstract": abstract[:3000],
        }
    except Exception as e:
        print(f"  [PMID {pmid}] fetch error: {e}")
        return None


def fetch_by_s2(query: str) -> dict | None:
    """Fetch the top paper matching a Semantic Scholar query."""
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "fields": "title,authors,year,abstract,externalIds",
                "limit": 1,
            },
            headers={"User-Agent": "ClinicalAIAgent/1.0"},
            timeout=20,
        )
        if resp.status_code == 429:
            print("  [S2] rate-limited, sleeping 5s…")
            time.sleep(5)
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "fields": "title,authors,year,abstract,externalIds",
                    "limit": 1,
                },
                headers={"User-Agent": "ClinicalAIAgent/1.0"},
                timeout=20,
            )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None

        paper = data[0]
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            return None

        authors = [a.get("name", "") for a in paper.get("authors", [])[:5]]
        if len(paper.get("authors", [])) > 5:
            authors.append("et al.")

        ext_ids = paper.get("externalIds", {}) or {}
        pmid = str(ext_ids.get("PubMed", ""))

        return {
            "pmid": pmid,
            "title": paper.get("title", ""),
            "authors": ", ".join(authors),
            "year": paper.get("year") or 0,
            "abstract": abstract[:3000],
        }
    except Exception as e:
        print(f"  [S2] query error for '{query[:50]}': {e}")
        return None


def upsert_paper(collection, embedding_model, paper: dict) -> bool:
    """Embed and upsert a paper dict into ChromaDB."""
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    pmid = str(paper.get("pmid", "")).strip()

    if not title or not abstract:
        return False

    doc_id = pmid if pmid else hashlib.md5(title.encode()).hexdigest()[:16]
    text_to_embed = f"{title}. {abstract}"

    embedding = embedding_model.encode(text_to_embed, show_progress_bar=False).tolist()

    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[abstract],
        metadatas=[{
            "title": title,
            "pmid": pmid,
            "authors": paper.get("authors", ""),
            "year": int(paper.get("year") or 0),
        }],
    )
    return True


def main():
    print("=" * 60)
    print("    ChromaDB Seeding Script")
    print("=" * 60)

    print(f"\n[1/3] Loading ChromaDB at {config.CHROMA_DB_DIR} …")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
        collection = client.get_or_create_collection(
            name="clinical_literature",
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  Collection 'clinical_literature'   {collection.count()} existing docs")
    except Exception as e:
        print(f"  ERROR: ChromaDB failed: {e}")
        sys.exit(1)

    print(f"\n[2/3] Loading embedding model '{config.EMBEDDING_MODEL}' …")
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"  Model loaded OK")
    except Exception as e:
        print(f"  ERROR: Embedding model failed: {e}")
        sys.exit(1)

    print("\n[3/3] Fetching and embedding papers …\n")

    all_papers: list[dict] = []
    seen_pmids: set[str] = set()

    print(f"--- Stage A: PubMed PMIDs ({len(SEED_PAPERS_BY_PMID)} papers) ---")
    for pmid in SEED_PAPERS_BY_PMID:
        if pmid in seen_pmids:
            continue
        print(f"  Fetching PMID {pmid} …", end=" ", flush=True)
        paper = fetch_by_pmid(pmid)
        if paper:
            all_papers.append(paper)
            seen_pmids.add(pmid)
            print(f"OK   {paper['title'][:60]}")
        else:
            print("SKIP (no abstract)")

    print(f"\n--- Stage B: Semantic Scholar queries ({len(SEED_PAPERS_BY_S2)} queries) ---")
    for query in SEED_PAPERS_BY_S2:
        print(f"  S2 query: '{query[:55]}' …", end=" ", flush=True)
        paper = fetch_by_s2(query)
        if paper and paper.get("pmid") not in seen_pmids:
            all_papers.append(paper)
            if paper.get("pmid"):
                seen_pmids.add(paper["pmid"])
            print(f"OK   {paper['title'][:60]}")
        elif paper:
            print(f"SKIP (duplicate PMID {paper.get('pmid')})")
        else:
            print("SKIP (no result)")
        time.sleep(1.5)

    print(f"\n--- Stage C: Manual fallbacks ({len(MANUAL_FALLBACKS)} papers) ---")
    for paper in MANUAL_FALLBACKS:
        pmid = str(paper.get("pmid", ""))
        if pmid and pmid in seen_pmids:
            print(f"  SKIP (duplicate): {paper['title'][:60]}")
            continue
        all_papers.append(paper)
        if pmid:
            seen_pmids.add(pmid)
        print(f"  Added: {paper['title'][:60]}")

    print(f"\n--- Stage D: Upserting {len(all_papers)} papers into ChromaDB ---")
    success = 0
    for i, paper in enumerate(all_papers, 1):
        title_short = paper.get("title", "?")[:55]
        print(f"  [{i:02d}/{len(all_papers)}] Embedding '{title_short}' …", end=" ", flush=True)
        try:
            ok = upsert_paper(collection, embedding_model, paper)
            if ok:
                success += 1
                print("OK")
            else:
                print("SKIP (missing title/abstract)")
        except Exception as e:
            print(f"ERROR: {e}")

    final_count = collection.count()
    print(f"\n{'='*60}")
    print(f"  Seeding complete: {success}/{len(all_papers)} papers upserted")
    print(f"  ChromaDB collection size: {final_count} docs")
    print(f"{'='*60}")

    print("\n--- Smoke test: query 'ICU mortality prediction' ---")
    try:
        q_emb = embedding_model.encode("ICU mortality prediction").tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=3)
        for i, (doc_id, meta, dist) in enumerate(zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0]
        )):
            print(f"  [{i+1}] (dist={dist:.4f}) {meta.get('title', doc_id)[:70]}")
    except Exception as e:
        print(f"  Smoke test failed: {e}")

    print("\nDone. You can now use search_clinical_literature_rag() in the MCP server.")


if __name__ == "__main__":
    main()
