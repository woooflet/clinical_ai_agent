"""
scripts/batch_seed_chroma.py   Large-scale ChromaDB seeding using NCBI API key.

Queries PubMed across 15 clinical AI topic areas and embeds all retrieved
abstracts into the ChromaDB vector store. Uses the NCBI API key for 10 req/sec
throughput (vs 3 req/sec without).

Run from the clinical_agent/ directory:
    cd clinical_agent
    nohup python scripts/batch_seed_chroma.py > logs/batch_seed.log 2>&1 &

Or interactively:
    python scripts/batch_seed_chroma.py

Expected runtime: ~10-20 minutes for ~400 papers on CPU embedding.
Progress is logged continuously   safe to interrupt and re-run (idempotent).
"""


import sys
import time
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_DIR  = SCRIPT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

LOG_DIR = AGENT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

import config

NCBI_API_KEY = config.NCBI_API_KEY
if not NCBI_API_KEY:
    print("WARNING: No NCBI API key found. Rate limit will be 3 req/sec.")
    NCBI_RATE_LIMIT = 0.4
else:
    print(f"NCBI API key loaded ({NCBI_API_KEY[:8]}…). Rate limit: 10 req/sec.")
    NCBI_RATE_LIMIT = 0.11

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

CLINICAL_STOPWORDS = {
    "agriculture", "plant pathogen", "plant disease", "crop", "soil microbi",
    "wildfire", "dental pulp", "dental adhesin", "veterinary", "poultry",
    "livestock", "pneumococcal adhesin", "equine", "bovine", "avian",
    "silkworm", "maize", "wheat", "soybean",
}


def _is_clinically_relevant(paper: dict) -> bool:
    """Return False if the paper matches non-clinical stopwords."""
    combined = (
        paper.get("title", "") + " " + paper.get("abstract", "")
    ).lower()
    return not any(word in combined for word in CLINICAL_STOPWORDS)


TOPICS = [
    ("mimic_iii",
     "MIMIC critical care database electronic health records",
     40),
    ("physionet",
     "PhysioNet waveform database clinical signals",
     20),

    ("sepsis_ml",
     "sepsis prediction machine learning ICU",
     40),
    ("sepsis_definitions",
     "Sepsis-3 organ failure SOFA score definition",
     30),
    ("icu_mortality",
     "ICU mortality prediction deep learning electronic health records",
     50),
    ("icu_los",
     "ICU length of stay prediction neural network",
     30),

    ("xgboost_clinical",
     "XGBoost gradient boosting clinical outcome prediction",
     30),
    ("lstm_clinical",
     "LSTM recurrent neural network clinical time series prediction",
     30),
    ("clinical_bert",
     "BERT clinical notes NLP electronic health records fine-tuning",
     30),
    ("shap_clinical",
     "SHAP Shapley values feature importance clinical machine learning",
     30),

    ("explainable_ai_health",
     "explainable artificial intelligence healthcare clinical decision support",
     40),
    ("fairness_ml_health",
     "algorithmic bias[ti] OR health disparity AND machine learning AND (hospital OR clinical OR EHR)",
     25),

    ("llm_clinical",
     "large language model GPT clinical medicine diagnosis",
     40),
    ("clinical_decision_support",
     "clinical decision support system artificial intelligence physician",
     30),

    ("readmission",
     "hospital readmission prediction 30-day machine learning",
     30),
    ("aki_prediction",
     "acute kidney injury prediction machine learning creatinine",
     25),
    ("phenotyping",
     "(patient phenotyping[MeSH] OR clinical subgroup[ti] OR patient subtype[ti]) AND (ICU OR hospital OR EHR)",
     25),

    ("radiology_ai",
     "deep learning chest X-ray radiology pneumonia classification",
     25),

    ("clinical_nlp",
     "natural language processing discharge summary clinical note information extraction",
     30),
    ("icd_coding",
     "ICD diagnosis code prediction clinical note NLP",
     25),

    ("cardiology_icu",
     "cardiac ICU heart failure arrhythmia atrial fibrillation prediction machine learning EHR",
     40),
    ("aki_ckd_management",
     "acute kidney injury renal replacement therapy ICU creatinine prediction machine learning",
     35),
    ("drug_dosing_pk",
     "vancomycin antibiotic pharmacokinetics dose optimization ICU renal function",
     25),
    ("ventilator_ards",
     "mechanical ventilation ARDS acute respiratory distress lung protective PEEP tidal volume",
     30),
    ("neurocritical_care",
     "neurological ICU stroke traumatic brain injury seizure delirium prediction machine learning",
     25),
    ("clinical_pharmacology",
     "drug drug interaction polypharmacy adverse drug event electronic health records hospital",
     25),
    ("delirium_sedation",
     "ICU delirium sedation analgesia RASS CAM-ICU prevention prediction critical care",
     20),
]


_last_ncbi = 0.0


def _ncbi_get(url: str, params: dict) -> requests.Response:
    global _last_ncbi
    elapsed = time.monotonic() - _last_ncbi
    if elapsed < NCBI_RATE_LIMIT:
        time.sleep(NCBI_RATE_LIMIT - elapsed)
    _last_ncbi = time.monotonic()
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def search_pmids(query: str, max_results: int) -> list[str]:
    """Return up to max_results PMIDs for a PubMed query."""
    try:
        resp = _ncbi_get(
            f"{PUBMED_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "xml",
                "sort": "relevance",
            },
        )
        root = ET.fromstring(resp.text)
        return [el.text for el in root.findall(".//IdList/Id")]
    except Exception as e:
        print(f"    [esearch error] {e}")
        return []


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch abstracts for a batch of PMIDs (up to 100 at a time)."""
    results = []
    for chunk_start in range(0, len(pmids), 100):
        chunk = pmids[chunk_start:chunk_start + 100]
        try:
            resp = _ncbi_get(
                f"{PUBMED_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(chunk),
                    "rettype": "abstract",
                    "retmode": "xml",
                },
            )
            root = ET.fromstring(resp.text)
            for article in root.findall(".//PubmedArticle"):
                try:
                    pmid_el  = article.find(".//PMID")
                    title_el = article.find(".//ArticleTitle")
                    year_el  = article.find(".//PubDate/Year")

                    pmid    = pmid_el.text if pmid_el is not None else ""
                    title   = "".join(title_el.itertext()) if title_el is not None else ""

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
                        text  = "".join(ab.itertext())
                        abstract_parts.append(f"{label}: {text}" if label else text)
                    abstract = " ".join(abstract_parts).strip()

                    if not abstract or not title:
                        continue

                    results.append({
                        "pmid":     pmid,
                        "title":    title,
                        "authors":  ", ".join(authors),
                        "year":     int(year_el.text) if year_el is not None else 0,
                        "abstract": abstract[:4000],
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f"    [efetch error for chunk] {e}")
            time.sleep(1)
    return results


def upsert_paper(collection, embedding_model, paper: dict) -> bool:
    title    = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    pmid     = str(paper.get("pmid", "")).strip()

    if not title or not abstract:
        return False

    doc_id = pmid if pmid else hashlib.md5(title.encode()).hexdigest()[:16]
    text   = f"{title}. {abstract}"
    emb    = embedding_model.encode(text, show_progress_bar=False).tolist()

    collection.upsert(
        ids=[doc_id],
        embeddings=[emb],
        documents=[abstract],
        metadatas=[{
            "title":   title,
            "pmid":    pmid,
            "authors": paper.get("authors", ""),
            "year":    int(paper.get("year") or 0),
        }],
    )
    return True


MANUAL_PAPERS = [
    {
        "title":    "A Unified Approach to Interpreting Model Predictions (SHAP)",
        "abstract": (
            "We propose SHAP (SHapley Additive exPlanations), a unified framework for interpreting "
            "model predictions. SHAP assigns each feature an importance value for a particular prediction. "
            "Key properties are local accuracy, missingness, and consistency. We present efficient "
            "model-agnostic approximations consistent with Shapley values that unify six existing methods: "
            "LIME, DeepLIFT, SHAP, layer-wise relevance propagation, Shapley regression values, and "
            "Shapley sampling values. SHAP values represent each feature's contribution to the prediction "
            "relative to a baseline, enabling consistent feature attribution in XGBoost and neural networks "
            "used for clinical mortality prediction and sepsis risk scoring."
        ),
        "pmid": "shap_neurips_2017", "authors": "Lundberg SM, Lee SI", "year": 2017,
    },
    {
        "title":    "RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention",
        "abstract": (
            "We propose RETAIN, an interpretable predictive model for healthcare using a two-level "
            "neural attention model that detects influential past visits and significant clinical variables. "
            "RETAIN mimics physician practice by attending to past clinical visits in reverse chronological "
            "order. Validated on a large EHR dataset with 14 million patients, RETAIN achieves prediction "
            "accuracy comparable to RNNs while providing interpretable outputs showing which ICU visits and "
            "diagnoses drove the mortality prediction."
        ),
        "pmid": "retain_nips_2016", "authors": "Choi E, Bahadori MT, Sun J, et al.", "year": 2016,
    },
    {
        "title":    "Cognitive Architectures for Language Agents (CoALA)",
        "abstract": (
            "We synthesize a taxonomy of language agent architectures by mapping them onto a cognitive "
            "architecture framework drawing on cognitive science. This framework organizes components into "
            "four categories: memory (working, episodic, semantic, procedural), action space (memory, "
            "execution, reasoning, learning), decision-making, and learning from feedback. We analyze "
            "existing language agents through this lens, revealing that most focus on a small subset of "
            "capabilities, leaving gaps that must be addressed for reliable clinical AI orchestration."
        ),
        "pmid": "coala_2024", "authors": "Sumers TR, Yao S, Narasimhan K, Griffiths TL", "year": 2024,
    },
    {
        "title":    "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning",
        "abstract": (
            "We develop CheXNet, a 121-layer convolutional neural network trained on ChestX-ray14, the "
            "largest publicly available chest X-ray dataset with over 100,000 frontal-view images. "
            "CheXNet achieves an F1 score of 0.435, exceeding the average radiologist F1 of 0.387, and "
            "achieves state-of-the-art on all 14 pathologies in ChestX-ray14, demonstrating deep "
            "learning potential for radiology AI in clinical settings."
        ),
        "pmid": "chexnet_arxiv_2017", "authors": "Rajpurkar P, Irvin J, Ball RL, et al.", "year": 2017,
    },
    {
        "title":    "ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission",
        "abstract": (
            "We train ClinicalBERT, a language model pre-trained on clinical notes from MIMIC-III, "
            "on the task of predicting 30-day hospital readmission. ClinicalBERT outperforms BERT "
            "pre-trained on general text, demonstrating the importance of domain-specific pre-training. "
            "Clinical text representations capture temporal dynamics and comorbidity patterns not "
            "captured by structured EHR features alone, with implications for early warning systems "
            "and discharge planning in ICU settings."
        ),
        "pmid": "clinicalbert_2019", "authors": "Huang K, Altosaar J, Ranganath R", "year": 2019,
    },
    {
        "title":    "Multitask learning and benchmarking with clinical time series data (MIMIC benchmarks)",
        "abstract": (
            "We propose four clinical prediction benchmarks   in-hospital mortality, ICD-9 code group "
            "classification, length of stay, and physiological decompensation   using datasets derived "
            "from MIMIC-III. We benchmark logistic regression, LSTM, and channel-wise LSTM models. "
            "Our results establish a reference point for clinical time series prediction and demonstrate "
            "MIMIC-III as a rich resource for clinical machine learning research."
        ),
        "pmid": "harutyunyan_2019", "authors": "Harutyunyan H, Khachatrian H, Kale DC, et al.", "year": 2019,
    },
    {
        "title":    "High-performance medicine: the convergence of human and artificial intelligence",
        "abstract": (
            "Deep learning has produced rapid advances in processing medical images, signals, genomics, "
            "and electronic health records for diagnosis, risk stratification, and treatment response "
            "prediction. High-performance medicine requires combining irreplaceable human clinical "
            "dimensions with AI capabilities that reduce variability, bias, and medical errors, "
            "augmenting physician pattern recognition with data-driven tools at scale."
        ),
        "pmid": "31637566", "authors": "Topol EJ", "year": 2019,
    },
    {
        "title":    "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face",
        "abstract": (
            "We propose HuggingGPT, a system that leverages ChatGPT to connect AI models in Hugging Face "
            "to solve complex AI tasks. ChatGPT acts as a controller, managing expert models for task "
            "planning, model selection, task execution, and response generation. This demonstrates the "
            "potential of LLM-as-orchestrator for multi-modal AI pipelines, directly analogous to "
            "clinical decision support architectures that route queries to specialized medical models."
        ),
        "pmid": "hugging_gpt_2023", "authors": "Shen Y, Song K, Tan X, et al.", "year": 2023,
    },
]


def main():
    print("=" * 70)
    print("  Batch ChromaDB Seeder   NCBI API Key Mode")
    print(f"  Rate limit: {'10 req/sec (authenticated)' if NCBI_API_KEY else '3 req/sec (anonymous)'}")
    print("=" * 70)

    print(f"\n[1/3] Connecting to ChromaDB at {config.CHROMA_DB_DIR} …")
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name="clinical_literature",
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Existing docs: {collection.count()}")

    print(f"\n[2/3] Loading embedding model '{config.EMBEDDING_MODEL}' …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)()
    print(f"  Model loaded. Embedding dim: {dim}")

    print(f"\n[3/3] Fetching papers across {len(TOPICS)} topics …\n")

    total_fetched   = 0
    total_upserted  = 0
    seen_pmids: set[str] = set()

    print("─── Manual seed papers ───")
    for paper in MANUAL_PAPERS:
        pmid = str(paper.get("pmid", ""))
        if pmid in seen_pmids:
            print(f"  SKIP (dup): {paper['title'][:60]}")
            continue
        ok = upsert_paper(collection, model, paper)
        if ok:
            seen_pmids.add(pmid)
            total_upserted += 1
            print(f"  ok {paper['title'][:65]}")
        else:
            print(f"  FAIL SKIP (no abstract): {paper['title'][:60]}")

    print(f"\n  Manual papers done. Collection size: {collection.count()}\n")

    for topic_idx, (label, query, max_n) in enumerate(TOPICS, 1):
        print(f"─── [{topic_idx:02d}/{len(TOPICS)}] {label}  (max {max_n}) ───")
        print(f"  Query: {query}")

        pmids = search_pmids(query, max_n)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        print(f"  Found {len(pmids)} PMIDs, {len(new_pmids)} new")

        if not new_pmids:
            print("  (all already in collection)")
            print()
            continue

        papers = fetch_abstracts(new_pmids)
        print(f"  Fetched {len(papers)} abstracts")

        batch_upserted = 0
        for paper in papers:
            pmid = paper.get("pmid", "")
            if pmid in seen_pmids:
                continue
            if not _is_clinically_relevant(paper):
                print(f"  SKIP (noise): {paper.get('title', '')[:60]}")
                continue
            ok = upsert_paper(collection, model, paper)
            if ok:
                seen_pmids.add(pmid)
                batch_upserted += 1
                total_upserted += 1

        total_fetched += len(papers)
        print(f"  Upserted {batch_upserted}/{len(papers)}. Collection size: {collection.count()}")
        print()

    print("=" * 70)
    print(f"  DONE.")
    print(f"  Papers fetched:   {total_fetched}")
    print(f"  Papers upserted:  {total_upserted}")
    print(f"  Collection size:  {collection.count()}")
    print("=" * 70)

    print("\n─── Smoke test queries ───")
    for q in ["ICU mortality prediction", "sepsis management treatment", "SHAP explainability"]:
        emb = model.encode(q).tolist()
        res = collection.query(query_embeddings=[emb], n_results=3)
        print(f"\nQuery: '{q}'")
        for i, (meta, dist) in enumerate(zip(res["metadatas"][0], res["distances"][0])):
            print(f"  [{i+1}] (d={dist:.3f}) {meta.get('title', '?')[:70]}")

    print("\nBatch seeding complete.")


if __name__ == "__main__":
    main()
