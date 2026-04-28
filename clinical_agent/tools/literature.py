
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

import requests

_chroma_collection  = None   # 'clinical_literature'   PubMed abstracts
_chroma_guidelines  = None   # 'clinical_guidelines'    full-text guideline chunks
_chroma_notes       = None   # 'mimic_notes'            de-identified MIMIC discharge summaries (_embedding_model    = None

_bm25_index    = None
_bm25_doc_ids  = []

_reranker = None

_db_con = None

def set_db_connection(con) -> None:
    """Inject the shared read-only DuckDB connection for MIMIC note look-ups."""
    global _db_con
    _db_con = con

_NCBI_RATE_LIMIT_SECS = 0.35
_last_ncbi_request: float = 0.0

PUBMED_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
S2_BASE      = "https://api.semanticscholar.org/graph/v1"

_S2_FIELDS = "title,authors,year,abstract,citationCount,openAccessPdf,externalIds,venue"

def _ncbi_get(url: str, params: dict) -> requests.Response:
    """Rate-limited GET to NCBI E-utilities."""
    global _last_ncbi_request
    elapsed = time.monotonic() - _last_ncbi_request
    if elapsed < _NCBI_RATE_LIMIT_SECS:
        time.sleep(_NCBI_RATE_LIMIT_SECS - elapsed)
    _last_ncbi_request = time.monotonic()
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp

def load_rag_assets(
    chroma_db_dir,
    embedding_model_name: str,
    chroma_guidelines_dir=None,
    chroma_notes_dir=None,
) -> None:
    """
    Initialize ChromaDB persistent clients and sentence-transformer embedding model.
    Called by server.py at startup.

    Args:
        chroma_db_dir:         Path to 'clinical_literature' ChromaDB directory.
        embedding_model_name:  Sentence-transformer model name.
        chroma_guidelines_dir: Path to 'clinical_guidelines' ChromaDB directory (chroma_notes_dir:      Path to 'mimic_notes' ChromaDB directory ("""
    global _chroma_collection, _chroma_guidelines, _chroma_notes, _embedding_model
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_db_dir))
        _chroma_collection = client.get_or_create_collection(
            name="clinical_literature",
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[literature] ChromaDB 'clinical_literature' loaded ({_chroma_collection.count()} docs)")
    except Exception as e:
        print(f"[literature] WARNING: ChromaDB 'clinical_literature' init failed: {e}")

    if chroma_guidelines_dir is not None:
        try:
            import chromadb
            g_client = chromadb.PersistentClient(path=str(chroma_guidelines_dir))
            _chroma_guidelines = g_client.get_or_create_collection(
                name="clinical_guidelines",
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[literature] ChromaDB 'clinical_guidelines' loaded ({_chroma_guidelines.count()} chunks)")
        except Exception as e:
            print(f"[literature] WARNING: ChromaDB 'clinical_guidelines' init failed: {e}")

    if chroma_notes_dir is not None:
        try:
            import chromadb
            n_client = chromadb.PersistentClient(path=str(chroma_notes_dir))
            _chroma_notes = n_client.get_or_create_collection(
                name="mimic_notes",
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[literature] ChromaDB 'mimic_notes' loaded ({_chroma_notes.count()} chunks)")
        except Exception as e:
            print(f"[literature] WARNING: ChromaDB 'mimic_notes' init failed: {e}")

    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(embedding_model_name)
        print(f"[literature] Embedding model '{embedding_model_name}' loaded")
    except Exception as e:
        print(f"[literature] WARNING: Embedding model load failed: {e}")

    if _chroma_collection is not None and _chroma_collection.count() > 0:
        _build_bm25_index(_chroma_collection)

def _build_bm25_index(collection) -> None:
    """
    Build a BM25Okapi index over all documents in *collection* at startup.
    Stores results in module-level globals _bm25_index and _bm25_doc_ids.
    Safe to call multiple times (rebuilds from scratch).
    """
    global _bm25_index, _bm25_doc_ids
    try:
        from rank_bm25 import BM25Okapi
        result = collection.get(include=["documents"])
        docs   = result.get("documents", []) or []
        ids    = result.get("ids", []) or []
        if not docs:
            print("[literature] BM25: no documents found   skipping index build")
            return
        _bm25_doc_ids = ids
        tokenized     = [doc.lower().split() for doc in docs]
        _bm25_index   = BM25Okapi(tokenized)
        print(f"[literature] BM25 index built over {len(docs)} documents")
    except Exception as e:
        print(f"[literature] WARNING: BM25 index build failed: {e}")

def _rrf_merge(
    dense_ids:   list,
    dense_scores: list,
    bm25_ids:    list,
    bm25_scores: list,
    k: int = 60,
) -> list:
    """
    Reciprocal Rank Fusion (RRF) combining dense and BM25 ranked lists.

    score(doc) = 1 / (k + rank_dense + 1) + 1 / (k + rank_bm25 + 1)

    Returns a list of (doc_id, rrf_score) sorted by descending RRF score.
    """
    from collections import defaultdict
    rrf: dict = defaultdict(float)
    for rank, doc_id in enumerate(dense_ids):
        rrf[doc_id] += 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        rrf[doc_id] += 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANK_POOL = 15

def load_reranker() -> None:
    """
    Load the cross-encoder re-ranker at server startup.
    Called from server.py after load_rag_assets().
    Downloads ~90 MB on first run; cached to ~/.cache/huggingface.
    """
    global _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        print(f"[literature] Cross-encoder re-ranker loaded ({CROSS_ENCODER_MODEL})")
    except Exception as e:
        print(f"[literature] WARNING: Cross-encoder load failed (re-ranking disabled): {e}")

def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Re-rank *candidates* using the cross-encoder.

    Each candidate must contain either an 'abstract' key (papers) or a 'text'
    key (guidelines).  The cross-encoder scores (query, passage) pairs; higher
    is better (unlike cosine distance).

    The cross-encoder score is stored in each dict as 'rerank_score' for
    transparency, and the list is sorted by that score before slicing to top_k.
    Falls back to returning candidates[:top_k] if _reranker is None.
    """
    if _reranker is None or not candidates:
        return candidates[:top_k]

    pairs = [
        (query, c.get("abstract", c.get("text", ""))[:1000])
        for c in candidates
    ]
    try:
        scores = _reranker.predict(pairs)
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = round(float(score), 4)
        ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return ranked[:top_k]
    except Exception as e:
        print(f"[literature] WARNING: re-ranking failed ({e}); returning pre-ranked results")
        return candidates[:top_k]

def register(mcp) -> None:
    """Attach all literature tools to the FastMCP instance."""

    @mcp.tool()
    def search_pubmed(query: str, max_results: int = 5) -> list:
        """
        Search PubMed using NCBI E-utilities and return article abstracts.

        Uses esearch.fcgi to retrieve PMIDs, then efetch.fcgi to fetch abstracts.
        Rate-limited to <3 requests/second (no API key required).

        Args:
            query: PubMed search query string.
            max_results: Maximum number of abstracts to return (default 5).

        Returns: list of {pmid, title, authors, journal, year, abstract, url}
        """
        try:
            max_results = min(int(max_results), 20)

            search_resp = _ncbi_get(
                f"{PUBMED_BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "xml",
                    "usehistory": "y",
                },
            )
            search_root = ET.fromstring(search_resp.text)
            pmids = [id_el.text for id_el in search_root.findall(".//IdList/Id")]

            if not pmids:
                return []

            fetch_resp = _ncbi_get(
                f"{PUBMED_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "rettype": "abstract",
                    "retmode": "xml",
                },
            )
            fetch_root = ET.fromstring(fetch_resp.text)

            results = []
            for article in fetch_root.findall(".//PubmedArticle"):
                try:
                    pmid_el = article.find(".//PMID")
                    pmid = pmid_el.text if pmid_el is not None else ""

                    title_el = article.find(".//ArticleTitle")
                    title = "".join(title_el.itertext()) if title_el is not None else ""

                    authors = []
                    for author in article.findall(".//AuthorList/Author")[:5]:
                        last = author.findtext("LastName", "")
                        fore = author.findtext("ForeName", "")
                        initials = author.findtext("Initials", "")
                        name = f"{last} {initials or fore}".strip()
                        if name.strip():
                            authors.append(name)
                    if len(article.findall(".//AuthorList/Author")) > 5:
                        authors.append("et al.")

                    journal_el = article.find(".//Journal/Title")
                    journal = journal_el.text if journal_el is not None else ""

                    year_el = article.find(".//PubDate/Year")
                    year = int(year_el.text) if year_el is not None else 0

                    abstract_parts = []
                    for ab in article.findall(".//AbstractText"):
                        label = ab.get("Label", "")
                        text = "".join(ab.itertext())
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                    abstract = " ".join(abstract_parts).strip()

                    results.append({
                        "pmid": pmid,
                        "title": title,
                        "authors": ", ".join(authors),
                        "journal": journal,
                        "year": year,
                        "abstract": abstract[:2000] if abstract else "",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    })
                except Exception:
                    continue

            return results

        except requests.RequestException as e:
            return [{"error": f"PubMed request failed: {e}", "query": query}]
        except Exception as e:
            return [{"error": str(e), "query": query}]

    @mcp.tool()
    def search_semantic_scholar(query: str, max_results: int = 5) -> list:
        """
        Search Semantic Scholar for ML/AI-relevant clinical papers.

        Uses the free Semantic Scholar Graph API. Returns papers with citation
        counts and open-access PDF links when available.

        Args:
            query: Search query string.
            max_results: Maximum papers to return (default 5).

        Returns: list of {paper_id, title, authors, year, abstract,
                          citations, open_access_pdf, url}
        """
        try:
            max_results = min(int(max_results), 20)

            resp = requests.get(
                f"{S2_BASE}/paper/search",
                params={
                    "query": query,
                    "fields": _S2_FIELDS,
                    "limit": max_results,
                },
                headers={"User-Agent": "ClinicalAIAgent/1.0"},
                timeout=15,
            )

            if resp.status_code == 429:
                time.sleep(2)
                resp = requests.get(
                    f"{S2_BASE}/paper/search",
                    params={
                        "query": query,
                        "fields": _S2_FIELDS,
                        "limit": max_results,
                    },
                    headers={"User-Agent": "ClinicalAIAgent/1.0"},
                    timeout=15,
                )

            resp.raise_for_status()
            data = resp.json()

            results = []
            for paper in data.get("data", []):
                authors = [a.get("name", "") for a in paper.get("authors", [])[:5]]
                if len(paper.get("authors", [])) > 5:
                    authors.append("et al.")

                oa_pdf = None
                if paper.get("openAccessPdf"):
                    oa_pdf = paper["openAccessPdf"].get("url")

                paper_id = paper.get("paperId", "")
                url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""

                results.append({
                    "paper_id": paper_id,
                    "title": paper.get("title", ""),
                    "authors": ", ".join(authors),
                    "year": paper.get("year"),
                    "abstract": (paper.get("abstract") or "")[:2000],
                    "citations": paper.get("citationCount", 0),
                    "open_access_pdf": oa_pdf,
                    "url": url,
                })

            return results

        except requests.RequestException as e:
            return [{"error": f"Semantic Scholar request failed: {e}", "query": query}]
        except Exception as e:
            return [{"error": str(e), "query": query}]

    @mcp.tool()
    def search_clinical_literature_rag(
        query: str,
        top_k: int = 5,
        min_year: int = 2010,
        topic_filter: str = None,
    ) -> list:
        """
        Semantic search over the local ChromaDB vector store of seeded clinical papers.

        Encodes the query with the loaded embedding model, then retrieves the top-k
        nearest documents by cosine similarity.

        Args:
            query: Natural language clinical question.
            top_k: Number of results to return (default 5).
            min_year: Only return papers published >= this year (default 2010).
            topic_filter: Optional substring to filter results by title or abstract
                          (e.g. 'sepsis', 'cardiology'). Applied after retrieval.

        Returns: list of {
            title, abstract, pmid, authors, year,
            relevance_score (0-1, higher=better),
            citation (pre-formatted author/year/title/PMID string),
            url
        }
        A leading 'warning' dict is prepended when no result exceeds
        relevance_score 0.35   the LLM should then call search_pubmed instead.
        """
        if _chroma_collection is None:
            return [{"error": "ChromaDB not initialized. Check server startup logs."}]
        if _chroma_collection.count() == 0:
            return [{"error": "ChromaDB collection is empty. Add papers with add_paper_to_rag first."}]
        if _embedding_model is None:
            return [{"error": "Embedding model not loaded. Check server startup logs."}]

        try:
            top_k = min(int(top_k), 20)

            candidate_n = min(top_k * 3, _chroma_collection.count())

            query_embedding = _embedding_model.encode(query).tolist()
            dense_results = _chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_n,
                include=["metadatas", "documents", "distances"],
            )
            dense_ids    = dense_results["ids"][0]
            dense_scores = [1.0 - d for d in dense_results["distances"][0]]

            bm25_ids: list    = []
            bm25_scores: list = []
            if _bm25_index is not None:
                query_tokens = query.lower().split()
                raw_scores   = _bm25_index.get_scores(query_tokens)
                import numpy as _np
                top_indices  = _np.argsort(raw_scores)[::-1][:candidate_n].tolist()
                bm25_ids     = [_bm25_doc_ids[i] for i in top_indices]
                bm25_scores  = [float(raw_scores[i]) for i in top_indices]

            if bm25_ids:
                merged = _rrf_merge(dense_ids, dense_scores, bm25_ids, bm25_scores)
                merged_ids = [doc_id for doc_id, _ in merged]
            else:
                merged_ids = dense_ids

            fetched = _chroma_collection.get(
                ids=merged_ids,
                include=["metadatas", "documents"],
            )
            meta_by_id = {
                fid: (fetched["metadatas"][j], fetched["documents"][j])
                for j, fid in enumerate(fetched["ids"])
            }

            rerank_candidates = []
            output = []
            for doc_id in merged_ids:
                if doc_id not in meta_by_id:
                    continue
                meta, abstract_text = meta_by_id[doc_id]
                year = int(meta.get("year", 0) or 0)

                if year < min_year:
                    continue

                if topic_filter:
                    haystack = (meta.get("title", "") + " " + abstract_text).lower()
                    if topic_filter.lower() not in haystack:
                        continue

                if doc_id in dense_ids:
                    dense_rank = dense_ids.index(doc_id)
                    relevance  = round(dense_scores[dense_rank], 3)
                else:
                    relevance = 0.5

                pmid    = meta.get("pmid", "")
                authors = meta.get("authors", "")
                title   = meta.get("title", doc_id)

                rerank_candidates.append({
                    "title":           title,
                    "abstract":        abstract_text[:1500],
                    "pmid":            pmid,
                    "authors":         authors,
                    "year":            year,
                    "relevance_score": relevance,
                    "citation":        f"{authors} ({year}) {title}. PMID:{pmid}".strip(". "),
                    "url":             f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                    "retrieval_mode":  "hybrid" if bm25_ids else "dense",
                })

                if len(rerank_candidates) >= _RERANK_POOL:
                    break

            output = _rerank(query, rerank_candidates, top_k)

            import math as _math
            for item in output:
                if item.get("relevance_score") == 0.5 and "rerank_score" in item:
                    logit = item["rerank_score"]
                    item["relevance_score"] = round(1.0 / (1.0 + _math.exp(-logit)), 3)

            LOW_RELEVANCE_THRESHOLD = 0.65
            if not output or output[0].get("relevance_score", 0) < LOW_RELEVANCE_THRESHOLD:
                best = output[0].get("relevance_score", 0) if output else 0.0
                output.insert(0, {
                    "warning": (
                        f"No highly relevant documents found in local RAG corpus "
                        f"(best relevance: {best:.2f}, threshold: {LOW_RELEVANCE_THRESHOLD}). "
                        "Consider calling search_pubmed for live PubMed retrieval."
                    ),
                    "best_available_relevance": best,
                })

            return output

        except Exception as e:
            return [{"error": str(e)}]

    @mcp.tool()
    def search_clinical_guidelines_rag(query: str, top_k: int = 5) -> list:
        """
        Semantic search over indexed clinical practice guidelines.

        Searches the local ChromaDB 'clinical_guidelines' collection, which
        contains section-level chunks from:
          - Surviving Sepsis Campaign Guidelines 2021
          - KDIGO AKI Guidelines 2012
          - KDIGO CKD Guidelines 2024
          - ARDSNet Low Tidal Volume Ventilation Protocol
          - AHA/ACC/HFSA Heart Failure Guideline 2022
          - AHA/ACC/HRS Atrial Fibrillation Guideline 2023
          - ASHP/IDSA Vancomycin AUC-Guided Dosing Guidelines 2020
          - IDSA/ATS Community-Acquired Pneumonia Guidelines 2019
          - NICE NG253 Sepsis Guidelines 2024
          - WHO Sepsis Management 2017

        Use this for questions about RECOMMENDED TREATMENTS, DOSING THRESHOLDS,
        DIAGNOSTIC CRITERIA, and CLINICAL PROTOCOLS   NOT for research evidence.

        The distinction matters: guidelines are authoritative consensus
        recommendations; use search_clinical_literature_rag for research papers.

        Args:
            query: Natural language clinical question.
            top_k: Number of guideline chunks to return (default 5).

        Returns: list of {
            text (the actual guideline section text),
            source (guideline name),
            tags, url,
            relevance_score (0-1, higher=better),
        }
        A leading 'warning' dict is prepended when no result exceeds
        relevance_score 0.60.
        """
        if _chroma_guidelines is None:
            return [{
                "error": (
                    "Clinical guidelines collection not initialized. "
                    "Run scripts/seed_guidelines.py first, then restart the server."
                )
            }]
        if _chroma_guidelines.count() == 0:
            return [{"error": "Clinical guidelines collection is empty. Run scripts/seed_guidelines.py."}]
        if _embedding_model is None:
            return [{"error": "Embedding model not loaded. Check server startup logs."}]

        try:
            top_k = min(int(top_k), 20)
            pool_n = min(max(top_k * 2, _RERANK_POOL), _chroma_guidelines.count())
            query_embedding = _embedding_model.encode(query).tolist()
            results = _chroma_guidelines.query(
                query_embeddings=[query_embedding],
                n_results=pool_n,
                include=["metadatas", "documents", "distances"],
            )

            candidates = []
            for i in range(len(results["ids"][0])):
                meta      = results["metadatas"][0][i] if results.get("metadatas") else {}
                raw_dist  = results["distances"][0][i] if results.get("distances") else 1.0
                relevance = round(max(0.0, 1.0 - raw_dist), 3)
                text      = results["documents"][0][i] if results.get("documents") else ""

                candidates.append({
                    "text":            text[:2000],
                    "source":          meta.get("source", ""),
                    "tags":            meta.get("tags", ""),
                    "url":             meta.get("url", ""),
                    "relevance_score": relevance,
                    "chunk_index":     meta.get("chunk_index", -1),
                })

            output = _rerank(query, candidates, top_k)

            LOW_RELEVANCE_THRESHOLD = 0.60
            if not output or output[0].get("relevance_score", 0) < LOW_RELEVANCE_THRESHOLD:
                best = output[0].get("relevance_score", 0) if output else 0.0
                output.insert(0, {
                    "warning": (
                        f"No highly relevant guideline chunks found "
                        f"(best relevance: {best:.2f}, threshold: {LOW_RELEVANCE_THRESHOLD}). "
                        "Consider calling search_clinical_literature_rag or search_pubmed."
                    ),
                    "best_available_relevance": best,
                })

            return output

        except Exception as e:
            return [{"error": str(e)}]

    @mcp.tool()
    def add_paper_to_rag(
        title: str,
        abstract: str,
        pmid: str = "",
        authors: str = "",
        year: int = 0,
    ) -> dict:
        """
        Embed and add a paper to the local ChromaDB vector store.

        Args:
            title: Paper title.
            abstract: Paper abstract text.
            pmid: PubMed ID (optional).
            authors: Author list string (optional).
            year: Publication year (optional).

        Returns: {status, doc_id, collection_size}
        """
        if _chroma_collection is None:
            return {"error": "ChromaDB not initialized."}
        if _embedding_model is None:
            return {"error": "Embedding model not loaded."}

        try:
            import hashlib
            doc_id = pmid if pmid else hashlib.md5(title.encode()).hexdigest()[:12]

            text_to_embed = f"{title}. {abstract}"
            embedding = _embedding_model.encode(text_to_embed).tolist()

            _chroma_collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[abstract],
                metadatas=[{
                    "title": title,
                    "pmid": pmid,
                    "authors": authors,
                    "year": int(year),
                }],
            )

            return {
                "status": "ok",
                "doc_id": doc_id,
                "collection_size": _chroma_collection.count(),
            }

        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def find_similar_cases_rag(
        patient_id: int,
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> list:
        """
        Find historical MIMIC-III ICU patients with clinically similar
        presentations using semantic search over de-identified discharge
        summaries.

        If query_text is not provided, the tool automatically fetches the most
        recent discharge summary for patient_id from MIMIC and uses it as the
        query.  This enables case-based reasoning:
          "Show me prior patients with a presentation similar to patient 27513."

        The MIMIC notes collection was seeded with 5,000 de-identified
        discharge summaries stratified by acuity cluster
        (2,500 cluster-2 high-acuity, 1,500 cluster-0, 1,000 cluster-1).

        Args:
            patient_id: MIMIC subject_id of the index patient.
            query_text: Optional free-text query to use instead of fetching the
                        patient's own discharge note.  Useful when you already
                        have a clinical description or want to search by
                        diagnosis / symptom phrase.
            top_k:      Number of similar cases to return (default 5).

        Returns: list of {
            subject_id,
            relevance_score  (0 1, higher = more similar),
            cluster          (0=moderate, 1=low, 2=high-acuity),
            mortality_label  (0=survived, 1=died, -1=unknown),
            diagnoses_snippet,
            note_snippet     (first 500 chars of the matching note chunk),
        }
        A leading 'warning' dict is prepended when the best result < 0.55.
        """
        if _chroma_notes is None:
            return [{
                "error": (
                    "MIMIC notes collection not initialized. "
                    "Run scripts/seed_mimic_notes.py first, then restart the server."
                )
            }]
        if _chroma_notes.count() == 0:
            return [{"error": "MIMIC notes collection is empty. Run scripts/seed_mimic_notes.py."}]
        if _embedding_model is None:
            return [{"error": "Embedding model not loaded. Check server startup logs."}]

        try:
            if query_text and query_text.strip():
                search_text = query_text.strip()[:8_000]
            else:
                if _db_con is None:
                    return [{
                        "error": (
                            "DuckDB connection not available and no query_text provided. "
                            "Provide query_text or ensure the server has a MIMIC connection."
                        )
                    }]
                try:
                    df = _db_con.execute(
                        """
                        SELECT TEXT FROM noteevents
                        WHERE SUBJECT_ID = ?
                          AND CATEGORY = 'Discharge summary'
                          AND ISERROR IS NULL
                        ORDER BY CHARTDATE DESC
                        LIMIT 1
                        """,
                        [patient_id],
                    ).df()
                    if df.empty or df.iloc[0]["TEXT"] is None:
                        return [{
                            "error": (
                                f"No discharge summary found for patient {patient_id} in MIMIC. "
                                "Provide query_text manually."
                            )
                        }]
                    search_text = str(df.iloc[0]["TEXT"])[:3_000]
                except Exception as e:
                    return [{"error": f"Failed to fetch discharge note: {e}"}]

            top_k = min(int(top_k), 20)
            query_embedding = _embedding_model.encode(search_text).tolist()
            results = _chroma_notes.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 3, _chroma_notes.count()),
                include=["metadatas", "documents", "distances"],
            )

            seen_subjects: dict[int, dict] = {}

            for i in range(len(results["ids"][0])):
                meta      = results["metadatas"][0][i] if results.get("metadatas") else {}
                raw_dist  = results["distances"][0][i] if results.get("distances") else 1.0
                relevance = round(max(0.0, 1.0 - raw_dist), 3)
                text      = results["documents"][0][i] if results.get("documents") else ""

                sid = int(meta.get("subject_id", -1))

                if sid == patient_id:
                    continue

                if sid not in seen_subjects or relevance > seen_subjects[sid]["relevance_score"]:
                    seen_subjects[sid] = {
                        "subject_id":        sid,
                        "relevance_score":   relevance,
                        "cluster":           int(meta.get("cluster", -1)),
                        "mortality_label":   int(meta.get("mortality_label", -1)),
                        "diagnoses_snippet": meta.get("diagnoses_snippet", ""),
                        "note_snippet":      text[:500],
                    }

            output = sorted(seen_subjects.values(), key=lambda x: x["relevance_score"], reverse=True)[:top_k]

            LOW_RELEVANCE_THRESHOLD = 0.55
            if not output or output[0].get("relevance_score", 0) < LOW_RELEVANCE_THRESHOLD:
                best = output[0].get("relevance_score", 0) if output else 0.0
                output.insert(0, {
                    "warning": (
                        f"Low similarity to existing indexed cases "
                        f"(best relevance: {best:.2f}, threshold: {LOW_RELEVANCE_THRESHOLD}). "
                        "Results may not represent truly similar presentations."
                    ),
                    "best_available_relevance": best,
                })

            return output

        except Exception as e:
            return [{"error": str(e)}]
