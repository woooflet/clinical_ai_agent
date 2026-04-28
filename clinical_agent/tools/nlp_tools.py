
import re
from pathlib import Path
from typing import Optional

_nlp = None
_analyzer = None
_anonymizer = None

_db_con = None

def set_db_connection(con) -> None:
    """Inject the shared DuckDB connection so analyze_patient_notes can query MIMIC."""
    global _db_con
    _db_con = con

_LAB_PATTERN = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z0-9\-/]*(?:\s+[A-Za-z][A-Za-z0-9\-/]*){0,2})"
    r"[\s:]+(?P<value>[0-9]+(?:\.[0-9]+)?)"
    r"(?:\s*(?P<unit>[A-Za-z/%µμ][A-Za-z0-9/µμ%\.]*))?"
)

_VITAL_PATTERN = re.compile(
    r"\b(?P<name>BP|Blood Pressure|HR|Heart Rate|SpO2|O2 Sat|Temp(?:erature)?|RR|Resp(?:iratory)? Rate|MAP|GCS|SBP|DBP)"
    r"[\s:]+(?P<value>[0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)"
    r"(?:\s*(?P<unit>[A-Za-z/%°µ][A-Za-z0-9/°%\.]*))?"
    , re.IGNORECASE
)

_LAB_TABLE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9 \-/]{2,25})\s{2,}(?P<value>[0-9]+(?:\.[0-9]+)?)",
    re.MULTILINE
)

_KNOWN_LABS = {
    "WBC", "RBC", "Hgb", "Hct", "MCV", "PLT", "Platelets",
    "Na", "Sodium", "K", "Potassium", "Cl", "Chloride",
    "CO2", "BUN", "Creatinine", "Glucose", "Ca", "Calcium",
    "Mg", "Magnesium", "Phos", "Phosphate", "Albumin", "Bilirubin",
    "ALT", "AST", "ALP", "GGT", "Total Bilirubin", "Direct Bilirubin",
    "INR", "PT", "PTT", "aPTT", "D-dimer", "Fibrinogen",
    "Troponin", "BNP", "proBNP", "NT-proBNP", "CRP", "ESR", "Procalcitonin",
    "Lactate", "pH", "PaO2", "PaCO2", "HCO3", "FiO2", "SaO2",
    "TSH", "T4", "T3", "Free T4", "Cortisol", "HbA1c", "A1c",
    "eGFR", "GFR", "LDH", "CK", "CKMB", "Amylase", "Lipase",
    "Uric Acid", "Iron", "Ferritin", "TIBC", "Transferrin",
    "Total Protein", "Total Cholesterol", "LDL", "HDL", "Triglycerides",
    "WBC count", "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils",
}
_KNOWN_LABS_LOWER = {lab.lower() for lab in _KNOWN_LABS}

def _patch_scispacy_config(model_name: str) -> None:
    """
    Patch en_ner_bc5cdr_md's config.cfg for spaCy 3.8 compatibility.

    The model was trained on spaCy 3.7 and stores include_static_vectors as the
    string 'True' rather than the boolean true, which causes a ConfigValidationError
    under spaCy 3.8's stricter type checking.  We fix the config file in-place.
    """
    import importlib, pathlib, re
    try:
        pkg = importlib.import_module(model_name)
        pkg_dir = pathlib.Path(pkg.__file__).parent
        candidates = list(pkg_dir.glob("**/config.cfg"))
        if not candidates:
            return
        cfg_path = candidates[0]
        text = cfg_path.read_text()
        patched = re.sub(
            r'(include_static_vectors\s*=\s*)"(True|False)"',
            lambda m: m.group(1) + m.group(2).lower(),
            text,
        )
        if patched != text:
            cfg_path.write_text(patched)
            print(f"[nlp_tools] Patched {cfg_path} for spaCy 3.8 compatibility")
    except Exception as e:
        print(f"[nlp_tools] Config patch skipped: {e}")

def load_nlp_models(model_name: str = "en_ner_bc5cdr_md") -> None:
    """Load scispaCy NER model and Presidio engines at server startup."""
    global _nlp, _analyzer, _anonymizer

    _patch_scispacy_config(model_name)
    try:
        import spacy
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nlp = spacy.load(model_name)
        print(f"[nlp_tools] scispaCy model '{model_name}' loaded")
    except Exception as e:
        print(f"[nlp_tools] WARNING: Could not load '{model_name}': {e}")
        print(
            "[nlp_tools] Install with: pip install --no-deps "
            "https://s3-us-west-2.amazonaws.com/"
            "ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
        )

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
        print("[nlp_tools] Presidio AnalyzerEngine + AnonymizerEngine loaded")
    except Exception as e:
        print(f"[nlp_tools] WARNING: Could not load Presidio: {e}")

def _run_ner(text: str) -> tuple[list[str], list[str]]:
    """
    Run scispaCy NER on text, return (diseases, chemicals).
    Falls back to empty lists if model is not loaded.
    """
    if _nlp is None:
        return [], []
    doc = _nlp(text)
    diseases: list[str] = []
    chemicals: list[str] = []
    seen_disease: set[str] = set()
    seen_chem: set[str] = set()
    for ent in doc.ents:
        label = ent.label_.upper()
        norm = ent.text.strip()
        if not norm:
            continue
        if label == "DISEASE":
            key = norm.lower()
            if key not in seen_disease:
                diseases.append(norm)
                seen_disease.add(key)
        elif label in ("CHEMICAL", "DRUG"):
            key = norm.lower()
            if key not in seen_chem:
                chemicals.append(norm)
                seen_chem.add(key)
    return diseases, chemicals

def _extract_vitals(text: str) -> list[dict]:
    """Extract structured vital sign measurements via regex."""
    results: list[dict] = []
    seen: set[str] = set()
    for m in _VITAL_PATTERN.finditer(text):
        name = m.group("name").strip()
        value = m.group("value").strip()
        unit = (m.group("unit") or "").strip()
        key = f"{name.lower()}:{value}"
        if key not in seen:
            results.append({"name": name, "value": value, "unit": unit})
            seen.add(key)
    return results

def _extract_labs(text: str) -> list[dict]:
    """
    Extract structured lab values via regex.
    Uses a known-lab-name whitelist to suppress spurious matches.
    Also handles MIMIC whitespace-padded table format (two-column layout).
    """
    results: list[dict] = []
    seen: set[str] = set()

    for m in _LAB_PATTERN.finditer(text):
        name = m.group("name").strip()
        if name.lower() not in _KNOWN_LABS_LOWER:
            continue
        value = m.group("value").strip()
        unit = (m.group("unit") or "").strip()
        key = f"{name.lower()}:{value}"
        if key not in seen:
            results.append({"name": name, "value": value, "unit": ""})
            seen.add(key)

    for m in _LAB_TABLE_PATTERN.finditer(text):
        name = m.group("name").strip()
        if name.lower() not in _KNOWN_LABS_LOWER:
            continue
        value = m.group("value").strip()
        key = f"{name.lower()}:{value}"
        if key not in seen:
            results.append({"name": name, "value": value, "unit": ""})
            seen.add(key)

    return results

def _build_entity_result(
    diseases: list[str],
    chemicals: list[str],
    lab_values: list[dict],
    vitals: list[dict],
    **extras,
) -> dict:
    """Package NER results into a consistent return dict."""
    out: dict = {
        "diseases": diseases,
        "chemicals": chemicals,
        "lab_values": lab_values,
        "vitals": vitals,
    }
    out.update(extras)
    return out

def register(mcp) -> None:
    """Attach all NLP tools to the FastMCP instance."""

    @mcp.tool()
    def extract_clinical_entities(text: str) -> dict:
        """
        Extract clinical entities (diseases, chemicals/drugs) from free text
        using the scispaCy en_ner_bc5cdr_md NER model, plus regex extraction
        of structured lab values and vital sign patterns.

        Args:
            text: Clinical note text or snippet.

        Returns: {
            diseases:   [str]                  disease / condition names,
            chemicals:  [str]                  drug / chemical names,
            lab_values: [{name, value, unit}]   structured lab results,
            vitals:     [{name, value, unit}]   structured vital signs,
        }
        """
        if not text or not text.strip():
            return _build_entity_result([], [], [], [])

        diseases, chemicals = _run_ner(text)
        lab_values = _extract_labs(text)
        vitals = _extract_vitals(text)

        return _build_entity_result(diseases, chemicals, lab_values, vitals)

    @mcp.tool()
    def ingest_clinical_pdf(file_path: str) -> dict:
        """
        Extract text from all pages of a clinical PDF and run NER entity extraction.

        Uses pdfplumber for text extraction, then calls extract_clinical_entities.

        Args:
            file_path: Absolute path to a PDF file on the server filesystem.

        Returns same structure as extract_clinical_entities, plus:
            page_count (int) and raw_text (str, truncated to 8000 chars).
        """
        path = Path(file_path)
        if not path.exists():
            return {
                "error": f"File not found: {file_path}",
                "diseases": [], "chemicals": [], "lab_values": [], "vitals": [],
                "page_count": 0, "raw_text": "",
            }
        if path.suffix.lower() != ".pdf":
            return {
                "error": f"Not a PDF file: {file_path}",
                "diseases": [], "chemicals": [], "lab_values": [], "vitals": [],
                "page_count": 0, "raw_text": "",
            }

        try:
            import pdfplumber
        except ImportError:
            return {
                "error": "pdfplumber not installed. Run: pip install pdfplumber",
                "diseases": [], "chemicals": [], "lab_values": [], "vitals": [],
                "page_count": 0, "raw_text": "",
            }

        try:
            pages_text: list[str] = []
            with pdfplumber.open(str(path)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages_text.append(t)
        except Exception as e:
            return {
                "error": f"Failed to open PDF: {e}",
                "diseases": [], "chemicals": [], "lab_values": [], "vitals": [],
                "page_count": 0, "raw_text": "",
            }

        raw_text = "\n\n".join(pages_text)
        ner_text = raw_text[:50_000]

        diseases, chemicals = _run_ner(ner_text)
        lab_values = _extract_labs(ner_text)
        vitals = _extract_vitals(ner_text)

        return _build_entity_result(
            diseases,
            chemicals,
            lab_values,
            vitals,
            page_count=page_count,
            raw_text=raw_text[:8_000],
        )

    @mcp.tool()
    def deidentify_note(text: str) -> dict:
        """
        Strip PHI from a clinical note using Microsoft Presidio.

        Detects and anonymizes: PERSON, DATE_TIME, PHONE_NUMBER, EMAIL_ADDRESS,
        US_SSN, US_DRIVER_LICENSE, MEDICAL_LICENSE, NRP (MRN-like patterns),
        LOCATION, ORGANIZATION, URL, IP_ADDRESS.

        Args:
            text: Raw clinical note text.

        Returns: {
            anonymized_text: str,
            entities_found:  [{type, start, end, score}],
            entity_count:    int,
        }
        """
        if not text or not text.strip():
            return {
                "anonymized_text": text,
                "entities_found": [],
                "entity_count": 0,
            }

        if _analyzer is None or _anonymizer is None:
            return {
                "anonymized_text": text,
                "entities_found": [],
                "entity_count": 0,
                "warning": "Presidio not loaded. PHI is NOT anonymized.",
            }

        entities_to_detect = [
            "PERSON",
            "DATE_TIME",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "US_SSN",
            "US_DRIVER_LICENSE",
            "MEDICAL_LICENSE",
            "NRP",
            "LOCATION",
            "ORGANIZATION",
            "URL",
            "IP_ADDRESS",
        ]

        try:
            analyzer_results = _analyzer.analyze(
                text=text,
                entities=entities_to_detect,
                language="en",
            )

            from presidio_anonymizer.entities import OperatorConfig
            operators = {
                entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
                for entity in entities_to_detect
            }

            anonymized = _anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators,
            )

            entities_found = [
                {
                    "type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": round(r.score, 3),
                }
                for r in sorted(analyzer_results, key=lambda x: x.start)
            ]

            return {
                "anonymized_text": anonymized.text,
                "entities_found": entities_found,
                "entity_count": len(entities_found),
            }

        except Exception as e:
            return {
                "anonymized_text": text,
                "entities_found": [],
                "entity_count": 0,
                "error": f"Presidio anonymization failed: {e}",
            }

    @mcp.tool()
    def analyze_patient_notes(
        patient_id: int,
        category: Optional[str] = None,
        deidentify: bool = True,
        max_notes: int = 3,
    ) -> dict:
        """
        Fetch clinical notes for a patient from MIMIC-III, optionally
        de-identify them with Presidio, then run scispaCy NER + regex
        extraction for labs and vitals.

        Args:
            patient_id: MIMIC subject_id.
            category:   Optional note category filter, e.g. 'Discharge summary',
                        'Nursing', 'Radiology'.  None = all categories.
            deidentify: If True (default), strip PHI with Presidio before NER
                        so patient names never appear in the entity output.
            max_notes:  Maximum number of notes to process (default 3).

        Returns: {
            notes_analyzed: int,
            diseases:       [str],
            chemicals:      [str],
            lab_values:     [{name, value, unit}],
            vitals:         [{name, value, unit}],
            notes_preview:  [{category, chartdate, text_snippet}]
        }
        """
        if _db_con is None:
            return {
                "error": "DuckDB connection not available. "
                         "Call nlp_tools.set_db_connection(con) at startup.",
                "notes_analyzed": 0,
                "diseases": [], "chemicals": [], "lab_values": [], "vitals": [],
                "notes_preview": [],
            }

        if category:
            notes_sql = """
                SELECT CHARTDATE, CATEGORY, DESCRIPTION, TEXT
                FROM noteevents
                WHERE SUBJECT_ID = ?
                  AND ISERROR IS NULL
                  AND LOWER(CATEGORY) = LOWER(?)
                ORDER BY CHARTDATE DESC
                LIMIT ?
            """
            notes_df = _db_con.execute(notes_sql, [patient_id, category, max_notes]).df()
        else:
            notes_sql = """
                SELECT CHARTDATE, CATEGORY, DESCRIPTION, TEXT
                FROM noteevents
                WHERE SUBJECT_ID = ?
                  AND ISERROR IS NULL
                ORDER BY CHARTDATE DESC
                LIMIT ?
            """
            notes_df = _db_con.execute(notes_sql, [patient_id, max_notes]).df()

        if notes_df.empty:
            return {
                "notes_analyzed": 0,
                "diseases": [], "chemicals": [], "lab_values": [], "vitals": [],
                "notes_preview": [],
                "message": f"No notes found for patient {patient_id}.",
            }

        notes_df["CHARTDATE"] = notes_df["CHARTDATE"].astype(str)

        all_diseases:   list[str] = []
        all_chemicals:  list[str] = []
        all_labs:       list[dict] = []
        all_vitals:     list[dict] = []
        notes_preview:  list[dict] = []

        seen_disease: set[str] = set()
        seen_chem:    set[str] = set()
        seen_lab:     set[str] = set()
        seen_vital:   set[str] = set()

        for _, note_row in notes_df.iterrows():
            raw_text = note_row["TEXT"] or ""
            ner_text = raw_text[:50_000]

            if deidentify and _analyzer is not None and _anonymizer is not None:
                try:
                    from presidio_anonymizer.entities import OperatorConfig
                    entities_to_detect = [
                        "PERSON", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS",
                        "US_SSN", "LOCATION", "ORGANIZATION", "URL", "IP_ADDRESS",
                    ]
                    ar = _analyzer.analyze(
                        text=ner_text, entities=entities_to_detect, language="en"
                    )
                    ops = {
                        e: OperatorConfig("replace", {"new_value": f"<{e}>"})
                        for e in entities_to_detect
                    }
                    ner_text = _anonymizer.anonymize(
                        text=ner_text, analyzer_results=ar, operators=ops
                    ).text
                except Exception:
                    pass

            diseases, chemicals = _run_ner(ner_text)
            labs   = _extract_labs(ner_text)
            vitals = _extract_vitals(ner_text)

            for d in diseases:
                key = d.lower()
                if key not in seen_disease:
                    all_diseases.append(d)
                    seen_disease.add(key)
            for c in chemicals:
                key = c.lower()
                if key not in seen_chem:
                    all_chemicals.append(c)
                    seen_chem.add(key)
            for lab in labs:
                key = f"{lab['name'].lower()}:{lab['value']}"
                if key not in seen_lab:
                    all_labs.append(lab)
                    seen_lab.add(key)
            for vit in vitals:
                key = f"{vit['name'].lower()}:{vit['value']}"
                if key not in seen_vital:
                    all_vitals.append(vit)
                    seen_vital.add(key)

            snippet = (raw_text[:500] + "…") if len(raw_text) > 500 else raw_text
            notes_preview.append({
                "category":     note_row["CATEGORY"],
                "chartdate":    note_row["CHARTDATE"],
                "text_snippet": snippet,
            })

        return {
            "notes_analyzed": len(notes_df),
            "diseases":       all_diseases,
            "chemicals":      all_chemicals,
            "lab_values":     all_labs,
            "vitals":         all_vitals,
            "notes_preview":  notes_preview,
        }
