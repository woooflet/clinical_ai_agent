import os
import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT   = _PACKAGE_DIR.parent
_DRUGBANK_DB = Path(
    os.environ.get(
        "DRUGBANK_DB",
        str(_REPO_ROOT / "drugbank-mcp-server" / "data" / "drugbank.db"),
    )
)

_HIGH_SEVERITY_PATTERNS = re.compile(
    r"\b(death|fatal|life.?threaten|cardiac arrest|severe|serious|"
    r"significantly (increase|decrease)|contraindicated|QT prolongat|"
    r"torsades|agranulocytosis|serotonin syndrome|neuroleptic malignant|"
    r"rhabdomyolysis|hemorrhag|respiratory depression|anaphylax)",
    re.IGNORECASE,
)
_MODERATE_SEVERITY_PATTERNS = re.compile(
    r"\b(increas|decreas|risk|caution|monitor|may enhance|may reduc|"
    r"elevat|hypotension|hypertension|bradycardia|tachycardia|"
    r"nephrotoxic|hepatotoxic|ototoxic|neurotoxic)",
    re.IGNORECASE,
)

def _infer_severity(description: str) -> str:
    """Infer severity level from interaction description text."""
    if _HIGH_SEVERITY_PATTERNS.search(description):
        return "high"
    if _MODERATE_SEVERITY_PATTERNS.search(description):
        return "moderate"
    return "minor"

def load_drugbank(csv_path: Path) -> None:
    """
    Load the DrugBank SQLite database at server startup.

    Ignores the legacy csv_path argument (kept for API compatibility with server.py)
    and instead opens the DrugBank SQLite directly.
    """
    global _db_con
    if _DRUGBANK_DB.exists():
        _db_con = sqlite3.connect(str(_DRUGBANK_DB), check_same_thread=False)
        _db_con.row_factory = sqlite3.Row
        cur = _db_con.execute("SELECT COUNT(*) FROM drugs")
        count = cur.fetchone()[0]
        print(f"[drug_tools] DrugBank SQLite loaded: {count:,} drugs")
    else:
        print(f"[drug_tools] WARNING: DrugBank SQLite not found at {_DRUGBANK_DB}")
        print("[drug_tools] Run: cd drugbank-mcp-server && "
              "npm install --ignore-scripts && node scripts/download-db.js")
        if csv_path.exists():
            print(f"[drug_tools] Falling back to legacy CSV at {csv_path}")

def _lookup_drug(name: str) -> Optional[dict]:
    """
    Look up a drug by name, synonym, or brand name.
    Returns {drugbank_id, canonical_name, drug_interactions (JSON list)}.
    """
    if _db_con is None:
        return None

    name_clean = name.strip().lower()

    cur = _db_con.execute(
        "SELECT drugbank_id, name, drug_interactions, synonyms FROM drugs "
        "WHERE LOWER(name) = ? LIMIT 1",
        (name_clean,),
    )
    row = cur.fetchone()
    if row:
        return dict(row)

    try:
        cur = _db_con.execute(
            "SELECT drugs.drugbank_id, drugs.name, drugs.drug_interactions, drugs.synonyms "
            "FROM drugs_fts "
            "JOIN drugs ON drugs_fts.drugbank_id = drugs.drugbank_id "
            "WHERE drugs_fts MATCH ? "
            "LIMIT 3",
            (name_clean,),
        )
        rows = cur.fetchall()
        for row in rows:
            if row["name"].lower().startswith(name_clean):
                return dict(row)
        if rows:
            return dict(rows[0])
    except sqlite3.OperationalError:
        pass

    cur = _db_con.execute(
        "SELECT drugbank_id, name, drug_interactions, synonyms FROM drugs "
        "WHERE LOWER(name) LIKE ? LIMIT 1",
        (f"%{name_clean}%",),
    )
    row = cur.fetchone()
    if row:
        return dict(row)

    cur = _db_con.execute(
        "SELECT drugbank_id, name, drug_interactions, synonyms FROM drugs "
        "WHERE LOWER(synonyms) LIKE ? LIMIT 1",
        (f"%{name_clean}%",),
    )
    row = cur.fetchone()
    if row:
        return dict(row)

    return None

def register(mcp) -> None:
    """Attach all drug tools to the FastMCP instance."""

    @mcp.tool()
    def check_drug_interactions(drug_list: list[str]) -> dict:
        """
        Check for known drug-drug interactions between a list of medications.

        Uses the DrugBank open-data SQLite (~19,842 drugs, interactions stored
        as JSON arrays). Severity is inferred from description keywords:
        high / moderate / minor.

        Args:
            drug_list: List of drug names, e.g. ['metoprolol', 'amiodarone', 'warfarin'].

        Returns: {
            drugs_checked: [{name, canonical_name, found}],
            interactions: [{drug_a, drug_b, severity, description}],
            total_interactions: int,
            high_severity_count: int,
            moderate_severity_count: int,
            minor_severity_count: int,
            not_found: [str],
        }
        """
        if _db_con is None:
            return {
                "error": "DrugBank SQLite not loaded.",
                "setup": (
                    "cd drugbank-mcp-server && "
                    "npm install --ignore-scripts && node scripts/download-db.js"
                ),
                "drug_list": drug_list,
            }

        if not drug_list or len(drug_list) < 2:
            return {
                "error": "Provide at least 2 drug names to check for interactions.",
                "drug_list": drug_list,
            }

        drug_records = {}
        not_found = []
        drugs_checked = []

        for name in drug_list:
            record = _lookup_drug(name)
            if record:
                drug_records[name] = record
                drugs_checked.append({
                    "input_name": name,
                    "canonical_name": record["name"],
                    "drugbank_id": record["drugbank_id"],
                    "found": True,
                })
            else:
                not_found.append(name)
                drugs_checked.append({
                    "input_name": name,
                    "canonical_name": None,
                    "drugbank_id": None,
                    "found": False,
                })

        if len(drug_records) < 2:
            return {
                "drugs_checked": drugs_checked,
                "not_found": not_found,
                "interactions": [],
                "total_interactions": 0,
                "high_severity_count": 0,
                "moderate_severity_count": 0,
                "minor_severity_count": 0,
                "message": "Not enough recognized drugs to check interactions.",
            }

        found_pairs: set[tuple[str, str]] = set()
        interactions = []

        drug_names = list(drug_records.keys())
        for i in range(len(drug_names)):
            for j in range(i + 1, len(drug_names)):
                drug_a_name = drug_names[i]
                drug_b_name = drug_names[j]
                record_a = drug_records[drug_a_name]
                canonical_a = record_a["name"]
                canonical_b = drug_records[drug_b_name]["name"]

                ix_list = json.loads(record_a["drug_interactions"] or "[]")
                for ix in ix_list:
                    ix_drug_name = ix.get("name", "")
                    if canonical_b.lower() in ix_drug_name.lower() or \
                       ix_drug_name.lower() in canonical_b.lower():
                        pair_key = tuple(sorted([canonical_a, ix_drug_name]))
                        if pair_key not in found_pairs:
                            found_pairs.add(pair_key)
                            desc = ix.get("description", "")
                            severity = _infer_severity(desc)
                            interactions.append({
                                "drug_a": canonical_a,
                                "drug_b": ix_drug_name,
                                "severity": severity,
                                "description": desc,
                            })

        severity_order = {"high": 0, "moderate": 1, "minor": 2}
        interactions.sort(key=lambda x: severity_order.get(x["severity"], 3))

        high_count = sum(1 for ix in interactions if ix["severity"] == "high")
        moderate_count = sum(1 for ix in interactions if ix["severity"] == "moderate")
        minor_count = sum(1 for ix in interactions if ix["severity"] == "minor")

        return {
            "drugs_checked": drugs_checked,
            "interactions": interactions,
            "total_interactions": len(interactions),
            "high_severity_count": high_count,
            "moderate_severity_count": moderate_count,
            "minor_severity_count": minor_count,
            "not_found": not_found,
        }

    @mcp.tool()
    def get_drug_info(drug_name: str) -> dict:
        """
        Retrieve detailed information about a drug from the DrugBank database.

        Returns pharmacological data including mechanism of action, indications,
        half-life, and drug categories.

        Args:
            drug_name: Drug name (generic or brand).

        Returns: {name, drugbank_id, indication, mechanism_of_action, half_life,
                  categories, groups, pharmacodynamics}
        """
        if _db_con is None:
            return {
                "error": "DrugBank SQLite not loaded.",
                "drug_name": drug_name,
            }

        record = _lookup_drug(drug_name)
        if not record:
            return {
                "error": f"Drug '{drug_name}' not found in DrugBank.",
                "drug_name": drug_name,
            }

        cur = _db_con.execute(
            "SELECT name, drugbank_id, indication, mechanism_of_action, "
            "pharmacodynamics, half_life, categories, groups, absorption, "
            "metabolism, toxicity FROM drugs WHERE drugbank_id = ?",
            (record["drugbank_id"],),
        )
        row = cur.fetchone()
        if not row:
            return {"error": "Record not found.", "drug_name": drug_name}

        result = dict(row)
        if result.get("categories"):
            try:
                result["categories"] = json.loads(result["categories"])
            except (json.JSONDecodeError, TypeError):
                pass
        result.pop("drug_interactions", None)
        for field in ["indication", "mechanism_of_action", "pharmacodynamics",
                      "absorption", "metabolism", "toxicity"]:
            if result.get(field) and len(str(result[field])) > 1000:
                result[field] = str(result[field])[:1000] + "... [truncated]"

        return result
