"""
scripts/build_mimic_duckdb.py   Build a DuckDB database from raw MIMIC-III CSVs.

MIMIC-III is distributed as plain CSV files by PhysioNet. This script reads
those CSVs and loads them into a single DuckDB database file for fast SQL
queries used by the clinical decision support server.

Usage:
    python clinical_agent/scripts/build_mimic_duckdb.py \\
        --csv-dir /path/to/mimic-iii-clinical-database-1.4 \\
        --db-path /path/to/mimic_duck.db

    python clinical_agent/scripts/build_mimic_duckdb.py

The CSVs may be plain (.csv) or gzip-compressed (.csv.gz)   DuckDB handles both.
CHARTEVENTS is the largest table (~33 GB uncompressed, ~3M rows). Expect
10-40 minutes depending on hardware.

Tables imported (all standard MIMIC-III 1.4 tables needed by the agent):
    ADMISSIONS, CALLOUT, CAREGIVERS, CHARTEVENTS, CPTEVENTS,
    D_CPT, D_ICD_DIAGNOSES, D_ICD_PROCEDURES, D_ITEMS, D_LABITEMS,
    DATETIMEEVENTS, DIAGNOSES_ICD, DRGCODES, ICUSTAYS, INPUTEVENTS_CV,
    INPUTEVENTS_MV, LABEVENTS, MICROBIOLOGYEVENTS, NOTEEVENTS, OUTPUTEVENTS,
    PATIENTS, PRESCRIPTIONS, PROCEDUREEVENTS_MV, PROCEDURES_ICD,
    SERVICES, TRANSFERS
"""


import argparse
import os
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_AGENT_DIR   = _SCRIPTS_DIR.parent
_REPO_ROOT   = _AGENT_DIR.parent
sys.path.insert(0, str(_AGENT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_AGENT_DIR / ".env")
except ImportError:
    pass

import duckdb

MIMIC_TABLES = [
    "ADMISSIONS",
    "CALLOUT",
    "CAREGIVERS",
    "CHARTEVENTS",
    "CPTEVENTS",
    "D_CPT",
    "D_ICD_DIAGNOSES",
    "D_ICD_PROCEDURES",
    "D_ITEMS",
    "D_LABITEMS",
    "DATETIMEEVENTS",
    "DIAGNOSES_ICD",
    "DRGCODES",
    "ICUSTAYS",
    "INPUTEVENTS_CV",
    "INPUTEVENTS_MV",
    "LABEVENTS",
    "MICROBIOLOGYEVENTS",
    "NOTEEVENTS",
    "OUTPUTEVENTS",
    "PATIENTS",
    "PRESCRIPTIONS",
    "PROCEDUREEVENTS_MV",
    "PROCEDURES_ICD",
    "SERVICES",
    "TRANSFERS",
]

REQUIRED_TABLES = {
    "ADMISSIONS", "PATIENTS", "ICUSTAYS", "LABEVENTS", "D_LABITEMS",
    "CHARTEVENTS", "PRESCRIPTIONS", "NOTEEVENTS", "MICROBIOLOGYEVENTS",
    "DIAGNOSES_ICD", "D_ICD_DIAGNOSES", "D_ITEMS",
}


def find_csv(csv_dir: Path, table_name: str) -> Path | None:
    """Find the CSV file for a given table (handles .csv and .csv.gz)."""
    for suffix in (".csv.gz", ".csv"):
        p = csv_dir / f"{table_name}{suffix}"
        if p.exists():
            return p
    return None


def load_table(con: duckdb.DuckDBPyConnection, table_name: str, csv_path: Path) -> int:
    """Load a single CSV into DuckDB. Returns row count."""
    path_str = str(csv_path)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} AS
        SELECT * FROM read_csv_auto('{path_str}', header=true, ignore_errors=true)
    """)
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return count


def build_indexes(con: duckdb.DuckDBPyConnection) -> None:
    """Create indexes on the most frequently joined columns for query speed."""
    indexes = [
        ("ADMISSIONS",      "SUBJECT_ID"),
        ("ADMISSIONS",      "HADM_ID"),
        ("ICUSTAYS",        "SUBJECT_ID"),
        ("ICUSTAYS",        "HADM_ID"),
        ("ICUSTAYS",        "ICUSTAY_ID"),
        ("LABEVENTS",       "SUBJECT_ID"),
        ("LABEVENTS",       "HADM_ID"),
        ("LABEVENTS",       "ITEMID"),
        ("CHARTEVENTS",     "SUBJECT_ID"),
        ("CHARTEVENTS",     "ICUSTAY_ID"),
        ("CHARTEVENTS",     "ITEMID"),
        ("NOTEEVENTS",      "SUBJECT_ID"),
        ("NOTEEVENTS",      "HADM_ID"),
        ("DIAGNOSES_ICD",   "SUBJECT_ID"),
        ("DIAGNOSES_ICD",   "HADM_ID"),
        ("PRESCRIPTIONS",   "SUBJECT_ID"),
        ("MICROBIOLOGYEVENTS", "SUBJECT_ID"),
    ]
    print("\n[build_mimic_duckdb] Creating query indexes...")
    for table, col in indexes:
        idx_name = f"idx_{table.lower()}_{col.lower()}"
        try:
            con.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({col})")
            print(f"  ok {idx_name}")
        except Exception as e:
            print(f"  FAIL {idx_name}   {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DuckDB database from raw MIMIC-III CSV files."
    )
    parser.add_argument(
        "--csv-dir",
        default=os.environ.get("MIMIC_CSV_DIR", ""),
        help="Directory containing raw MIMIC-III CSV files (from PhysioNet).",
    )
    parser.add_argument(
        "--db-path",
        default=os.environ.get("MIMIC_DB", ""),
        help="Path where mimic_duck.db will be created.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help="Only import these table names (default: all). Useful for testing.",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip index creation (faster build, slower queries).",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else None
    db_path = Path(args.db_path) if args.db_path else None

    if not csv_dir:
        print("[ERROR] --csv-dir is required (or set MIMIC_CSV_DIR in .env)")
        sys.exit(1)
    if not csv_dir.is_dir():
        print(f"[ERROR] CSV directory not found: {csv_dir}")
        sys.exit(1)
    if not db_path:
        db_path = csv_dir / "mimic_duck.db"
        print(f"[INFO]  No --db-path given; using {db_path}")

    tables_to_load = args.tables or MIMIC_TABLES

    print("=" * 65)
    print("  MIMIC-III → DuckDB Builder")
    print("=" * 65)
    print(f"  CSV source : {csv_dir}")
    print(f"  DuckDB out : {db_path}")
    print(f"  Tables     : {len(tables_to_load)}")
    print()

    available: dict[str, Path] = {}
    missing: list[str] = []
    for table in tables_to_load:
        p = find_csv(csv_dir, table)
        if p:
            available[table] = p
        else:
            missing.append(table)

    missing_required = [t for t in missing if t in REQUIRED_TABLES]
    if missing_required:
        print(f"[WARN] Missing REQUIRED tables: {missing_required}")
        print("       The agent will have limited functionality.")

    if missing:
        print(f"[INFO] Tables not found (will be skipped): {missing}")
    print()

    print(f"[INFO] Connecting to {db_path} ...")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    total_start = time.time()
    loaded = 0
    errors = []

    for i, (table, csv_path) in enumerate(available.items(), 1):
        existing = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
            [table]
        ).fetchone()
        if existing:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  [{i:02d}/{len(available)}] {table:30s} already loaded ({count:,} rows)   skipping")
            loaded += 1
            continue

        t0 = time.time()
        size_mb = csv_path.stat().st_size / 1_048_576
        print(f"  [{i:02d}/{len(available)}] {table:30s} ({size_mb:6.0f} MB) ... ", end="", flush=True)
        try:
            count = load_table(con, table, csv_path)
            elapsed = time.time() - t0
            print(f"{count:>10,} rows  ({elapsed:.1f}s)")
            loaded += 1
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"ERROR   {exc}  ({elapsed:.1f}s)")
            errors.append((table, str(exc)))

    if not args.skip_indexes:
        build_indexes(con)

    con.close()

    total_elapsed = time.time() - total_start
    print()
    print("=" * 65)
    print(f"  Loaded  : {loaded}/{len(available)} tables  ({total_elapsed/60:.1f} min)")
    print(f"  DB path : {db_path}")
    print(f"  DB size : {db_path.stat().st_size / 1_073_741_824:.2f} GB")
    if errors:
        print(f"  Errors  : {len(errors)}")
        for tbl, err in errors:
            print(f"    {tbl}: {err}")
    print()
    print("  Next step   add to your .env:")
    print(f"    MIMIC_CSV_DIR={csv_dir}")
    print(f"    MIMIC_DB={db_path}")
    print("  Then restart the MCP server: ./deploy.sh --restart")
    print("=" * 65)


if __name__ == "__main__":
    main()
