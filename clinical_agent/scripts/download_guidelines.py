"""
scripts/download_guidelines.py   Download clinical guideline PDFs for .

Tries multiple known-working URLs per guideline (primary + fallbacks).
Saves PDFs to clinical_agent/data/guidelines/.
Writes download_manifest.json so seed_guidelines.py knows what's available.

Run from clinical_agent/ directory:
    nohup python scripts/download_guidelines.py > logs/download_guidelines.log 2>&1 &

100 Mbps connection: expect 30 120s per PDF depending on size (5 30 MB each).
"""


import json
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR  = Path(__file__).resolve().parent
AGENT_DIR   = SCRIPT_DIR.parent
GUIDELINES_DIR = AGENT_DIR / "data" / "guidelines"
GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = AGENT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MANIFEST_PATH = GUIDELINES_DIR / "download_manifest.json"

GUIDELINES = [
    {
        "filename": "surviving_sepsis_2021.pdf",
        "source": "Surviving Sepsis Campaign Guidelines 2021",
        "tags": "sepsis,antibiotics,vasopressors,critical_care,ICU",
        "urls": [
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8594550/pdf/ccm-49-e1032.pdf",
            "https://journals.lww.com/ccmjournal/Fulltext/2021/11000/Surviving_Sepsis_Campaign__International.21.aspx",
            "https://link.springer.com/content/pdf/10.1007/s00134-021-06506-y.pdf",
        ],
        "notes": "Evans L et al. 2021 SSC guidelines. Open access via PMC/Springer.",
    },
    {
        "filename": "kdigo_aki_2012.pdf",
        "source": "KDIGO Clinical Practice Guideline for Acute Kidney Injury 2012",
        "tags": "acute_kidney_injury,renal,creatinine,ICU,dialysis",
        "urls": [
            "https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf",
            "https://www.theisn.org/images/ISN_Outreach/KDIGO-2012-AKI-Guideline-English.pdf",
        ],
        "notes": "KDIGO 2012 AKI guideline   freely available from kdigo.org.",
    },
    {
        "filename": "kdigo_ckd_2024.pdf",
        "source": "KDIGO Clinical Practice Guideline for CKD Evaluation and Management 2024",
        "tags": "chronic_kidney_disease,renal,egfr,proteinuria",
        "urls": [
            "https://kdigo.org/wp-content/uploads/2024/07/KDIGO-2024-CKD-Guideline.pdf",
            "https://kdigo.org/wp-content/uploads/2017/02/KDIGO_2012_CKD_GL.pdf",
        ],
        "notes": "KDIGO 2024 CKD update   URL may differ; check https://kdigo.org/guidelines/ckd-mbd/ if failed.",
    },
    {
        "filename": "aha_heart_failure_2022.pdf",
        "source": "AHA/ACC/HFSA Heart Failure Guideline 2022",
        "tags": "heart_failure,cardiology,ejection_fraction,diuretics",
        "urls": [
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9238937/pdf/nihms-1801069.pdf",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9238937/pdf/",
            "https://www.onlinejacc.org/content/accj/79/17/e263.full.pdf",
        ],
        "notes": "Heidenreich PA et al. 2022 AHA/ACC/HFSA Guideline. PMC9238937.",
    },
    {
        "filename": "ardsnet_ventilation_protocol.pdf",
        "source": "ARDSNet Low Tidal Volume Ventilation Protocol",
        "tags": "ARDS,mechanical_ventilation,PEEP,lung_protective,tidal_volume",
        "urls": [
            "http://www.ardsnet.org/files/ventilator_protocol_2008-07.pdf",
            "https://www.ardsnet.org/files/ventilator_protocol_2008-07.pdf",
        ],
        "notes": "ARDSNet ARMA trial ventilation protocol. Small file (~400KB).",
    },
    {
        "filename": "who_sepsis_2017.pdf",
        "source": "WHO Improving Prevention, Diagnosis and Clinical Management of Sepsis 2017",
        "tags": "sepsis,global_health,diagnosis,management,WHO",
        "urls": [
            "https://iris.who.int/bitstream/handle/10665/274742/A70_13-en.pdf",
            "https://iris.who.int/bitstream/handle/10665/273181/B140_12-en.pdf",
        ],
        "notes": "WHO WHA70.7 sepsis resolution documents   IRIS repository.",
    },
    {
        "filename": "aha_atrial_fibrillation_2023.pdf",
        "source": "AHA/ACC/HRS Atrial Fibrillation Guideline 2023",
        "tags": "atrial_fibrillation,cardiology,anticoagulation,rate_control",
        "urls": [
            "https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000001193",
        ],
        "notes": "2023 AHA AF guideline.",
    },
    {
        "filename": "nice_sepsis_ng253.pdf",
        "source": "NICE Guideline NG253   Suspected Sepsis (Adults 16+) 2024",
        "tags": "sepsis,diagnosis,NEWS2,antibiotics,NICE,UK",
        "urls": [
            "https://www.nice.org.uk/guidance/ng253/resources/suspected-sepsis-recognition-diagnosis-and-early-management-pdf-66145040453573",
            "https://www.nice.org.uk/guidance/ng253/evidence/full-guideline-pdf-66145040453574",
        ],
        "notes": "NG253 is the 2024 update replacing NG51. Covers adults 16+.",
    },
    {
        "filename": "idsa_community_pneumonia_2019.pdf",
        "source": "IDSA/ATS Community-Acquired Pneumonia Guidelines 2019",
        "tags": "pneumonia,antibiotics,infectious_disease,respiratory",
        "urls": [
            "https://academic.oup.com/cid/article-pdf/71/7/e45/34027655/ciaa314.pdf",
        ],
        "notes": "Metlay JP et al. 2019 IDSA/ATS CAP guidelines.",
    },
    {
        "filename": "ashp_vancomycin_dosing_2020.pdf",
        "source": "ASHP/IDSA/SIDP Vancomycin AUC-Guided Dosing Guidelines 2020",
        "tags": "vancomycin,pharmacokinetics,dosing,antibiotics,AUC,renal",
        "urls": [
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7260580/pdf/",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7260580/pdf/zxaa036.pdf",
        ],
        "notes": "Rybak MJ et al. 2020 vancomycin consensus guidelines. PMC7260580.",
    },
]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download url → dest. Returns True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True, allow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        content_length = resp.headers.get("Content-Length", "?")

        if "text/html" in content_type and "pdf" not in content_type:
            print(f"    FAIL Got HTML instead of PDF (login wall?). Content-Type: {content_type}")
            return False

        bytes_written = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

        if bytes_written < 10_240:
            dest.unlink(missing_ok=True)
            print(f"    FAIL File too small ({bytes_written} bytes)   likely an error page")
            return False

        print(f"    ok Downloaded {bytes_written / 1024 / 1024:.1f} MB  [{content_length} declared]")
        return True

    except requests.exceptions.Timeout:
        print(f"    FAIL Timeout after {timeout}s")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"    FAIL HTTP {e.response.status_code}: {e}")
        return False
    except Exception as e:
        print(f"    FAIL Error: {e}")
        return False


def main():
    print("=" * 70)
    print("  Clinical Guideline PDF Downloader   ")
    print(f"  Destination: {GUIDELINES_DIR}")
    print("=" * 70)

    manifest = {}

    for idx, g in enumerate(GUIDELINES, 1):
        filename  = g["filename"]
        source    = g["source"]
        dest_path = GUIDELINES_DIR / filename
        urls      = g["urls"]

        print(f"\n[{idx:02d}/{len(GUIDELINES)}] {source}")
        print(f"  → {filename}")

        if dest_path.exists() and dest_path.stat().st_size > 10_240:
            size_mb = dest_path.stat().st_size / 1024 / 1024
            print(f"  ⏭  Already exists ({size_mb:.1f} MB)   skipping")
            manifest[filename] = {
                "status":   "ok",
                "source":   source,
                "tags":     g["tags"],
                "path":     str(dest_path),
                "notes":    g.get("notes", ""),
            }
            continue

        success = False
        for url_idx, url in enumerate(urls, 1):
            print(f"  Trying URL {url_idx}/{len(urls)}: {url[:80]}…")
            success = download_file(url, dest_path)
            if success:
                break
            time.sleep(1)

        if success:
            size_mb = dest_path.stat().st_size / 1024 / 1024
            manifest[filename] = {
                "status":   "ok",
                "source":   source,
                "tags":     g["tags"],
                "path":     str(dest_path),
                "size_mb":  round(size_mb, 2),
                "notes":    g.get("notes", ""),
            }
            print(f"  ok Saved ({size_mb:.1f} MB)")
        else:
            manifest[filename] = {
                "status":  "failed",
                "source":  source,
                "tags":    g["tags"],
                "path":    str(dest_path),
                "notes":   g.get("notes", "") + " | DOWNLOAD FAILED   manual download required.",
            }
            print(f"  FAIL: All URLs failed   manual download required.")
            print(f"     See RAG_IMPROVEMENT_PLAN.md for alternative URLs.")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    ok      = [k for k, v in manifest.items() if v["status"] == "ok"]
    failed  = [k for k, v in manifest.items() if v["status"] == "failed"]

    print("\n" + "=" * 70)
    print(f"  DONE. {len(ok)}/{len(GUIDELINES)} downloaded successfully.")
    if ok:
        print(f"  ok Success: {', '.join(ok)}")
    if failed:
        print(f"  FAIL: Failed (manual download needed): {', '.join(failed)}")
    print(f"\n  Manifest written to: {MANIFEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
