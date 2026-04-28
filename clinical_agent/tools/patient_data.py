import re
import duckdb
import pandas as pd
from typing import Optional

_con = None

VITAL_ITEM_IDS = {
    "Heart Rate":          220045,
    "SBP":                 220179,
    "DBP":                 220180,
    "SpO2":                220277,
    "Temperature (C)":     223762,
    "Respiratory Rate":    220210,
}

_CAREVUE_VITAL_IDS = {
    211:  "Heart Rate",
    455:  "SBP",
    8441: "DBP",
    646:  "SpO2",
    678:  "Temperature (F)",
    618:  "Respiratory Rate",
}

_ITEMID_TO_VITAL: dict[int, str] = {v: k for k, v in VITAL_ITEM_IDS.items()}
_ITEMID_TO_VITAL.update(_CAREVUE_VITAL_IDS)

_VITAL_ID_LIST = list(_ITEMID_TO_VITAL.keys())

_DEFAULT_LAB_HOURS    = 48
_DEFAULT_VITAL_HOURS  = 72
_DEFAULT_NOTE_LIMIT   = 5
_DEFAULT_SQL_LIMIT    = 500


def set_db_connection(con: duckdb.DuckDBPyConnection) -> None:
    global _con
    _con = con




def register(mcp) -> None:

    @mcp.tool()
    def get_patient_info(patient_id: int) -> dict:
        """
        Retrieve basic demographics and admission details for a MIMIC-III patient.

        Returns a dict with subject_id, gender, dob, admission_type, insurance,
        language, religion, marital_status, ethnicity, hospital_expire_flag,
        admittime, dischtime, diagnosis, and age_at_admission (computed).
        """
        con = _con
        if con is None:
            return {"error": "MIMIC-III database not available"}

        sql = """
            SELECT
                a.SUBJECT_ID,
                p.GENDER,
                p.DOB,
                a.ADMITTIME,
                a.DISCHTIME,
                a.ADMISSION_TYPE,
                a.ADMISSION_LOCATION,
                a.DISCHARGE_LOCATION,
                a.INSURANCE,
                a.LANGUAGE,
                a.RELIGION,
                a.MARITAL_STATUS,
                a.ETHNICITY,
                a.DIAGNOSIS,
                a.HOSPITAL_EXPIRE_FLAG,
                a.HADM_ID
            FROM admissions a
            JOIN patients p ON a.SUBJECT_ID = p.SUBJECT_ID
            WHERE a.SUBJECT_ID = ?
            ORDER BY a.ADMITTIME DESC
            LIMIT 1
        """
        df = con.execute(sql, [patient_id]).df()

        if df.empty:
            return {
                "error": f"No admission record found for subject_id={patient_id}",
                "subject_id": patient_id,
            }

        row = df.iloc[0].to_dict()

        try:
            dob = pd.to_datetime(row["DOB"])
            admit = pd.to_datetime(row["ADMITTIME"])
            age = int((admit - dob).days / 365.25)
            row["age_at_admission"] = age if age <= 89 else ">=90 (anonymised)"
        except Exception:
            row["age_at_admission"] = None

        for k in ("DOB", "ADMITTIME", "DISCHTIME"):
            if row.get(k) is not None:
                try:
                    row[k] = str(row[k])
                except Exception:
                    pass

        return {k.lower(): v for k, v in row.items()}

    @mcp.tool()
    def get_patient_labs(patient_id: int, hours: int = _DEFAULT_LAB_HOURS) -> dict:
        """
        Retrieve lab values for a patient over the first N hours of their ICU stay.

        Returns a dict keyed by lab item name; each value is a list of
        {charttime, value, valuenum, valueuom, flag} records sorted by charttime.
        """
        con = _con
        if con is None:
            return {"error": "MIMIC-III database not available", "subject_id": patient_id}

        admit_sql = """
            SELECT MIN(ADMITTIME) AS admittime
            FROM admissions
            WHERE SUBJECT_ID = ?
        """
        admit_row = con.execute(admit_sql, [patient_id]).fetchone()
        if admit_row is None or admit_row[0] is None:
            return {"error": f"No admission found for subject_id={patient_id}"}

        admittime = admit_row[0]

        sql = """
            SELECT
                d.LABEL              AS lab_name,
                l.CHARTTIME,
                l.VALUE,
                l.VALUENUM,
                l.VALUEUOM,
                l.FLAG
            FROM labevents l
            JOIN d_labitems d ON l.ITEMID = d.ITEMID
            WHERE l.SUBJECT_ID = ?
              AND l.VALUENUM IS NOT NULL
              AND l.CHARTTIME >= CAST(? AS TIMESTAMP)
              AND l.CHARTTIME <  CAST(? AS TIMESTAMP) + INTERVAL (?) HOUR
            ORDER BY l.CHARTTIME ASC
            LIMIT ?
        """
        df = con.execute(
            sql,
            [patient_id, admittime, admittime, hours, _DEFAULT_SQL_LIMIT],
        ).df()

        if df.empty:
            return {
                "subject_id": patient_id,
                "hours": hours,
                "labs": {},
                "total_records": 0,
            }

        df["CHARTTIME"] = df["CHARTTIME"].astype(str)
        labs: dict = {}
        for name, grp in df.groupby("lab_name"):
            labs[name] = grp.drop(columns=["lab_name"]).rename(
                columns=str.lower
            ).to_dict(orient="records")

        return {
            "subject_id": patient_id,
            "hours": hours,
            "admittime": str(admittime),
            "total_records": len(df),
            "labs": labs,
        }

    @mcp.tool()
    def get_patient_vitals(patient_id: int, hours: int = _DEFAULT_VITAL_HOURS) -> dict:
        """
        Retrieve vital signs (HR, SBP, DBP, SpO2, temperature, RR) for a patient
        over the first N hours of their ICU stay from CHARTEVENTS.

        Returns a dict keyed by vital name; each value is a list of
        {charttime, valuenum, valueuom} records.
        """
        con = _con
        if con is None:
            return {"error": "MIMIC-III database not available", "subject_id": patient_id}

        admit_sql = "SELECT MIN(ADMITTIME) FROM admissions WHERE SUBJECT_ID = ?"
        admit_row = con.execute(admit_sql, [patient_id]).fetchone()
        if admit_row is None or admit_row[0] is None:
            return {"error": f"No admission found for subject_id={patient_id}"}

        admittime = admit_row[0]
        id_list_str = ", ".join(str(i) for i in _VITAL_ID_LIST)

        sql = f"""
            SELECT
                c.ITEMID,
                c.CHARTTIME,
                c.VALUENUM,
                c.VALUEUOM
            FROM chartevents c
            WHERE c.SUBJECT_ID = ?
              AND c.ITEMID IN ({id_list_str})
              AND c.VALUENUM IS NOT NULL
              AND c.ERROR IS DISTINCT FROM 1
              AND c.CHARTTIME >= CAST(? AS TIMESTAMP)
              AND c.CHARTTIME <  CAST(? AS TIMESTAMP) + INTERVAL (?) HOUR
            ORDER BY c.CHARTTIME ASC
            LIMIT ?
        """
        df = con.execute(
            sql,
            [patient_id, admittime, admittime, hours, _DEFAULT_SQL_LIMIT],
        ).df()

        if df.empty:
            return {
                "subject_id": patient_id,
                "hours": hours,
                "vitals": {},
                "total_records": 0,
            }

        df["vital_name"] = df["ITEMID"].map(_ITEMID_TO_VITAL)
        df["CHARTTIME"]  = df["CHARTTIME"].astype(str)

        vitals: dict = {}
        for name, grp in df.groupby("vital_name"):
            vitals[name] = (
                grp[["CHARTTIME", "VALUENUM", "VALUEUOM"]]
                .rename(columns=str.lower)
                .to_dict(orient="records")
            )

        return {
            "subject_id": patient_id,
            "hours": hours,
            "admittime": str(admittime),
            "total_records": len(df),
            "vitals": vitals,
        }

    @mcp.tool()
    def get_patient_medications(patient_id: int) -> list:
        """
        Retrieve prescription records for a patient from PRESCRIPTIONS.

        Returns a list of {drug, drug_name_generic, dose_val_rx, dose_unit_rx,
        route, startdate, enddate, drug_type}.
        """
        con = _con
        if con is None:
            return [{"error": "MIMIC-III database not available"}]

        sql = """
            SELECT
                DRUG,
                DRUG_NAME_GENERIC,
                DOSE_VAL_RX,
                DOSE_UNIT_RX,
                ROUTE,
                DRUG_TYPE,
                STARTDATE,
                ENDDATE
            FROM prescriptions
            WHERE SUBJECT_ID = ?
            ORDER BY STARTDATE ASC
            LIMIT ?
        """
        df = con.execute(sql, [patient_id, _DEFAULT_SQL_LIMIT]).df()

        if df.empty:
            return [{"subject_id": patient_id, "message": "No prescriptions found."}]

        for col in ("STARTDATE", "ENDDATE"):
            df[col] = df[col].astype(str)

        return df.rename(columns=str.lower).to_dict(orient="records")

    @mcp.tool()
    def get_patient_notes(
        patient_id: int,
        category: Optional[str] = None,
        max_notes: int = _DEFAULT_NOTE_LIMIT,
    ) -> list:
        """
        Retrieve clinical notes for a patient from NOTEEVENTS.

        Args:
            patient_id: MIMIC subject_id.
            category: Optional filter, e.g. 'Discharge summary', 'Radiology',
                      'Nursing', 'Physician'. If None, all categories are returned.
            max_notes: Maximum number of notes to return (default 5, max 20).

        Returns a list of {chartdate, category, description, text} records.
        """
        con = _con
        if con is None:
            return [{"error": "MIMIC-III database not available"}]

        limit = min(int(max_notes), 20)

        if category:
            sql = """
                SELECT CHARTDATE, CATEGORY, DESCRIPTION, TEXT
                FROM noteevents
                WHERE SUBJECT_ID = ?
                  AND ISERROR IS NULL
                  AND LOWER(CATEGORY) = LOWER(?)
                ORDER BY CHARTDATE DESC
                LIMIT ?
            """
            df = con.execute(sql, [patient_id, category, limit]).df()
        else:
            sql = """
                SELECT CHARTDATE, CATEGORY, DESCRIPTION, TEXT
                FROM noteevents
                WHERE SUBJECT_ID = ?
                  AND ISERROR IS NULL
                ORDER BY CHARTDATE DESC
                LIMIT ?
            """
            df = con.execute(sql, [patient_id, limit]).df()

        if df.empty:
            return [
                {
                    "subject_id": patient_id,
                    "category": category,
                    "message": "No notes found.",
                }
            ]

        df["CHARTDATE"] = df["CHARTDATE"].astype(str)
        df["TEXT"] = df["TEXT"].apply(
            lambda t: t[:4000] + "...[truncated]" if isinstance(t, str) and len(t) > 4000 else t
        )
        return df.rename(columns=str.lower).to_dict(orient="records")


    @mcp.tool()
    def get_patient_diagnoses(patient_id: int) -> list:
        """
        Retrieve ICD-9 diagnosis codes and long titles for a patient.

        Joins DIAGNOSES_ICD with D_ICD_DIAGNOSES.
        Returns a list of {seq_num, icd9_code, short_title, long_title}.
        """
        con = _con
        if con is None:
            return [{"error": "MIMIC-III database not available"}]

        sql = """
            SELECT
                di.SEQ_NUM,
                di.ICD9_CODE,
                d.SHORT_TITLE,
                d.LONG_TITLE
            FROM diagnoses_icd di
            LEFT JOIN d_icd_diagnoses d ON di.ICD9_CODE = d.ICD9_CODE
            WHERE di.SUBJECT_ID = ?
            ORDER BY di.SEQ_NUM ASC
            LIMIT ?
        """
        df = con.execute(sql, [patient_id, _DEFAULT_SQL_LIMIT]).df()

        if df.empty:
            return [{"subject_id": patient_id, "message": "No diagnoses found."}]

        return df.rename(columns=str.lower).to_dict(orient="records")

    @mcp.tool()
    def get_patient_microbiology(patient_id: int) -> list:
        """
        Retrieve microbiology culture results for a patient from MICROBIOLOGYEVENTS.

        Returns a list of {chartdate, spec_type_desc, org_name,
        ab_name, interpretation} (antibiotic sensitivity).
        """
        con = _con
        if con is None:
            return [{"error": "MIMIC-III database not available"}]

        sql = """
            SELECT
                CHARTDATE,
                SPEC_TYPE_DESC,
                ORG_NAME,
                AB_NAME,
                INTERPRETATION
            FROM microbiologyevents
            WHERE SUBJECT_ID = ?
            ORDER BY CHARTDATE ASC
            LIMIT ?
        """
        df = con.execute(sql, [patient_id, _DEFAULT_SQL_LIMIT]).df()

        if df.empty:
            return [
                {
                    "subject_id": patient_id,
                    "message": "No microbiology results found.",
                }
            ]

        df["CHARTDATE"] = df["CHARTDATE"].astype(str)
        records = df.rename(columns=str.lower).to_dict(orient="records")
        return [
            {k: (None if (isinstance(v, float) and v != v) else v) for k, v in row.items()}
            for row in records
        ]

    @mcp.tool()
    def query_cohort_sql(sql: str) -> list:
        """
        Execute arbitrary read-only DuckDB SQL against the MIMIC-III database.

        The query MUST include a LIMIT clause (max 500 rows enforced).
        Returns a list of row dicts.

        Example:
            query_cohort_sql("SELECT subject_id, hadm_id FROM admissions LIMIT 5")
        """
        con = _con
        if con is None:
            return [{"error": "MIMIC-III database not available"}]

        sql_upper = sql.upper().strip()
        for forbidden in ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
                          "ALTER", "TRUNCATE", "REPLACE", "ATTACH", "DETACH",
                          "COPY", "EXPORT", "IMPORT"):
            if re.search(rf"\b{forbidden}\b", sql_upper):
                return [
                    {
                        "error": f"Mutation statements are not allowed: {forbidden}",
                        "sql": sql,
                    }
                ]

        if "LIMIT" not in sql_upper:
            sql = sql.rstrip("; ") + f" LIMIT {_DEFAULT_SQL_LIMIT}"

        try:
            df = con.execute(sql).df()
        except Exception as exc:
            return [{"error": str(exc), "sql": sql}]

        if len(df) > _DEFAULT_SQL_LIMIT:
            df = df.iloc[: _DEFAULT_SQL_LIMIT]

        import numpy as np
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        for col in dt_cols:
            df[col] = df[col].astype(str)

        return df.rename(columns=str.lower).to_dict(orient="records")

    @mcp.tool()
    def get_patient_summary(patient_id: int) -> dict:
        """
        One-shot patient overview combining demographics, last 72h vitals
        (mean, min, max per vital), top-10 most recent labs, and all ICD-9
        diagnoses. Designed to be the first tool called in any patient
        analysis chain.

        Returns: {
            patient_info:    {...}
            vitals_summary:  {HR: {mean, min, max}, ...}
            recent_labs:     [{name, value, unit, charttime}]
            diagnoses:       [{icd9_code, long_title}]
        }
        """
        con = _con
        if con is None:
            return {"error": "MIMIC-III database not available"}

        info_sql = """
            SELECT
                a.SUBJECT_ID, p.GENDER, p.DOB,
                a.ADMITTIME, a.DISCHTIME, a.ADMISSION_TYPE,
                a.INSURANCE, a.LANGUAGE, a.MARITAL_STATUS,
                a.ETHNICITY, a.DIAGNOSIS, a.HOSPITAL_EXPIRE_FLAG, a.HADM_ID
            FROM admissions a
            JOIN patients p ON a.SUBJECT_ID = p.SUBJECT_ID
            WHERE a.SUBJECT_ID = ?
            ORDER BY a.ADMITTIME DESC
            LIMIT 1
        """
        info_df = con.execute(info_sql, [patient_id]).df()
        if info_df.empty:
            return {"error": f"No admission record found for subject_id={patient_id}"}

        row = info_df.iloc[0].to_dict()
        try:
            dob   = pd.to_datetime(row["DOB"])
            admit = pd.to_datetime(row["ADMITTIME"])
            age   = int((admit - dob).days / 365.25)
            row["age_at_admission"] = age if age <= 89 else ">=90 (anonymised)"
        except Exception:
            row["age_at_admission"] = None
        for k in ("DOB", "ADMITTIME", "DISCHTIME"):
            if row.get(k) is not None:
                row[k] = str(row[k])
        patient_info = {k.lower(): v for k, v in row.items()}

        id_list_str = ", ".join(str(i) for i in _VITAL_ID_LIST)
        vitals_sql = f"""
            SELECT
                c.ITEMID,
                AVG(c.VALUENUM)  AS v_mean,
                MIN(c.VALUENUM)  AS v_min,
                MAX(c.VALUENUM)  AS v_max,
                COUNT(*)         AS n_readings
            FROM admissions a
            JOIN chartevents c
              ON c.SUBJECT_ID = a.SUBJECT_ID
            WHERE a.SUBJECT_ID = ?
              AND c.ITEMID IN ({id_list_str})
              AND c.VALUENUM IS NOT NULL
              AND c.ERROR IS DISTINCT FROM 1
              AND c.CHARTTIME >= a.ADMITTIME
              AND c.CHARTTIME <  a.ADMITTIME + INTERVAL 72 HOUR
            GROUP BY c.ITEMID
        """
        vitals_df = con.execute(vitals_sql, [patient_id]).df()
        vitals_summary: dict = {}
        for _, vrow in vitals_df.iterrows():
            name = _ITEMID_TO_VITAL.get(int(vrow["ITEMID"]), str(int(vrow["ITEMID"])))
            vitals_summary[name] = {
                "mean":       round(float(vrow["v_mean"]), 2),
                "min":        round(float(vrow["v_min"]),  2),
                "max":        round(float(vrow["v_max"]),  2),
                "n_readings": int(vrow["n_readings"]),
            }

        labs_sql = """
            SELECT
                d.LABEL AS name,
                l.VALUENUM  AS value,
                l.VALUEUOM  AS unit,
                l.CHARTTIME AS charttime
            FROM labevents l
            JOIN d_labitems d ON l.ITEMID = d.ITEMID
            WHERE l.SUBJECT_ID = ?
              AND l.VALUENUM IS NOT NULL
            ORDER BY l.CHARTTIME DESC
            LIMIT 10
        """
        labs_df = con.execute(labs_sql, [patient_id]).df()
        labs_df["charttime"] = labs_df["charttime"].astype(str)
        recent_labs = labs_df.to_dict(orient="records")

        dx_sql = """
            SELECT di.ICD9_CODE AS icd9_code, d.LONG_TITLE AS long_title
            FROM diagnoses_icd di
            LEFT JOIN d_icd_diagnoses d ON di.ICD9_CODE = d.ICD9_CODE
            WHERE di.SUBJECT_ID = ?
            ORDER BY di.SEQ_NUM ASC
            LIMIT 25
        """
        dx_df = con.execute(dx_sql, [patient_id]).df()
        diagnoses = dx_df.to_dict(orient="records")

        return {
            "patient_info":   patient_info,
            "vitals_summary": vitals_summary,
            "recent_labs":    recent_labs,
            "diagnoses":      diagnoses,
        }

    @mcp.tool()
    def get_icd9_description(code: str) -> dict:
        """
        Look up the short and long title for an ICD-9-CM code.

        Args:
            code: ICD-9 code string, e.g. '41401' or '410.01'
                  (dots are stripped automatically).

        Returns {icd9_code, short_title, long_title, category}.
        """
        con = _con
        if con is None:
            return {"error": "MIMIC-III database not available"}

        normalised = code.replace(".", "").strip()

        sql = """
            SELECT ICD9_CODE, SHORT_TITLE, LONG_TITLE
            FROM d_icd_diagnoses
            WHERE ICD9_CODE = ?
            LIMIT 1
        """
        df = con.execute(sql, [normalised]).df()

        if df.empty:
            prefix_sql = """
                SELECT ICD9_CODE, SHORT_TITLE, LONG_TITLE
                FROM d_icd_diagnoses
                WHERE ICD9_CODE LIKE ?
                ORDER BY ICD9_CODE ASC
                LIMIT 5
            """
            df = con.execute(prefix_sql, [normalised + "%"]).df()
            if df.empty:
                return {
                    "error": f"ICD-9 code '{code}' not found in D_ICD_DIAGNOSES.",
                    "icd9_code": code,
                }
            row = df.iloc[0].to_dict()
            row["note"] = f"Exact code '{code}' not found; showing closest match."
        else:
            row = df.iloc[0].to_dict()

        icd_num = normalised[:3]
        row["category"] = _icd9_chapter(icd_num)

        return {k.lower(): v for k, v in row.items()}


def _icd9_chapter(prefix3: str) -> str:
    try:
        n = int(prefix3)
    except ValueError:
        if prefix3.startswith("V"):
            return "Supplementary Classification (V-codes)"
        if prefix3.startswith("E"):
            return "External Causes of Injury (E-codes)"
        return "Unknown"

    chapters = [
        (1,   139,  "Infectious and Parasitic Diseases"),
        (140, 239,  "Neoplasms"),
        (240, 279,  "Endocrine, Nutritional, Metabolic and Immunity"),
        (280, 289,  "Diseases of Blood and Blood-Forming Organs"),
        (290, 319,  "Mental Disorders"),
        (320, 389,  "Nervous System and Sense Organs"),
        (390, 459,  "Circulatory System"),
        (460, 519,  "Respiratory System"),
        (520, 579,  "Digestive System"),
        (580, 629,  "Genitourinary System"),
        (630, 679,  "Pregnancy, Childbirth, and Puerperium"),
        (680, 709,  "Skin and Subcutaneous Tissue"),
        (710, 739,  "Musculoskeletal System and Connective Tissue"),
        (740, 759,  "Congenital Anomalies"),
        (760, 779,  "Perinatal Conditions"),
        (780, 799,  "Symptoms, Signs, and Ill-Defined Conditions"),
        (800, 999,  "Injury and Poisoning"),
    ]
    for lo, hi, name in chapters:
        if lo <= n <= hi:
            return name
    return "Unknown"
