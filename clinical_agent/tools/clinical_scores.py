from typing import Optional

from tools._scores_sepsis      import sofa, qsofa
from tools._scores_severity    import apache2, news2, mews
from tools._scores_comorbidity import charlson, curb65, wells_pe, wells_dvt
from tools._scores_labs        import ckd_epi, anion_gap, corrected_calcium, meld, child_pugh

_con = None


def set_db_connection(con) -> None:
    global _con
    _con = con


def register(mcp) -> None:

    @mcp.tool()
    def calculate_sofa(
        pao2_fio2: float,
        platelets: float,
        bilirubin: float,
        map_mmhg: float,
        vasopressors: str = "none",
        creatinine: float = 1.0,
        gcs: int = 15,
    ) -> dict:
        """
        Calculate the Sequential Organ Failure Assessment (SOFA) score.
        vasopressors: 'none', 'dopamine_low', 'dopamine_high', 'norepinephrine_low',
                      'epinephrine_low', 'dobutamine', 'high_dose'.
        Returns {score, interpretation, risk_level, component_breakdown}.
        """
        return sofa(pao2_fio2, platelets, bilirubin, map_mmhg, vasopressors, creatinine, gcs)

    @mcp.tool()
    def calculate_qsofa(
        respiratory_rate: float,
        altered_mentation: bool,
        sbp: float,
    ) -> dict:
        """
        Calculate the quick SOFA (qSOFA) score for sepsis screening outside the ICU.
        Returns {score, interpretation, risk_level, component_breakdown}.
        """
        return qsofa(respiratory_rate, altered_mentation, sbp)

    @mcp.tool()
    def calculate_apache2(
        temperature: float,
        map_mmhg: float,
        heart_rate: float,
        respiratory_rate: float,
        pao2: float,
        ph: float,
        sodium: float,
        potassium: float,
        creatinine: float,
        hematocrit: float,
        wbc: float,
        gcs: int,
        age: int,
        chronic_health_points: int = 0,
        acute_renal_failure: bool = False,
    ) -> dict:
        """
        Calculate the APACHE II score.
        Returns {score, predicted_mortality_pct, interpretation, risk_level, component_breakdown}.
        """
        return apache2(temperature, map_mmhg, heart_rate, respiratory_rate, pao2, ph,
                       sodium, potassium, creatinine, hematocrit, wbc, gcs, age,
                       chronic_health_points, acute_renal_failure)

    @mcp.tool()
    def calculate_news2(
        spo2: float,
        on_oxygen: bool,
        temperature: float,
        sbp: float,
        heart_rate: float,
        avpu: str,
        respiratory_rate: float,
    ) -> dict:
        """
        Calculate NEWS2. avpu: 'A', 'V', 'P', 'U'.
        Returns {score, interpretation, risk_level, component_breakdown}.
        """
        return news2(spo2, on_oxygen, temperature, sbp, heart_rate, avpu, respiratory_rate)

    @mcp.tool()
    def calculate_mews(
        sbp: float,
        heart_rate: float,
        respiratory_rate: float,
        temperature: float,
        avpu: str,
    ) -> dict:
        """
        Calculate MEWS. avpu: 'A', 'V', 'P', 'U'.
        Returns {score, interpretation, risk_level, component_breakdown}.
        """
        return mews(sbp, heart_rate, respiratory_rate, temperature, avpu)

    @mcp.tool()
    def calculate_charlson(icd9_codes: list[str]) -> dict:
        """
        Calculate Charlson Comorbidity Index from ICD-9-CM codes.
        Returns {score, interpretation, 10yr_survival_pct, component_breakdown}.
        """
        return charlson(icd9_codes)

    @mcp.tool()
    def calculate_curb65(
        confusion: bool,
        urea: float,
        respiratory_rate: float,
        sbp: float,
        dbp: float,
        age: int,
    ) -> dict:
        """
        Calculate CURB-65 for community-acquired pneumonia severity.
        urea in mg/dL (threshold >19.6 mg/dL equals >7 mmol/L).
        Returns {score, interpretation, risk_level, 30_day_mortality_pct, component_breakdown}.
        """
        return curb65(confusion, urea, respiratory_rate, sbp, dbp, age)

    @mcp.tool()
    def calculate_wells_pe(
        clinical_signs_dvt: bool,
        pe_most_likely: bool,
        heart_rate_over_100: bool,
        immobilisation_or_surgery: bool,
        prior_dvt_or_pe: bool,
        hemoptysis: bool,
        malignancy: bool,
    ) -> dict:
        """
        Calculate Wells score for PE pre-test probability.
        Returns {score, probability, risk_level, component_breakdown}.
        """
        return wells_pe(clinical_signs_dvt, pe_most_likely, heart_rate_over_100,
                        immobilisation_or_surgery, prior_dvt_or_pe, hemoptysis, malignancy)

    @mcp.tool()
    def calculate_wells_dvt(
        active_cancer: bool,
        paralysis_or_recent_immobilisation: bool,
        bedridden_over_3d_or_major_surgery: bool,
        localised_tenderness: bool,
        entire_leg_swollen: bool,
        calf_swelling_over_3cm: bool,
        pitting_edema: bool,
        collateral_superficial_veins: bool,
        prior_dvt: bool,
        alternative_diagnosis_likely: bool,
    ) -> dict:
        """
        Calculate Wells score for DVT pre-test probability.
        Returns {score, probability, risk_level, component_breakdown}.
        """
        return wells_dvt(active_cancer, paralysis_or_recent_immobilisation,
                         bedridden_over_3d_or_major_surgery, localised_tenderness,
                         entire_leg_swollen, calf_swelling_over_3cm, pitting_edema,
                         collateral_superficial_veins, prior_dvt, alternative_diagnosis_likely)

    @mcp.tool()
    def calculate_ckd_epi(
        creatinine: float,
        age: int,
        sex: str,
    ) -> dict:
        """
        Calculate eGFR using CKD-EPI 2021. sex: 'male' or 'female'.
        Returns {egfr, ckd_stage, interpretation}.
        """
        return ckd_epi(creatinine, age, sex)

    @mcp.tool()
    def calculate_anion_gap(
        sodium: float,
        chloride: float,
        bicarbonate: float,
    ) -> dict:
        """
        Calculate anion gap: Na minus (Cl plus HCO3).
        Returns {anion_gap, interpretation, normal_range}.
        """
        return anion_gap(sodium, chloride, bicarbonate)

    @mcp.tool()
    def calculate_corrected_calcium(
        calcium: float,
        albumin: float,
    ) -> dict:
        """
        Calculate albumin-corrected serum calcium.
        Returns {measured_calcium, albumin, corrected_calcium, interpretation}.
        """
        return corrected_calcium(calcium, albumin)

    @mcp.tool()
    def calculate_meld(
        bilirubin: float,
        inr: float,
        creatinine: float,
        dialysis: bool = False,
    ) -> dict:
        """
        Calculate MELD score for end-stage liver disease.
        Returns {score, 3month_mortality_pct, interpretation}.
        """
        return meld(bilirubin, inr, creatinine, dialysis)

    @mcp.tool()
    def calculate_child_pugh(
        bilirubin: float,
        albumin: float,
        pt_seconds_prolonged: float,
        ascites: str,
        encephalopathy: int,
    ) -> dict:
        """
        Calculate Child-Pugh score. ascites: 'none', 'mild', 'moderate_severe'.
        encephalopathy: 0-4.
        Returns {score, class_label, 1yr_survival_pct, 2yr_survival_pct, component_breakdown}.
        """
        return child_pugh(bilirubin, albumin, pt_seconds_prolonged, ascites, encephalopathy)

    @mcp.tool()
    def calculate_sofa_from_patient(patient_id: int, hours: int = 24) -> dict:
        """
        Pull lab and vital values from MIMIC-III for the specified patient over
        the first N hours of their ICU stay, then compute the SOFA score.
        Uses the worst (most abnormal) value per component.

        MIMIC item IDs used:
          PaO2: LABEVENTS itemid 490 or 779
          FiO2: CHARTEVENTS itemid 3420
          Platelets: LABEVENTS itemid 51265
          Bilirubin: LABEVENTS itemid 50885
          MAP: CHARTEVENTS itemid 220052 (or derived from SBP and DBP)
          Creatinine: LABEVENTS itemid 50912
          GCS total: CHARTEVENTS itemid 198

        Returns same structure as calculate_sofa, plus {values_used: {}}.
        """
        if _con is None:
            return {"error": "DuckDB connection not initialised. Call set_db_connection first."}

        admit_row = _con.execute(
            "SELECT MIN(ADMITTIME) FROM admissions WHERE SUBJECT_ID = ?",
            [patient_id]
        ).fetchone()
        if admit_row is None or admit_row[0] is None:
            return {"error": f"No admission found for subject_id={patient_id}"}
        admittime = admit_row[0]

        def _lab_worst(itemids: list[int], agg: str = "MIN") -> Optional[float]:
            id_str = ", ".join(str(i) for i in itemids)
            row = _con.execute(f"""
                SELECT {agg}(VALUENUM)
                FROM labevents
                WHERE SUBJECT_ID = ?
                  AND ITEMID IN ({id_str})
                  AND VALUENUM IS NOT NULL
                  AND CHARTTIME >= CAST(? AS TIMESTAMP)
                  AND CHARTTIME <  CAST(? AS TIMESTAMP) + INTERVAL (?) HOUR
            """, [patient_id, admittime, admittime, hours]).fetchone()
            return float(row[0]) if row and row[0] is not None else None

        def _chart_worst(itemids: list[int], agg: str = "MIN") -> Optional[float]:
            id_str = ", ".join(str(i) for i in itemids)
            row = _con.execute(f"""
                SELECT {agg}(VALUENUM)
                FROM chartevents
                WHERE SUBJECT_ID = ?
                  AND ITEMID IN ({id_str})
                  AND VALUENUM IS NOT NULL
                  AND ERROR IS DISTINCT FROM 1
                  AND CHARTTIME >= CAST(? AS TIMESTAMP)
                  AND CHARTTIME <  CAST(? AS TIMESTAMP) + INTERVAL (?) HOUR
            """, [patient_id, admittime, admittime, hours]).fetchone()
            return float(row[0]) if row and row[0] is not None else None

        pao2      = _lab_worst([490, 779], "MIN")
        fio2_raw  = _chart_worst([3420], "MAX")
        platelets = _lab_worst([51265], "MIN")
        bilirubin = _lab_worst([50885], "MAX")
        creatinine= _lab_worst([50912], "MAX")
        map_val   = _chart_worst([220052], "MIN")
        gcs_val   = _chart_worst([198], "MIN")

        if map_val is None:
            sbp = _chart_worst([220179], "MIN")
            dbp = _chart_worst([220180], "MIN")
            if sbp is not None and dbp is not None:
                map_val = round((sbp + 2 * dbp) / 3, 1)

        pao2_fio2  = (pao2 / (fio2_raw if fio2_raw and fio2_raw > 0 else 0.21)
                      ) if pao2 is not None else 300.0
        platelets  = platelets  if platelets  is not None else 150.0
        bilirubin  = bilirubin  if bilirubin  is not None else 0.8
        map_val    = map_val    if map_val    is not None else 75.0
        creatinine = creatinine if creatinine is not None else 0.9
        gcs_val    = int(gcs_val) if gcs_val is not None else 15

        values_used = {
            "pao2": pao2, "fio2": fio2_raw, "pao2_fio2_ratio": round(pao2_fio2, 1),
            "platelets": platelets, "bilirubin": bilirubin,
            "map": map_val, "creatinine": creatinine, "gcs": gcs_val,
            "hours_window": hours,
        }

        result = sofa(pao2_fio2, platelets, bilirubin, map_val, "none", creatinine, gcs_val)
        result["values_used"] = values_used
        result["note"] = (
            "Vasopressor status not auto-detected from MIMIC. CV component computed "
            "from MAP only. Manual override via calculate_sofa() if vasopressors are known."
        )
        return result

    @mcp.tool()
    def calculate_qsofa_from_patient(patient_id: int, hours: int = 24) -> dict:
        """
        Pull RR, SBP, and GCS from MIMIC-III for the patient over the first
        N hours of their ICU stay, then compute qSOFA. GCS less than 15 is treated as
        altered mentation.

        Returns same structure as calculate_qsofa, plus {values_used: {}}.
        """
        if _con is None:
            return {"error": "DuckDB connection not initialised. Call set_db_connection first."}

        admit_row = _con.execute(
            "SELECT MIN(ADMITTIME) FROM admissions WHERE SUBJECT_ID = ?",
            [patient_id]
        ).fetchone()
        if admit_row is None or admit_row[0] is None:
            return {"error": f"No admission found for subject_id={patient_id}"}
        admittime = admit_row[0]

        def _chart_worst(itemids: list[int], agg: str) -> Optional[float]:
            id_str = ", ".join(str(i) for i in itemids)
            row = _con.execute(f"""
                SELECT {agg}(VALUENUM)
                FROM chartevents
                WHERE SUBJECT_ID = ?
                  AND ITEMID IN ({id_str})
                  AND VALUENUM IS NOT NULL
                  AND ERROR IS DISTINCT FROM 1
                  AND CHARTTIME >= CAST(? AS TIMESTAMP)
                  AND CHARTTIME <  CAST(? AS TIMESTAMP) + INTERVAL (?) HOUR
            """, [patient_id, admittime, admittime, hours]).fetchone()
            return float(row[0]) if row and row[0] is not None else None

        rr_val  = _chart_worst([220210], "MAX")
        sbp_val = _chart_worst([220179], "MIN")
        gcs_val = _chart_worst([198], "MIN")

        rr_val  = rr_val  if rr_val  is not None else 16.0
        sbp_val = sbp_val if sbp_val is not None else 120.0
        gcs_val = gcs_val if gcs_val is not None else 15.0

        altered_mentation = (gcs_val < 15)

        values_used = {
            "respiratory_rate": rr_val,
            "sbp": sbp_val,
            "gcs": gcs_val,
            "altered_mentation_inferred": altered_mentation,
            "hours_window": hours,
        }

        result = qsofa(rr_val, altered_mentation, sbp_val)
        result["values_used"] = values_used
        return result
