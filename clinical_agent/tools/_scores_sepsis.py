

def _sofa_respiration(pao2_fio2: float) -> int:
    if pao2_fio2 >= 400: return 0
    if pao2_fio2 >= 300: return 1
    if pao2_fio2 >= 200: return 2
    if pao2_fio2 >= 100: return 3
    return 4


def _sofa_coagulation(platelets: float) -> int:
    if platelets >= 150: return 0
    if platelets >= 100: return 1
    if platelets >= 50:  return 2
    if platelets >= 20:  return 3
    return 4


def _sofa_liver(bilirubin: float) -> int:
    if bilirubin < 1.2:  return 0
    if bilirubin < 2.0:  return 1
    if bilirubin < 6.0:  return 2
    if bilirubin < 12.0: return 3
    return 4


def _sofa_cardiovascular(map_mmhg: float, vasopressors: str) -> int:
    vaso = vasopressors.lower().strip()
    if vaso == "none":
        return 0 if map_mmhg >= 70 else 1
    if vaso in ("dopamine_low", "dobutamine"):
        return 2
    if vaso in ("dopamine_high", "epinephrine_low", "norepinephrine_low"):
        return 3
    return 4


def _sofa_cns(gcs: int) -> int:
    if gcs == 15: return 0
    if gcs >= 13: return 1
    if gcs >= 10: return 2
    if gcs >= 6:  return 3
    return 4


def _sofa_renal(creatinine: float) -> int:
    if creatinine < 1.2:  return 0
    if creatinine < 2.0:  return 1
    if creatinine < 3.5:  return 2
    if creatinine < 5.0:  return 3
    return 4


def sofa(pao2_fio2, platelets, bilirubin, map_mmhg, vasopressors, creatinine, gcs) -> dict:
    r = _sofa_respiration(pao2_fio2)
    c = _sofa_coagulation(platelets)
    l = _sofa_liver(bilirubin)
    cv = _sofa_cardiovascular(map_mmhg, vasopressors)
    cns = _sofa_cns(gcs)
    ren = _sofa_renal(creatinine)
    score = r + c + l + cv + cns + ren

    if score <= 1:
        risk, interp = "low", "Minimal organ dysfunction"
    elif score <= 5:
        risk, interp = "moderate", "Moderate organ dysfunction, approx 10% ICU mortality"
    elif score <= 9:
        risk, interp = "high", "Significant organ dysfunction, approx 20-40% ICU mortality"
    elif score <= 12:
        risk, interp = "very_high", "Severe organ dysfunction, approx 40-60% ICU mortality"
    else:
        risk, interp = "critical", "Critical organ failure, greater than 80% ICU mortality"

    return {
        "score": score,
        "interpretation": interp,
        "risk_level": risk,
        "component_breakdown": {
            "respiration_pao2fio2": {"value": pao2_fio2, "points": r},
            "coagulation_platelets": {"value": platelets, "points": c},
            "liver_bilirubin":       {"value": bilirubin,  "points": l},
            "cardiovascular":        {"value": vasopressors, "map": map_mmhg, "points": cv},
            "cns_gcs":               {"value": gcs,         "points": cns},
            "renal_creatinine":      {"value": creatinine,  "points": ren},
        },
    }


def qsofa(respiratory_rate, altered_mentation, sbp) -> dict:
    rr_pt  = 1 if respiratory_rate >= 22 else 0
    am_pt  = 1 if altered_mentation else 0
    sbp_pt = 1 if sbp <= 100 else 0
    score  = rr_pt + am_pt + sbp_pt

    if score == 0:
        risk, interp = "low", "Low risk for sepsis. qSOFA less than 2."
    elif score == 1:
        risk, interp = "low_moderate", "Borderline. Monitor closely."
    else:
        risk, interp = "high", "High risk for sepsis (qSOFA 2 or higher). Assess for organ dysfunction."

    return {
        "score": score,
        "interpretation": interp,
        "risk_level": risk,
        "component_breakdown": {
            "respiratory_rate": {"value": respiratory_rate, "threshold": ">=22", "points": rr_pt},
            "altered_mentation": {"value": altered_mentation, "points": am_pt},
            "sbp":              {"value": sbp, "threshold": "<=100", "points": sbp_pt},
        },
    }
