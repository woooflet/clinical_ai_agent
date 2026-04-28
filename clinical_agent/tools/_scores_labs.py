import math


def ckd_epi(creatinine: float, age: int, sex: str) -> dict:
    sex = sex.lower().strip()
    if sex == "female":
        kappa, alpha, sex_factor = 0.7, -0.241, 1.012
    else:
        kappa, alpha, sex_factor = 0.9, -0.302, 1.0

    cr_kappa = creatinine / kappa
    egfr = (142
            * (min(cr_kappa, 1) ** alpha)
            * (max(cr_kappa, 1) ** -1.200)
            * (0.9938 ** age)
            * sex_factor)
    egfr = round(egfr, 1)

    if egfr >= 90:   stage, interp = "G1", "Normal or high kidney function"
    elif egfr >= 60: stage, interp = "G2", "Mildly decreased kidney function"
    elif egfr >= 45: stage, interp = "G3a","Mildly to moderately decreased"
    elif egfr >= 30: stage, interp = "G3b","Moderately to severely decreased"
    elif egfr >= 15: stage, interp = "G4", "Severely decreased kidney function"
    else:            stage, interp = "G5", "Kidney failure (ESRD)"

    return {"egfr": egfr, "ckd_stage": stage, "interpretation": interp}


def anion_gap(sodium: float, chloride: float, bicarbonate: float) -> dict:
    ag = round(sodium - (chloride + bicarbonate), 1)
    normal_low, normal_high = 8, 12

    if ag <= normal_high:
        interp = "Normal anion gap. Non-anion gap acidosis or normal."
    elif ag <= 20:
        interp = "Mildly elevated anion gap. Consider early AGMA."
    else:
        interp = (
            "Elevated anion gap. Consider MUDPILES "
            "(Methanol, Uremia, DKA, Propylene glycol, "
            "Isoniazid or Iron, Lactic acidosis, Ethylene glycol, Salicylates)."
        )

    return {
        "anion_gap": ag,
        "interpretation": interp,
        "normal_range": f"{normal_low} to {normal_high} mEq/L",
    }


def corrected_calcium(calcium: float, albumin: float) -> dict:
    corrected = round(calcium + 0.8 * (4.0 - albumin), 2)

    if corrected < 8.5:   interp = "Corrected hypocalcemia"
    elif corrected <= 10.5: interp = "Normal corrected calcium"
    else:                 interp = "Corrected hypercalcemia"

    return {
        "measured_calcium": calcium,
        "albumin": albumin,
        "corrected_calcium": corrected,
        "interpretation": interp,
    }


def meld(bilirubin: float, inr: float, creatinine: float, dialysis: bool = False) -> dict:
    cr = min(4.0, creatinine) if not dialysis else 4.0
    bili = max(1.0, bilirubin)
    inr_v = max(1.0, inr)
    cr_v  = max(1.0, cr)

    score = round(3.78 * math.log(bili) + 11.2 * math.log(inr_v) + 9.57 * math.log(cr_v) + 6.43)

    mortality_map = [
        (9,  "less than 2% 3-month mortality"),
        (19, "6% 3-month mortality"),
        (29, "20% 3-month mortality"),
        (39, "52% 3-month mortality"),
    ]
    mort_interp = "greater than 71% 3-month mortality"
    for cutoff, msg in mortality_map:
        if score <= cutoff:
            mort_interp = msg
            break

    return {
        "score": score,
        "3month_mortality_pct": mort_interp,
        "interpretation": f"MELD {score}: {mort_interp}",
    }


def child_pugh(bilirubin: float, albumin: float, pt_seconds_prolonged: float,
               ascites: str, encephalopathy: int) -> dict:

    def _bili_pts(b):
        if b < 2:   return 1
        if b <= 3:  return 2
        return 3

    def _alb_pts(a):
        if a > 3.5: return 1
        if a >= 2.8: return 2
        return 3

    def _pt_pts(pt):
        if pt < 4:  return 1
        if pt <= 6: return 2
        return 3

    def _ascites_pts(a):
        a = a.lower()
        if "none" in a or a == "absent": return 1
        if "mild" in a or "slight" in a: return 2
        return 3

    def _enceph_pts(e):
        if e == 0:      return 1
        if e in (1, 2): return 2
        return 3

    pts = {
        "bilirubin":     _bili_pts(bilirubin),
        "albumin":       _alb_pts(albumin),
        "pt_prolonged":  _pt_pts(pt_seconds_prolonged),
        "ascites":       _ascites_pts(ascites),
        "encephalopathy":_enceph_pts(encephalopathy),
    }
    score = sum(pts.values())

    if score <= 6:
        cls, s1, s2 = "A", 100, 85
    elif score <= 9:
        cls, s1, s2 = "B", 81, 57
    else:
        cls, s1, s2 = "C", 45, 35

    return {
        "score": score,
        "class_label": cls,
        "1yr_survival_pct": s1,
        "2yr_survival_pct": s2,
        "component_breakdown": pts,
    }
