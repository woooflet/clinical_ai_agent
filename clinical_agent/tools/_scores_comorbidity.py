import math

_CHARLSON_MAP = [
    (1, "myocardial_infarction",        ["410", "412"]),
    (1, "congestive_heart_failure",     ["428"]),
    (1, "peripheral_vascular_disease",  ["440", "441", "443", "4439", "4471", "5571", "5579", "V434"]),
    (1, "cerebrovascular_disease",      ["430", "431", "432", "433", "434", "435", "436", "437", "438"]),
    (1, "dementia",                     ["290"]),
    (1, "chronic_pulmonary_disease",    ["490", "491", "492", "493", "494", "495", "496", "500", "501",
                                         "502", "503", "504", "505", "5064", "5081", "5088"]),
    (1, "rheumatic_disease",            ["4465", "7100", "7101", "7102", "7103", "7104", "7140",
                                         "7141", "7142", "7148", "725"]),
    (1, "peptic_ulcer_disease",         ["531", "532", "533", "534"]),
    (1, "mild_liver_disease",           ["5712", "5714", "5715", "5716"]),
    (1, "diabetes_uncomplicated",       ["2500", "2501", "2502", "2503", "2508", "2509"]),
    (2, "diabetes_complicated",         ["2504", "2505", "2506", "2507"]),
    (2, "hemiplegia_paraplegia",        ["3341", "342", "343", "3440", "3441", "3442", "3443",
                                         "3444", "3445", "3446", "3449"]),
    (2, "renal_disease",               ["403", "404", "405", "582", "5830", "5831", "5832", "5834",
                                         "5836", "5837", "585", "586", "5880", "V420", "V451", "V56"]),
    (2, "malignancy",                  ["140", "141", "142", "143", "144", "145", "146", "147", "148",
                                         "149", "150", "151", "152", "153", "154", "155", "156", "157",
                                         "158", "159", "160", "161", "162", "163", "164", "165", "170",
                                         "171", "172", "174", "175", "176", "179", "180", "181", "182",
                                         "183", "184", "185", "186", "187", "188", "189", "190", "191",
                                         "192", "193", "194", "195", "200", "201", "202", "203", "204",
                                         "205", "206", "207", "208", "2386"]),
    (3, "moderate_severe_liver_disease", ["4560", "4561", "4562", "5722", "5723", "5724", "5728"]),
    (6, "metastatic_solid_tumor",       ["196", "197", "198", "199"]),
    (6, "aids_hiv",                     ["042", "043", "044"]),
]

def charlson(icd9_codes: list[str]) -> dict:
    codes_clean = [c.replace(".", "").upper() for c in icd9_codes]
    breakdown = {}
    score = 0
    for weight, name, prefixes in _CHARLSON_MAP:
        hit = any(c.startswith(p) for c in codes_clean for p in prefixes)
        if hit:
            breakdown[name] = weight
            score += weight

    survival_10yr = round(0.983 ** (math.e ** (score * 0.9)) * 100, 1)

    if score == 0:    interp = "No significant comorbidities"
    elif score <= 2:  interp = "Mild comorbidity burden"
    elif score <= 4:  interp = "Moderate comorbidity burden"
    else:             interp = "Severe comorbidity burden"

    return {
        "score": score,
        "interpretation": interp,
        "10yr_survival_pct": survival_10yr,
        "component_breakdown": breakdown,
    }


def curb65(confusion, urea, respiratory_rate, sbp, dbp, age) -> dict:
    pts = {
        "confusion":        1 if confusion else 0,
        "urea_gt19":        1 if urea > 19.6 else 0,
        "respiratory_rate": 1 if respiratory_rate >= 30 else 0,
        "low_bp":           1 if sbp < 90 or dbp <= 60 else 0,
        "age_ge65":         1 if age >= 65 else 0,
    }
    score = sum(pts.values())

    mortality_map = {0: 0.7, 1: 3.2, 2: 13.0, 3: 17.0, 4: 41.5, 5: 57.0}
    mortality = mortality_map.get(score, 57.0)

    if score <= 1:  risk, interp = "low",    "Low severity. Likely suitable for home treatment."
    elif score == 2: risk, interp = "medium", "Moderate severity. Consider hospital admission."
    else:            risk, interp = "high",   "High severity. Consider ICU assessment."

    return {
        "score": score,
        "interpretation": interp,
        "risk_level": risk,
        "30_day_mortality_pct": mortality,
        "component_breakdown": pts,
    }


def wells_pe(clinical_signs_dvt, pe_most_likely, heart_rate_over_100,
             immobilisation_or_surgery, prior_dvt_or_pe, hemoptysis, malignancy) -> dict:
    pts = {
        "clinical_signs_dvt":        3.0 if clinical_signs_dvt else 0,
        "pe_most_likely_diagnosis":  3.0 if pe_most_likely else 0,
        "heart_rate_over_100":       1.5 if heart_rate_over_100 else 0,
        "immobilisation_or_surgery": 1.5 if immobilisation_or_surgery else 0,
        "prior_dvt_or_pe":           1.5 if prior_dvt_or_pe else 0,
        "hemoptysis":                1.0 if hemoptysis else 0,
        "malignancy":                1.0 if malignancy else 0,
    }
    score = sum(pts.values())

    if score <= 1:    prob, risk = "Low (~1.3%)",    "low"
    elif score <= 6:  prob, risk = "Moderate (~16%)", "moderate"
    else:             prob, risk = "High (~38%)",     "high"

    return {"score": score, "probability": prob, "risk_level": risk,
            "component_breakdown": pts}


def wells_dvt(active_cancer, paralysis_or_recent_immobilisation,
              bedridden_over_3d_or_major_surgery, localised_tenderness,
              entire_leg_swollen, calf_swelling_over_3cm, pitting_edema,
              collateral_superficial_veins, prior_dvt,
              alternative_diagnosis_likely) -> dict:
    pts = {
        "active_cancer":                     1 if active_cancer else 0,
        "paralysis_or_immobilisation":        1 if paralysis_or_recent_immobilisation else 0,
        "bedridden_3d_or_major_surgery":      1 if bedridden_over_3d_or_major_surgery else 0,
        "localised_tenderness":               1 if localised_tenderness else 0,
        "entire_leg_swollen":                 1 if entire_leg_swollen else 0,
        "calf_swelling_over_3cm":             1 if calf_swelling_over_3cm else 0,
        "pitting_edema":                      1 if pitting_edema else 0,
        "collateral_superficial_veins":       1 if collateral_superficial_veins else 0,
        "prior_dvt":                          1 if prior_dvt else 0,
        "alternative_diagnosis_likely":      -2 if alternative_diagnosis_likely else 0,
    }
    score = sum(pts.values())

    if score <= 0:   prob, risk = "Low (~3%)",      "low"
    elif score <= 2: prob, risk = "Moderate (~17%)", "moderate"
    else:            prob, risk = "High (~75%)",     "high"

    return {"score": score, "probability": prob, "risk_level": risk,
            "component_breakdown": pts}
