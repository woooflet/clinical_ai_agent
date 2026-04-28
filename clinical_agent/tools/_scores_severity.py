import math


def _aps_temp(t: float) -> int:
    if t >= 41 or t <= 29.9: return 4
    if t >= 39 or t <= 31.9: return 3
    if t >= 38.5 or t == 30: return 1
    if 36 <= t <= 38.4:      return 0
    if 34 <= t <= 35.9:      return 1
    if 32 <= t <= 33.9:      return 2
    return 3

def _aps_map(m: float) -> int:
    if m >= 160 or m <= 49:  return 4
    if m >= 130:             return 3
    if m >= 110 or m <= 69:  return 2
    return 0

def _aps_hr(hr: float) -> int:
    if hr >= 180 or hr <= 39: return 4
    if hr >= 140 or hr <= 54: return 3
    if hr >= 110 or hr <= 69: return 2
    return 0

def _aps_rr(rr: float) -> int:
    if rr >= 50 or rr <= 5:   return 4
    if rr >= 35:               return 3
    if rr >= 25 or rr <= 9:    return 1
    if 12 <= rr <= 24:         return 0
    return 2

def _aps_pao2(pao2: float) -> int:
    if pao2 < 55:  return 4
    if pao2 < 61:  return 3
    if pao2 < 71:  return 1
    return 0

def _aps_ph(ph: float) -> int:
    if ph >= 7.7 or ph < 7.15: return 4
    if ph >= 7.6 or ph < 7.25: return 3
    if ph >= 7.5:               return 1
    if 7.33 <= ph <= 7.49:      return 0
    return 2

def _aps_sodium(na: float) -> int:
    if na >= 180 or na <= 110: return 4
    if na >= 160 or na <= 119: return 3
    if na >= 155 or na <= 129: return 2
    if na >= 150:              return 1
    return 0

def _aps_potassium(k: float) -> int:
    if k >= 7 or k < 2.5: return 4
    if k >= 6:             return 3
    if k >= 5.5 or k < 3: return 2
    if k >= 5:             return 1
    return 0

def _aps_creatinine(cr: float, arf: bool) -> int:
    pts = 0
    if cr >= 3.5:   pts = 4
    elif cr >= 2:   pts = 3
    elif cr >= 1.5: pts = 2
    elif cr < 0.6:  pts = 2
    else:           pts = 0
    return pts * 2 if arf else pts

def _aps_hematocrit(hct: float) -> int:
    if hct >= 60 or hct < 20:  return 4
    if hct >= 50 or hct < 30:  return 2
    if hct >= 46:               return 1
    return 0

def _aps_wbc(wbc: float) -> int:
    if wbc >= 40 or wbc < 1:   return 4
    if wbc >= 20 or wbc < 3:   return 2
    if wbc >= 15:               return 1
    return 0

def _aps_gcs(gcs: int) -> int:
    return 15 - gcs

def _apache2_age(age: int) -> int:
    if age < 45:  return 0
    if age < 55:  return 2
    if age < 65:  return 3
    if age < 75:  return 5
    return 6

def apache2(temperature, map_mmhg, heart_rate, respiratory_rate, pao2, ph,
            sodium, potassium, creatinine, hematocrit, wbc, gcs, age,
            chronic_health_points=0, acute_renal_failure=False) -> dict:
    aps = sum([
        _aps_temp(temperature),
        _aps_map(map_mmhg),
        _aps_hr(heart_rate),
        _aps_rr(respiratory_rate),
        _aps_pao2(pao2),
        _aps_ph(ph),
        _aps_sodium(sodium),
        _aps_potassium(potassium),
        _aps_creatinine(creatinine, acute_renal_failure),
        _aps_hematocrit(hematocrit),
        _aps_wbc(wbc),
        _aps_gcs(gcs),
    ])
    age_pts = _apache2_age(age)
    score   = aps + age_pts + chronic_health_points

    log_odds = -3.517 + (score * 0.146)
    mortality_pct = round(100 / (1 + math.exp(-log_odds)), 1)

    if score <= 9:   risk, interp = "low",       "Low severity, predicted mortality less than 10%"
    elif score <= 19: risk, interp = "moderate", "Moderate severity"
    elif score <= 29: risk, interp = "high",     "High severity"
    else:             risk, interp = "critical", "Critical severity, predicted mortality greater than 80%"

    return {
        "score": score,
        "predicted_mortality_pct": mortality_pct,
        "interpretation": interp,
        "risk_level": risk,
        "component_breakdown": {
            "acute_physiology_score": aps,
            "age_points": age_pts,
            "chronic_health_points": chronic_health_points,
        },
    }


def _news2_spo2(spo2, on_oxygen) -> int:
    if on_oxygen:
        if spo2 >= 97: return 3
        if spo2 >= 95: return 2
        if spo2 >= 93: return 1
        return 0
    else:
        if spo2 >= 96: return 0
        if spo2 >= 94: return 1
        if spo2 >= 92: return 2
        return 3

def _news2_temp(t) -> int:
    if t <= 35.0:         return 3
    if t <= 36.0:         return 1
    if t <= 38.0:         return 0
    if t <= 39.0:         return 1
    return 2

def _news2_sbp(sbp) -> int:
    if sbp <= 90:         return 3
    if sbp <= 100:        return 2
    if sbp <= 110:        return 1
    if sbp <= 219:        return 0
    return 3

def _news2_hr(hr) -> int:
    if hr <= 40:          return 3
    if hr <= 50:          return 1
    if hr <= 90:          return 0
    if hr <= 110:         return 1
    if hr <= 130:         return 2
    return 3

def _news2_avpu(avpu: str) -> int:
    return 0 if avpu.upper() == "A" else 3

def _news2_rr(rr) -> int:
    if rr <= 8:           return 3
    if rr <= 11:          return 1
    if rr <= 20:          return 0
    if rr <= 24:          return 2
    return 3

def news2(spo2, on_oxygen, temperature, sbp, heart_rate, avpu, respiratory_rate) -> dict:
    pts = {
        "spo2":             _news2_spo2(spo2, on_oxygen),
        "on_oxygen":        2 if on_oxygen else 0,
        "temperature":      _news2_temp(temperature),
        "sbp":              _news2_sbp(sbp),
        "heart_rate":       _news2_hr(heart_rate),
        "avpu":             _news2_avpu(avpu),
        "respiratory_rate": _news2_rr(respiratory_rate),
    }
    score = sum(pts.values())

    if score <= 4:
        risk, interp = "low",      "Low clinical risk. Routine monitoring."
    elif score <= 6:
        risk, interp = "medium",   "Medium risk. Increased monitoring frequency."
    elif score <= 8 or pts["avpu"] == 3:
        risk, interp = "high",     "High risk. Urgent clinical review."
    else:
        risk, interp = "very_high","Very high risk. Emergency response."

    return {"score": score, "interpretation": interp, "risk_level": risk,
            "component_breakdown": pts}


def _mews_sbp(sbp) -> int:
    if sbp <= 70:   return 3
    if sbp <= 80:   return 2
    if sbp <= 100:  return 1
    if sbp <= 199:  return 0
    return 2

def _mews_hr(hr) -> int:
    if hr < 40:     return 2
    if hr <= 50:    return 1
    if hr <= 100:   return 0
    if hr <= 110:   return 1
    if hr <= 129:   return 2
    return 3

def _mews_rr(rr) -> int:
    if rr < 9:      return 2
    if rr <= 14:    return 0
    if rr <= 20:    return 1
    if rr <= 29:    return 2
    return 3

def _mews_temp(t) -> int:
    if t < 35:      return 2
    if t <= 38.4:   return 0
    return 2

def _mews_avpu(avpu: str) -> int:
    avpu = avpu.upper()
    return {"A": 0, "V": 1, "P": 2, "U": 3}.get(avpu, 3)

def mews(sbp, heart_rate, respiratory_rate, temperature, avpu) -> dict:
    pts = {
        "sbp":              _mews_sbp(sbp),
        "heart_rate":       _mews_hr(heart_rate),
        "respiratory_rate": _mews_rr(respiratory_rate),
        "temperature":      _mews_temp(temperature),
        "avpu":             _mews_avpu(avpu),
    }
    score = sum(pts.values())

    if score <= 2:  risk, interp = "low",    "Low risk. Standard monitoring."
    elif score <= 4: risk, interp = "medium", "Increased risk. Notify nurse or doctor."
    else:            risk, interp = "high",   "High risk. Urgent medical review required."

    return {"score": score, "interpretation": interp, "risk_level": risk,
            "component_breakdown": pts}
