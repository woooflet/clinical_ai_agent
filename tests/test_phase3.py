"""
test_phase3.py — Validate all 14 clinical scoring calculators.
Run from the repo root:
    ./run_tests.sh
    ./run_tests.sh --module test_phase3
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from tools._scores_sepsis      import sofa, qsofa
from tools._scores_severity    import apache2, news2, mews
from tools._scores_comorbidity import charlson, curb65, wells_pe, wells_dvt
from tools._scores_labs        import ckd_epi, anion_gap, corrected_calcium, meld, child_pugh

PASS = 0; FAIL = 0

def check(name, result, key, expected, op="=="):
    global PASS, FAIL
    val = result[key]
    ok = (val == expected) if op == "==" else \
         (val <= expected) if op == "<=" else \
         (val >= expected) if op == ">=" else False
    status = "PASS" if ok else "FAIL"
    if not ok: FAIL += 1
    else: PASS += 1
    print(f"  [{status}] {name}: {key}={val} (expected {op}{expected})")

print("=== SOFA ===")
r = sofa(200, 80, 3.0, 65, "dopamine_low", 2.5, 12)
check("SOFA typical", r, "score", 12, "==")
check("SOFA risk", r, "risk_level", "very_high", "==")
r0 = sofa(450, 200, 0.8, 80, "none", 0.8, 15)
check("SOFA normal", r0, "score", 0, "==")

print("=== qSOFA ===")
r = qsofa(24, True, 95)
check("qSOFA=3", r, "score", 3, "==")
check("qSOFA high", r, "risk_level", "high", "==")
r = qsofa(15, False, 120)
check("qSOFA=0", r, "score", 0, "==")

print("=== APACHE II ===")
r = apache2(38, 70, 110, 22, 80, 7.35, 140, 4.0, 1.5, 42, 11, 13, 65, 2)
check("APACHE II score>=10", r, "score", 10, ">=")
check("APACHE II mortality>0", r, "predicted_mortality_pct", 0, ">=")

print("=== NEWS2 ===")
r = news2(94, False, 37.5, 105, 95, "A", 18)
check("NEWS2 low", r, "risk_level", "low", "==")
r = news2(88, True, 36.0, 88, 115, "V", 25)
check("NEWS2 high risk score>=5", r, "score", 5, ">=")

print("=== MEWS ===")
r = mews(70, 130, 25, 38.8, "V")
check("MEWS high", r, "risk_level", "high", "==")

print("=== CHARLSON ===")
r = charlson(["41001", "250.00", "5850"])
check("Charlson w/ 3 conditions", r, "score", 3, ">=")
r = charlson(["19600"])  # metastatic tumor
check("Charlson metastatic=6", r, "score", 6, "==")

print("=== CURB-65 ===")
r = curb65(True, 22, 32, 85, 55, 70)
check("CURB-65=5", r, "score", 5, "==")
check("CURB-65 high", r, "risk_level", "high", "==")
r = curb65(False, 10, 18, 120, 70, 45)
check("CURB-65=0 low", r, "risk_level", "low", "==")

print("=== WELLS PE ===")
r = wells_pe(True, True, True, False, False, False, False)
check("Wells PE moderate", r, "score", 4.0, ">=")
r = wells_pe(True, True, True, True, True, True, True)
check("Wells PE high>6", r, "score", 6.0, ">=")

print("=== WELLS DVT ===")
r = wells_dvt(True, True, True, True, True, True, False, False, False, False)
check("Wells DVT high", r, "risk_level", "high", "==")
r = wells_dvt(False, False, False, False, False, False, False, False, False, True)
check("Wells DVT alt dx likely score<0", r, "score", 0, "<=")

print("=== CKD-EPI ===")
r = ckd_epi(1.0, 50, "male")
check("CKD-EPI normal eGFR>=60", r, "egfr", 60, ">=")
r = ckd_epi(4.0, 70, "female")
check("CKD-EPI ESRD stage=G5", r, "ckd_stage", "G5", "==")

print("=== ANION GAP ===")
r = anion_gap(140, 100, 24)
check("AG normal=16", r, "anion_gap", 16, "==")
r = anion_gap(140, 90, 12)
check("AG elevated=38", r, "anion_gap", 38, "==")

print("=== CORRECTED CALCIUM ===")
r = corrected_calcium(8.0, 2.5)
check("Corrected Ca=9.2", r, "corrected_calcium", 9.2, "==")

print("=== MELD ===")
r = meld(1.0, 1.0, 1.0)
check("MELD min=6", r, "score", 6, "==")
r = meld(10.0, 3.5, 4.0, dialysis=True)
check("MELD high>=30", r, "score", 30, ">=")

print("=== CHILD-PUGH ===")
r = child_pugh(1.5, 4.0, 2, "none", 0)
check("Child-Pugh A", r, "class_label", "A", "==")
r = child_pugh(5.0, 2.0, 8, "moderate_severe", 3)
check("Child-Pugh C", r, "class_label", "C", "==")

print(f"\n{'='*40}")
print(f"clinical scoring Results: {PASS} PASSED / {PASS+FAIL} total")
if FAIL: print(f"  FAILED: {FAIL}")
else: print("  ALL TESTS PASSED ok")
