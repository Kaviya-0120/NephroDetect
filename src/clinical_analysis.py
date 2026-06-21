"""Clinical analysis helpers for NephroDetect result page."""


def _check_bp(bp: float) -> tuple[bool, str]:
    if bp < 90:
        return False, "Lower than normal"
    if bp <= 120:
        return True, "Normal"
    if bp <= 139:
        return False, "Elevated — pre-hypertension range"
    return False, "Higher than normal"


def _check_sg(sg: float) -> tuple[bool, str]:
    if 1.010 <= sg <= 1.025:
        return True, "Normal"
    if sg < 1.010:
        return False, "Below normal concentration"
    return False, "Above normal concentration"


def _check_al(al: float) -> tuple[bool, str]:
    if al == 0:
        return True, "Normal"
    return False, f"Level {int(al)} — protein leakage detected"


def _check_urine_sugar(su: int) -> tuple[bool, str]:
    if su == 0:
        return True, "Normal"
    return False, f"Level {su} — glucose detected in urine"


def _check_blood_sugar(sugar: float) -> tuple[bool, str]:
    if 70 <= sugar <= 100:
        return True, "Normal fasting range"
    if sugar < 140:
        return True, "Within acceptable post-meal range"
    if sugar <= 200:
        return False, "Elevated — may indicate poor glucose control"
    return False, "High — diabetes-related kidney risk"


def _check_bgr(bgr: float) -> tuple[bool, str]:
    if bgr <= 140:
        return True, "Normal"
    if bgr <= 200:
        return False, "Elevated random glucose"
    return False, "High random glucose"


def _check_sc(sc: float) -> tuple[bool, str]:
    if 0.6 <= sc <= 1.2:
        return True, "Normal"
    if sc < 0.6:
        return False, "Below typical range"
    return False, "Elevated — reduced kidney filtration possible"


def _check_hemo(hemo: float) -> tuple[bool, str]:
    if 12 <= hemo <= 17:
        return True, "Normal"
    if hemo < 12:
        return False, "Low — possible anemia"
    return False, "Above typical range"


def build_clinical_analysis(inputs: dict, is_ckd: bool) -> dict:
    """Build structured clinical analysis for the result page."""
    checks = [
        ("Blood Pressure", f"{inputs['bp']} mmHg", *_check_bp(inputs["bp"])),
        ("Albumin", f"Level {int(inputs['al'])}", *_check_al(inputs["al"])),
        ("Urine Sugar", f"Level {int(inputs['su'])}", *_check_urine_sugar(int(inputs["su"]))),
        ("Blood Sugar", f"{inputs['sugar']} mg/dL", *_check_blood_sugar(inputs["sugar"])),
        ("Blood Glucose Random", f"{inputs['bgr']} mg/dL", *_check_bgr(inputs["bgr"])),
        ("Serum Creatinine", f"{inputs['sc']} mg/dL", *_check_sc(inputs["sc"])),
        ("Hemoglobin", f"{inputs['hemo']} g/dL", *_check_hemo(inputs["hemo"])),
    ]

    abnormal = [
        {"label": label, "value": value, "note": note}
        for label, value, ok, note in checks
        if not ok
    ]
    healthy = [
        f"{label}: {note}"
        for label, _, ok, note in checks
        if ok
    ]

    prevention = [
        "Maintain proper hydration",
        "Exercise regularly",
        "Reduce excess salt intake",
        "Schedule regular kidney screening",
        "Maintain healthy blood sugar levels",
        "Avoid smoking and alcohol",
        "Avoid overusing painkillers",
    ]

    if is_ckd:
        reasons = []
        bp_ok, bp_note = _check_bp(inputs["bp"])
        sc_ok, sc_note = _check_sc(inputs["sc"])
        al_ok, al_note = _check_al(inputs["al"])
        hemo_ok, hemo_note = _check_hemo(inputs["hemo"])
        su_ok, su_note = _check_urine_sugar(int(inputs["su"]))

        if not bp_ok:
            reasons.append("High blood pressure can damage kidney blood vessels.")
        if not sc_ok:
            reasons.append("Elevated creatinine may indicate reduced kidney filtration.")
        if not al_ok:
            reasons.append("Protein in urine may indicate kidney filter damage.")
        if not hemo_ok:
            reasons.append("Low hemoglobin may indicate advanced CKD-related anemia.")
        if not su_ok:
            reasons.append("Glucose in urine may indicate kidney dysfunction or poor sugar control.")

        if not reasons:
            reasons.append("Combined clinical markers suggest possible kidney stress requiring evaluation.")

        actions = [
            "Consult a nephrologist immediately",
            "Monitor blood pressure regularly",
            "Reduce salt intake",
            "Maintain proper hydration",
            "Control diabetes and blood sugar levels",
            "Avoid overusing painkillers",
            "Follow a kidney-friendly diet",
        ]
        consult = [
            "Persistent swelling in legs or ankles",
            "Uncontrolled blood pressure",
            "Fatigue or loss of appetite",
            "Changes in urination patterns",
            "Any abnormal repeat lab results",
        ]
        intro = (
            "The following parameters appear abnormal and may indicate kidney dysfunction:"
            if abnormal
            else "Clinical markers suggest further kidney evaluation is recommended."
        )
    else:
        reasons = []
        actions = [
            "Maintain a balanced, kidney-friendly diet",
            "Stay physically active",
            "Monitor blood pressure periodically",
            "Get annual health checkups",
        ]
        consult = [
            "Blood pressure consistently above 140/90",
            "Family history of kidney disease",
            "Diabetes or high blood sugar",
            "Persistent fatigue or swelling",
        ]
        intro = (
            "Most entered values appear within healthy ranges."
            if len(abnormal) <= 1
            else "Some values need attention, but overall screening did not indicate CKD."
        )

    return {
        "abnormal_findings": abnormal,
        "healthy_indicators": healthy,
        "analysis_intro": intro,
        "possible_reasons": reasons,
        "recommended_actions": actions,
        "prevention_tips": prevention,
        "consult_when": consult,
    }
