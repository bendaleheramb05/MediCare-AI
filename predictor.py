"""Prediction and explanation layer for MediCare AI.

This module produces an educational screening score. It must not be interpreted as
an individual diagnosis or as a calibrated clinical probability.
"""
import json
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "risk_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "data", "cardio_train.csv")

SYMPTOMS = {
    "chest_pain": ("Chest pain / tightness", 14),
    "breathlessness": ("Shortness of breath", 11),
    "palpitations": ("Palpitations / irregular heartbeat", 8),
    "dizziness": ("Dizziness or fainting", 7),
    "fatigue": ("Unusual fatigue / weakness", 6),
    "headache": ("Frequent headaches", 5),
    "swelling": ("Swelling in legs or ankles", 7),
    "blurred_vision": ("Blurred vision", 5),
    "excess_thirst": ("Excessive thirst / frequent urination", 8),
    "numbness": ("Numbness or tingling in hands/feet", 5),
}

_bundle = None
_metrics = None


def load_model():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("ML model not found. Run: python ml/train_model.py")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_metrics():
    global _metrics
    if _metrics is None:
        if not os.path.exists(METRICS_PATH):
            return {"best_model": "Not trained", "records": 0, "accuracy": 0, "roc_auc": 0,
                    "precision": 0, "recall": 0, "f1_score": 0, "dataset": "Not trained",
                    "dataset_type": "unknown"}
        with open(METRICS_PATH, encoding="utf-8") as f:
            _metrics = json.load(f)
    return _metrics


def _safe_float(value, default=None):
    try:
        value = float(value)
        return default if pd.isna(value) else value
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    try:
        value = int(float(value))
        return default if pd.isna(value) else value
    except (TypeError, ValueError):
        return default


def _category_label(value, labels):
    category = _safe_int(value)
    return labels.get(category, "Unavailable")


def get_dataset_patient_rows(limit=None):
    """Return records from the real cardio_train.csv dataset."""
    if not os.path.exists(DATASET_PATH):
        return []
    try:
        df = pd.read_csv(DATASET_PATH, sep=";")
    except Exception:
        return []
    if df.empty:
        return []
    required = {"id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
                "cholesterol", "gluc", "smoke", "alco", "active", "cardio"}
    if not required.issubset(df.columns):
        return []

    rows = []
    for idx, row in df.iterrows():
        age = _safe_int(row.get("age"))
        if age is not None and age > 120:
            age = round(age / 365.25)
        gender_value = _safe_int(row.get("gender"))
        gender = {1: "Female", 2: "Male"}.get(gender_value, "Other")
        height = _safe_float(row.get("height"))
        weight = _safe_float(row.get("weight"))
        bmi = None
        if height and weight and height > 0:
            bmi = round(weight / (height / 100) ** 2, 1)

        record = {
            "id": _safe_int(row.get("id"), idx),
            "age": age,
            "gender": gender,
            "height": round(height, 1) if height is not None else "",
            "weight": round(weight, 1) if weight is not None else "",
            "bmi": bmi if bmi is not None else "",
            "systolic_bp": _safe_int(row.get("ap_hi")),
            "diastolic_bp": _safe_int(row.get("ap_lo")),
            "cholesterol_category": _safe_int(row.get("cholesterol")),
            "cholesterol_label": _category_label(row.get("cholesterol"), {1: "Normal", 2: "Above normal", 3: "Well above normal"}),
            "glucose_category": _safe_int(row.get("gluc")),
            "glucose_label": _category_label(row.get("gluc"), {1: "Normal", 2: "Above normal", 3: "Well above normal"}),
            "smoking": _safe_int(row.get("smoke")),
            "alcohol": _safe_int(row.get("alco")),
            "physical_activity": _safe_int(row.get("active")),
            "target": _safe_int(row.get("cardio")),
        }
        rows.append(record)

    if limit is not None:
        return rows[:limit]
    return rows


def search_dataset_patient_rows(query, limit=25):
    """Search every loaded CSV record and return a small result window."""
    query = str(query or "").strip().lower()
    rows = get_dataset_patient_rows()
    if not query:
        return [{"index": index, **row} for index, row in enumerate(rows[:limit])]

    matches = []
    for index, row in enumerate(rows):
        haystack = " ".join(str(row.get(key, "")) for key in (
            "id", "age", "gender", "systolic_bp", "diastolic_bp", "target"
        )).lower()
        if query in haystack:
            matches.append({"index": index, **row})
            if len(matches) >= limit:
                break
    return matches


def dataset_row_to_form(row):
    """Map a CSV row into the form structure used by the assessment page."""
    form = {
        "age": row.get("age", "") if row.get("age") not in (None, "") else "",
        "gender": row.get("gender", "Other"),
        "height": row.get("height", ""),
        "weight": row.get("weight", ""),
        "bmi": row.get("bmi", ""),
        "systolic_bp": row.get("systolic_bp", ""),
        "diastolic_bp": row.get("diastolic_bp", ""),
        "blood_sugar": "",
        "glucose_category": row.get("glucose_category", ""),
        "glucose_label": row.get("glucose_label", "Unavailable"),
        "cholesterol": "",
        "cholesterol_category": row.get("cholesterol_category", ""),
        "cholesterol_label": row.get("cholesterol_label", "Unavailable"),
        "heart_rate": "",
        "smoking": 1 if row.get("smoking") else 0,
        "alcohol": 1 if row.get("alcohol") else 0,
        "physical_activity": 1 if row.get("physical_activity", 1) else 0,
        "family_history": 0,
        "symptoms": [],
        "dataset_id": row.get("id", ""),
        "target": row.get("target", ""),
    }
    return form


def calculate_bmi(weight_kg, height_cm):
    return round(weight_kg / (height_cm / 100) ** 2, 1)


def bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    if bmi < 25: return "Normal"
    if bmi < 30: return "Overweight"
    return "Obese"


def bp_category(sys_bp, dia_bp):
    if sys_bp is None or dia_bp is None:
        return "Not provided"
    if sys_bp > 180 or dia_bp > 120:
        return "Severe hypertension range"
    if sys_bp >= 140 or dia_bp >= 90:
        return "Hypertension Stage 2 range"
    if sys_bp >= 130 or dia_bp >= 80:
        return "Hypertension Stage 1 range"
    if 120 <= sys_bp <= 129 and dia_bp < 80:
        return "Elevated range"
    if sys_bp < 90 or dia_bp < 60:
        return "Below usual adult range"
    return "Normal range"


def sugar_category(sugar, fasting=True):
    if fasting:
        if sugar < 70: return "Low glucose range"
        if sugar < 100: return "Normal fasting range"
        if sugar < 126: return "Impaired fasting glucose range"
        return "Diabetes diagnostic threshold reached"
    if sugar < 140: return "Below 2-hour diagnostic threshold"
    if sugar < 200: return "Above usual 2-hour target range"
    return "Diabetes diagnostic threshold may be reached if criteria are met"


def pulse_category(hr):
    if hr in (None, ""):
        return "Unavailable in dataset"
    if hr < 60: return "Below usual resting range"
    if hr <= 100: return "Typical resting range"
    return "Above usual resting range"


def cholesterol_level(chol):
    if chol < 200: return 1, "Desirable"
    if chol < 240: return 2, "Borderline high"
    return 3, "High"


def glucose_level(sugar):
    if sugar < 100: return 1, "Normal"
    if sugar < 126: return 2, "Above normal"
    return 3, "High"


def _urgent_flags(data, bp_cat, sugar_cat, score):
    # This is an emergency-safety nudge, not a diagnosis.
    symptoms = set(data.get("symptoms", []))
    return (
        bp_cat == "Severe hypertension range" or
        "chest_pain" in symptoms or
        "breathlessness" in symptoms or
        "fainting" in symptoms or
        score >= 75
    )


def detect_health_factors(data, bmi, bp_cat, sugar_cat, chol_code, chol_text, selected):
    """Build separate measurable and healthcare-discussion factor lists."""
    key_factors = []
    doctor_factors = []
    if bp_cat not in {"Not provided", "Normal range"}:
        key_factors.append(("Blood pressure status", f"Entered reading: {data['systolic_bp']}/{data['diastolic_bp']} mmHg ({bp_cat}).", "high"))
        doctor_factors.append(("Persistently elevated blood pressure", "Repeated elevated readings should be discussed with a healthcare professional.", "high"))
    if chol_code in {2, 3}:
        key_factors.append(("Cholesterol category", chol_text, "medium"))
        doctor_factors.append(("Above-normal cholesterol", "The selected category may be worth discussing with a healthcare professional.", "medium"))
    if _safe_int(data.get("glucose_category")) in {2, 3}:
        key_factors.append(("Glucose category", sugar_cat, "medium"))
        doctor_factors.append(("Above-normal blood glucose", "The selected category may be worth discussing with a healthcare professional; this does not diagnose diabetes.", "medium"))
    if bmi >= 25:
        bmi_text = "overweight range" if bmi < 30 else "obesity range"
        key_factors.append(("BMI status", f"Calculated BMI: {bmi} ({bmi_text}).", "medium"))
        if bmi >= 30:
            doctor_factors.append(("High BMI", "A BMI in this range may be worth discussing with a healthcare professional; BMI does not diagnose a condition.", "medium"))
    if data.get("smoking"):
        key_factors.append(("Smoking status", "Smoking was reported.", "high"))
    if data.get("alcohol"):
        key_factors.append(("Alcohol status", "Alcohol use was reported.", "medium"))
    if not data.get("physical_activity"):
        key_factors.append(("Physical activity status", "Low physical activity was reported.", "medium"))
    return {
        "positive": key_factors,
        "key_factors": key_factors,
        "doctor_factors": doctor_factors,
        "conditions": [(name, description) for name, description, _ in doctor_factors],
        "bp_elevated": bp_cat not in {"Normal range", "Not provided"},
        "glucose_high": _safe_int(data.get("glucose_category")) in {2, 3},
        "cholesterol_high": chol_code in {2, 3},
        "bmi_concern": bmi >= 25,
        "has_positive": bool(key_factors),
    }


def predict(data):
    bmi = calculate_bmi(float(data["weight"]), float(data["height"]))
    chol_code = _safe_int(data.get("cholesterol_category"))
    if chol_code is None and data.get("cholesterol") is not None:
        chol_code, chol_text = cholesterol_level(float(data["cholesterol"]))
    elif chol_code is not None:
        chol_text = {1: "Desirable category", 2: "Above normal category", 3: "Well above normal category"}.get(chol_code, "Unavailable")
    else:
        chol_text = "Not provided"
    gluc_code = _safe_int(data.get("glucose_category"))
    if gluc_code is None and data.get("blood_sugar") is not None:
        gluc_code, _ = glucose_level(float(data["blood_sugar"]))
    bundle = load_model()

    row = pd.DataFrame([{
        "age": data["age"],
        "gender": 1 if str(data["gender"]).lower().startswith("m") else 0,
        "bmi": bmi, "ap_hi": data["systolic_bp"], "ap_lo": data["diastolic_bp"],
        "cholesterol": chol_code, "glucose": gluc_code,
        "smoke": data["smoking"], "alcohol": data["alcohol"], "active": data["physical_activity"],
    }])[bundle["features"]]

    model_probability = float(bundle["model"].predict_proba(row)[0][1]) * 100

    selected = data.get("symptoms", [])
    symptom_score = min(sum(SYMPTOMS[s][1] for s in selected if s in SYMPTOMS), 35)
    # This is intentionally called a score, not a medical probability.
    screening_score = round(min(model_probability * 0.75 + symptom_score * 0.5, 99), 1)

    if screening_score < 30: level, color = "Lower screening score", "green"
    elif screening_score < 60: level, color = "Moderate screening score", "amber"
    else: level, color = "Higher screening score", "red"

    bp_cat = bp_category(data["systolic_bp"], data["diastolic_bp"])
    if gluc_code is not None:
        sugar_cat = {1: "Normal glucose category", 2: "Above normal glucose category", 3: "Well above normal glucose category"}.get(gluc_code, "Unavailable")
    else:
        sugar_cat = "Not provided"
    bmi_cat = bmi_category(bmi)
    hr_cat = pulse_category(data.get("heart_rate"))
    detected = detect_health_factors(data, bmi, bp_cat, sugar_cat, chol_code, chol_text, selected)

    urgent = _urgent_flags(data, bp_cat, sugar_cat, screening_score)
    return {
        "bmi": bmi, "bmi_category": bmi_cat, "bp_category": bp_cat,
        "sugar_category": sugar_cat, "cholesterol_category": chol_text, "pulse_category": hr_cat,
        "glucose_label": data.get("glucose_label") or sugar_cat,
        "cholesterol_label": data.get("cholesterol_label") or chol_text,
        "heart_rate_display": "Unavailable in dataset" if data.get("heart_rate") in (None, "") else f"{data['heart_rate']} bpm",
        "ml_probability": round(model_probability, 1), "symptom_score": symptom_score,
        "risk_score": screening_score, "risk_level": level, "risk_color": color,
        "score_label": "Overall Health Screening Score",
        "factors": detected["key_factors"],
        "symptoms": [SYMPTOMS[s][0] for s in selected if s in SYMPTOMS],
        "conditions": detected["conditions"], "urgent": urgent,
        "recommendations": build_recommendations(data, bmi, bp_cat, sugar_cat, level, detected),
        "exercises": build_exercise_plan(data, level),
    }


def build_recommendations(data, bmi, bp_cat, sugar_cat, level, detected):
    bp_elevated = detected["bp_elevated"]
    glucose_high = detected["glucose_high"]
    cholesterol_high = detected["cholesterol_high"]
    bmi_concern = detected["bmi_concern"]
    smoking = bool(data.get("smoking"))
    alcohol = bool(data.get("alcohol"))
    why_guidance = []
    if bp_elevated:
        why_guidance.append("Your blood-pressure reading is elevated, so guidance focuses on lower-sodium choices and regular activity.")
    if glucose_high:
        why_guidance.append("Your glucose category is above normal, so guidance focuses on high-fiber foods and limiting added sugar.")
    if cholesterol_high:
        why_guidance.append("Your cholesterol category is above normal, so guidance focuses on fiber-rich foods and limiting saturated fat.")
    if bmi_concern:
        bmi_range = "overweight range" if bmi < 30 else "obesity range"
        why_guidance.append(f"Your BMI is in the {bmi_range}, so guidance focuses on balanced portions and nutrient-dense foods.")
    if smoking:
        why_guidance.append("Smoking was reported, so cessation support is included.")
    if alcohol:
        why_guidance.append("Alcohol use was reported, so moderation guidance is included.")

    offset = int(data.get("age", 0)) + int(round(float(data.get("weight", 0)))) + int(round(bmi * 10))
    fruit_pools = {
        "bp": ["Orange", "Guava", "Watermelon", "Pear"],
        "glucose": ["Berries", "Apple", "Pear", "Guava"],
        "cholesterol": ["Apple", "Pear", "Berries", "Orange"],
        "bmi": ["Berries", "Orange", "Papaya", "Guava"],
        "balanced": ["Papaya", "Apple", "Guava", "Orange", "Pear", "Berries"],
    }
    vegetable_pools = {
        "bp": ["Cucumber", "Tomato", "Cauliflower", "Spinach"],
        "glucose": ["Spinach", "Broccoli", "Cauliflower", "Cucumber"],
        "cholesterol": ["Broccoli", "Carrot", "Beans", "Spinach"],
        "bmi": ["Cauliflower", "Cucumber", "Carrot", "Tomato"],
        "balanced": ["Tomato", "Broccoli", "Spinach", "Carrot", "Cucumber", "Cauliflower"],
    }
    active_keys = [key for key, active in (("bp", bp_elevated), ("glucose", glucose_high), ("cholesterol", cholesterol_high), ("bmi", bmi_concern)) if active]
    active_keys = active_keys or ["balanced"]
    fruit_candidates = sum((fruit_pools[key] for key in active_keys), [])
    vegetable_candidates = sum((vegetable_pools[key] for key in active_keys), [])

    def select(values, count=4):
        rotated = values[offset % len(values):] + values[:offset % len(values)]
        return list(dict.fromkeys(rotated))[:count]

    recommended_fruits = select(fruit_candidates)
    recommended_vegetables = select(vegetable_candidates)
    if glucose_high or cholesterol_high:
        healthy_protein_other = ["Oats", "Lentils", "Beans", "Whole grains"]
    elif bmi_concern:
        healthy_protein_other = ["Dal / pulses", "Chickpeas", "Peas"]
    elif bp_elevated:
        healthy_protein_other = ["Dal / pulses", "Beans", "Whole grains"]
    else:
        healthy_protein_other = ["Dal / pulses", "Beans", "Low-fat dairy if suitable"]
    if cholesterol_high or bmi_concern:
        healthy_protein_other.append("Nuts or seeds in appropriate portions")

    diet_guidance = []
    if bp_elevated:
        diet_guidance.append("Choose lower-sodium preparations with vegetables, fruits, pulses and whole grains.")
    if glucose_high:
        diet_guidance.append("Choose high-fiber, minimally processed foods and whole fruit rather than juice.")
    if cholesterol_high:
        diet_guidance.append("Choose fiber-rich foods such as oats, beans, lentils and vegetables; limit saturated fat.")
    if bmi_concern:
        diet_guidance.append("Use portion awareness with balanced, nutrient-dense meals.")
    if not diet_guidance:
        diet_guidance.append("Your assessment did not identify a specific factor requiring targeted dietary guidance. Focus on balanced eating and regular activity.")

    foods_to_limit = []
    if glucose_high:
        foods_to_limit.extend(["Sugary drinks", "Sweets", "Highly refined carbohydrates"])
    if cholesterol_high:
        foods_to_limit.extend(["Foods high in saturated or trans fats", "Deep-fried foods", "Highly processed foods"])
    if bp_elevated:
        foods_to_limit.extend(["High-sodium foods", "Excessively salty packaged foods"])
    if bmi_concern:
        foods_to_limit.extend(["Sugary drinks", "Large portions of energy-dense foods"])
    foods_to_limit = list(dict.fromkeys(foods_to_limit)) or ["Highly processed foods"]
    lifestyle = []
    if bp_elevated:
        lifestyle.append("Monitor repeated BP readings and discuss persistent elevation with a healthcare professional.")
    if bmi_concern:
        lifestyle.append("Build regular activity gradually and discuss an appropriate plan with a qualified professional.")
    if smoking:
        lifestyle.append("Consider professional smoking-cessation support.")
    if alcohol:
        lifestyle.append("Consider moderating alcohol intake and discuss personal needs with a healthcare professional.")
    if not lifestyle:
        lifestyle.append("Aim for regular activity, adequate sleep and a balanced pattern of minimally processed foods.")
    lifestyle.append("Do not start, stop or change medicines based on this application.")
    monitor = ["Use recent, appropriately measured health values and discuss persistent concerns with a healthcare professional."]
    if data.get("symptoms"):
        monitor.insert(0, "Because symptoms were reported, seek appropriate medical evaluation if they persist, worsen or feel urgent.")
    if glucose_high:
        monitor.insert(0, "Discuss the above-normal glucose category; this screening result does not diagnose diabetes.")
    if cholesterol_high:
        monitor.insert(0, "Discuss the above-normal cholesterol category with a healthcare professional.")
    return {"why_guidance": why_guidance or ["Your assessment did not identify a specific factor requiring targeted guidance. Balanced preventive health guidance is shown."], "recommended_fruits": recommended_fruits, "recommended_vegetables": recommended_vegetables, "healthy_protein_other": list(dict.fromkeys(healthy_protein_other))[:5], "diet_guidance": diet_guidance, "foods_to_limit": foods_to_limit[:8], "lifestyle": lifestyle, "monitoring": monitor}


def build_exercise_plan(data, level):
    if level == "Higher screening score":
        first = "If you currently have chest pain, severe breathlessness, fainting or another acute symptom, seek urgent medical care rather than exercising."
    else:
        first = "If you are not currently active, start gradually and choose activity appropriate for your age, fitness and health status."
    return [
        ("General activity", first),
        ("Aerobic activity", "For adults who can exercise safely, general WHO guidance is 150–300 minutes of moderate activity per week or an equivalent amount of vigorous activity."),
        ("Strength", "Muscle-strengthening activity on 2 or more days per week can be included when appropriate."),
        ("Stop signs", "Stop exercise and seek appropriate medical help for chest pain, severe breathlessness, fainting or other concerning symptoms."),
    ]


DISCLAIMER = (
    "IMPORTANT MEDICAL DISCLAIMER: MediCare AI is an educational software project that produces "
    "a preliminary screening score from user-entered information. It is NOT a medical diagnosis, "
    "does not establish that a person has or does not have a disease, and is not a substitute for "
    "a qualified healthcare professional. Model outputs are not clinical probabilities unless the "
    "model has been independently validated and calibrated for the intended population. Do not use "
    "this application to choose, start, stop or change treatment. For an emergency, contact local "
    "emergency services or go to the nearest emergency department."
)
