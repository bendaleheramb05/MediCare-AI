"""MediCare AI Flask application.

Educational/demo application only. It is not a medical device or diagnostic system.
"""
import json
import os
from datetime import datetime
from functools import wraps

from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import database as db
from pdf_report import build_pdf
from predictor import DISCLAIMER, SYMPTOMS, get_metrics, predict

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("MEDICARE_SECRET_KEY", "dev-only-change-me"),
    MAX_CONTENT_LENGTH=1 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("MEDICARE_HTTPS", "0") == "1",
)


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if "patient_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapper


def current_patient():
    return db.get_patient(session["patient_id"])


def pretty_date(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return iso


app.jinja_env.filters["prettydate"] = pretty_date


@app.context_processor
def inject_globals():
    return {
        "disclaimer": DISCLAIMER,
        "patient_name": session.get("patient_name"),
        "year": datetime.now().year,
    }


@app.route("/")
def index():
    return render_template("index.html", metrics=get_metrics())


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        f = request.form
        full_name = f.get("full_name", "").strip()
        email = f.get("email", "").strip().lower()
        password = f.get("password", "")
        confirm = f.get("confirm_password", "")

        try:
            age = int(f.get("age", ""))
        except ValueError:
            age = 0

        if not full_name or len(full_name) > 100:
            flash("Please enter a valid name.", "error")
        elif not email or "@" not in email or len(email) > 254:
            flash("Please enter a valid email address.", "error")
        elif db.get_patient_by_email(email):
            flash("An account with this email already exists.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
        elif password != confirm:
            flash("The two passwords do not match.", "error")
        elif not 1 <= age <= 120:
            flash("Age must be between 1 and 120 years.", "error")
        elif f.get("gender") not in {"Male", "Female", "Other"}:
            flash("Please select a valid gender.", "error")
        else:
            pid = db.create_patient(
                full_name, email, generate_password_hash(password), age,
                f["gender"], f.get("phone", "").strip()[:30]
            )
            session.clear()
            session["patient_id"], session["patient_name"] = pid, full_name
            flash("Account created successfully. Welcome to MediCare AI!", "success")
            return redirect(url_for("dashboard"))
        return render_template("register.html", form=f)
    return render_template("register.html", form={})


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        p = db.get_patient_by_email(email)
        if p and check_password_hash(p["password_hash"], password):
            session.clear()
            session["patient_id"], session["patient_name"] = p["id"], p["full_name"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    p = current_patient()
    if not p:
        session.clear()
        return redirect(url_for("login"))
    history = db.get_history(p["id"], limit=10)
    trend = [{"date": pretty_date(r["created_at"])[:6], "score": r["risk_score"]}
             for r in reversed(history)]
    return render_template("dashboard.html", patient=p, history=history,
                           latest=history[0] if history else None,
                           trend=trend, metrics=get_metrics())


# Server-side bounds are deliberately conservative. They are input validation,
# not medical diagnosis thresholds.
RANGES = {
    "age": (1, 120), "height": (50, 250), "weight": (10, 300),
    "systolic_bp": (70, 260), "diastolic_bp": (40, 160),
    "blood_sugar": (20, 700), "cholesterol": (50, 600), "heart_rate": (30, 220),
}


def _number(name, raw, cast):
    value = cast(raw)
    low, high = RANGES[name]
    if not low <= value <= high:
        raise ValueError(f"{name} is outside the accepted input range")
    return value


def _assessment_from_form(f):
    gender = f.get("gender", "")
    if gender not in {"Male", "Female", "Other"}:
        raise ValueError("Invalid gender")
    cholesterol_category = f.get("cholesterol_category", "").strip()
    glucose_category = f.get("glucose_category", "").strip()
    if cholesterol_category not in {"", "1", "2", "3"}:
        raise ValueError("Please select a valid cholesterol category")
    if glucose_category not in {"", "1", "2", "3"}:
        raise ValueError("Please select a valid blood sugar category")
    category_labels = {"1": "Normal", "2": "Above normal", "3": "Well above normal"}
    bp_unknown = f.get("bp_unknown") == "1"
    systolic_raw = f.get("systolic_bp", "").strip()
    diastolic_raw = f.get("diastolic_bp", "").strip()
    if bp_unknown:
        systolic_bp = diastolic_bp = None
    else:
        if not systolic_raw or not diastolic_raw:
            raise ValueError("Enter both blood-pressure values or select 'I don't know my blood pressure'")
        systolic_bp = _number("systolic_bp", systolic_raw, int)
        diastolic_bp = _number("diastolic_bp", diastolic_raw, int)
        if diastolic_bp >= systolic_bp:
            raise ValueError("Diastolic blood pressure must be less than systolic blood pressure")

    return {
        "age": _number("age", f.get("age", ""), int),
        "gender": gender,
        "height": _number("height", f.get("height", ""), float),
        "weight": _number("weight", f.get("weight", ""), float),
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "bp_unknown": bp_unknown,
        "blood_sugar": None,
        "cholesterol": None,
        "cholesterol_category": int(cholesterol_category) if cholesterol_category else None,
        "glucose_category": int(glucose_category) if glucose_category else None,
        "glucose_label": category_labels.get(glucose_category, "Not provided"),
        "cholesterol_label": category_labels.get(cholesterol_category, "Not provided"),
        "heart_rate": None,
        "sugar_fasting": 1 if f.get("sugar_fasting", "1") == "1" else 0,
        "smoking": 1 if f.get("smoking") else 0,
        "alcohol": 1 if f.get("alcohol") else 0,
        "physical_activity": 1 if f.get("physical_activity") else 0,
        "family_history": 1 if f.get("family_history") else 0,
        "symptoms": [s for s in f.getlist("symptoms") if s in SYMPTOMS],
        "dataset_id": f.get("dataset_id", ""),
        "target": f.get("target", ""),
    }


@app.route("/assess", methods=["GET", "POST"])
@login_required
def assess():
    p = current_patient()
    if request.method == "POST":
        try:
            form = _assessment_from_form(request.form)
            result = predict(form)
        except (ValueError, KeyError, TypeError) as exc:
            flash("Please check all fields and enter valid values.", "error")
            return render_template("assess.html", patient=p, symptoms=SYMPTOMS,
                                   form=request.form, error=str(exc))
        aid = db.save_assessment(p["id"], form, result)
        return redirect(url_for("result", aid=aid))
    return render_template("assess.html", patient=p, symptoms=SYMPTOMS,
                           form={})


@app.route("/result/<int:aid>")
@login_required
def result(aid):
    row = db.get_assessment(aid, session["patient_id"])
    if not row:
        flash("Report not found.", "error")
        return redirect(url_for("dashboard"))
    return render_template("result.html", patient=current_patient(), row=row,
                           r=json.loads(row["result_json"]), metrics=get_metrics(), aid=aid)


@app.route("/history")
@login_required
def history():
    return render_template("history.html", patient=current_patient(),
                           history=db.get_history(session["patient_id"]))


@app.route("/delete/<int:aid>", methods=["POST"])
@login_required
def delete(aid):
    db.delete_assessment(aid, session["patient_id"])
    flash("Report deleted.", "success")
    return redirect(url_for("history"))


@app.route("/download/<int:aid>")
@login_required
def download(aid):
    row = db.get_assessment(aid, session["patient_id"])
    if not row:
        flash("Report not found.", "error")
        return redirect(url_for("dashboard"))
    p = current_patient()
    form = {k: row[k] for k in ("age", "gender", "height", "weight", "systolic_bp",
                                "diastolic_bp", "blood_sugar", "cholesterol", "heart_rate",
                                "smoking", "alcohol", "physical_activity", "family_history")}
    form["assessment_id"] = aid
    buf = build_pdf(p, form, json.loads(row["result_json"]),
                    created_at=pretty_date(row["created_at"]), metrics=get_metrics())
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"MediCareAI_Report_{aid}.pdf")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    try:
        data = request.get_json(silent=False)
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        # Reuse the same validation as the web form.
        class FormAdapter:
            def get(self, key, default=None):
                return data.get(key, default)
            def getlist(self, key):
                value = data.get(key, [])
                return value if isinstance(value, list) else []
        result = predict(_assessment_from_form(FormAdapter()))
        return jsonify(result)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return jsonify({"error": "Invalid assessment payload"}), 400


@app.route("/api/metrics")
def api_metrics():
    return jsonify(get_metrics())


if __name__ == "__main__":
    db.init_db()
    # Debug is intentionally disabled by default because the Flask debugger can expose
    # sensitive application internals. Set MEDICARE_DEBUG=1 only for local development.
    debug = os.environ.get("MEDICARE_DEBUG", "0") == "1"
    app.run(debug=debug, host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
