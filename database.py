"""
MediCare AI - SQLite database layer
===================================
Two tables:
  patients     -> login account + basic profile
  assessments  -> one row per health assessment (history)
"""

import json
import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "medicare.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # rows behave like dictionaries
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS patients (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name     TEXT    NOT NULL,
        email         TEXT    NOT NULL UNIQUE,
        password_hash TEXT    NOT NULL,
        age           INTEGER,
        gender        TEXT,
        phone         TEXT,
        created_at    TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS assessments (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id    INTEGER NOT NULL,
        created_at    TEXT    NOT NULL,
        age           INTEGER, gender TEXT,
        height        REAL,    weight REAL,   bmi REAL,
        systolic_bp   INTEGER, diastolic_bp INTEGER,
        blood_sugar   REAL,    cholesterol REAL, heart_rate INTEGER,
        smoking       INTEGER, alcohol INTEGER,
        physical_activity INTEGER, family_history INTEGER,
        symptoms      TEXT,
        risk_score    REAL,    risk_level TEXT,
        ml_probability REAL,
        result_json   TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    );
    """)
    conn.commit()
    conn.close()


# --------------------------- patients --------------------------------------
def create_patient(full_name, email, password_hash, age, gender, phone):
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO patients (full_name, email, password_hash, age, gender,
                                 phone, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (full_name, email.lower().strip(), password_hash, age, gender, phone,
         datetime.now().isoformat(timespec="seconds")))
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid


def get_patient_by_email(email):
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE email = ?",
                       (email.lower().strip(),)).fetchone()
    conn.close()
    return row


def get_patient(pid):
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()
    conn.close()
    return row


# --------------------------- assessments -----------------------------------
def save_assessment(patient_id, form, result):
    conn = get_db()
    cur = conn.execute("""
        INSERT INTO assessments (patient_id, created_at, age, gender, height,
            weight, bmi, systolic_bp, diastolic_bp, blood_sugar, cholesterol,
            heart_rate, smoking, alcohol, physical_activity, family_history,
            symptoms, risk_score, risk_level, ml_probability, result_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (patient_id, datetime.now().isoformat(timespec="seconds"),
         form["age"], form["gender"], form["height"], form["weight"],
         result["bmi"], form["systolic_bp"], form["diastolic_bp"],
         form["blood_sugar"], form["cholesterol"], form["heart_rate"],
         form["smoking"], form["alcohol"], form["physical_activity"],
         form["family_history"], json.dumps(form.get("symptoms", [])),
         result["risk_score"], result["risk_level"], result["ml_probability"],
         json.dumps(result)))
    conn.commit()
    aid = cur.lastrowid
    conn.close()
    return aid


def get_assessment(aid, patient_id):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM assessments WHERE id = ? AND patient_id = ?",
        (aid, patient_id)).fetchone()
    conn.close()
    return row


def get_history(patient_id, limit=50):
    conn = get_db()
    rows = conn.execute(
        """SELECT * FROM assessments WHERE patient_id = ?
           ORDER BY created_at DESC LIMIT ?""", (patient_id, limit)).fetchall()
    conn.close()
    return rows


def delete_assessment(aid, patient_id):
    conn = get_db()
    conn.execute("DELETE FROM assessments WHERE id = ? AND patient_id = ?",
                 (aid, patient_id))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database ready at {DB_PATH}")
