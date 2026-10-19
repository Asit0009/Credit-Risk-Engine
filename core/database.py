import os
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

DEFAULT_DB_PATH = os.path.join("data", "decision_history.db")  # local sqlite database file


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)  # ensure data folder exists
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    # create table if not already created
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_uuid TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                cibil_score INTEGER NOT NULL,
                annual_income REAL NOT NULL,
                age INTEGER NOT NULL,
                employment_type TEXT NOT NULL,
                requested_limit REAL NOT NULL,
                monthly_emi REAL NOT NULL,
                prior_defaults INTEGER NOT NULL,
                dti_ratio REAL NOT NULL,
                limit_ratio REAL NOT NULL,
                risk_probability REAL NOT NULL,
                verdict TEXT NOT NULL,
                threshold REAL NOT NULL,
                model_version TEXT NOT NULL,
                top_factors_json TEXT NOT NULL,
                summary TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON assessments(timestamp DESC);")
        conn.commit()


def save_assessment(assessment_data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> int:
    init_db(db_path)
    assessment_uuid = assessment_data.get("assessment_uuid") or str(uuid.uuid4())
    ts = assessment_data.get("timestamp") or datetime.now(timezone.utc).isoformat()

    top_factors = assessment_data.get("top_factors", [])
    top_factors_json = json.dumps(top_factors) if isinstance(top_factors, list) else str(top_factors)

    # insert record using parameterized sql
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO assessments (
                assessment_uuid, timestamp, cibil_score, annual_income, age,
                employment_type, requested_limit, monthly_emi, prior_defaults,
                dti_ratio, limit_ratio, risk_probability, verdict, threshold,
                model_version, top_factors_json, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assessment_uuid,
            ts,
            int(assessment_data["cibil_score"]),
            float(assessment_data["annual_income"]),
            int(assessment_data["age"]),
            str(assessment_data.get("employment_label", assessment_data.get("employment_type", "Salaried"))),
            float(assessment_data["requested_limit"]),
            float(assessment_data["monthly_emi"]),
            int(assessment_data["prior_defaults"]),
            float(assessment_data.get("dti_ratio", 0.0)),
            float(assessment_data.get("limit_ratio", 0.0)),
            float(assessment_data["risk_probability"]),
            str(assessment_data["verdict"]),
            float(assessment_data.get("threshold", 0.10)),
            str(assessment_data.get("model_version", "LightGBM-v1.0")),
            top_factors_json,
            str(assessment_data.get("summary", ""))
        ))
        conn.commit()
        return cursor.lastrowid


def get_all_assessments(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    # fetch all records newest first
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM assessments ORDER BY id DESC")
        rows = cursor.fetchall()

    results = []
    for r in rows:
        item = dict(r)
        try:
            item["top_factors"] = json.loads(item["top_factors_json"])
        except Exception:
            item["top_factors"] = []
        results.append(item)
    return results


def get_assessment_by_id(assessment_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    # fetch single record by id
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM assessments WHERE id = ?", (assessment_id,))
        row = cursor.fetchone()

    if not row:
        return None

    item = dict(row)
    try:
        item["top_factors"] = json.loads(item["top_factors_json"])
    except Exception:
        item["top_factors"] = []
    return item


def delete_assessment(assessment_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    # delete single record by id
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM assessments WHERE id = ?", (assessment_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_assessments(db_path: str = DEFAULT_DB_PATH) -> int:
    # erase all records
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM assessments")
        conn.commit()
        return cursor.rowcount
