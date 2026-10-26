import os
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager
from config import Config

def get_db_path() -> str:
    path = Path(Config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)

def init_db():
    schema_path = Path(__file__).resolve().parent / 'schema.sql'
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()

@contextmanager
def get_db_connection():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    
    # Auto-initialize schema if system_settings table does not yet exist
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_settings'")
        if not cursor.fetchone():
            schema_path = Path(__file__).resolve().parent / 'schema.sql'
            if schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    conn.executescript(f.read())
                conn.commit()
    except Exception:
        pass

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def get_setting(key: str, default=None):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT setting_value FROM system_settings WHERE setting_key = ?", (key,))
        row = cursor.fetchone()
        return row['setting_value'] if row else default

def set_setting(key: str, value: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO system_settings (setting_key, setting_value, updated_at) 
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value, updated_at = CURRENT_TIMESTAMP""",
            (key, str(value))
        )

def save_prediction(
    model_version: str,
    input_payload: dict,
    predicted_probability: float,
    classification_result: int,
    risk_category: str,
    decision_threshold: float,
    notes: str = ""
) -> int:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO predictions (
                model_version, customer_age, driving_experience, annual_mileage, credit_score,
                vehicle_year, vehicle_type, past_accidents, speeding_violations, duis,
                input_payload_json, predicted_probability, classification_result, risk_category,
                decision_threshold, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_version,
                int(input_payload.get('age', 0)),
                int(input_payload.get('driving_experience', 0)),
                int(input_payload.get('annual_mileage', 0)),
                float(input_payload.get('credit_score', 0.0)),
                str(input_payload.get('vehicle_year', '')),
                str(input_payload.get('vehicle_type', '')),
                int(input_payload.get('past_accidents', 0)),
                int(input_payload.get('speeding_violations', 0)),
                int(input_payload.get('duis', 0)),
                json.dumps(input_payload),
                round(float(predicted_probability), 4),
                int(classification_result),
                str(risk_category),
                round(float(decision_threshold), 4),
                notes
            )
        )
        return cursor.lastrowid

def get_prediction_history(limit: int = 50, offset: int = 0, risk_filter: str = None, search: str = None):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if risk_filter and risk_filter.lower() != 'all':
            query += " AND LOWER(risk_category) = LOWER(?)"
            params.append(risk_filter)
            
        if search:
            query += " AND (notes LIKE ? OR input_payload_json LIKE ?)"
            term = f"%{search}%"
            params.extend([term, term])
            
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_prediction_by_id(pred_id: int):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_prediction_count() -> int:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM predictions")
        return cursor.fetchone()['cnt']

def get_prediction_risk_distribution():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT risk_category, COUNT(*) as count 
               FROM predictions 
               GROUP BY risk_category"""
        )
        rows = cursor.fetchall()
        return {row['risk_category']: row['count'] for row in rows}

def save_model_run(
    model_name: str,
    dataset_name: str,
    train_samples: int,
    test_samples: int,
    accuracy: float,
    precision_score: float,
    recall_score: float,
    f1_score: float,
    roc_auc: float,
    metrics_json: dict
) -> int:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO model_runs (
                model_name, dataset_name, train_samples, test_samples,
                accuracy, precision_score, recall_score, f1_score, roc_auc, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_name,
                dataset_name,
                train_samples,
                test_samples,
                round(accuracy, 4),
                round(precision_score, 4),
                round(recall_score, 4),
                round(f1_score, 4),
                round(roc_auc, 4) if roc_auc is not None else None,
                json.dumps(metrics_json)
            )
        )
        return cursor.lastrowid

def get_latest_model_run():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_runs ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            d = dict(row)
            if d.get('metrics_json'):
                try:
                    d['metrics_data'] = json.loads(d['metrics_json'])
                except Exception:
                    d['metrics_data'] = {}
            return d
        return None
