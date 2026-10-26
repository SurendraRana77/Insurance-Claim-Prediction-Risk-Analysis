-- SQLite Schema for Insurance Claim Prediction & Risk Analysis System

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_version TEXT NOT NULL,
    customer_age INTEGER,
    driving_experience INTEGER,
    annual_mileage INTEGER,
    credit_score REAL,
    vehicle_year TEXT,
    vehicle_type TEXT,
    past_accidents INTEGER,
    speeding_violations INTEGER,
    duis INTEGER,
    input_payload_json TEXT NOT NULL,
    predicted_probability REAL NOT NULL,
    classification_result INTEGER NOT NULL,
    risk_category TEXT NOT NULL,
    decision_threshold REAL NOT NULL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_category);

CREATE TABLE IF NOT EXISTS system_settings (
    setting_key TEXT PRIMARY KEY,
    setting_value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    train_samples INTEGER NOT NULL,
    test_samples INTEGER NOT NULL,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    roc_auc REAL,
    metrics_json TEXT
);

-- Seed initial default configuration settings
INSERT OR IGNORE INTO system_settings (setting_key, setting_value) VALUES 
('risk_threshold_low', '0.18'),
('risk_threshold_moderate', '0.32'),
('risk_threshold_elevated', '0.45'),
('default_decision_threshold', '0.30'),
('active_dataset', 'sample_insurance_data.csv'),
('active_target_col', 'claim_occurred');
