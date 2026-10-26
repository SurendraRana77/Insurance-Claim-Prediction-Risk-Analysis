import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get('SECRET_KEY', 'academic-insurance-risk-secret-key-2026')
    DATA_DIR = os.environ.get('DATA_DIR', str(BASE_DIR / 'data'))
    ARTIFACTS_DIR = os.environ.get('ARTIFACTS_DIR', str(BASE_DIR / 'artifacts'))
    
    # SQLite Database Configuration (supports persistent disk mount via DB_DIR or DB_PATH)
    DB_PATH = os.environ.get('DB_PATH', str(BASE_DIR / 'database' / 'insurance_app.db'))
    
    # Default Benchmark Dataset
    DEFAULT_DATASET_NAME = 'sample_insurance_data.csv'
    
    # Upload limits: 16 MB max upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Default Risk Category Cutoffs (Calibrated against ~23% empirical claim rate)
    DEFAULT_RISK_THRESHOLDS = {
        'low': 0.18,       # p < 0.18 -> Low Risk
        'moderate': 0.32,  # 0.18 <= p < 0.32 -> Moderate Risk
        'elevated': 0.45,  # 0.32 <= p < 0.45 -> Elevated Risk
        'high': 1.00       # p >= 0.45 -> High Risk
    }
    
    DEFAULT_DECISION_THRESHOLD = 0.30
    
    # Supported Model Candidates
    MODEL_CANDIDATES = ['logistic_regression', 'random_forest']
