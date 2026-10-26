import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config
from ml.pipeline import build_model_pipeline
from ml.evaluate import evaluate_model_pipeline
from database.db import save_model_run, set_setting, get_setting, init_db
from services.data_service import IDENTIFIER_KEYWORDS, LEAKAGE_KEYWORDS
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_and_evaluate(
    dataset_path: str = None,
    target_col: str = None,
    preferred_model: str = None
) -> dict:
    # 1. Guarantee database schema is initialized before any operations
    init_db()

    if dataset_path is None:
        active_name = get_setting('active_dataset', Config.DEFAULT_DATASET_NAME)
        dataset_path = str(Path(Config.DATA_DIR) / active_name)
        
    if not os.path.exists(dataset_path):
        # Auto-generate benchmark seed dataset if missing in clean deployment
        if Path(dataset_path).name == Config.DEFAULT_DATASET_NAME:
            from ml.generate_seed_data import generate_benchmark_dataset
            generate_benchmark_dataset(dataset_path)
        else:
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
        
    if target_col is None:
        target_col = get_setting('active_target_col', 'claim_occurred')
        
    df = pd.read_csv(dataset_path)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")
        
    # Clean target
    df = df.dropna(subset=[target_col])
    # Ensure binary target
    unique_targets = df[target_col].unique()
    if len(unique_targets) != 2:
        raise ValueError(f"Target column '{target_col}' must contain exactly 2 unique classes for binary claim prediction. Found: {unique_targets}")
        
    # Map target to 0 and 1 if boolean or string
    target_mapping = None
    if set(unique_targets).issubset({0, 1}):
        y = df[target_col].astype(int)
    else:
        sorted_targets = sorted(list(unique_targets))
        target_mapping = {sorted_targets[0]: 0, sorted_targets[1]: 1}
        y = df[target_col].map(target_mapping).astype(int)
        
    # Feature selection
    feature_cols = []
    dropped_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        lower_col = col.lower()
        if lower_col in IDENTIFIER_KEYWORDS:
            dropped_cols.append((col, 'Identifier column'))
            continue
        if any(leak in lower_col for leak in LEAKAGE_KEYWORDS):
            dropped_cols.append((col, 'Potential target leakage'))
            continue
        feature_cols.append(col)
        
    if len(feature_cols) < 2:
        raise ValueError("Insufficient feature columns remaining after dropping identifiers and leakage features.")
        
    X = df[feature_cols]
    
    # Identify numerical and categorical features
    numerical_cols = []
    categorical_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
            
    # Stratified Train/Test Split (80% train, 20% test held-out)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    models_to_test = ['logistic_regression', 'random_forest']
    
    # 2. Model Selection via 5-Fold Stratified Cross-Validation on X_train ONLY
    cv = StratifiedKFold(n_splits=min(5, max(2, int(y_train.sum()))), shuffle=True, random_state=42)
    cv_scores = {}
    for m_type in models_to_test:
        base_pipe = build_model_pipeline(numerical_cols, categorical_cols, model_type=m_type)
        try:
            scores = cross_val_score(base_pipe, X_train, y_train, cv=cv, scoring='roc_auc')
            cv_scores[m_type] = float(np.mean(scores))
        except Exception:
            cv_scores[m_type] = 0.5
            
    if preferred_model and preferred_model in models_to_test:
        best_model_name = preferred_model
    else:
        # Automatic selection based strictly on training-set cross-validation
        best_model_name = max(models_to_test, key=lambda m: cv_scores[m])
        
    # 3. Probability Calibration on X_train using CalibratedClassifierCV
    cv_calib = min(5, max(2, int(y_train.sum())))
    candidate_results = {}
    fitted_pipelines = {}
    
    decision_thresh = float(get_setting('default_decision_threshold', Config.DEFAULT_DECISION_THRESHOLD))
    
    for m_type in models_to_test:
        base_pipe = build_model_pipeline(numerical_cols, categorical_cols, model_type=m_type)
        calibrated_pipeline = CalibratedClassifierCV(estimator=base_pipe, method='sigmoid', cv=cv_calib)
        calibrated_pipeline.fit(X_train, y_train)
        
        # Evaluate on the completely held-out test set
        metrics = evaluate_model_pipeline(
            calibrated_pipeline, X_test, y_test,
            numerical_cols, categorical_cols,
            decision_threshold=decision_thresh
        )
        metrics['train_cv_roc_auc'] = round(cv_scores[m_type], 4)
        candidate_results[m_type] = metrics
        fitted_pipelines[m_type] = calibrated_pipeline
        
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = candidate_results[best_model_name]
    
    # Save artifacts
    artifacts_dir = Path(Config.ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    model_artifact_path = artifacts_dir / 'model_pipeline.joblib'
    metrics_artifact_path = artifacts_dir / 'metrics.json'
    
    # Extract feature sample schema for input form generation & validation
    feature_schema = {}
    for col in feature_cols:
        if col in numerical_cols:
            series = X[col].dropna()
            feature_schema[col] = {
                'type': 'numerical',
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
                'mean': float(series.mean()),
                'default': float(series.median())
            }
        else:
            top_vals = list(X[col].dropna().astype(str).unique()[:15])
            mode_val = str(X[col].mode()[0]) if not X[col].empty else top_vals[0]
            feature_schema[col] = {
                'type': 'categorical',
                'options': top_vals,
                'default': mode_val
            }
            
    artifact_payload = {
        'pipeline': best_pipeline,
        'model_name': best_model_name,
        'feature_cols': feature_cols,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'target_col': target_col,
        'target_mapping': target_mapping,
        'dataset_name': Path(dataset_path).name,
        'timestamp': datetime.now().isoformat(),
        'feature_schema': feature_schema,
        'is_calibrated': True
    }
    
    joblib.dump(artifact_payload, str(model_artifact_path))
    
    full_metrics_payload = {
        'model_name': best_model_name,
        'dataset_name': Path(dataset_path).name,
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'target_col': target_col,
        'feature_cols': feature_cols,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'dropped_cols': dropped_cols,
        'is_calibrated': True,
        'candidate_comparison': {
            m: {
                'accuracy': candidate_results[m]['accuracy'],
                'precision': candidate_results[m]['precision'],
                'recall': candidate_results[m]['recall'],
                'f1_score': candidate_results[m]['f1_score'],
                'roc_auc': candidate_results[m]['roc_auc'],
                'brier_score': candidate_results[m]['brier_score'],
                'mean_predicted_probability': candidate_results[m]['mean_predicted_probability'],
                'observed_claim_rate': candidate_results[m]['observed_claim_rate'],
                'calibration_error': candidate_results[m]['calibration_error'],
                'train_cv_roc_auc': candidate_results[m]['train_cv_roc_auc']
            } for m in models_to_test
        },
        'best_metrics': best_metrics,
        'feature_schema': feature_schema
    }
    
    with open(metrics_artifact_path, 'w', encoding='utf-8') as f:
        json.dump(full_metrics_payload, f, indent=2)
        
    # Record to SQLite
    save_model_run(
        model_name=best_model_name,
        dataset_name=Path(dataset_path).name,
        train_samples=len(X_train),
        test_samples=len(X_test),
        accuracy=best_metrics['accuracy'],
        precision_score=best_metrics['precision'],
        recall_score=best_metrics['recall'],
        f1_score=best_metrics['f1_score'],
        roc_auc=best_metrics['roc_auc'],
        metrics_json=full_metrics_payload
    )
    
    set_setting('active_model_name', best_model_name)
    
    print(f"Model Training Complete! Selected: {best_model_name}")
    print(f"ROC-AUC: {best_metrics.get('roc_auc')} | Recall: {best_metrics['recall']} | F1: {best_metrics['f1_score']}")
    
    return full_metrics_payload

if __name__ == '__main__':
    train_and_evaluate()
