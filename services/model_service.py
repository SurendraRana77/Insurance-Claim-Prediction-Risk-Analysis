import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from config import Config
from database.db import save_prediction
from services.risk_service import categorize_risk, get_decision_threshold

_CACHED_ARTIFACT = None
_CACHED_MTIME = 0

def get_artifact_path() -> Path:
    return Path(Config.ARTIFACTS_DIR) / 'model_pipeline.joblib'

def get_metrics_path() -> Path:
    return Path(Config.ARTIFACTS_DIR) / 'metrics.json'

def load_model_artifact(force_reload: bool = False):
    global _CACHED_ARTIFACT, _CACHED_MTIME
    artifact_path = get_artifact_path()
    
    if not artifact_path.exists():
        return None
        
    try:
        current_mtime = artifact_path.stat().st_mtime
        if _CACHED_ARTIFACT is None or force_reload or current_mtime > _CACHED_MTIME:
            _CACHED_ARTIFACT = joblib.load(str(artifact_path))
            _CACHED_MTIME = current_mtime
        return _CACHED_ARTIFACT
    except Exception as e:
        print(f"Error loading model artifact: {e}")
        return None

def get_model_status() -> dict:
    artifact_path = get_artifact_path()
    metrics_path = get_metrics_path()
    
    if not artifact_path.exists() or not metrics_path.exists():
        return {
            'is_ready': False,
            'message': 'No trained model artifact is currently available. Please train a model via the Dataset Explorer or Model Performance page.'
        }
        
    try:
        artifact = load_model_artifact()
        if not artifact:
            return {'is_ready': False, 'message': 'Failed to deserialize model artifact.'}
            
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            
        best_metrics = metrics.get('best_metrics', {})
        
        return {
            'is_ready': True,
            'model_name': artifact.get('model_name', 'Unknown'),
            'dataset_name': artifact.get('dataset_name', 'Unknown'),
            'timestamp': artifact.get('timestamp', 'Unknown'),
            'feature_count': len(artifact.get('feature_cols', [])),
            'feature_cols': artifact.get('feature_cols', []),
            'feature_schema': artifact.get('feature_schema', {}),
            'accuracy': best_metrics.get('accuracy'),
            'roc_auc': best_metrics.get('roc_auc'),
            'f1_score': best_metrics.get('f1_score'),
            'recall': best_metrics.get('recall')
        }
    except Exception as e:
        return {'is_ready': False, 'message': f'Error reading model metadata: {str(e)}'}

def get_model_metrics() -> dict:
    metrics_path = get_metrics_path()
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def explain_prediction(pipeline, input_df: pd.DataFrame, feature_cols: list) -> list:
    """Provides local feature impact estimation for the individual prediction."""
    explanations = []
    try:
        # Extract preprocessor and classifier whether standard Pipeline or CalibratedClassifierCV
        if hasattr(pipeline, 'calibrated_classifiers_'):
            first_est = pipeline.calibrated_classifiers_[0].estimator
            preprocessor = first_est.named_steps['preprocessor']
            classifier = first_est.named_steps['classifier']
            is_calibrated = True
        else:
            preprocessor = pipeline.named_steps['preprocessor']
            classifier = pipeline.named_steps['classifier']
            is_calibrated = False
        
        if hasattr(classifier, 'coef_'):
            transformed = preprocessor.transform(input_df)
            if is_calibrated:
                coefs = np.mean([clf.estimator.named_steps['classifier'].coef_[0] for clf in pipeline.calibrated_classifiers_], axis=0)
            else:
                coefs = classifier.coef_[0]
                
            feature_names = list(preprocessor.get_feature_names_out())
            contributions = transformed[0] * coefs
            ranked_indices = np.argsort(np.abs(contributions))[::-1][:6]
            
            for idx in ranked_indices:
                raw_name = feature_names[idx]
                cleaned_name = raw_name.replace('num__', '').replace('cat__', '')
                val_contrib = contributions[idx]
                impact = 'Pushed risk higher' if val_contrib > 0 else 'Pushed risk lower'
                explanations.append({
                    'factor': cleaned_name,
                    'impact': impact,
                    'strength': round(float(abs(val_contrib)), 3),
                    'direction': 'risk-increase' if val_contrib > 0 else 'risk-decrease'
                })
        else:
            # Tree-based fallback explanation
            explanations.append({
                'factor': 'Tree Ensemble Decision Path',
                'impact': 'Combined leaf voting across decision trees',
                'strength': 1.0,
                'direction': 'neutral'
            })
    except Exception as e:
        print(f"Explanation extraction error: {e}")
        
    return explanations

def validate_prediction_input(input_dict: dict, feature_cols: list, feature_schema: dict, custom_threshold=None) -> tuple:
    """Strictly validates prediction inputs; returns (is_valid, error_list, cleaned_row_data)."""
    errors = []
    cleaned_data = {}
    
    if not input_dict or not isinstance(input_dict, dict) or len(input_dict) == 0:
        return False, ["Prediction payload cannot be empty. Please provide policyholder attributes."], {}
        
    for col in feature_cols:
        val = input_dict.get(col)
        schema = feature_schema.get(col, {})
        
        if val is None or str(val).strip() == '':
            errors.append(f"Missing required field: '{col}'.")
            continue
            
        val_str = str(val).strip()
        
        if schema.get('type') == 'numerical':
            try:
                num_val = float(val_str)
            except (ValueError, TypeError):
                errors.append(f"Field '{col}' must be a valid numeric value.")
                continue
                
            # Domain bounds validation
            if col == 'age':
                if not (16 <= num_val <= 100):
                    errors.append("Driver Age must be between 16 and 100 years.")
            elif col == 'driving_experience':
                if not (0 <= num_val <= 85):
                    errors.append("Driving Experience must be between 0 and 85 years.")
                if 'age' in input_dict:
                    try:
                        age_val = float(input_dict['age'])
                        if num_val > (age_val - 14):
                            errors.append("Driving Experience cannot exceed (Age - 14).")
                    except Exception:
                        pass
            elif col == 'credit_score':
                if not (0.0 <= num_val <= 1.0):
                    errors.append("Credit Score must be a normalized score between 0.0 and 1.0.")
            elif col == 'annual_mileage':
                if not (500 <= num_val <= 100000):
                    errors.append("Annual Mileage must be between 500 and 100,000 miles.")
            elif col in ('speeding_violations', 'duis', 'past_accidents'):
                if not (0 <= num_val <= 50):
                    errors.append(f"Field '{col}' must be between 0 and 50.")
            elif col == 'vehicle_ownership':
                if val_str not in ('0', '1', '0.0', '1.0', 'True', 'False', 'true', 'false'):
                    errors.append("Vehicle Ownership must be 1 (Owned) or 0 (Leased/Financed).")
                else:
                    num_val = 1.0 if val_str in ('1', '1.0', 'True', 'true') else 0.0
                    
            cleaned_data[col] = num_val
        else:
            # Categorical validation
            options = schema.get('options', [])
            if options:
                allowed_lower = [str(opt).lower() for opt in options]
                if val_str.lower() not in allowed_lower:
                    errors.append(f"Invalid value for '{col}': '{val_str}'. Allowed options: {', '.join(options)}.")
                    continue
                # Normalize case to match schema option
                match_opt = next(opt for opt in options if str(opt).lower() == val_str.lower())
                cleaned_data[col] = match_opt
            else:
                cleaned_data[col] = val_str
                
    if custom_threshold is not None:
        try:
            thresh_val = float(custom_threshold)
            if not (0.01 <= thresh_val <= 0.99):
                errors.append("Decision Threshold must be between 0.01 (1%) and 0.99 (99%).")
        except (ValueError, TypeError):
            errors.append("Decision Threshold must be a valid numeric value.")
            
    is_valid = (len(errors) == 0)
    return is_valid, errors, cleaned_data

def predict_claim(input_dict: dict, custom_threshold: float = None, notes: str = "") -> dict:
    artifact = load_model_artifact()
    if not artifact:
        raise ValueError("Model pipeline is not ready for predictions.")
        
    pipeline = artifact['pipeline']
    feature_cols = artifact['feature_cols']
    feature_schema = artifact.get('feature_schema', {})
    
    # 1. Enforce strict input validation
    is_valid, errors, row_data = validate_prediction_input(
        input_dict, feature_cols, feature_schema, custom_threshold=custom_threshold
    )
    if not is_valid:
        raise ValueError("; ".join(errors))
        
    input_df = pd.DataFrame([row_data])
    
    # 2. Generate calibrated probability
    probabilities = pipeline.predict_proba(input_df)[0]
    claim_probability = float(probabilities[1])
    
    # 3. Determine threshold
    decision_threshold = custom_threshold if custom_threshold is not None else get_decision_threshold()
    classification_result = int(claim_probability >= decision_threshold)
    
    # 4. Risk categorization
    risk_info = categorize_risk(claim_probability)
    
    # 5. Explainability factors
    explanations = explain_prediction(pipeline, input_df, feature_cols)
    
    # 6. Save to SQLite database
    model_version = f"{artifact.get('model_name', 'model')}-{artifact.get('timestamp', '')[:10]}"
    pred_id = save_prediction(
        model_version=model_version,
        input_payload=row_data,
        predicted_probability=claim_probability,
        classification_result=classification_result,
        risk_category=risk_info['tier'],
        decision_threshold=decision_threshold,
        notes=notes
    )
    
    return {
        'id': pred_id,
        'probability': round(claim_probability, 4),
        'probability_percent': round(claim_probability * 100, 1),
        'classification_result': classification_result,
        'classification_label': 'Claim Expected' if classification_result == 1 else 'No Claim Expected',
        'risk_tier': risk_info['tier'],
        'risk_css': risk_info['css_class'],
        'risk_summary': risk_info['summary'],
        'risk_guidance': risk_info['guidance'],
        'decision_threshold': decision_threshold,
        'model_version': model_version,
        'explanations': explanations,
        'input_data': row_data
    }
