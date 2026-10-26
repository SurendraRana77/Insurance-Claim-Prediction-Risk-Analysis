import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss
)

def extract_feature_names(preprocessor, numerical_cols: list, categorical_cols: list) -> list:
    names = []
    try:
        names = list(preprocessor.get_feature_names_out())
        # Clean prefix like 'num__' and 'cat__' for cleaner display
        cleaned = []
        for n in names:
            if n.startswith('num__'):
                cleaned.append(n[5:])
            elif n.startswith('cat__'):
                cleaned.append(n[5:])
            else:
                cleaned.append(n)
        return cleaned
    except Exception:
        pass
    
    # Fallback name construction
    names.extend(numerical_cols)
    try:
        cat_trans = preprocessor.named_transformers_['cat']
        encoder = cat_trans.named_steps['encoder']
        encoded_cats = encoder.get_feature_names_out(categorical_cols)
        names.extend(encoded_cats)
    except Exception:
        names.extend(categorical_cols)
    return names

def evaluate_model_pipeline(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numerical_cols: list,
    categorical_cols: list,
    decision_threshold: float = 0.30
) -> dict:
    # 1. Probabilities
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= decision_threshold).astype(int)
    
    # 2. Standard Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = None
        
    try:
        brier = float(brier_score_loss(y_test, y_proba))
    except Exception:
        brier = None
        
    # Calibration Diagnostics
    mean_pred_prob = float(np.mean(y_proba))
    obs_rate = float(np.mean(y_test))
    cal_err = float(abs(mean_pred_prob - obs_rate))
    
    # 3. Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 4. ROC Curve (downsampled to max 40 points for smooth Chart.js rendering)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    if len(fpr) > 40:
        indices = np.linspace(0, len(fpr) - 1, 40).astype(int)
        roc_points = [{'x': round(float(fpr[i]), 4), 'y': round(float(tpr[i]), 4)} for i in indices]
    else:
        roc_points = [{'x': round(float(fpr[i]), 4), 'y': round(float(tpr[i]), 4)} for i in range(len(fpr))]
        
    # 5. Precision-Recall Curve
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_proba)
    if len(pr_rec) > 40:
        indices = np.linspace(0, len(pr_rec) - 1, 40).astype(int)
        pr_points = [{'x': round(float(pr_rec[i]), 4), 'y': round(float(pr_prec[i]), 4)} for i in indices]
    else:
        pr_points = [{'x': round(float(pr_rec[i]), 4), 'y': round(float(pr_prec[i]), 4)} for i in range(len(pr_rec))]
        
    # 6. Feature Importance Extraction (handles both standard Pipeline and CalibratedClassifierCV)
    if hasattr(pipeline, 'calibrated_classifiers_'):
        first_est = pipeline.calibrated_classifiers_[0].estimator
        preprocessor = first_est.named_steps['preprocessor']
        classifier = first_est.named_steps['classifier']
        is_calibrated = True
    else:
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        is_calibrated = False
        
    feature_names = extract_feature_names(preprocessor, numerical_cols, categorical_cols)
    
    importances = []
    if hasattr(classifier, 'coef_'):
        # If calibrated, average coefficients across folds for robustness
        if is_calibrated:
            coefs = np.mean([clf.estimator.named_steps['classifier'].coef_[0] for clf in pipeline.calibrated_classifiers_], axis=0)
        else:
            coefs = classifier.coef_[0]
            
        for name, coef in zip(feature_names, coefs):
            direction = 'Increases Risk' if coef > 0 else 'Decreases Risk'
            importances.append({
                'feature': name,
                'weight': round(float(coef), 4),
                'abs_weight': round(float(abs(coef)), 4),
                'direction': direction
            })
        importances.sort(key=lambda x: x['abs_weight'], reverse=True)
    elif hasattr(classifier, 'feature_importances_'):
        if is_calibrated:
            mdi = np.mean([clf.estimator.named_steps['classifier'].feature_importances_ for clf in pipeline.calibrated_classifiers_], axis=0)
        else:
            mdi = classifier.feature_importances_
            
        for name, val in zip(feature_names, mdi):
            importances.append({
                'feature': name,
                'weight': round(float(val), 4),
                'abs_weight': round(float(val), 4),
                'direction': 'Feature Importance (MDI)'
            })
        importances.sort(key=lambda x: x['weight'], reverse=True)
        
    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4) if auc is not None else None,
        'brier_score': round(brier, 4) if brier is not None else None,
        'mean_predicted_probability': round(mean_pred_prob, 4),
        'observed_claim_rate': round(obs_rate, 4),
        'calibration_error': round(cal_err, 4),
        'is_calibrated': is_calibrated,
        'decision_threshold': decision_threshold,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'roc_curve': roc_points,
        'pr_curve': pr_points,
        'feature_importances': importances[:15],
        'test_samples': int(len(y_test)),
        'positive_test_samples': int(y_test.sum())
    }
