import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, flash, redirect, url_for
from services.data_service import load_dataset, get_active_target_col
from services.model_service import get_model_status, get_model_metrics
from services.risk_service import get_risk_thresholds
from ml.train import train_and_evaluate
from database.db import set_setting

analytics_bp = Blueprint('analytics', __name__)

def calculate_eda_metrics(df: pd.DataFrame, target_col: str) -> dict:
    if df is None or df.empty or target_col not in df.columns:
        return {}
        
    eda = {}
    
    # 1. Driving Experience Bins
    if 'driving_experience' in df.columns:
        bins = [-1, 5, 15, 25, 100]
        labels = ['0–5 yrs', '6–15 yrs', '16–25 yrs', '26+ yrs']
        df['exp_group'] = pd.cut(df['driving_experience'], bins=bins, labels=labels)
        grouped = df.groupby('exp_group', observed=False)[target_col].mean() * 100
        eda['experience'] = {
            'labels': labels,
            'rates': [round(float(grouped.get(l, 0)), 1) for l in labels]
        }
        
    # 2. Past Accidents
    if 'past_accidents' in df.columns:
        df['acc_group'] = df['past_accidents'].apply(lambda x: '3+' if x >= 3 else str(int(x)))
        labels = ['0', '1', '2', '3+']
        grouped = df.groupby('acc_group', observed=False)[target_col].mean() * 100
        eda['accidents'] = {
            'labels': [f'{l} Acc' for l in labels],
            'rates': [round(float(grouped.get(l, 0)), 1) for l in labels]
        }
        
    # 3. Vehicle Type
    if 'vehicle_type' in df.columns:
        grouped = df.groupby('vehicle_type')[target_col].mean() * 100
        labels = list(grouped.index)
        eda['vehicle_type'] = {
            'labels': labels,
            'rates': [round(float(v), 1) for v in grouped.values]
        }
        
    # 4. Credit Score Quartiles
    if 'credit_score' in df.columns:
        try:
            df['credit_tier'] = pd.qcut(df['credit_score'], q=4, labels=['Q1 Low', 'Q2 Medium', 'Q3 High', 'Q4 Prime'])
            grouped = df.groupby('credit_tier', observed=False)[target_col].mean() * 100
            labels = ['Q1 Low', 'Q2 Medium', 'Q3 High', 'Q4 Prime']
            eda['credit_score'] = {
                'labels': labels,
                'rates': [round(float(grouped.get(l, 0)), 1) for l in labels]
            }
        except Exception:
            pass
            
    return eda

@analytics_bp.route('/analytics')
def risk_analytics_view():
    df = load_dataset()
    target_col = get_active_target_col()
    eda_data = calculate_eda_metrics(df, target_col)
    risk_thresholds = get_risk_thresholds()
    model_status = get_model_status()
    
    return render_template(
        'analytics.html',
        eda_data=eda_data,
        risk_thresholds=risk_thresholds,
        model_status=model_status
    )

@analytics_bp.route('/model')
def model_performance_view():
    model_status = get_model_status()
    metrics = get_model_metrics()
    
    return render_template(
        'model.html',
        model_status=model_status,
        metrics=metrics
    )

@analytics_bp.route('/model/train', methods=['POST'])
def trigger_training():
    model_type = request.form.get('model_type')
    preferred = model_type if model_type in ('logistic_regression', 'random_forest') else None
    try:
        results = train_and_evaluate(preferred_model=preferred)
        best_m = results['best_metrics']
        flash(f"Model pipeline successfully trained! Selected: {results['model_name'].replace('_', ' ').title()} (ROC-AUC: {best_m.get('roc_auc')}, Recall: {round(best_m['recall']*100, 1)}%)", "success")
    except Exception as e:
        flash(f"Model training failed: {str(e)}", "danger")
        
    return redirect(url_for('analytics.model_performance_view'))

@analytics_bp.route('/analytics/thresholds', methods=['POST'])
def update_risk_thresholds():
    try:
        from config import Config
        low = float(request.form.get('risk_threshold_low', Config.DEFAULT_RISK_THRESHOLDS['low']))
        mod = float(request.form.get('risk_threshold_moderate', Config.DEFAULT_RISK_THRESHOLDS['moderate']))
        elev = float(request.form.get('risk_threshold_elevated', Config.DEFAULT_RISK_THRESHOLDS['elevated']))
        
        if not (0 < low < mod < elev < 1.0):
            flash("Invalid threshold hierarchy. Ensure Low < Moderate < Elevated.", "danger")
        else:
            set_setting('risk_threshold_low', str(low))
            set_setting('risk_threshold_moderate', str(mod))
            set_setting('risk_threshold_elevated', str(elev))
            flash("Risk tier boundaries successfully updated.", "success")
    except Exception as e:
        flash(f"Error saving thresholds: {str(e)}", "danger")
        
    return redirect(url_for('analytics.risk_analytics_view'))
