from flask import Blueprint, render_template
from services.data_service import get_dataset_summary, load_dataset, get_active_target_col
from services.model_service import get_model_status
from database.db import get_prediction_history, get_prediction_count, get_prediction_risk_distribution

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    df = load_dataset()
    target_col = get_active_target_col()
    summary = get_dataset_summary(df, target_col)
    model_status = get_model_status()
    total_preds = get_prediction_count()
    risk_dist = get_prediction_risk_distribution()
    recent_preds = get_prediction_history(limit=5)
    
    return render_template(
        'dashboard.html',
        dataset_summary=summary,
        model_status=model_status,
        total_predictions=total_preds,
        risk_distribution=risk_dist,
        recent_predictions=recent_preds
    )

@dashboard_bp.route('/about')
def about_view():
    model_status = get_model_status()
    return render_template('about.html', model_status=model_status)
