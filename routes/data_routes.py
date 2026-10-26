from flask import Blueprint, render_template, request, flash, redirect, url_for
from services.data_service import (
    load_dataset, get_dataset_summary, get_active_target_col,
    save_uploaded_dataset, get_available_datasets, get_active_dataset_filename
)
from database.db import set_setting
from services.model_service import get_model_status

data_bp = Blueprint('data_bp', __name__)

@data_bp.route('/dataset')
def dataset_view():
    df = load_dataset()
    target_col = get_active_target_col()
    summary = get_dataset_summary(df, target_col)
    available_datasets = get_available_datasets()
    model_status = get_model_status()
    
    return render_template(
        'dataset.html',
        summary=summary,
        available_datasets=available_datasets,
        model_status=model_status
    )

@data_bp.route('/dataset/upload', methods=['POST'])
def upload_dataset():
    file = request.files.get('csv_file')
    if not file:
        flash("No file was uploaded.", "danger")
        return redirect(url_for('data_bp.dataset_view'))
        
    result = save_uploaded_dataset(file)
    if result['valid']:
        flash(f"Dataset '{result['filename']}' ({result['rows']} rows, {result['cols']} cols) uploaded and activated successfully!", "success")
    else:
        flash(f"Upload rejected: {result.get('error')}", "danger")
        
    return redirect(url_for('data_bp.dataset_view'))

@data_bp.route('/dataset/target', methods=['POST'])
def set_target():
    target_col = request.form.get('target_column')
    if target_col:
        set_setting('active_target_col', target_col)
        flash(f"Supervised target column updated to '{target_col}'.", "info")
    return redirect(url_for('data_bp.dataset_view'))

@data_bp.route('/dataset/switch', methods=['POST'])
def switch_dataset():
    dataset_name = request.form.get('dataset_name')
    if dataset_name:
        set_setting('active_dataset', dataset_name)
        flash(f"Active dataset switched to '{dataset_name}'.", "info")
    return redirect(url_for('data_bp.dataset_view'))
