from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from services.model_service import get_model_status, predict_claim
from database.db import get_prediction_history, get_prediction_by_id

prediction_bp = Blueprint('predictions', __name__)

@prediction_bp.route('/predict', methods=['GET', 'POST'])
def predict_view():
    model_status = get_model_status()
    prediction_result = None
    
    if request.method == 'POST' and model_status.get('is_ready'):
        try:
            form_data = request.form.to_dict()
            custom_thresh = float(form_data['custom_threshold']) if form_data.get('custom_threshold') else None
            notes = form_data.get('notes', '')
            
            prediction_result = predict_claim(form_data, custom_threshold=custom_thresh, notes=notes)
            flash(f"Policyholder risk evaluated: {prediction_result['risk_tier']} ({prediction_result['probability_percent']}%)", "success")
        except ValueError as ve:
            flash(f"Input validation error: {str(ve)}", "danger")
        except Exception as e:
            flash(f"Prediction evaluation failed: {str(e)}", "danger")
            
    return render_template(
        'predict.html',
        model_status=model_status,
        prediction_result=prediction_result
    )

@prediction_bp.route('/api/predict', methods=['POST'])
def api_predict():
    model_status = get_model_status()
    if not model_status.get('is_ready'):
        return jsonify({'success': False, 'error': 'Model pipeline is not ready for predictions.'}), 400
        
    try:
        if request.is_json:
            data = request.get_json(silent=True)
        else:
            data = request.form.to_dict() if request.form else None

        if not data or not isinstance(data, dict) or len(data) == 0:
            return jsonify({
                'success': False,
                'error': 'Prediction payload cannot be empty. Please provide policyholder attributes.',
                'errors': ['Prediction payload cannot be empty. Please provide policyholder attributes.']
            }), 400

        custom_thresh = float(data['custom_threshold']) if data.get('custom_threshold') not in (None, '') else None
        notes = data.get('notes', '')
        
        result = predict_claim(data, custom_threshold=custom_thresh, notes=notes)
        return jsonify({'success': True, 'result': result})
    except ValueError as ve:
        error_msg = str(ve)
        errors = [err.strip() for err in error_msg.split(';') if err.strip()]
        return jsonify({'success': False, 'error': error_msg, 'errors': errors}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@prediction_bp.route('/history')
def history_view():
    model_status = get_model_status()
    risk_filter = request.args.get('risk_filter')
    search = request.args.get('search')
    
    predictions = get_prediction_history(limit=100, risk_filter=risk_filter, search=search)
    
    return render_template(
        'history.html',
        model_status=model_status,
        predictions=predictions,
        risk_filter=risk_filter,
        search=search
    )

@prediction_bp.route('/api/history/<int:pred_id>')
def api_prediction_detail(pred_id: int):
    pred = get_prediction_by_id(pred_id)
    if not pred:
        return jsonify({'success': False, 'error': 'Record not found.'}), 404
    return jsonify({'success': True, 'prediction': pred})
