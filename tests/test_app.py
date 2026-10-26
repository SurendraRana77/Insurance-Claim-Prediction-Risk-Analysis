import pytest
import os
import json
import io
import sys
import tempfile
import sqlite3
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app
from database.db import get_prediction_history, get_prediction_count, init_db, get_setting, set_setting
from services.model_service import get_model_status, load_model_artifact, validate_prediction_input
from services.data_service import validate_uploaded_csv
from config import Config

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_app_and_database_init(client):
    """Verify application initializes cleanly and database tables exist."""
    init_db()
    count = get_prediction_count()
    assert isinstance(count, int)
    
    # Test setting operations
    set_setting('test_key', 'test_value')
    assert get_setting('test_key') == 'test_value'

def test_clean_environment_db_init():
    """Verify database initialization works in a totally clean environment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db_path = Path(temp_dir) / "fresh_test.db"
        # Database does not exist yet
        assert not test_db_path.exists()
        
        # Initialize directly
        schema_path = Path(Config.BASE_DIR) / 'database' / 'schema.sql'
        conn = sqlite3.connect(str(test_db_path))
        try:
            with open(schema_path, 'r') as f:
                conn.executescript(f.read())
            conn.commit()
            
            # Verify tables and default settings
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            assert table_count >= 3
            
            cursor.execute("SELECT setting_value FROM system_settings WHERE setting_key='default_decision_threshold'")
            row = cursor.fetchone()
            assert row is not None
            assert float(row[0]) == 0.30
        finally:
            conn.close()

def test_all_pages_render_successfully(client):
    """Verify all 7 main application pages render HTTP 200 with HTML content."""
    routes = [
        ('/', 'Executive Analytics Dashboard'),
        ('/predict', 'Policyholder Risk Assessment'),
        ('/dataset', 'Dataset Explorer & Validation'),
        ('/analytics', 'Underwriting Risk Analytics'),
        ('/model', 'Machine Learning Model Diagnostics'),
        ('/history', 'Inference Audit Trail'),
        ('/about', 'Project Architecture & Methodology')
    ]
    
    for route, expected_title in routes:
        response = client.get(route)
        assert response.status_code == 200, f"Route {route} returned status {response.status_code}"
        assert expected_title in response.get_data(as_text=True), f"Expected title '{expected_title}' not found in {route}"

def test_model_status_and_calibration_metrics(client):
    """Verify model artifact is ready and has valid calibration metrics."""
    status = get_model_status()
    assert status['is_ready'] is True
    assert 'roc_auc' in status
    assert len(status['feature_cols']) > 0
    
    # Verify probability calibration metrics
    metrics_path = Path(Config.BASE_DIR) / 'artifacts' / 'metrics.json'
    assert metrics_path.exists()
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    assert metrics.get('is_calibrated') is True
    best_metrics = metrics.get('best_metrics', {})
    assert 'brier_score' in best_metrics
    # Brier score should be well below 0.20 (typically ~0.15 for calibrated models on this dataset)
    assert best_metrics['brier_score'] < 0.18
    # Mean predicted probability should closely match empirical claim rate (within 3 percentage points)
    assert abs(best_metrics['mean_predicted_probability'] - best_metrics['observed_claim_rate']) < 0.03
    assert best_metrics['calibration_error'] < 0.03

def test_prediction_api_success(client):
    """Verify JSON prediction endpoint evaluates input and persists to SQLite."""
    payload = {
        'age': 45,
        'driving_experience': 22,
        'credit_score': 0.78,
        'gender': 'female',
        'education': 'bachelors',
        'income_level': 'high',
        'vehicle_year': 'after_2015',
        'vehicle_type': 'sedan',
        'vehicle_ownership': '1',
        'annual_mileage': 11000,
        'speeding_violations': 0,
        'duis': 0,
        'past_accidents': 0,
        'custom_threshold': 0.50,
        'notes': 'Automated Test Evaluation'
    }
    
    initial_count = get_prediction_count()
    
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['success'] is True
    res = data['result']
    
    assert 'probability' in res
    assert 0.0 <= res['probability'] <= 1.0
    assert res['risk_tier'] in ['Low Risk', 'Moderate Risk', 'Elevated Risk', 'High Risk']
    assert len(res['explanations']) > 0
    
    # Verify saved in SQLite
    new_count = get_prediction_count()
    assert new_count == initial_count + 1

def test_prediction_validation_empty_payload(client):
    """Verify empty JSON payload is rejected with HTTP 400 and structured error."""
    response = client.post('/api/predict', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert 'empty' in data['error'].lower()
    assert isinstance(data.get('errors'), list)

def test_prediction_validation_missing_required_field(client):
    """Verify payload missing required fields is rejected with HTTP 400."""
    payload = {
        'age': 35,
        'driving_experience': 10
        # Missing all other required fields
    }
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert any('Missing required field' in err for err in data.get('errors', []))

def test_prediction_validation_out_of_range(client):
    """Verify payload with out-of-range numeric fields is rejected with HTTP 400."""
    payload = {
        'age': 150,  # Invalid age (> 100)
        'driving_experience': 22,
        'credit_score': 0.78,
        'gender': 'female',
        'education': 'bachelors',
        'income_level': 'high',
        'vehicle_year': 'after_2015',
        'vehicle_type': 'sedan',
        'vehicle_ownership': '1',
        'annual_mileage': 11000,
        'speeding_violations': 0,
        'duis': 0,
        'past_accidents': 0
    }
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert any('Driver Age must be between 16 and 100' in err for err in data.get('errors', []))

def test_prediction_validation_invalid_categorical(client):
    """Verify payload with invalid categorical options is rejected with HTTP 400."""
    payload = {
        'age': 35,
        'driving_experience': 12,
        'credit_score': 0.70,
        'gender': 'invalid_gender',  # Not male/female
        'education': 'bachelors',
        'income_level': 'high',
        'vehicle_year': 'after_2015',
        'vehicle_type': 'sedan',
        'vehicle_ownership': '1',
        'annual_mileage': 11000,
        'speeding_violations': 0,
        'duis': 0,
        'past_accidents': 0
    }
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert any('Invalid value for \'gender\'' in err for err in data.get('errors', []))

def test_prediction_history_search(client):
    """Verify prediction history search and filtering works."""
    response = client.get('/history?search=Automated+Test')
    assert response.status_code == 200
    assert 'Automated Test' in response.get_data(as_text=True)

def test_empty_csv_upload_rejected(client):
    """Verify empty CSV upload is cleanly rejected with a friendly message."""
    data = {
        'csv_file': (io.BytesIO(b""), 'empty.csv')
    }
    response = client.post('/dataset/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert response.status_code == 200
    assert b"empty" in response.data.lower() or b"rejected" in response.data.lower()

def test_upload_without_binary_target_rejected(client):
    """Verify uploading a CSV without binary target column is rejected."""
    # CSV with 25 rows, 4 columns, but no binary column (all random floats/strings)
    csv_content = "feat1,feat2,feat3,category\n" + "\n".join([f"{i},{i*1.5},{i*2.5},group_{i%5}" for i in range(25)])
    data = {
        'csv_file': (io.BytesIO(csv_content.encode('utf-8')), 'test_no_binary.csv')
    }
    response = client.post('/dataset/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert response.status_code == 200
    assert b"rejected" in response.data.lower() or b"no suitable binary" in response.data.lower()

def test_upload_benchmark_filename_prevented(client):
    """Verify uploading a file named sample_insurance_data.csv is rejected to protect benchmark."""
    csv_content = "col1,col2,col3,target\n" + "\n".join([f"1,2,3,{i%2}" for i in range(25)])
    data = {
        'csv_file': (io.BytesIO(csv_content.encode('utf-8')), Config.DEFAULT_DATASET_NAME)
    }
    response = client.post('/dataset/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert response.status_code == 200
    assert b"cannot overwrite" in response.data.lower()

def test_invalid_threshold_rejected(client):
    """Verify invalid threshold order is caught and rejected."""
    data = {
        'risk_threshold_low': '0.80',
        'risk_threshold_moderate': '0.50',
        'risk_threshold_elevated': '0.20'
    }
    response = client.post('/analytics/thresholds', data=data, follow_redirects=True)
    assert response.status_code == 200
    assert b"Invalid threshold hierarchy" in response.data

def test_prediction_validation_malformed_numeric(client):
    """Verify payload with non-numeric text in numeric field is rejected with HTTP 400."""
    payload = {
        'age': 'twenty-five',  # String instead of number
        'driving_experience': 5,
        'credit_score': 0.70,
        'gender': 'female',
        'education': 'bachelors',
        'income_level': 'high',
        'vehicle_year': 'after_2015',
        'vehicle_type': 'sedan',
        'vehicle_ownership': '1',
        'annual_mileage': 11000,
        'speeding_violations': 0,
        'duis': 0,
        'past_accidents': 0
    }
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert any("must be a valid numeric value" in err for err in data.get('errors', []))

def test_missing_model_artifact_behavior(monkeypatch, client):
    """Verify application handles missing model artifacts gracefully."""
    # Point artifacts dir to an empty directory
    with tempfile.TemporaryDirectory() as empty_dir:
        monkeypatch.setattr(Config, 'ARTIFACTS_DIR', empty_dir)
        status = get_model_status()
        assert status['is_ready'] is False
        
        # Prediction API should reject with 400 when model not ready
        response = client.post('/api/predict', json={'age': 30})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'not ready' in data.get('error', '').lower()
        
        # Prediction HTML view renders the model not ready card
        get_res = client.get('/predict')
        assert get_res.status_code == 200
        assert 'Model Pipeline Not Initialized' in get_res.get_data(as_text=True)

def test_model_artifact_loading_and_metadata_consistency():
    """Verify loaded model artifact contains valid pipeline, schema, and matches metrics.json."""
    artifact = load_model_artifact()
    assert artifact is not None
    assert 'pipeline' in artifact
    assert 'feature_cols' in artifact
    assert 'feature_schema' in artifact
    assert len(artifact['feature_cols']) == 13
    
    status = get_model_status()
    assert status['is_ready'] is True
    assert status['model_name'] == artifact['model_name']
    assert status['feature_cols'] == artifact['feature_cols']

def test_404_error_handler_renders_claymorphism_page(client):
    """Verify requesting a non-existent route returns 404 and renders custom Claymorphism error page."""
    response = client.get('/non_existent_page_path_12345')
    assert response.status_code == 404
    html = response.get_data(as_text=True)
    assert '404' in html
    assert 'Page Not Found' in html
    assert 'Return to Dashboard' in html

