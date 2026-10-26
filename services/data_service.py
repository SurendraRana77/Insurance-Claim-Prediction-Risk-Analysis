import os
import io
import pandas as pd
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename
from config import Config
from database.db import get_setting, set_setting

LEAKAGE_KEYWORDS = {'amount', 'paid', 'cost', 'settlement', 'payout', 'loss', 'fraud', 'reimbursement'}
IDENTIFIER_KEYWORDS = {'id', 'uuid', 'policy_id', 'ssn', 'customer_id'}

def get_active_dataset_filename() -> str:
    filename = get_setting('active_dataset', Config.DEFAULT_DATASET_NAME)
    full_path = Path(Config.DATA_DIR) / filename
    if not full_path.exists():
        # Fallback to default benchmark
        filename = Config.DEFAULT_DATASET_NAME
        set_setting('active_dataset', filename)
    return filename

def get_active_dataset_path() -> str:
    filename = get_active_dataset_filename()
    return str(Path(Config.DATA_DIR) / filename)

def get_active_target_col() -> str:
    return get_setting('active_target_col', 'claim_occurred')

def load_dataset(filepath: str = None) -> pd.DataFrame:
    if filepath is None:
        filepath = get_active_dataset_path()
    if not os.path.exists(filepath):
        return None
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading CSV from {filepath}: {e}")
        return None

def detect_target_candidates(df: pd.DataFrame) -> list:
    """Finds binary or integer columns suitable as supervised classification targets."""
    candidates = []
    for col in df.columns:
        if col.lower() in IDENTIFIER_KEYWORDS:
            continue
        vals = df[col].dropna().unique()
        if len(vals) == 2 and set(vals).issubset({0, 1, 0.0, 1.0, '0', '1', True, False, 'yes', 'no', 'Y', 'N'}):
            candidates.append(col)
        elif len(vals) == 2:
            candidates.append(col)
    return candidates

def detect_potential_leakage(columns: list, target_col: str = None) -> list:
    """Identifies columns that likely represent post-incident leakage."""
    warnings = []
    for col in columns:
        if target_col and col == target_col:
            continue
        lower = col.lower()
        if any(keyword in lower for keyword in LEAKAGE_KEYWORDS):
            warnings.append(f"Column '{col}' matches leakage keyword. Ensure it is known PRIOR to claim occurrence.")
    return warnings

def get_dataset_summary(df: pd.DataFrame, target_col: str = None) -> dict:
    if df is None or df.empty:
        return {
            'is_loaded': False,
            'message': 'No active dataset is currently loaded.'
        }
        
    total_rows, total_cols = df.shape
    missing_series = df.isnull().sum()
    duplicate_count = int(df.duplicated().sum())
    
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col.lower() in IDENTIFIER_KEYWORDS:
            continue
        if target_col and col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
            
    num_summary = {}
    for col in numerical_cols:
        series = df[col].dropna()
        if not series.empty:
            num_summary[col] = {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': round(float(series.mean()), 2),
                'median': round(float(series.median()), 2),
                'std': round(float(series.std()), 2)
            }
            
    cat_summary = {}
    for col in categorical_cols:
        series = df[col].dropna().astype(str)
        val_counts = series.value_counts().head(6).to_dict()
        cat_summary[col] = {
            'unique_count': int(series.nunique()),
            'top_values': val_counts
        }
        
    target_info = {'exists': False}
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts().to_dict()
        total_valid = sum(counts.values())
        pos_count = counts.get(1, counts.get(1.0, counts.get('1', 0)))
        claim_rate = (pos_count / total_valid) if total_valid > 0 else 0.0
        
        target_info = {
            'exists': True,
            'name': target_col,
            'counts': {str(k): int(v) for k, v in counts.items()},
            'claim_rate': round(claim_rate, 4),
            'positive_count': int(pos_count),
            'is_binary': len(counts) == 2,
            'is_imbalanced': claim_rate < 0.35 or claim_rate > 0.65
        }
        
    leakage_warnings = detect_potential_leakage(df.columns.tolist(), target_col)
    
    # Check if this is the default synthetic dataset
    filename = get_active_dataset_filename()
    is_synthetic = ('sample' in filename.lower() or 'synthetic' in filename.lower())
    
    return {
        'is_loaded': True,
        'filename': filename,
        'is_synthetic': is_synthetic,
        'total_rows': total_rows,
        'total_cols': total_cols,
        'missing_counts': {k: int(v) for k, v in missing_series.items() if v > 0},
        'duplicate_count': duplicate_count,
        'columns': df.columns.tolist(),
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'numerical_summary': num_summary,
        'categorical_summary': cat_summary,
        'target_info': target_info,
        'leakage_warnings': leakage_warnings,
        'target_candidates': detect_target_candidates(df)
    }

def validate_uploaded_csv(file_storage) -> dict:
    if not file_storage or file_storage.filename == '':
        return {'valid': False, 'error': 'No file was selected for upload.'}
        
    filename = secure_filename(file_storage.filename)
    if not filename.lower().endswith('.csv'):
        return {'valid': False, 'error': 'Unsupported file format. Please upload a standard comma-separated (.csv) file.'}
        
    # Prevent overwriting default benchmark dataset
    if filename.lower() == Config.DEFAULT_DATASET_NAME.lower():
        return {
            'valid': False,
            'error': f"Cannot overwrite the default benchmark dataset '{Config.DEFAULT_DATASET_NAME}'. Please rename your file before uploading."
        }
        
    try:
        content = file_storage.read()
        file_storage.seek(0)
        
        if len(content) == 0:
            return {'valid': False, 'error': 'The uploaded CSV file is empty.'}
            
        if len(content) > Config.MAX_CONTENT_LENGTH:
            return {'valid': False, 'error': f'File exceeds maximum permitted size of {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB.'}
            
        df = pd.read_csv(io.BytesIO(content))
        
        if df.empty or df.shape[0] < 20:
            return {'valid': False, 'error': f'Dataset contains only {df.shape[0]} rows. A minimum of 20 policyholder records is required.'}
            
        if df.shape[1] < 3:
            return {'valid': False, 'error': f'Dataset contains only {df.shape[1]} columns. An insurance dataset requires at least 3 features and a target.'}
            
        candidates = detect_target_candidates(df)
        if len(candidates) == 0:
            return {
                'valid': False,
                'error': 'Dataset rejected: No suitable binary classification target column found (expected a column with binary 0/1, True/False, or Yes/No outcomes).'
            }
        
        return {
            'valid': True,
            'filename': filename,
            'df': df,
            'rows': df.shape[0],
            'cols': df.shape[1],
            'target_candidates': candidates
        }
    except Exception as e:
        return {'valid': False, 'error': f'Failed to parse CSV file: {str(e)}'}

def save_uploaded_dataset(file_storage, target_col: str = None) -> dict:
    validation = validate_uploaded_csv(file_storage)
    if not validation['valid']:
        return validation
        
    df = validation['df']
    if target_col:
        if target_col not in df.columns:
            return {'valid': False, 'error': f"Specified target column '{target_col}' not found in uploaded dataset."}
        if target_col not in validation['target_candidates']:
            return {'valid': False, 'error': f"Specified target column '{target_col}' is not a valid binary classification target."}
            
    filename = validation['filename']
    save_path = Path(Config.DATA_DIR) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_storage.seek(0)
    file_storage.save(str(save_path))
    
    set_setting('active_dataset', filename)
    if target_col:
        set_setting('active_target_col', target_col)
    elif validation['target_candidates']:
        set_setting('active_target_col', validation['target_candidates'][0])
        
    return {
        'valid': True,
        'filename': filename,
        'rows': validation['rows'],
        'cols': validation['cols'],
        'active_target_col': get_active_target_col()
    }

def get_available_datasets() -> list:
    data_dir = Path(Config.DATA_DIR)
    if not data_dir.exists():
        return []
    files = []
    active = get_active_dataset_filename()
    for f in data_dir.glob('*.csv'):
        stat = f.stat()
        files.append({
            'name': f.name,
            'size_kb': round(stat.st_size / 1024, 1),
            'modified': stat.st_mtime,
            'is_active': (f.name == active),
            'is_benchmark': ('sample' in f.name.lower())
        })
    return sorted(files, key=lambda x: x['is_active'], reverse=True)
