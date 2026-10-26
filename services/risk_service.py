from config import Config
from database.db import get_setting

def get_risk_thresholds() -> dict:
    try:
        low = float(get_setting('risk_threshold_low', Config.DEFAULT_RISK_THRESHOLDS['low']))
        mod = float(get_setting('risk_threshold_moderate', Config.DEFAULT_RISK_THRESHOLDS['moderate']))
        elev = float(get_setting('risk_threshold_elevated', Config.DEFAULT_RISK_THRESHOLDS['elevated']))
    except Exception:
        low = Config.DEFAULT_RISK_THRESHOLDS['low']
        mod = Config.DEFAULT_RISK_THRESHOLDS['moderate']
        elev = Config.DEFAULT_RISK_THRESHOLDS['elevated']
        
    return {
        'low': low,
        'moderate': mod,
        'elevated': elev,
        'high': 1.00
    }

def categorize_risk(probability: float, thresholds: dict = None) -> dict:
    if thresholds is None:
        thresholds = get_risk_thresholds()
        
    prob_percent = round(probability * 100, 1)
    
    if probability < thresholds['low']:
        return {
            'tier': 'Low Risk',
            'css_class': 'risk-low',
            'probability_percent': prob_percent,
            'summary': 'Policyholder demonstrates low incident propensity.',
            'guidance': 'Policyholder characteristics align with low historical claim frequencies. Standard underwriting terms indicated.'
        }
    elif probability < thresholds['moderate']:
        return {
            'tier': 'Moderate Risk',
            'css_class': 'risk-moderate',
            'probability_percent': prob_percent,
            'summary': 'Policyholder demonstrates moderate incident propensity.',
            'guidance': 'Profile exhibits average claim probability. Normal portfolio distribution expected.'
        }
    elif probability < thresholds['elevated']:
        return {
            'tier': 'Elevated Risk',
            'css_class': 'risk-elevated',
            'probability_percent': prob_percent,
            'summary': 'Policyholder demonstrates elevated claim likelihood.',
            'guidance': 'Notable risk factors detected (e.g., higher mileage or violation history). Closer underwriting review or risk-adjusted terms suggested.'
        }
    else:
        return {
            'tier': 'High Risk',
            'css_class': 'risk-high',
            'probability_percent': prob_percent,
            'summary': 'Policyholder demonstrates substantially high claim likelihood.',
            'guidance': 'High concentration of historical risk drivers. Increased likelihood of claim filing during policy period.'
        }

def get_decision_threshold() -> float:
    try:
        return float(get_setting('default_decision_threshold', Config.DEFAULT_DECISION_THRESHOLD))
    except Exception:
        return Config.DEFAULT_DECISION_THRESHOLD
