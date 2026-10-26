"""
Benchmark Insurance Dataset Generator
Creates a realistic 10,000-record benchmark automobile insurance dataset for academic demonstration.
Features reflect genuine underwriting risk factors and produce ~21% claim occurrence.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

def generate_benchmark_dataset(output_path: str, n_samples: int = 10000, seed: int = 42):
    np.random.seed(seed)
    
    # 1. Driver Demographics
    ages = np.random.randint(18, 76, size=n_samples)
    genders = np.random.choice(['male', 'female'], size=n_samples, p=[0.51, 0.49])
    
    # Driving experience correlates with age
    driving_experience = np.clip(ages - 18 - np.random.randint(0, 6, size=n_samples), 0, 55)
    
    educations = np.random.choice(['high_school', 'bachelors', 'advanced'], size=n_samples, p=[0.42, 0.43, 0.15])
    incomes = np.random.choice(['low', 'middle', 'high', 'very_high'], size=n_samples, p=[0.28, 0.45, 0.20, 0.07])
    
    # Credit score (beta distribution shifted to 0.2 - 0.95)
    credit_scores = np.round(np.clip(np.random.beta(5, 3, size=n_samples) * 0.75 + 0.20, 0.15, 0.98), 3)
    
    # 2. Vehicle Attributes
    vehicle_ownership = np.random.choice([1, 0], size=n_samples, p=[0.68, 0.32]) # 1 = Owned, 0 = Leased/Financed
    vehicle_years = np.random.choice(['before_2015', 'after_2015'], size=n_samples, p=[0.42, 0.58])
    vehicle_types = np.random.choice(['sedan', 'suv', 'sports_car', 'compact'], size=n_samples, p=[0.45, 0.30, 0.08, 0.17])
    annual_mileage = np.random.normal(12500, 3500, size=n_samples).astype(int)
    annual_mileage = np.clip(annual_mileage, 3000, 38000)
    
    # 3. Driving History
    # Poisson distributed violations with experience attenuation
    exp_factor = np.exp(-driving_experience / 20.0)
    speeding_violations = np.random.poisson(lam=np.clip(1.2 * exp_factor + 0.4, 0.1, 4.0), size=n_samples)
    duis = np.random.poisson(lam=0.15 * exp_factor, size=n_samples)
    past_accidents = np.random.poisson(lam=np.clip(0.8 * exp_factor + 0.2, 0.05, 3.0), size=n_samples)
    
    # 4. Latent Risk Function for Claim Occurrence
    # Log-odds calculation based on actuarial risk factors
    log_odds = (
        -1.75
        - 0.035 * (driving_experience - 15)
        - 0.85 * (credit_scores - 0.60)
        + 0.00004 * (annual_mileage - 12000)
        + 0.28 * speeding_violations
        + 0.65 * duis
        + 0.45 * past_accidents
        + np.where(vehicle_years == 'before_2015', 0.35, 0.0)
        + np.where(vehicle_types == 'sports_car', 0.45, 0.0)
        + np.where(vehicle_types == 'compact', 0.12, 0.0)
        + np.where(vehicle_ownership == 0, 0.15, 0.0)
        + np.where(incomes == 'low', 0.20, 0.0)
        - np.where(incomes == 'very_high', 0.18, 0.0)
    )
    
    probabilities = 1.0 / (1.0 + np.exp(-log_odds))
    claim_occurred = (np.random.rand(n_samples) < probabilities).astype(int)
    
    df = pd.DataFrame({
        'id': np.arange(1001, 1001 + n_samples),
        'age': ages,
        'gender': genders,
        'driving_experience': driving_experience,
        'education': educations,
        'income_level': incomes,
        'credit_score': credit_scores,
        'vehicle_ownership': vehicle_ownership,
        'vehicle_year': vehicle_years,
        'vehicle_type': vehicle_types,
        'annual_mileage': annual_mileage,
        'speeding_violations': speeding_violations,
        'duis': duis,
        'past_accidents': past_accidents,
        'claim_occurred': claim_occurred
    })
    
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records in '{output_path}'. Claim rate: {df['claim_occurred'].mean():.2%}")
    return df

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import Config
    target_csv = os.path.join(Config.DATA_DIR, Config.DEFAULT_DATASET_NAME)
    generate_benchmark_dataset(target_csv)
