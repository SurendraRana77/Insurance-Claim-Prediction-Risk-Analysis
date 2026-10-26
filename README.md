# Insurance Claim Prediction & Risk Analysis System

An end-to-end, explainable, and deployable academic insurance decision-support system built with **Python**, **Flask**, **Scikit-learn**, **SQLite**, and a **Claymorphism** visual interface.

---

## 1. Project Overview & Scope

In motor vehicle insurance underwriting, estimating the probability of a policyholder filing a claim is critical for balanced portfolio risk management. This project builds a functional web application that provides:

1. **Executive Underwriting Dashboard**: High-level portfolio risk posture, dataset characteristics, and historical prediction volume.
2. **Individual Policyholder Prediction**: Real-time evaluation of claim likelihood using fitted Scikit-learn pipelines.
3. **Interactive Risk Categorization**: Mapping continuous claim probabilities into actionable risk tiers (Low, Moderate, Elevated, High) with configurable thresholds.
4. **Local Feature Attribution**: Interpretable explanations detailing how specific driver characteristics (e.g., annual mileage, speeding violations, credit score) influenced the prediction.
5. **Dataset Explorer & Validation**: Inspection of policyholder schemas, missing values, class distributions, and upload validation for custom CSV files with automatic data leakage detection.
6. **Model Diagnostics**: Real ROC curves, Precision-Recall curves, confusion matrices, and candidate algorithm benchmarking (Logistic Regression vs. Random Forest).
7. **Immutable Audit Trail**: SQLite-backed prediction history with search and filtering capabilities.

### Important Academic Boundaries
- **Probability vs. Severity**: The system models the **likelihood of event occurrence** (`claim_occurred`), *not* the dollar payout amount or loss reserve valuation (claim severity).
- **Claim Occurrence vs. Fraud**: A predicted claim reflects genuine insured risk, *not* deceptive or fraudulent claims.
- **Decision Support**: Risk categories are illustrative application thresholds and do not represent actuarially certified rate-making.

---

## 2. Technology Stack & Architecture

- **Backend Web Framework**: Python 3, Flask, Jinja2 template engine.
- **Machine Learning & Pipeline**: Scikit-learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`, `RandomForestClassifier`), Pandas, NumPy, Joblib.
- **Data Persistence**: SQLite3 with parameterized queries and thread-safe connection pooling.
- **Frontend & Visualization**: Semantic HTML5, Claymorphism CSS3 design system, Vanilla JavaScript, Chart.js.
- **Production Server**: Gunicorn WSGI server.
- **Deployment Platform**: Render Web Service (Single-service deployment).

---

## 3. Directory Structure

```text
insurance-claim-prediction/
├── app.py                      # Application entry point & factory
├── config.py                   # Centralized configuration (dev vs prod, paths)
├── requirements.txt            # Python dependencies
├── Procfile                    # Render / Gunicorn process definition
├── .gitignore                  # Git exclusions
├── README.md                   # Complete academic documentation
├── data/
│   ├── sample_insurance_data.csv # 10,000-record benchmark insurance dataset
│   └── README.md               # Data dictionary & schema notes
├── artifacts/
│   ├── model_pipeline.joblib   # Trained preprocessing + model pipeline
│   ├── metrics.json            # ROC-AUC, confusion matrix, and feature weights
│   └── README.md
├── database/
│   ├── db.py                   # SQLite helper functions & query logic
│   └── schema.sql              # Table definitions for predictions & settings
├── ml/
│   ├── pipeline.py             # Scikit-learn ColumnTransformer pipeline builder
│   ├── train.py                # Standalone training script & model comparator
│   ├── evaluate.py             # Metrics calculation & curve generation
│   └── generate_seed_data.py   # Benchmark dataset generation utility
├── services/
│   ├── data_service.py         # CSV ingestion, validation, and leakage checks
│   ├── model_service.py        # Model loading, inference, and explanations
│   └── risk_service.py         # Risk categorization & threshold logic
├── routes/
│   ├── dashboard_routes.py     # Executive overview routes
│   ├── prediction_routes.py    # Prediction form & AJAX endpoint
│   ├── data_routes.py          # Dataset explorer & upload handling
│   └── analytics_routes.py     # Risk analytics EDA & model performance
├── static/
│   ├── css/
│   │   ├── claymorphism.css    # Clay cards, tactile inputs, and risk pills
│   │   └── style.css           # Responsive grids, layout, tables, and alerts
│   └── js/
│       └── app.js              # Sidebar toggle, toast alerts, and safe charts
├── templates/
│   ├── base.html               # Shared layout & navigation
│   ├── dashboard.html          # Executive KPI overview
│   ├── predict.html            # Individual prediction workspace & scorecard
│   ├── dataset.html            # Dataset profiler & CSV upload zone
│   ├── analytics.html          # Underwriting exploratory risk analysis
│   ├── model.html              # Model evaluation, calibration metrics & benchmarks
│   ├── history.html            # Searchable prediction audit trail
│   ├── about.html              # Methodology, problem statement, and ethics
│   └── error.html              # Custom Claymorphism error pages (404, 413, 500)
```

---

## 4. Local Quickstart Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.14)
- Git

### 1. Clone & Set Up Virtual Environment
```bash
git clone <repository-url>
cd rana
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Benchmark Dataset & Train Initial Model
*(Note: If pre-generated artifacts exist, this step can be skipped, but running it guarantees a fresh state)*
```bash
python ml/generate_seed_data.py
python ml/train.py
```

### 4. Run the Application Locally
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 5. Render Deployment Instructions (Single-Service)

This repository is designed to deploy to **Render** as a single Web Service without external database or frontend services.

### Step-by-Step Deployment Guide:
1. **Push to GitHub**:
   Ensure all files are committed and pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Implement complete Insurance Claim Prediction System"
   git push origin main
   ```
2. **Create New Web Service on Render**:
   - Log into [Render.com](https://render.com).
   - Click **New +** -> **Web Service**.
   - Connect your GitHub repository.
3. **Configure Service Settings**:
   - **Name**: `insurance-claim-prediction` (or preferred name)
   - **Region**: Choose the closest region (e.g., Oregon, Frankfurt, Singapore)
   - **Branch**: `main`
   - **Root Directory**: Leave blank (root of repository)
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt && python ml/train.py
     ```
   - **Start Command**:
     ```bash
     gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120
     ```
   - **Plan**: `Free`
4. **Environment Variables**:
   Add the following environment variables under the **Environment** tab:
   - `SECRET_KEY`: Set to a strong random string (e.g., `render-prod-insurance-key-2026`).
   - `FLASK_ENV`: `production`
5. **Deploy**:
   - Click **Create Web Service**.
   - Render will build dependencies, execute model training on the seed dataset, and start the Gunicorn server.
   - Your live URL will be: `https://<service-name>.onrender.com`.

### Persistence & Ephemeral Filesystem Notes
- **Render's Free Tier** uses an ephemeral container filesystem. This means SQLite prediction records and user-uploaded CSVs reset when the dyno restarts or redeploys.
- The application is engineered to handle this gracefully:
  - The benchmark dataset (`sample_insurance_data.csv`) is bundled in Git.
  - The SQLite database is automatically initialized on startup via `init_db()`.
  - For durable multi-month persistence in production, attach a **Render Persistent Disk** and set `DB_PATH=/var/data/insurance_app.db`.

---

## 6. Academic Viva & Defense Talking Points

When presenting this project to an examination committee or evaluator:

1. **Why Calibrated Probabilities and ROC-AUC over Raw Accuracy?**
   - In insurance, claims are minority events (~23% empirical rate). While `class_weight='balanced'` allows models to effectively separate classes, it artificially inflates uncalibrated predicted probabilities (e.g., ~44% average predicted probability). We apply `CalibratedClassifierCV` (sigmoid / Platt scaling) using internal cross-validation on training data, restoring reliable probability estimates (Brier score: ~0.1498; mean predicted probability: 22.69% vs. 23.35% observed claim rate) while preserving ROC-AUC ranking power (0.75+).
2. **How is Data Leakage Prevented?**
   - Variables that only exist *after* an accident occurs (such as repair cost, claim settlement amount, or fraud flags) are strictly screened out during ingestion. Preprocessing steps (imputation, scaling, one-hot encoding) are fitted strictly on training data within cross-validation folds.
3. **Why Scikit-learn Pipelines with CalibratedClassifierCV?**
   - Packaging preprocessing and estimation into unified Scikit-learn pipeline objects ensures that runtime single-record inferences receive exact transformations without manual feature manipulation.
4. **How are Predictions Explained?**
   - For calibrated linear models, average feature coefficient weights across calibration estimators produce localized contribution vectors ($w_i \cdot x_i$), clearly demonstrating which policyholder characteristics pushed claim risk higher or lower.
5. **Why Claymorphism?**
   - Custom-engineered CSS tokens, soft dual-inset shadows, and tactile cards create a distinctive, clean visual hierarchy without heavy third-party CSS frameworks.
