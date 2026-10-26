from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def create_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = []
    if numerical_cols:
        transformers.append(('num', num_pipeline, numerical_cols))
    if categorical_cols:
        transformers.append(('cat', cat_pipeline, categorical_cols))
        
    return ColumnTransformer(transformers=transformers, remainder='drop')

def build_model_pipeline(
    numerical_cols: list,
    categorical_cols: list,
    model_type: str = 'logistic_regression',
    random_state: int = 42
) -> Pipeline:
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    if model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )
    else:
        # Default to interpretable Logistic Regression
        classifier = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        )
        
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
