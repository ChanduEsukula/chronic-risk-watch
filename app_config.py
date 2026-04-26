from pathlib import Path


# ============================================================
# app_config.py
# Project paths and page names
# ============================================================


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR / "untitled folder"


STATE_MODEL_PATH = PROJECT_ROOT / "models" / "best_diabetes_warning_model.pkl"
STATE_FEATURES_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"
STATE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_state_year_modeling_data.csv"
STATE_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "state_level_predictions_with_results.csv"
STATE_COEFFICIENTS_PATH = PROJECT_ROOT / "data" / "processed" / "logistic_regression_coefficients.csv"
STATE_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "model_comparison_results.csv"


PERSONAL_MODEL_PATH = PROJECT_ROOT / "models" / "personal_diabetes_risk_model.pkl"
PERSONAL_FEATURES_PATH = PROJECT_ROOT / "models" / "personal_diabetes_feature_columns.pkl"
PERSONAL_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "personal_diabetes_model_results.csv"
PERSONAL_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "personal_diabetes_feature_importance.csv"


PAGES = [
    "Home",
    "Personal Risk Check",
    "State Risk Explorer",
    "Combined Risk View",
    "Compare States",
    "Model Insights",
]