import streamlit as st
import pandas as pd
import joblib

from app_config import (
    STATE_MODEL_PATH,
    STATE_FEATURES_PATH,
    STATE_DATA_PATH,
    STATE_PREDICTIONS_PATH,
    STATE_COEFFICIENTS_PATH,
    STATE_RESULTS_PATH,
    PERSONAL_MODEL_PATH,
    PERSONAL_FEATURES_PATH,
    PERSONAL_RESULTS_PATH,
    PERSONAL_IMPORTANCE_PATH,
)


# ============================================================
# data_loader.py
# Loads saved models and processed project files
# ============================================================


@st.cache_resource
def load_models():
    state_model = joblib.load(STATE_MODEL_PATH)
    state_features = joblib.load(STATE_FEATURES_PATH)

    personal_model = joblib.load(PERSONAL_MODEL_PATH)
    personal_features = joblib.load(PERSONAL_FEATURES_PATH)

    return {
        "state_model": state_model,
        "state_features": state_features,
        "personal_model": personal_model,
        "personal_features": personal_features,
    }


@st.cache_data
def load_data():
    state_df = pd.read_csv(STATE_DATA_PATH)
    state_predictions = pd.read_csv(STATE_PREDICTIONS_PATH)
    state_coefficients = pd.read_csv(STATE_COEFFICIENTS_PATH)
    state_results = pd.read_csv(STATE_RESULTS_PATH)

    personal_results = pd.read_csv(PERSONAL_RESULTS_PATH)
    personal_importance = pd.read_csv(PERSONAL_IMPORTANCE_PATH)

    return {
        "state_df": state_df,
        "state_predictions": state_predictions,
        "state_coefficients": state_coefficients,
        "state_results": state_results,
        "personal_results": personal_results,
        "personal_importance": personal_importance,
    }