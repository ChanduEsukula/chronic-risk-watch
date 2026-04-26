import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib


# ============================================================
# 03_train_models.py
# Chronic Disease Early Warning System
#
# Purpose:
# Train first classification models to predict whether a state
# becomes high-risk for adult diabetes in the following year.
#
# Models:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# ============================================================


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

modeling_file_path = Path(
    "/Users/chanduesukula/Downloads/ML Project Self/chronic-disease-warning-system/untitled folder/data/processed/diabetes_state_year_modeling_data.csv"
)

project_root = modeling_file_path.parents[2]

models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

results_dir = project_root / "data" / "processed"
results_dir.mkdir(parents=True, exist_ok=True)

best_model_output_path = models_dir / "best_diabetes_warning_model.pkl"
feature_columns_output_path = models_dir / "feature_columns.pkl"
model_results_output_path = results_dir / "model_comparison_results.csv"
state_predictions_output_path = results_dir / "state_level_predictions.csv"


# ------------------------------------------------------------
# 2. Load modeling dataset
# ------------------------------------------------------------

print("\nLoading modeling dataset...\n")

df = pd.read_csv(modeling_file_path)

print("Modeling dataset loaded.")
print(f"Shape: {df.shape}")

print("\nColumns:")
for col in df.columns:
    print(f"- {col}")


# ------------------------------------------------------------
# 3. Confirm target
# ------------------------------------------------------------

target_col = "high_risk_diabetes_next_year"

if target_col not in df.columns:
    raise ValueError(f"Target column not found: {target_col}")

df[target_col] = df[target_col].astype(int)

print("\nTarget distribution:")
print(df[target_col].value_counts())

print("\nTarget distribution percentage:")
print((df[target_col].value_counts(normalize=True) * 100).round(2))


# ------------------------------------------------------------
# 4. Choose feature columns
# ------------------------------------------------------------

# We do NOT use:
# - LocationAbbr / LocationDesc as model features yet
# - diabetes_next_year because that is future information
# - diabetes_high_risk_threshold because it is used to create the label
# - target column itself

feature_cols = [
    "current_smoking_adults",
    "diabetes_adults",
    "high_blood_pressure_adults",
    "high_cholesterol_screened_adults",
    "lack_health_insurance_adults_18_64",
    "no_leisure_physical_activity_adults",
    "obesity_adults",
    "routine_checkup_past_year_adults",
    "diabetes_previous_year",
    "diabetes_change_from_previous_year",
]

missing_features = [col for col in feature_cols if col not in df.columns]

if missing_features:
    raise ValueError(f"Missing feature columns: {missing_features}")

X = df[feature_cols].copy()
y = df[target_col].copy()

print("\nFeature columns:")
for col in feature_cols:
    print(f"- {col}")

print("\nMissing values in features:")
print(X.isna().sum())


# ------------------------------------------------------------
# 5. Time-based train/test split
# ------------------------------------------------------------

# Current available years after target creation are likely 2019, 2020, 2021.
# We use the latest available year as the test set.
# This is better than random split because we are predicting future risk.

latest_year = df["YearStart"].max()

train_df = df[df["YearStart"] < latest_year].copy()
test_df = df[df["YearStart"] == latest_year].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].astype(int).copy()

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].astype(int).copy()

print("\nTime-based split:")
print(f"Latest year used for test: {latest_year}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

print("\nTrain target distribution:")
print(y_train.value_counts())

print("\nTest target distribution:")
print(y_test.value_counts())


# ------------------------------------------------------------
# 6. Define models
# ------------------------------------------------------------

models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    ),

    "Decision Tree": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=4,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
}


# ------------------------------------------------------------
# 7. Evaluation function
# ------------------------------------------------------------

def evaluate_model(model_name, model, X_test, y_test):
    """
    Evaluate a trained classification model.
    """

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if y_proba is not None and len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = np.nan

    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    print("\nMetrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


# ------------------------------------------------------------
# 8. Train and evaluate models
# ------------------------------------------------------------

results = {}
results_list = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    print(f"{model_name} training complete.")

    metrics = evaluate_model(model_name, model, X_test, y_test)

    results[model_name] = model
    results_list.append(metrics)


# ------------------------------------------------------------
# 9. Save model comparison results
# ------------------------------------------------------------

results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values(
    by=["recall", "f1_score", "roc_auc"],
    ascending=False
)

results_df.to_csv(model_results_output_path, index=False)

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(results_df)

print(f"\nSaved model comparison results to:\n{model_results_output_path}")


# ------------------------------------------------------------
# 10. Select best model
# ------------------------------------------------------------

# For early warning, recall is most important.
# If recall ties, use F1-score and ROC-AUC.

best_model_name = results_df.iloc[0]["model"]
best_model = results[best_model_name]

print("\nBest model selected:")
print(best_model_name)


# ------------------------------------------------------------
# 11. Save best model and feature list
# ------------------------------------------------------------

joblib.dump(best_model, best_model_output_path)
joblib.dump(feature_cols, feature_columns_output_path)

print(f"\nSaved best model to:\n{best_model_output_path}")
print(f"Saved feature columns to:\n{feature_columns_output_path}")


# ------------------------------------------------------------
# 12. Save state-level test predictions
# ------------------------------------------------------------

test_predictions = test_df[
    [
        "LocationAbbr",
        "LocationDesc",
        "YearStart",
        "diabetes_adults",
        "diabetes_next_year",
        "high_risk_diabetes_next_year",
    ]
].copy()

test_predictions["predicted_high_risk"] = best_model.predict(X_test)

if hasattr(best_model, "predict_proba"):
    test_predictions["predicted_probability_high_risk"] = best_model.predict_proba(X_test)[:, 1]
else:
    test_predictions["predicted_probability_high_risk"] = np.nan

test_predictions = test_predictions.sort_values(
    by="predicted_probability_high_risk",
    ascending=False,
)

test_predictions.to_csv(state_predictions_output_path, index=False)

print(f"\nSaved state-level predictions to:\n{state_predictions_output_path}")

print("\nTop 15 states by predicted high-risk probability:")
print(test_predictions.head(15))


# ------------------------------------------------------------
# 13. Feature importance / coefficients
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("MODEL INTERPRETATION")
print("=" * 80)

if best_model_name == "Logistic Regression":
    coefficients = best_model.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values(by="absolute_coefficient", ascending=False)

    print("\nLogistic Regression coefficients:")
    print(coef_df)

    coef_output_path = results_dir / "logistic_regression_coefficients.csv"
    coef_df.to_csv(coef_output_path, index=False)

    print(f"\nSaved coefficients to:\n{coef_output_path}")

elif best_model_name in ["Decision Tree", "Random Forest"]:
    importances = best_model.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    print("\nFeature importances:")
    print(importance_df)

    importance_output_path = results_dir / "feature_importances.csv"
    importance_df.to_csv(importance_output_path, index=False)

    print(f"\nSaved feature importances to:\n{importance_output_path}")


# ------------------------------------------------------------
# 14. Done
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE")
print("=" * 80)

print(
    "\nNext step:\n"
    "Review model_comparison_results.csv and state_level_predictions.csv. "
    "Then we will create visualizations and improve the model if needed."
)