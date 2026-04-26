import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import joblib


# ============================================================
# 05_train_personal_diabetes_model.py
# Chronic Risk Watch
#
# Purpose:
# Train a personal diabetes risk-awareness model using the
# Kaggle/BRFSS Diabetes Health Indicators dataset.
#
# Dataset:
# diabetes_binary_health_indicators_BRFSS2015.csv
#
# Target:
# Diabetes_binary
# 0 = no diabetes
# 1 = prediabetes or diabetes
#
# Important:
# This model is for educational risk awareness only.
# It is NOT a medical diagnosis tool.
# ============================================================


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

raw_file_path = Path(
    "/Users/chanduesukula/Downloads/ML Project Self/chronic-disease-warning-system/untitled folder/data/raw/diabetes_binary_health_indicators_BRFSS2015.csv"
)

project_root = Path(
    "/Users/chanduesukula/Downloads/ML Project Self/chronic-disease-warning-system/untitled folder"
)

processed_dir = project_root / "data" / "processed"
models_dir = project_root / "models"
reports_dir = project_root / "reports"

processed_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

model_output_path = models_dir / "personal_diabetes_risk_model.pkl"
feature_columns_output_path = models_dir / "personal_diabetes_feature_columns.pkl"
results_output_path = processed_dir / "personal_diabetes_model_results.csv"
predictions_output_path = processed_dir / "personal_diabetes_test_predictions.csv"
feature_importance_output_path = processed_dir / "personal_diabetes_feature_importance.csv"


# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

print("\nLoading personal diabetes dataset...\n")

if not raw_file_path.exists():
    raise FileNotFoundError(
        f"File not found:\n{raw_file_path}\n\n"
        "Make sure diabetes_binary_health_indicators_BRFSS2015.csv is saved in data/raw."
    )

df = pd.read_csv(raw_file_path)

print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")

print("\nColumns:")
for col in df.columns:
    print(f"- {col}")


# ------------------------------------------------------------
# 3. Basic data check
# ------------------------------------------------------------

target_col = "Diabetes_binary"

if target_col not in df.columns:
    raise ValueError(f"Target column not found: {target_col}")

print("\nTarget distribution:")
print(df[target_col].value_counts())

print("\nTarget distribution percentage:")
print((df[target_col].value_counts(normalize=True) * 100).round(2))

print("\nMissing values:")
print(df.isna().sum())


# ------------------------------------------------------------
# 4. Define features
# ------------------------------------------------------------

# All columns except the target are used as features.
# These are BRFSS health/lifestyle indicators.

feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols].copy()
y = df[target_col].astype(int).copy()

print("\nFeature columns:")
for col in feature_cols:
    print(f"- {col}")

print(f"\nNumber of features: {len(feature_cols)}")


# ------------------------------------------------------------
# 5. Train/test split
# ------------------------------------------------------------

# Stratify keeps the diabetes/no-diabetes ratio similar in train and test.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTrain/test split complete.")
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
                    max_iter=1000,
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
                    n_estimators=250,
                    max_depth=12,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),

    "Gradient Boosting": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
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
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if y_proba is not None:
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

trained_models = {}
results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    print(f"{model_name} training complete.")

    metrics = evaluate_model(model_name, model, X_test, y_test)

    trained_models[model_name] = model
    results.append(metrics)


# ------------------------------------------------------------
# 9. Save model comparison results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

# For personal risk awareness, ROC-AUC and recall matter.
# Recall matters because missing a higher-risk person is worse.
results_df = results_df.sort_values(
    by=["recall", "f1_score", "roc_auc"],
    ascending=False,
)

results_df.to_csv(results_output_path, index=False)

print("\n" + "=" * 80)
print("PERSONAL MODEL COMPARISON")
print("=" * 80)

print(results_df)

print(f"\nSaved personal model results to:\n{results_output_path}")


# ------------------------------------------------------------
# 10. Select best model
# ------------------------------------------------------------

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

print("\nBest personal model selected:")
print(best_model_name)


# ------------------------------------------------------------
# 11. Save best model and feature columns
# ------------------------------------------------------------

joblib.dump(best_model, model_output_path)
joblib.dump(feature_cols, feature_columns_output_path)

print(f"\nSaved personal diabetes model to:\n{model_output_path}")
print(f"Saved personal feature columns to:\n{feature_columns_output_path}")


# ------------------------------------------------------------
# 12. Save test predictions
# ------------------------------------------------------------

test_predictions = X_test.copy()
test_predictions["actual_diabetes_binary"] = y_test.values
test_predictions["predicted_diabetes_binary"] = best_model.predict(X_test)

if hasattr(best_model, "predict_proba"):
    test_predictions["predicted_probability_diabetes"] = best_model.predict_proba(X_test)[:, 1]
else:
    test_predictions["predicted_probability_diabetes"] = np.nan

test_predictions.to_csv(predictions_output_path, index=False)

print(f"\nSaved personal test predictions to:\n{predictions_output_path}")


# ------------------------------------------------------------
# 13. Feature importance / coefficients
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PERSONAL MODEL INTERPRETATION")
print("=" * 80)

if best_model_name == "Logistic Regression":
    coefficients = best_model.named_steps["model"].coef_[0]

    interpretation_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    print("\nLogistic Regression coefficients:")
    print(interpretation_df)

elif best_model_name == "Random Forest":
    importances = best_model.named_steps["model"].feature_importances_

    interpretation_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    print("\nRandom Forest feature importances:")
    print(interpretation_df)

elif best_model_name == "Gradient Boosting":
    # HistGradientBoostingClassifier does not expose feature_importances_ directly.
    # We save a simple correlation-style summary as a lightweight interpretation.
    interpretation_rows = []

    for col in feature_cols:
        mean_no_diabetes = df[df[target_col] == 0][col].mean()
        mean_diabetes = df[df[target_col] == 1][col].mean()
        difference = mean_diabetes - mean_no_diabetes

        interpretation_rows.append(
            {
                "feature": col,
                "mean_no_diabetes": mean_no_diabetes,
                "mean_diabetes_or_prediabetes": mean_diabetes,
                "difference": difference,
                "absolute_difference": abs(difference),
            }
        )

    interpretation_df = pd.DataFrame(interpretation_rows).sort_values(
        "absolute_difference",
        ascending=False,
    )

    print("\nFeature mean differences by target group:")
    print(interpretation_df)

else:
    interpretation_df = pd.DataFrame({"feature": feature_cols})

interpretation_df.to_csv(feature_importance_output_path, index=False)

print(f"\nSaved personal model interpretation to:\n{feature_importance_output_path}")


# ------------------------------------------------------------
# 14. Create user-friendly risk bands
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("RISK BAND EXAMPLES")
print("=" * 80)

sample_probs = [0.10, 0.35, 0.55, 0.80]

for prob in sample_probs:
    if prob < 0.30:
        band = "Low"
    elif prob < 0.60:
        band = "Moderate"
    else:
        band = "High"

    print(f"Probability: {prob:.2f} -> Risk Awareness Level: {band}")


# ------------------------------------------------------------
# 15. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PERSONAL DIABETES MODEL TRAINING COMPLETE")
print("=" * 80)

print(f"\nBest model: {best_model_name}")

print("\nFiles created:")
print(f"- {model_output_path}")
print(f"- {feature_columns_output_path}")
print(f"- {results_output_path}")
print(f"- {predictions_output_path}")
print(f"- {feature_importance_output_path}")

print(
    "\nNext step:\n"
    "We will connect this personal diabetes model with the state-level model "
    "inside the Streamlit app."
)