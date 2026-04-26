import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


# ============================================================
# 04_create_visualizations.py
# Chronic Disease Early Warning System
#
# Purpose:
# Create visualizations for the project report and app.
#
# Visuals:
# 1. Model comparison - metrics
# 2. Logistic regression coefficients
# 3. Top predicted high-risk states
# 4. Confusion matrix
# 5. ROC curve
# 6. Diabetes prevalence trend for selected states
# 7. Actual vs predicted high-risk table/chart
# ============================================================


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

project_root = Path(
    "/Users/chanduesukula/Downloads/ML Project Self/chronic-disease-warning-system/untitled folder"
)

processed_dir = project_root / "data" / "processed"
models_dir = project_root / "models"
figures_dir = project_root / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

modeling_file_path = processed_dir / "diabetes_state_year_modeling_data.csv"
model_results_path = processed_dir / "model_comparison_results.csv"
predictions_path = processed_dir / "state_level_predictions.csv"
coefficients_path = processed_dir / "logistic_regression_coefficients.csv"

model_path = models_dir / "best_diabetes_warning_model.pkl"
feature_columns_path = models_dir / "feature_columns.pkl"


# ------------------------------------------------------------
# 2. Load files
# ------------------------------------------------------------

print("\nLoading project files...\n")

model_df = pd.read_csv(modeling_file_path)
results_df = pd.read_csv(model_results_path)
predictions_df = pd.read_csv(predictions_path)
coef_df = pd.read_csv(coefficients_path)

best_model = joblib.load(model_path)
feature_cols = joblib.load(feature_columns_path)

print("Files loaded successfully.")

print("\nModeling data shape:")
print(model_df.shape)

print("\nModel comparison:")
print(results_df)

print("\nPredictions preview:")
print(predictions_df.head())


# ------------------------------------------------------------
# 3. Helper function for saving figures
# ------------------------------------------------------------

def save_current_figure(filename):
    """
    Save the current matplotlib figure to the reports/figures folder.
    """
    output_path = figures_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.close()


# ------------------------------------------------------------
# 4. Model comparison chart
# ------------------------------------------------------------

print("\nCreating model comparison chart...")

metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

plot_df = results_df.set_index("model")[metrics_to_plot]

ax = plot_df.plot(kind="bar", figsize=(12, 6))

plt.title("Model Comparison for Diabetes Early Warning Prediction")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

save_current_figure("01_model_comparison.png")


# ------------------------------------------------------------
# 5. Logistic regression coefficient chart
# ------------------------------------------------------------

print("\nCreating logistic regression coefficient chart...")

coef_plot_df = coef_df.sort_values("coefficient", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(coef_plot_df["feature"], coef_plot_df["coefficient"])
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")

save_current_figure("02_logistic_regression_coefficients.png")


# ------------------------------------------------------------
# 6. Top predicted high-risk states
# ------------------------------------------------------------

print("\nCreating top predicted high-risk states chart...")

top_states = predictions_df.sort_values(
    "predicted_probability_high_risk",
    ascending=False
).head(15)

plt.figure(figsize=(10, 6))
plt.barh(
    top_states["LocationDesc"][::-1],
    top_states["predicted_probability_high_risk"][::-1],
)
plt.title("Top 15 States by Predicted High-Risk Diabetes Probability")
plt.xlabel("Predicted Probability of High Risk")
plt.ylabel("State")
plt.xlim(0, 1)

save_current_figure("03_top_predicted_high_risk_states.png")


# ------------------------------------------------------------
# 7. Recreate test set for confusion matrix and ROC curve
# ------------------------------------------------------------

print("\nRecreating test set for confusion matrix and ROC curve...")

target_col = "high_risk_diabetes_next_year"

latest_year = model_df["YearStart"].max()

test_df = model_df[model_df["YearStart"] == latest_year].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].astype(int).copy()

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(f"Latest year used for test: {latest_year}")
print(f"Test shape: {test_df.shape}")


# ------------------------------------------------------------
# 8. Confusion matrix
# ------------------------------------------------------------

print("\nCreating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not High Risk", "High Risk"],
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, values_format="d")
plt.title("Confusion Matrix - Logistic Regression")

save_current_figure("04_confusion_matrix.png")


# ------------------------------------------------------------
# 9. ROC curve
# ------------------------------------------------------------

print("\nCreating ROC curve...")

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC Curve, AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.title("ROC Curve - Diabetes Early Warning Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

save_current_figure("05_roc_curve.png")


# ------------------------------------------------------------
# 10. Diabetes trend for selected high-risk states
# ------------------------------------------------------------

print("\nCreating diabetes prevalence trend chart...")

selected_states = [
    "West Virginia",
    "Mississippi",
    "Alabama",
    "Louisiana",
    "Tennessee",
    "Minnesota",
]

trend_df = model_df[model_df["LocationDesc"].isin(selected_states)].copy()

plt.figure(figsize=(12, 6))

for state in selected_states:
    state_df = trend_df[trend_df["LocationDesc"] == state].sort_values("YearStart")
    if not state_df.empty:
        plt.plot(
            state_df["YearStart"],
            state_df["diabetes_adults"],
            marker="o",
            label=state,
        )

plt.title("Adult Diabetes Prevalence Trend for Selected States")
plt.xlabel("Year")
plt.ylabel("Adult Diabetes Prevalence")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

save_current_figure("06_diabetes_trend_selected_states.png")


# ------------------------------------------------------------
# 11. Actual vs predicted high-risk states
# ------------------------------------------------------------

print("\nCreating actual vs predicted chart...")

actual_pred_df = predictions_df.copy()

actual_pred_df["prediction_result"] = np.where(
    (actual_pred_df["high_risk_diabetes_next_year"] == 1)
    & (actual_pred_df["predicted_high_risk"] == 1),
    "True Positive",
    np.where(
        (actual_pred_df["high_risk_diabetes_next_year"] == 0)
        & (actual_pred_df["predicted_high_risk"] == 1),
        "False Positive",
        np.where(
            (actual_pred_df["high_risk_diabetes_next_year"] == 1)
            & (actual_pred_df["predicted_high_risk"] == 0),
            "False Negative",
            "True Negative",
        ),
    ),
)

result_counts = actual_pred_df["prediction_result"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(result_counts.index, result_counts.values)
plt.title("Prediction Result Breakdown")
plt.xlabel("Prediction Result")
plt.ylabel("Number of States")
plt.xticks(rotation=20)

save_current_figure("07_prediction_result_breakdown.png")


# ------------------------------------------------------------
# 12. Save enriched predictions with result labels
# ------------------------------------------------------------

enriched_predictions_path = processed_dir / "state_level_predictions_with_results.csv"
actual_pred_df.to_csv(enriched_predictions_path, index=False)

print(f"\nSaved enriched predictions to:\n{enriched_predictions_path}")


# ------------------------------------------------------------
# 13. Save figure summary file
# ------------------------------------------------------------

figure_summary = pd.DataFrame(
    {
        "figure": [
            "01_model_comparison.png",
            "02_logistic_regression_coefficients.png",
            "03_top_predicted_high_risk_states.png",
            "04_confusion_matrix.png",
            "05_roc_curve.png",
            "06_diabetes_trend_selected_states.png",
            "07_prediction_result_breakdown.png",
        ],
        "description": [
            "Compares accuracy, precision, recall, F1-score, and ROC-AUC across models.",
            "Shows which predictors have the strongest positive or negative relationship with high-risk diabetes prediction.",
            "Ranks the top 15 states by predicted probability of becoming high-risk for diabetes.",
            "Shows true positives, false positives, true negatives, and false negatives.",
            "Shows the model's ability to separate high-risk from not high-risk states.",
            "Shows diabetes prevalence trends for selected high-risk states and Minnesota.",
            "Summarizes prediction outcomes by result type.",
        ],
    }
)

figure_summary_path = figures_dir / "figure_summary.csv"
figure_summary.to_csv(figure_summary_path, index=False)

print(f"Saved figure summary to:\n{figure_summary_path}")


# ------------------------------------------------------------
# 14. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("VISUALIZATION CREATION COMPLETE")
print("=" * 80)

print("\nFigures saved in:")
print(figures_dir)

print("\nCreated figures:")
for fig_name in figure_summary["figure"]:
    print(f"- {fig_name}")

print(
    "\nNext step:\n"
    "Open the reports/figures folder and check the PNG files. "
    "After that, we can build the Streamlit app."
)