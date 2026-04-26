import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 02_build_modeling_dataset.py
# Chronic Disease Early Warning System
#
# Purpose:
# Build the first clean state-year modeling dataset.
#
# Target:
# Predict whether a state becomes high-risk for adult diabetes
# in the following year.
#
# Main target question:
# "Diabetes among adults"
# ============================================================


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

raw_file_path = Path(
    "/Users/chanduesukula/Downloads/ML Project Self/chronic-disease-warning-system/untitled folder/data/raw/U.S._Chronic_Disease_Indicators.csv"
)

project_root = raw_file_path.parents[2]

processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

output_file_path = processed_dir / "diabetes_state_year_modeling_data.csv"
selected_long_output_path = processed_dir / "selected_indicators_long_format.csv"


# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

print("\nLoading raw CDC Chronic Disease Indicators dataset...\n")

df = pd.read_csv(raw_file_path, low_memory=False)

print("Raw dataset loaded.")
print(f"Raw shape: {df.shape}")


# ------------------------------------------------------------
# 3. Remove columns that are fully empty or not needed
# ------------------------------------------------------------

columns_to_drop = [
    "Response",
    "ResponseID",
    "StratificationCategory2",
    "Stratification2",
    "StratificationCategory3",
    "Stratification3",
    "StratificationCategoryID2",
    "StratificationID2",
    "StratificationCategoryID3",
    "StratificationID3",
]

existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=existing_columns_to_drop)

print("\nDropped fully empty / unnecessary columns.")
print(f"Shape after dropping columns: {df.shape}")


# ------------------------------------------------------------
# 4. Filter to U.S. states only
# ------------------------------------------------------------

# We will keep 50 states + District of Columbia.
# We will remove United States total and territories like Guam.

state_abbrs = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY"
]

df_states = df[df["LocationAbbr"].isin(state_abbrs)].copy()

print("\nFiltered to 50 states + DC.")
print(f"Shape after state filter: {df_states.shape}")
print(f"Number of locations: {df_states['LocationDesc'].nunique()}")


# ------------------------------------------------------------
# 5. Filter to Overall / Overall rows
# ------------------------------------------------------------

df_overall = df_states[
    (df_states["StratificationCategory1"].astype(str).str.lower() == "overall")
    & (df_states["Stratification1"].astype(str).str.lower() == "overall")
].copy()

print("\nFiltered to Overall / Overall population rows.")
print(f"Shape after overall filter: {df_overall.shape}")


# ------------------------------------------------------------
# 6. Keep useful DataValueType values
# ------------------------------------------------------------

# First version focuses on prevalence percentages.
# We do not use mortality rates or raw counts for the main model yet.

valid_value_types = [
    "Age-adjusted Prevalence",
    "Crude Prevalence",
]

df_prev = df_overall[df_overall["DataValueType"].isin(valid_value_types)].copy()

print("\nFiltered to prevalence rows only.")
print(f"Shape after prevalence filter: {df_prev.shape}")
print("\nDataValueType counts:")
print(df_prev["DataValueType"].value_counts())


# ------------------------------------------------------------
# 7. Select useful questions for first model
# ------------------------------------------------------------

# These came from your first overview script.
# They are good early-warning predictors for diabetes risk.

selected_questions = [
    # Target
    "Diabetes among adults",

    # Lifestyle / behavioral risk indicators
    "Obesity among adults",
    "No leisure-time physical activity among adults",
    "Current cigarette smoking among adults",

    # Cardiovascular / metabolic risk indicators
    "High blood pressure among adults",
    "High cholesterol among adults who have been screened",

    # Prevention / access indicators
    "Routine checkup within the past year among adults",
    "Lack of health insurance among adults aged 18-64 years",
]

df_selected = df_prev[df_prev["Question"].isin(selected_questions)].copy()

print("\nFiltered to selected project questions.")
print(f"Shape after question filter: {df_selected.shape}")

print("\nSelected question counts:")
print(df_selected["Question"].value_counts())


# ------------------------------------------------------------
# 8. Check selected question availability by DataValueType
# ------------------------------------------------------------

print("\nSelected question availability by DataValueType:")
availability = pd.crosstab(df_selected["Question"], df_selected["DataValueType"])
print(availability)


# ------------------------------------------------------------
# 9. Prefer Age-adjusted Prevalence when duplicate exists
# ------------------------------------------------------------

# Some state-year-question combinations may have both crude and age-adjusted prevalence.
# We prefer age-adjusted prevalence because it is better for comparing states.
# If age-adjusted does not exist, we keep crude prevalence.

df_selected["value_type_priority"] = np.where(
    df_selected["DataValueType"] == "Age-adjusted Prevalence",
    1,
    2
)

df_selected = df_selected.sort_values(
    by=[
        "LocationAbbr",
        "LocationDesc",
        "YearStart",
        "Question",
        "value_type_priority",
    ]
)

df_dedup = df_selected.drop_duplicates(
    subset=["LocationAbbr", "LocationDesc", "YearStart", "Question"],
    keep="first"
).copy()

print("\nAfter choosing preferred DataValueType and removing duplicates:")
print(f"Shape: {df_dedup.shape}")


# ------------------------------------------------------------
# 10. Create clean feature names
# ------------------------------------------------------------

question_to_feature = {
    "Diabetes among adults": "diabetes_adults",
    "Obesity among adults": "obesity_adults",
    "No leisure-time physical activity among adults": "no_leisure_physical_activity_adults",
    "Current cigarette smoking among adults": "current_smoking_adults",
    "High blood pressure among adults": "high_blood_pressure_adults",
    "High cholesterol among adults who have been screened": "high_cholesterol_screened_adults",
    "Routine checkup within the past year among adults": "routine_checkup_past_year_adults",
    "Lack of health insurance among adults aged 18-64 years": "lack_health_insurance_adults_18_64",
}

df_dedup["feature_name"] = df_dedup["Question"].map(question_to_feature)


# ------------------------------------------------------------
# 11. Save selected long-format data
# ------------------------------------------------------------

long_columns = [
    "YearStart",
    "YearEnd",
    "LocationAbbr",
    "LocationDesc",
    "Topic",
    "Question",
    "feature_name",
    "DataValueType",
    "DataValue",
    "LowConfidenceLimit",
    "HighConfidenceLimit",
    "StratificationCategory1",
    "Stratification1",
]

existing_long_columns = [col for col in long_columns if col in df_dedup.columns]

df_dedup[existing_long_columns].to_csv(selected_long_output_path, index=False)

print(f"\nSaved selected long-format data to:\n{selected_long_output_path}")


# ------------------------------------------------------------
# 12. Pivot to wide state-year format
# ------------------------------------------------------------

model_df = df_dedup.pivot_table(
    index=["LocationAbbr", "LocationDesc", "YearStart"],
    columns="feature_name",
    values="DataValue",
    aggfunc="first"
).reset_index()

model_df.columns.name = None

print("\nCreated wide state-year modeling table.")
print(f"Shape: {model_df.shape}")

print("\nWide table preview:")
print(model_df.head())


# ------------------------------------------------------------
# 13. Sort by state and year
# ------------------------------------------------------------

model_df = model_df.sort_values(["LocationAbbr", "YearStart"]).reset_index(drop=True)


# ------------------------------------------------------------
# 14. Create next-year diabetes target
# ------------------------------------------------------------

model_df["diabetes_next_year"] = (
    model_df.groupby("LocationAbbr")["diabetes_adults"].shift(-1)
)

print("\nCreated diabetes_next_year target using next available row per state.")


# ------------------------------------------------------------
# 15. Create high-risk label using top 25% threshold by year
# ------------------------------------------------------------

# For each current year t, diabetes_next_year represents the outcome in t+1.
# We calculate the 75th percentile across states for that target year group.
# Since diabetes_next_year is stored on the current year row, we group by YearStart.

model_df["high_risk_diabetes_next_year"] = np.nan

for year in sorted(model_df["YearStart"].dropna().unique()):
    year_mask = model_df["YearStart"] == year
    year_values = model_df.loc[year_mask, "diabetes_next_year"].dropna()

    if len(year_values) == 0:
        continue

    threshold = year_values.quantile(0.75)

    model_df.loc[year_mask, "diabetes_high_risk_threshold"] = threshold
    model_df.loc[year_mask, "high_risk_diabetes_next_year"] = (
        model_df.loc[year_mask, "diabetes_next_year"] >= threshold
    ).astype(int)

print("\nCreated high-risk diabetes next-year label using top 25% threshold.")


# ------------------------------------------------------------
# 16. Add simple trend feature
# ------------------------------------------------------------

model_df["diabetes_previous_year"] = (
    model_df.groupby("LocationAbbr")["diabetes_adults"].shift(1)
)

model_df["diabetes_change_from_previous_year"] = (
    model_df["diabetes_adults"] - model_df["diabetes_previous_year"]
)

print("\nAdded diabetes previous-year and change features.")


# ------------------------------------------------------------
# 17. Remove rows without target
# ------------------------------------------------------------

model_df_before_drop = model_df.shape[0]

model_df = model_df.dropna(subset=["diabetes_next_year", "high_risk_diabetes_next_year"])

print("\nRemoved rows without next-year diabetes target.")
print(f"Rows before drop: {model_df_before_drop}")
print(f"Rows after drop: {model_df.shape[0]}")


# ------------------------------------------------------------
# 18. Report missing values in modeling data
# ------------------------------------------------------------

print("\nMissing values in modeling dataset:")
missing_model = model_df.isna().sum().sort_values(ascending=False)
print(missing_model)


# ------------------------------------------------------------
# 19. Save final modeling dataset
# ------------------------------------------------------------

model_df.to_csv(output_file_path, index=False)

print(f"\nSaved final modeling dataset to:\n{output_file_path}")


# ------------------------------------------------------------
# 20. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("MODELING DATASET BUILD COMPLETE")
print("=" * 80)

print(f"\nFinal modeling dataset shape: {model_df.shape}")

print("\nColumns:")
for col in model_df.columns:
    print(f"- {col}")

print("\nTarget distribution:")
print(model_df["high_risk_diabetes_next_year"].value_counts(dropna=False))

print("\nTarget distribution percentage:")
print(model_df["high_risk_diabetes_next_year"].value_counts(normalize=True, dropna=False) * 100)

print("\nPreview:")
print(model_df.head(20))

print(
    "\nNext step:\n"
    "Open diabetes_state_year_modeling_data.csv and check whether the rows look correct. "
    "Then we will train the first Logistic Regression and Random Forest models."
)