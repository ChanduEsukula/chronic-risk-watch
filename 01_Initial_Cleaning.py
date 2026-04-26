import pandas as pd
from pathlib import Path


# ============================================================
# 01_data_overview.py
# Chronic Disease Early Warning System
#
# Purpose:
# This script loads the CDC Chronic Disease Indicators dataset
# and performs the first basic inspection:
# - dataset shape
# - column names
# - data types
# - first rows
# - available topics
# - diabetes-related questions
# - missing values
# - years and locations available
# ============================================================


# ------------------------------------------------------------
# 1. File path
# ------------------------------------------------------------

file_path = Path(
    "/Users/chanduesukula/Downloads/ML Project Self/chronic-disease-warning-system/untitled folder/data/raw/U.S._Chronic_Disease_Indicators.csv"
)


# ------------------------------------------------------------
# 2. Check whether file exists
# ------------------------------------------------------------

if not file_path.exists():
    raise FileNotFoundError(
        f"\nFile not found:\n{file_path}\n\n"
        "Please check that the CSV file is in the correct folder."
    )


# ------------------------------------------------------------
# 3. Load dataset
# ------------------------------------------------------------

print("\nLoading CDC Chronic Disease Indicators dataset...\n")

df = pd.read_csv(file_path, low_memory=False)

print("Dataset loaded successfully.")


# ------------------------------------------------------------
# 4. Basic dataset overview
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("1. DATASET SHAPE")
print("=" * 80)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


print("\n" + "=" * 80)
print("2. COLUMN NAMES")
print("=" * 80)

for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")


print("\n" + "=" * 80)
print("3. FIRST 5 ROWS")
print("=" * 80)

print(df.head())


print("\n" + "=" * 80)
print("4. DATA TYPES")
print("=" * 80)

print(df.dtypes)


print("\n" + "=" * 80)
print("5. DATASET INFO")
print("=" * 80)

df.info()


# ------------------------------------------------------------
# 5. Important columns preview
# ------------------------------------------------------------

important_columns = [
    "YearStart",
    "YearEnd",
    "LocationAbbr",
    "LocationDesc",
    "Topic",
    "Question",
    "DataValue",
    "DataValueType",
    "StratificationCategory1",
    "Stratification1",
]

existing_important_columns = [
    col for col in important_columns if col in df.columns
]

print("\n" + "=" * 80)
print("6. IMPORTANT COLUMNS PREVIEW")
print("=" * 80)

if existing_important_columns:
    print(df[existing_important_columns].head(10))
else:
    print("None of the expected important columns were found.")
    print("Check the actual column names above.")


# ------------------------------------------------------------
# 6. Years available
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("7. YEARS AVAILABLE")
print("=" * 80)

if "YearStart" in df.columns:
    years = sorted(df["YearStart"].dropna().unique())
    print(years)
    print(f"\nEarliest year: {min(years)}")
    print(f"Latest year: {max(years)}")
else:
    print("Column 'YearStart' not found.")


# ------------------------------------------------------------
# 7. Locations available
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("8. LOCATIONS AVAILABLE")
print("=" * 80)

if "LocationDesc" in df.columns:
    locations = sorted(df["LocationDesc"].dropna().unique())
    print(f"Number of unique locations: {len(locations)}")
    print("\nFirst 25 locations:")
    for loc in locations[:25]:
        print(f"- {loc}")
else:
    print("Column 'LocationDesc' not found.")


# ------------------------------------------------------------
# 8. Topics available
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("9. TOPICS AVAILABLE")
print("=" * 80)

if "Topic" in df.columns:
    topic_counts = df["Topic"].value_counts()
    print(topic_counts)
else:
    print("Column 'Topic' not found.")


# ------------------------------------------------------------
# 9. DataValueType values
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("10. DATA VALUE TYPES")
print("=" * 80)

if "DataValueType" in df.columns:
    print(df["DataValueType"].value_counts(dropna=False))
else:
    print("Column 'DataValueType' not found.")


# ------------------------------------------------------------
# 10. Stratification values
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("11. STRATIFICATION CATEGORIES")
print("=" * 80)

if "StratificationCategory1" in df.columns:
    print(df["StratificationCategory1"].value_counts(dropna=False))
else:
    print("Column 'StratificationCategory1' not found.")


print("\n" + "=" * 80)
print("12. STRATIFICATION VALUES")
print("=" * 80)

if "Stratification1" in df.columns:
    print(df["Stratification1"].value_counts(dropna=False).head(30))
else:
    print("Column 'Stratification1' not found.")


# ------------------------------------------------------------
# 11. Missing values
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("13. MISSING VALUES BY COLUMN")
print("=" * 80)

missing_values = df.isna().sum().sort_values(ascending=False)
missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)

missing_summary = pd.DataFrame(
    {
        "missing_count": missing_values,
        "missing_percent": missing_percent.round(2),
    }
)

print(missing_summary)


# ------------------------------------------------------------
# 12. Diabetes-related questions
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("14. DIABETES-RELATED QUESTIONS")
print("=" * 80)

if "Topic" in df.columns and "Question" in df.columns:
    diabetes_df = df[
        df["Topic"].str.contains("Diabetes", case=False, na=False)
    ]

    diabetes_questions = diabetes_df["Question"].value_counts()

    print(f"Number of diabetes rows: {diabetes_df.shape[0]}")
    print(f"Number of unique diabetes questions: {diabetes_questions.shape[0]}")
    print("\nTop diabetes questions:")
    print(diabetes_questions.head(50))
else:
    print("Columns 'Topic' and/or 'Question' not found.")


# ------------------------------------------------------------
# 13. Search for useful chronic disease risk-factor keywords
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("15. USEFUL RISK-FACTOR QUESTIONS BY KEYWORD")
print("=" * 80)

keywords = [
    "diabetes",
    "obesity",
    "overweight",
    "physical activity",
    "physical inactivity",
    "smoking",
    "tobacco",
    "hypertension",
    "blood pressure",
    "cholesterol",
    "cardiovascular",
    "stroke",
    "coronary",
    "checkup",
    "screening",
    "insurance",
    "preventive",
]

if "Question" in df.columns:
    for keyword in keywords:
        keyword_questions = (
            df[df["Question"].str.contains(keyword, case=False, na=False)]
            ["Question"]
            .value_counts()
        )

        print("\n" + "-" * 80)
        print(f"Keyword: {keyword}")
        print("-" * 80)

        if keyword_questions.empty:
            print("No questions found.")
        else:
            print(keyword_questions.head(15))
else:
    print("Column 'Question' not found.")


# ------------------------------------------------------------
# 14. Overall population filter check
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("16. OVERALL POPULATION ROWS CHECK")
print("=" * 80)

if "StratificationCategory1" in df.columns and "Stratification1" in df.columns:
    overall_df = df[
        (df["StratificationCategory1"].astype(str).str.lower() == "overall")
        & (df["Stratification1"].astype(str).str.lower() == "overall")
    ]

    print(f"Rows with Overall / Overall stratification: {overall_df.shape[0]}")
    print(f"Percentage of dataset: {overall_df.shape[0] / df.shape[0] * 100:.2f}%")

    if "Topic" in overall_df.columns:
        print("\nTopics available in Overall rows:")
        print(overall_df["Topic"].value_counts())
else:
    print("Stratification columns not found.")


# ------------------------------------------------------------
# 15. Preview possible state-level overall diabetes data
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("17. POSSIBLE STATE-LEVEL OVERALL DIABETES DATA PREVIEW")
print("=" * 80)

required_cols = [
    "YearStart",
    "LocationDesc",
    "Topic",
    "Question",
    "DataValue",
    "DataValueType",
    "StratificationCategory1",
    "Stratification1",
]

if all(col in df.columns for col in required_cols):
    preview_diabetes = df[
        (df["Topic"].str.contains("Diabetes", case=False, na=False))
        & (df["StratificationCategory1"].astype(str).str.lower() == "overall")
        & (df["Stratification1"].astype(str).str.lower() == "overall")
    ][required_cols]

    print(preview_diabetes.head(25))
else:
    print("One or more required columns are missing.")


# ------------------------------------------------------------
# 16. Save overview outputs
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("18. SAVING OVERVIEW OUTPUTS")
print("=" * 80)

project_root = file_path.parents[2]
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

columns_output_path = output_dir / "column_names.csv"
topics_output_path = output_dir / "topic_counts.csv"
missing_output_path = output_dir / "missing_values_summary.csv"
diabetes_questions_output_path = output_dir / "diabetes_questions.csv"

pd.DataFrame({"column_name": df.columns}).to_csv(columns_output_path, index=False)

if "Topic" in df.columns:
    df["Topic"].value_counts().reset_index().rename(
        columns={"index": "Topic", "Topic": "count"}
    ).to_csv(topics_output_path, index=False)

missing_summary.to_csv(missing_output_path)

if "Topic" in df.columns and "Question" in df.columns:
    diabetes_questions.reset_index().rename(
        columns={"index": "Question", "Question": "count"}
    ).to_csv(diabetes_questions_output_path, index=False)

print(f"Saved column names to: {columns_output_path}")
print(f"Saved topic counts to: {topics_output_path}")
print(f"Saved missing values summary to: {missing_output_path}")
print(f"Saved diabetes questions to: {diabetes_questions_output_path}")


# ------------------------------------------------------------
# 17. Done
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DATA OVERVIEW COMPLETE")
print("=" * 80)

print(
    "\nNext step:\n"
    "Look at the diabetes questions printed above and choose the exact diabetes prevalence indicator "
    "we will use as the target variable."
)