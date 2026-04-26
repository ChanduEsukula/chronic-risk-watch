import pandas as pd


# ============================================================
# helpers.py
# Shared logic, mappings, BMI calculation, risk labels
# ============================================================


def get_risk_band(probability):
    """
    Used for personal risk-awareness model.
    Probability should be between 0 and 1.
    """
    if probability < 0.30:
        return "Low", "🟢"
    elif probability < 0.60:
        return "Moderate", "🟡"
    else:
        return "High", "🔴"


def get_state_signal_band(score):
    """
    Used for state-level model.

    This avoids calling the state output a real-world disease probability.
    Score should be between 0 and 1.
    """
    if score < 0.30:
        return "Low Signal", "🟢"
    elif score < 0.60:
        return "Moderate Signal", "🟡"
    elif score < 0.85:
        return "High Signal", "🟠"
    else:
        return "Very Strong Signal", "🔴"


def format_model_score(score):
    """
    Converts model score from 0-1 into 0-100 display format.
    Example: 0.999 -> 99.9 / 100
    """
    return f"{score * 100:.1f} / 100"


def state_signal_to_simple_band(state_signal):
    """
    Converts state signal labels into Low / Moderate / High
    so existing risk-card and combined-awareness functions can still work.
    """
    if state_signal in ["High Signal", "Very Strong Signal"]:
        return "High"
    elif state_signal == "Moderate Signal":
        return "Moderate"
    else:
        return "Low"


def get_combined_awareness(personal_band, state_band):
    if personal_band == "High" and state_band == "High":
        return "Very High Awareness", "🔴"
    elif personal_band == "High" and state_band in ["Low", "Moderate"]:
        return "High Awareness", "🟠"
    elif personal_band == "Moderate" and state_band == "High":
        return "Elevated Awareness", "🟠"
    elif personal_band == "Moderate" and state_band == "Moderate":
        return "Moderate Awareness", "🟡"
    elif personal_band == "Low" and state_band == "High":
        return "Watch Level", "🟡"
    else:
        return "Low Awareness", "🟢"


def yes_no_to_binary(value):
    return 1 if value == "Yes" else 0


def sex_to_binary(value):
    return 1 if value == "Male" else 0


def age_group_to_code(age_group):
    mapping = {
        "18-24": 1,
        "25-29": 2,
        "30-34": 3,
        "35-39": 4,
        "40-44": 5,
        "45-49": 6,
        "50-54": 7,
        "55-59": 8,
        "60-64": 9,
        "65-69": 10,
        "70-74": 11,
        "75-79": 12,
        "80+": 13,
    }
    return mapping[age_group]


def general_health_to_code(value):
    mapping = {
        "Excellent": 1,
        "Very good": 2,
        "Good": 3,
        "Fair": 4,
        "Poor": 5,
    }
    return mapping[value]


def education_to_code(value):
    mapping = {
        "Less than high school": 3,
        "High school graduate": 4,
        "Some college": 5,
        "College graduate": 6,
    }
    return mapping[value]


def income_to_code(value):
    mapping = {
        "Less than $25,000": 4,
        "$25,000-$35,000": 5,
        "$35,000-$50,000": 6,
        "$50,000-$75,000": 7,
        "$75,000 or more": 8,
    }
    return mapping[value]


def calculate_bmi_from_height_weight(height_feet, height_inches, weight_lbs):
    total_inches = (height_feet * 12) + height_inches

    if total_inches <= 0:
        return 0

    bmi = (weight_lbs / (total_inches ** 2)) * 703
    return round(bmi, 1)


def build_personal_input(
    personal_features,
    high_bp,
    high_chol,
    chol_check,
    bmi,
    smoker,
    stroke,
    heart_disease,
    phys_activity,
    fruits,
    veggies,
    heavy_alcohol,
    any_healthcare,
    no_doc_cost,
    general_health,
    mental_health_days,
    physical_health_days,
    diff_walk,
    sex,
    age_group,
    education,
    income,
):
    input_dict = {
        "HighBP": yes_no_to_binary(high_bp),
        "HighChol": yes_no_to_binary(high_chol),
        "CholCheck": yes_no_to_binary(chol_check),
        "BMI": bmi,
        "Smoker": yes_no_to_binary(smoker),
        "Stroke": yes_no_to_binary(stroke),
        "HeartDiseaseorAttack": yes_no_to_binary(heart_disease),
        "PhysActivity": yes_no_to_binary(phys_activity),
        "Fruits": yes_no_to_binary(fruits),
        "Veggies": yes_no_to_binary(veggies),
        "HvyAlcoholConsump": yes_no_to_binary(heavy_alcohol),
        "AnyHealthcare": yes_no_to_binary(any_healthcare),
        "NoDocbcCost": yes_no_to_binary(no_doc_cost),
        "GenHlth": general_health_to_code(general_health),
        "MentHlth": mental_health_days,
        "PhysHlth": physical_health_days,
        "DiffWalk": yes_no_to_binary(diff_walk),
        "Sex": sex_to_binary(sex),
        "Age": age_group_to_code(age_group),
        "Education": education_to_code(education),
        "Income": income_to_code(income),
    }

    personal_input = pd.DataFrame([input_dict])
    personal_input = personal_input[personal_features]

    return personal_input


def get_personal_risk_factors(input_df):
    row = input_df.iloc[0]
    factors = []

    if row["BMI"] >= 30:
        factors.append("BMI is in the obesity range.")
    elif row["BMI"] >= 25:
        factors.append("BMI is in the overweight range.")

    if row["HighBP"] == 1:
        factors.append("High blood pressure is present.")

    if row["HighChol"] == 1:
        factors.append("High cholesterol is present.")

    if row["Smoker"] == 1:
        factors.append("Smoking history is present.")

    if row["PhysActivity"] == 0:
        factors.append("No recent physical activity was reported.")

    if row["GenHlth"] >= 4:
        factors.append("General health was reported as fair or poor.")

    if row["DiffWalk"] == 1:
        factors.append("Difficulty walking was reported.")

    if row["HeartDiseaseorAttack"] == 1:
        factors.append("History of heart disease or heart attack was reported.")

    if row["Age"] >= 8:
        factors.append("Age group is 55 or older.")

    if len(factors) == 0:
        factors.append("No major self-reported risk factors were flagged from the inputs.")

    return factors


def get_state_prediction_for_selected_state(state_predictions, state_name):
    """
    Returns the selected state's row, model score, signal label, and icon.

    Important:
    The model_score is a model classification signal, not a real-world disease probability.
    """
    state_row = state_predictions[state_predictions["LocationDesc"] == state_name].copy()

    if state_row.empty:
        return None

    state_row = state_row.sort_values("YearStart", ascending=False).iloc[0]

    model_score = float(state_row["predicted_probability_high_risk"])
    signal_band, signal_icon = get_state_signal_band(model_score)

    return state_row, model_score, signal_band, signal_icon

def get_personal_guidance(personal_band):
    """
    Plain-English guidance for normal users based on personal risk-awareness band.
    This is educational only and not medical advice.
    """
    if personal_band == "Low":
        return {
            "title": "Your result looks lower risk based on your answers.",
            "meaning": (
                "Your answers do not strongly match the higher-risk diabetes or prediabetes "
                "profiles in the survey dataset. This does not mean zero risk, but it suggests "
                "your current profile is not strongly flagged by the model."
            ),
            "next_steps": [
                "Keep routine checkups.",
                "Stay physically active.",
                "Maintain healthy weight habits.",
                "Monitor blood pressure and cholesterol when possible.",
                "Ask about screening if you have symptoms or family history.",
            ],
            "dont_assume": [
                "This does not mean you have zero risk.",
                "This does not replace regular medical checkups.",
                "This does not rule out diabetes or prediabetes.",
            ],
        }

    elif personal_band == "Moderate":
        return {
            "title": "Your result shows some risk signals worth paying attention to.",
            "meaning": (
                "Your answers have some similarities with profiles linked to diabetes or prediabetes "
                "in the survey dataset. This does not mean you have diabetes, but it may be a good "
                "reminder to focus on prevention and screening awareness."
            ),
            "next_steps": [
                "Consider asking a healthcare professional whether diabetes screening is appropriate.",
                "Pay attention to BMI, blood pressure, cholesterol, and physical activity.",
                "Try to maintain consistent physical activity.",
                "Review eating habits and routine checkup patterns.",
                "Do not panic; use this as an awareness signal.",
            ],
            "dont_assume": [
                "This is not a diagnosis.",
                "This does not mean you definitely have diabetes.",
                "This score can change if inputs or health habits change.",
            ],
        }

    else:
        return {
            "title": "Your result shows stronger risk-awareness signals.",
            "meaning": (
                "Your answers more strongly match profiles linked to diabetes or prediabetes "
                "in the survey dataset. This does not mean you have diabetes, but it is a stronger "
                "reason to consider screening and prevention steps."
            ),
            "next_steps": [
                "Consider asking a healthcare professional about an A1C or fasting glucose test.",
                "Monitor blood pressure and cholesterol if relevant.",
                "Review weight, activity, and smoking-related risk factors.",
                "Do not panic; treat this as a prevention reminder.",
                "Seek medical advice if you have symptoms or family history.",
            ],
            "dont_assume": [
                "This does not confirm diabetes.",
                "This is not a medical diagnosis.",
                "A healthcare professional and lab tests are needed for diagnosis.",
            ],
        }


def get_state_guidance(state_signal):
    """
    Plain-English guidance for normal users based on state public-health signal.
    """
    if state_signal == "Low Signal":
        return {
            "title": "Your state has a lower public-health warning signal.",
            "meaning": (
                "The selected state is not strongly classified by the model as belonging to the "
                "higher diabetes-burden group. This is only state-level context and does not decide "
                "your personal health."
            ),
        }

    elif state_signal == "Moderate Signal":
        return {
            "title": "Your state has a moderate public-health warning signal.",
            "meaning": (
                "The selected state has some public-health patterns that the model associates with "
                "higher diabetes burden. This is useful context, but your personal health depends on "
                "your own risk factors and medical history."
            ),
        }

    elif state_signal == "High Signal":
        return {
            "title": "Your state has a high public-health warning signal.",
            "meaning": (
                "The selected state has stronger public-health patterns linked with higher diabetes burden. "
                "This does not mean you personally have diabetes, but it shows why prevention and screening "
                "awareness may matter in this state."
            ),
        }

    else:
        return {
            "title": "Your state has a very strong public-health warning signal.",
            "meaning": (
                "The selected state is strongly classified by the model as belonging to the higher "
                "diabetes-burden group. This is a state-level warning signal, not a personal diagnosis "
                "or a real-world disease probability."
            ),
        }


def format_feature_name(feature):
    names = {
        "current_smoking_adults": "Current smoking among adults",
        "diabetes_adults": "Adult diabetes prevalence",
        "high_blood_pressure_adults": "High blood pressure among adults",
        "high_cholesterol_screened_adults": "High cholesterol among screened adults",
        "lack_health_insurance_adults_18_64": "Lack of health insurance, adults 18-64",
        "no_leisure_physical_activity_adults": "No leisure-time physical activity",
        "obesity_adults": "Adult obesity prevalence",
        "routine_checkup_past_year_adults": "Routine checkup within past year",
        "diabetes_previous_year": "Previous-year diabetes prevalence",
        "diabetes_change_from_previous_year": "Diabetes change from previous year",
        "GenHlth": "General health",
        "BMI": "BMI",
        "Age": "Age group",
        "HighBP": "High blood pressure",
        "HighChol": "High cholesterol",
        "CholCheck": "Cholesterol check",
        "HvyAlcoholConsump": "Heavy alcohol consumption",
        "Sex": "Sex",
        "Income": "Income",
        "HeartDiseaseorAttack": "Heart disease or heart attack",
        "PhysHlth": "Physical health days",
        "Education": "Education",
        "DiffWalk": "Difficulty walking",
        "Stroke": "Stroke",
        "Fruits": "Fruit consumption",
        "MentHlth": "Mental health days",
        "PhysActivity": "Physical activity",
        "AnyHealthcare": "Any healthcare coverage",
        "NoDocbcCost": "Could not see doctor due to cost",
        "Veggies": "Vegetable consumption",
        "Smoker": "Smoking history",
    }
    return names.get(feature, feature)