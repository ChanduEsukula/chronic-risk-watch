import streamlit as st

from helpers import (
    calculate_bmi_from_height_weight,
    build_personal_input,
    get_risk_band,
    get_combined_awareness,
    get_personal_risk_factors,
    get_state_prediction_for_selected_state,
    format_model_score,
    state_signal_to_simple_band,
)

from ui_components import (
    render_risk_card,
    render_factor_cards,
    build_share_text,
    create_downloadable_report,
    create_pdf_report,
    render_share_box,
)


# ============================================================
# app_pages/combined_view.py
# Combined Personal + State Risk View page
# ============================================================


def render_combined_view_page(models, data):
    personal_model = models["personal_model"]
    personal_features = models["personal_features"]

    state_predictions = data["state_predictions"]

    st.title("🔗 Combined Personal + State Risk View")

    st.markdown(
        """
        This page combines two ideas:

        **1. Personal risk-awareness** based on a user's health and lifestyle profile.  
        **2. State public-health context** based on CDC chronic disease indicators.

        The result is a simple combined awareness level that is easier for a normal user to understand.
        """
    )

    states = sorted(state_predictions["LocationDesc"].unique())

    with st.form("combined_form"):
        st.markdown("### 🗺️ State context")

        selected_state = st.selectbox(
            "Choose your state",
            states,
            index=states.index("Minnesota") if "Minnesota" in states else 0,
        )

        st.markdown("### 👤 Personal profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            age_group = st.selectbox(
                "Age group",
                [
                    "18-24", "25-29", "30-34", "35-39", "40-44",
                    "45-49", "50-54", "55-59", "60-64", "65-69",
                    "70-74", "75-79", "80+",
                ],
                index=4,
                key="combined_age",
            )

            sex = st.selectbox(
                "Sex",
                ["Female", "Male"],
                key="combined_sex",
            )

            st.markdown("#### Height and weight")

            height_feet = st.number_input(
                "Height - feet",
                min_value=3,
                max_value=8,
                value=5,
                step=1,
                key="combined_height_feet",
            )

            height_inches = st.number_input(
                "Height - inches",
                min_value=0,
                max_value=11,
                value=7,
                step=1,
                key="combined_height_inches",
            )

            weight_lbs = st.number_input(
                "Weight - pounds",
                min_value=50.0,
                max_value=500.0,
                value=160.0,
                step=1.0,
                key="combined_weight_lbs",
            )

            bmi = calculate_bmi_from_height_weight(
                height_feet=height_feet,
                height_inches=height_inches,
                weight_lbs=weight_lbs,
            )

            st.info(f"Calculated BMI: **{bmi:.1f}**")

        with col2:
            high_bp = st.selectbox(
                "High blood pressure?",
                ["No", "Yes"],
                key="combined_bp",
            )

            high_chol = st.selectbox(
                "High cholesterol?",
                ["No", "Yes"],
                key="combined_chol",
            )

            chol_check = st.selectbox(
                "Cholesterol checked in last 5 years?",
                ["Yes", "No"],
                key="combined_cholcheck",
            )

            general_health = st.selectbox(
                "General health",
                ["Excellent", "Very good", "Good", "Fair", "Poor"],
                index=2,
                key="combined_genhlth",
            )

        with col3:
            smoker = st.selectbox(
                "Smoked at least 100 cigarettes in life?",
                ["No", "Yes"],
                key="combined_smoker",
            )

            stroke = st.selectbox(
                "Ever had a stroke?",
                ["No", "Yes"],
                key="combined_stroke",
            )

            heart_disease = st.selectbox(
                "Heart disease or heart attack?",
                ["No", "Yes"],
                key="combined_heart",
            )

            diff_walk = st.selectbox(
                "Difficulty walking?",
                ["No", "Yes"],
                key="combined_walk",
            )

        st.markdown("### 🏃 Lifestyle and access")

        col4, col5, col6 = st.columns(3)

        with col4:
            phys_activity = st.selectbox(
                "Physical activity in past 30 days?",
                ["Yes", "No"],
                key="combined_activity",
            )

            fruits = st.selectbox(
                "Eat fruit at least once per day?",
                ["Yes", "No"],
                key="combined_fruits",
            )

            veggies = st.selectbox(
                "Eat vegetables at least once per day?",
                ["Yes", "No"],
                key="combined_veggies",
            )

        with col5:
            heavy_alcohol = st.selectbox(
                "Heavy alcohol consumption?",
                ["No", "Yes"],
                key="combined_alcohol",
            )

            any_healthcare = st.selectbox(
                "Any healthcare coverage?",
                ["Yes", "No"],
                key="combined_healthcare",
            )

            no_doc_cost = st.selectbox(
                "Could not see doctor because of cost?",
                ["No", "Yes"],
                key="combined_cost",
            )

        with col6:
            mental_health_days = st.slider(
                "Poor mental health days in past 30 days",
                min_value=0,
                max_value=30,
                value=0,
                key="combined_mental",
            )

            physical_health_days = st.slider(
                "Poor physical health days in past 30 days",
                min_value=0,
                max_value=30,
                value=0,
                key="combined_physical",
            )

        st.markdown("### 🎓 Socioeconomic information")

        col7, col8 = st.columns(2)

        with col7:
            education = st.selectbox(
                "Education",
                [
                    "Less than high school",
                    "High school graduate",
                    "Some college",
                    "College graduate",
                ],
                index=2,
                key="combined_education",
            )

        with col8:
            income = st.selectbox(
                "Income",
                [
                    "Less than $25,000",
                    "$25,000-$35,000",
                    "$35,000-$50,000",
                    "$50,000-$75,000",
                    "$75,000 or more",
                ],
                index=3,
                key="combined_income",
            )

        submitted = st.form_submit_button("Generate combined awareness result")

    if submitted:
        personal_input = build_personal_input(
            personal_features=personal_features,
            high_bp=high_bp,
            high_chol=high_chol,
            chol_check=chol_check,
            bmi=bmi,
            smoker=smoker,
            stroke=stroke,
            heart_disease=heart_disease,
            phys_activity=phys_activity,
            fruits=fruits,
            veggies=veggies,
            heavy_alcohol=heavy_alcohol,
            any_healthcare=any_healthcare,
            no_doc_cost=no_doc_cost,
            general_health=general_health,
            mental_health_days=mental_health_days,
            physical_health_days=physical_health_days,
            diff_walk=diff_walk,
            sex=sex,
            age_group=age_group,
            education=education,
            income=income,
        )

        personal_probability = float(personal_model.predict_proba(personal_input)[0, 1])
        personal_band, personal_icon = get_risk_band(personal_probability)

        state_result = get_state_prediction_for_selected_state(
            state_predictions=state_predictions,
            state_name=selected_state,
        )

        if state_result is None:
            st.error("State prediction not found.")
            return

        state_row, state_score, state_signal, state_icon = state_result
        state_simple_band = state_signal_to_simple_band(state_signal)

        combined_level, combined_icon = get_combined_awareness(
            personal_band=personal_band,
            state_band=state_simple_band,
        )

        st.markdown("---")
        st.markdown(
            '<div class="section-header">🔗 Combined Awareness Result</div>',
            unsafe_allow_html=True,
        )

        col_result_1, col_result_2 = st.columns(2)

        with col_result_1:
            render_risk_card(
                title="Personal Risk-Awareness",
                probability=personal_probability,
                band=personal_band,
                icon=personal_icon,
                description="Based on your self-reported health, lifestyle, and access-to-care inputs.",
            )

            st.progress(personal_probability)

        with col_result_2:
            render_risk_card(
                title=f"{selected_state} State Context",
                probability=state_score,
                band=state_simple_band,
                icon=state_icon,
                description=(
                    f"Model High-Risk Score: {format_model_score(state_score)}. "
                    "This is a model signal, not a real-world disease probability."
                ),
            )

            st.progress(state_score)

        st.info(
            "The state score shows how strongly the model classifies the selected state as belonging "
            "to a higher diabetes-burden group. It is not a real-world disease probability."
        )

        st.markdown(
            f"""
            <div class="share-card">
                <h3>{combined_icon} Combined Awareness Level: {combined_level}</h3>
                <p>
                    Your personal profile is <b>{personal_band}</b>, and your selected state,
                    <b>{selected_state}</b>, has a <b>{state_signal}</b>.
                    Together, this creates a <b>{combined_level}</b> signal.
                </p>
                <p>
                    This is not a diagnosis. It is a simple educational summary that combines individual-level
                    survey-style risk awareness with broader public-health context.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-header">📌 State context details</div>',
            unsafe_allow_html=True,
        )

        state_col1, state_col2, state_col3, state_col4 = st.columns(4)

        with state_col1:
            st.metric("Prediction year", int(state_row["YearStart"]))

        with state_col2:
            st.metric(
                "Current adult diabetes %",
                f"{float(state_row['diabetes_adults']):.1f}%",
            )

        with state_col3:
            st.metric(
                "Next-year diabetes %",
                f"{float(state_row['diabetes_next_year']):.1f}%",
            )

        with state_col4:
            st.metric(
                "Model High-Risk Score",
                format_model_score(state_score),
            )

        factors = get_personal_risk_factors(personal_input)
        render_factor_cards(factors)

        st.markdown(
            '<div class="section-header">✅ Suggested Prevention Focus</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <ul style="font-size:17px; color:#334155; line-height:1.8;">
                    <li>Use this result as a prevention-awareness signal, not a diagnosis.</li>
                    <li>Consider discussing diabetes screening with a healthcare professional.</li>
                    <li>Pay attention to modifiable factors such as BMI, physical activity, blood pressure, cholesterol, and smoking.</li>
                    <li>State-level risk gives broader public-health context, but it does not determine your individual outcome.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        share_text = build_share_text(
            personal_band=personal_band,
            personal_probability=personal_probability,
            state_name=selected_state,
            state_band=state_signal,
            state_probability=state_score,
            combined_level=combined_level,
        )

        render_share_box(share_text)

        report_text = create_downloadable_report(
            report_title="Chronic Risk Watch Combined Awareness Report",
            personal_band=personal_band,
            personal_probability=personal_probability,
            state_name=selected_state,
            state_band=state_signal,
            state_probability=state_score,
            combined_level=combined_level,
            factors=factors,
        )

        pdf_report = create_pdf_report(report_text)

        st.download_button(
            label="📄 Download PDF report",
            data=pdf_report,
            file_name="chronic_risk_watch_combined_report.pdf",
            mime="application/pdf",
        )

        with st.expander("Show technical personal model input row"):
            st.dataframe(personal_input, width="stretch")