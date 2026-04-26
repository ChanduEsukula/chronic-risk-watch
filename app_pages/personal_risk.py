import streamlit as st

from helpers import (
    calculate_bmi_from_height_weight,
    build_personal_input,
    get_risk_band,
    get_personal_risk_factors,
    get_personal_guidance,
)

from ui_components import (
    render_risk_card,
    render_factor_cards,
    build_share_text,
    create_downloadable_report,
    create_pdf_report,
    render_share_box,
    render_guidance_card,
)


# ============================================================
# app_pages/personal_risk.py
# Personal Diabetes Risk Awareness Check page
# ============================================================


def render_personal_risk_page(models):
    personal_model = models["personal_model"]
    personal_features = models["personal_features"]

    st.title("👤 Personal Diabetes Risk Awareness Check")

    st.markdown(
        """
        Enter basic health and lifestyle information below.
        The app will estimate your **diabetes/prediabetes risk-awareness level**.

        This is not a diagnosis. It is only an educational screening-style tool.
        """
    )

    with st.form("personal_risk_form"):
        st.markdown("### Basic health information")

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
            )

            sex = st.selectbox("Sex", ["Female", "Male"])

            st.markdown("#### Height and weight")

            height_feet = st.number_input(
                "Height - feet",
                min_value=3,
                max_value=8,
                value=5,
                step=1,
            )

            height_inches = st.number_input(
                "Height - inches",
                min_value=0,
                max_value=11,
                value=7,
                step=1,
            )

            weight_lbs = st.number_input(
                "Weight - pounds",
                min_value=50.0,
                max_value=500.0,
                value=160.0,
                step=1.0,
            )

            bmi = calculate_bmi_from_height_weight(
                height_feet=height_feet,
                height_inches=height_inches,
                weight_lbs=weight_lbs,
            )

            st.info(f"Calculated BMI: **{bmi:.1f}**")

        with col2:
            high_bp = st.selectbox("High blood pressure?", ["No", "Yes"])
            high_chol = st.selectbox("High cholesterol?", ["No", "Yes"])
            chol_check = st.selectbox("Cholesterol checked in last 5 years?", ["Yes", "No"])
            general_health = st.selectbox(
                "General health",
                ["Excellent", "Very good", "Good", "Fair", "Poor"],
                index=2,
            )

        with col3:
            smoker = st.selectbox("Smoked at least 100 cigarettes in life?", ["No", "Yes"])
            stroke = st.selectbox("Ever had a stroke?", ["No", "Yes"])
            heart_disease = st.selectbox("Heart disease or heart attack?", ["No", "Yes"])
            diff_walk = st.selectbox("Difficulty walking?", ["No", "Yes"])

        st.markdown("### Lifestyle and access")

        col4, col5, col6 = st.columns(3)

        with col4:
            phys_activity = st.selectbox("Physical activity in past 30 days?", ["Yes", "No"])
            fruits = st.selectbox("Eat fruit at least once per day?", ["Yes", "No"])
            veggies = st.selectbox("Eat vegetables at least once per day?", ["Yes", "No"])

        with col5:
            heavy_alcohol = st.selectbox("Heavy alcohol consumption?", ["No", "Yes"])
            any_healthcare = st.selectbox("Any healthcare coverage?", ["Yes", "No"])
            no_doc_cost = st.selectbox("Could not see doctor because of cost?", ["No", "Yes"])

        with col6:
            mental_health_days = st.slider(
                "Poor mental health days in past 30 days",
                min_value=0,
                max_value=30,
                value=0,
            )

            physical_health_days = st.slider(
                "Poor physical health days in past 30 days",
                min_value=0,
                max_value=30,
                value=0,
            )

        st.markdown("### Socioeconomic information")

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
            )

        submitted = st.form_submit_button("Check my risk-awareness level")

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
        personal_prediction = int(personal_model.predict(personal_input)[0])
        personal_band, personal_icon = get_risk_band(personal_probability)

        st.markdown("---")
        st.markdown(
            '<div class="section-header">🎯 Your Personal Risk-Awareness Result</div>',
            unsafe_allow_html=True,
        )

        render_risk_card(
            title="Personal Diabetes Risk-Awareness",
            probability=personal_probability,
            band=personal_band,
            icon=personal_icon,
            description="Based on your self-reported health, lifestyle, and access-to-care inputs.",
        )

        st.progress(personal_probability)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Personal Risk-Awareness Score",
                f"{personal_probability * 100:.1f} / 100",
            )

        with c2:
            label_text = "Flagged for awareness" if personal_prediction == 1 else "Not flagged"
            st.metric("Model flag", label_text)

        with c3:
            st.metric("Risk level", personal_band)

        st.info(
            "This score is an awareness signal based on survey-style inputs. "
            "It is not a medical diagnosis and does not replace lab testing or professional advice."
        )

        if personal_band == "High":
            st.error(
                "Your profile is in a higher risk-awareness range. This does not mean you have diabetes, but it suggests screening and prevention awareness may be important."
            )
        elif personal_band == "Moderate":
            st.warning(
                "Your profile is in a moderate risk-awareness range. Monitoring risk factors and preventive screening may be useful."
            )
        else:
            st.success(
                "Your profile is in a lower risk-awareness range based on this model."
            )

        guidance = get_personal_guidance(personal_band)
        render_guidance_card(guidance)

        factors = get_personal_risk_factors(personal_input)
        render_factor_cards(factors)

        st.markdown(
            '<div class="section-header">🧪 What-if Simulator</div>',
            unsafe_allow_html=True,
        )

        st.write(
            "Try changing modifiable factors and see how the model score changes. "
            "This is educational only and does not guarantee medical outcomes."
        )

        sim_col1, sim_col2, sim_col3 = st.columns(3)

        with sim_col1:
            sim_bmi = st.slider(
                "Simulated BMI / weight-change effect",
                min_value=18.0,
                max_value=50.0,
                value=float(bmi),
                step=0.5,
            )

        with sim_col2:
            sim_activity = st.selectbox(
                "Simulated physical activity",
                ["Yes", "No"],
                index=0 if phys_activity == "Yes" else 1,
            )

        with sim_col3:
            sim_smoker = st.selectbox(
                "Simulated smoking history",
                ["No", "Yes"],
                index=0 if smoker == "No" else 1,
            )

        sim_input = build_personal_input(
            personal_features=personal_features,
            high_bp=high_bp,
            high_chol=high_chol,
            chol_check=chol_check,
            bmi=sim_bmi,
            smoker=sim_smoker,
            stroke=stroke,
            heart_disease=heart_disease,
            phys_activity=sim_activity,
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

        sim_probability = float(personal_model.predict_proba(sim_input)[0, 1])
        sim_band, sim_icon = get_risk_band(sim_probability)
        delta = sim_probability - personal_probability

        d1, d2, d3 = st.columns(3)

        with d1:
            st.metric("Current score", f"{personal_probability * 100:.1f} / 100")

        with d2:
            st.metric(
                "Simulated score",
                f"{sim_probability * 100:.1f} / 100",
                f"{delta * 100:.1f}",
            )

        with d3:
            st.metric("Simulated awareness level", f"{sim_icon} {sim_band}")

        st.progress(sim_probability)

        st.caption(
            "The what-if simulator shows how model output changes when selected inputs change. It is not a medical prediction."
        )

        st.markdown(
            '<div class="section-header">✅ Suggested Prevention Focus</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <ul style="font-size:17px; color:#334155; line-height:1.8;">
                    <li>Ask a healthcare professional about appropriate diabetes screening.</li>
                    <li>Monitor blood pressure and cholesterol if relevant.</li>
                    <li>Maintain regular physical activity.</li>
                    <li>Use this result as awareness, not as a diagnosis.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        share_text = build_share_text(
            personal_band=personal_band,
            personal_probability=personal_probability,
        )

        render_share_box(share_text)

        report_text = create_downloadable_report(
            report_title="Chronic Risk Watch Personal Risk-Awareness Report",
            personal_band=personal_band,
            personal_probability=personal_probability,
            factors=factors,
        )

        pdf_report = create_pdf_report(report_text)

        st.download_button(
            label="📄 Download PDF report",
            data=pdf_report,
            file_name="chronic_risk_watch_personal_report.pdf",
            mime="application/pdf",
        )

        with st.expander("Show technical model input row"):
            st.dataframe(personal_input, width="stretch")