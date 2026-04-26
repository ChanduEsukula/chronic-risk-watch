import streamlit as st

from helpers import format_feature_name


# ============================================================
# pages/model_insights.py
# Technical model explanation page
# ============================================================


def render_model_insights_page(data):
    state_results = data["state_results"]
    state_coefficients = data["state_coefficients"]
    personal_results = data["personal_results"]
    personal_importance = data["personal_importance"]

    st.title("📊 Model Insights")

    st.markdown(
        """
        This page explains the machine learning work behind **Chronic Risk Watch**.

        It is designed for professors, classmates, recruiters, and technical reviewers who want to understand:
        - what the models predict,
        - how well they performed,
        - which features mattered most,
        - and what the limitations are.
        """
    )

    st.markdown("---")

    # ========================================================
    # State-level model
    # ========================================================

    st.markdown(
        '<div class="section-header">🗺️ State-Level Early Warning Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        The state-level model uses CDC Chronic Disease Indicator data to predict whether a state may fall into a
        higher diabetes-burden group in the following year.

        This is a **public-health early warning prototype**, not an official forecast.
        """
    )

    st.markdown("### State model comparison")

    state_results_display = state_results.copy()

    rename_state_results = {
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "roc_auc": "ROC-AUC",
    }

    state_results_display = state_results_display.rename(columns=rename_state_results)

    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        if col in state_results_display.columns:
            state_results_display[col] = (state_results_display[col] * 100).round(1)

    st.dataframe(state_results_display, width="stretch")

    if not state_results.empty:
        best_state_model = state_results.sort_values("roc_auc", ascending=False).iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Best state model", str(best_state_model["model"]))

        with c2:
            st.metric("Accuracy", f"{best_state_model['accuracy']:.1%}")

        with c3:
            st.metric("Recall", f"{best_state_model['recall']:.1%}")

        with c4:
            st.metric("ROC-AUC", f"{best_state_model['roc_auc']:.1%}")

    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0;">Why recall matters here</h3>
            <p style="font-size:16px; color:#475569;">
                In a public-health early warning system, missing a truly high-risk state can be costly.
                That is why recall is important: it measures how many actual high-risk states the model successfully catches.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### State model feature drivers")

    coef_show = state_coefficients.copy()

    if "feature" in coef_show.columns:
        coef_show["Feature"] = coef_show["feature"].apply(format_feature_name)

    if "coefficient" in coef_show.columns:
        coef_show["Coefficient"] = coef_show["coefficient"].round(3)

    if "absolute_coefficient" in coef_show.columns:
        coef_show["Absolute Importance"] = coef_show["absolute_coefficient"].round(3)

    display_cols = [
        col for col in ["Feature", "Coefficient", "Absolute Importance"]
        if col in coef_show.columns
    ]

    if display_cols:
        st.dataframe(coef_show[display_cols], width="stretch")

        chart_df = coef_show.sort_values(
            "absolute_coefficient",
            ascending=False,
        ).head(10)

        chart_df = chart_df.set_index("Feature")["Absolute Importance"]

        st.bar_chart(chart_df)

    st.markdown("---")

    # ========================================================
    # Personal model
    # ========================================================

    st.markdown(
        '<div class="section-header">👤 Personal Diabetes Risk-Awareness Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        The personal model uses individual-level BRFSS survey responses.
        It predicts whether a user's self-reported profile resembles respondents labeled as diabetes or prediabetes.

        This is an **educational screening-style model**, not a medical diagnosis tool.
        """
    )

    st.markdown("### Personal model comparison")

    personal_results_display = personal_results.copy()

    rename_personal_results = {
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "roc_auc": "ROC-AUC",
    }

    personal_results_display = personal_results_display.rename(columns=rename_personal_results)

    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        if col in personal_results_display.columns:
            personal_results_display[col] = (personal_results_display[col] * 100).round(1)

    st.dataframe(personal_results_display, width="stretch")

    if not personal_results.empty:
        best_personal_model = personal_results.sort_values("recall", ascending=False).iloc[0]

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric("Selected personal model", str(best_personal_model["model"]))

        with p2:
            st.metric("Accuracy", f"{best_personal_model['accuracy']:.1%}")

        with p3:
            st.metric("Recall", f"{best_personal_model['recall']:.1%}")

        with p4:
            st.metric("ROC-AUC", f"{best_personal_model['roc_auc']:.1%}")

    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0;">Why the personal model favors recall</h3>
            <p style="font-size:16px; color:#475569;">
                For a risk-awareness tool, it is usually better to catch more potentially at-risk profiles,
                even if some users are flagged unnecessarily. This makes recall more important than simple accuracy.
            </p>
            <p style="font-size:16px; color:#475569;">
                However, a model flag should never be treated as a diagnosis. It should only encourage prevention awareness
                and appropriate screening conversations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Personal model feature interpretation")

    personal_show = personal_importance.copy()

    if "feature" in personal_show.columns:
        personal_show["Feature"] = personal_show["feature"].apply(format_feature_name)

    if "coefficient" in personal_show.columns:
        personal_show["Coefficient"] = personal_show["coefficient"].round(3)

    if "absolute_coefficient" in personal_show.columns:
        personal_show["Absolute Importance"] = personal_show["absolute_coefficient"].round(3)
        chart_col = "Absolute Importance"

    elif "importance" in personal_show.columns:
        personal_show["Importance"] = personal_show["importance"].round(3)
        chart_col = "Importance"

    elif "absolute_difference" in personal_show.columns:
        personal_show["Absolute Difference"] = personal_show["absolute_difference"].round(3)
        chart_col = "Absolute Difference"

    else:
        chart_col = None

    display_cols = [
        col for col in [
            "Feature",
            "Coefficient",
            "Absolute Importance",
            "Importance",
            "Absolute Difference",
        ]
        if col in personal_show.columns
    ]

    if display_cols:
        st.dataframe(personal_show[display_cols], width="stretch")

    if chart_col is not None:
        top_personal = personal_show.sort_values(
            chart_col,
            ascending=False,
        ).head(12)

        st.bar_chart(top_personal.set_index("Feature")[chart_col])

    st.markdown("---")

    # ========================================================
    # Limitations
    # ========================================================

    st.markdown(
        '<div class="section-header">⚠️ Important Limitations</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        """
        - This app is not a medical diagnosis tool.
        - The personal model is based on survey data, not lab tests.
        - The state model uses state-level public-health indicators and cannot predict individual health outcomes.
        - The state-level dataset is small, so results should be treated as a prototype.
        - Model scores are not the same as real-world disease probability.
        - Predictions depend on dataset definitions, feature choices, and model assumptions.
        """
    )

    st.markdown(
        """
        <div class="share-card">
            <h3>🎓 Suggested presentation wording</h3>
            <p>
                Chronic Risk Watch is an educational machine learning prototype that combines individual-level
                BRFSS survey indicators with state-level CDC chronic disease indicators. The goal is to make
                diabetes risk awareness more understandable through interactive tools, not to diagnose disease
                or replace professional healthcare advice.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )