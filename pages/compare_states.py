import streamlit as st
import pandas as pd
import altair as alt

from helpers import format_feature_name


# ============================================================
# pages/compare_states.py
# Compare two states side by side
# ============================================================


def render_compare_states_page(data):
    state_df = data["state_df"]
    state_predictions = data["state_predictions"]

    st.title("⚖️ Compare Two States")

    st.markdown(
        """
        Compare two states side by side using diabetes-related public-health indicators.
        This page helps explain **why one state may receive a higher model high-risk score than another**.
        """
    )

    states = sorted(state_df["LocationDesc"].unique())

    col1, col2 = st.columns(2)

    with col1:
        state_a = st.selectbox(
            "State A",
            states,
            index=states.index("Minnesota") if "Minnesota" in states else 0,
        )

    with col2:
        state_b = st.selectbox(
            "State B",
            states,
            index=states.index("Mississippi") if "Mississippi" in states else 1,
        )

    available_years = sorted(state_df["YearStart"].unique())

    selected_year = st.selectbox(
        "Choose year",
        available_years,
        index=len(available_years) - 1,
    )

    compare_df = state_df[
        (state_df["LocationDesc"].isin([state_a, state_b]))
        & (state_df["YearStart"] == selected_year)
    ].copy()

    st.markdown("---")

    st.markdown(
        '<div class="section-header">📊 Side-by-side indicator comparison</div>',
        unsafe_allow_html=True,
    )

    if compare_df.empty:
        st.warning("No comparison data available for the selected states and year.")

    else:
        indicators = [
            "diabetes_adults",
            "obesity_adults",
            "current_smoking_adults",
            "no_leisure_physical_activity_adults",
            "lack_health_insurance_adults_18_64",
            "routine_checkup_past_year_adults",
        ]

        available_indicators = [col for col in indicators if col in compare_df.columns]

        display_df = compare_df[["LocationDesc"] + available_indicators].copy()

        rename_dict = {col: format_feature_name(col) for col in available_indicators}
        display_df = display_df.rename(columns=rename_dict)
        display_df = display_df.rename(columns={"LocationDesc": "State"})

        st.dataframe(display_df, width="stretch")

        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0;">Plain-English comparison</h3>
                <p style="font-size:16px; color:#475569;">
                    Higher values in diabetes prevalence, obesity, smoking, physical inactivity,
                    and lack of insurance may help explain why a state receives a higher model high-risk score.
                    Routine checkup percentage is different because it can reflect healthcare access and prevention behavior.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Indicator chart")

        st.caption(
            "Each bar shows the indicator value for one state. This is a comparison of public-health indicators, not individual health risk."
        )

        chart_long = display_df.melt(
            id_vars="State",
            var_name="Indicator",
            value_name="Value",
        )

        grouped_chart = (
            alt.Chart(chart_long)
            .mark_bar()
            .encode(
                y=alt.Y("Indicator:N", title="Indicator", sort=None),
                x=alt.X("Value:Q", title="Percent / Indicator Value"),
                color=alt.Color("State:N", title="State"),
                yOffset=alt.YOffset("State:N"),
                tooltip=["State", "Indicator", "Value"],
            )
            .properties(height=360)
        )

        st.altair_chart(grouped_chart, use_container_width=True)

    st.markdown(
        '<div class="section-header">🔮 State model score comparison</div>',
        unsafe_allow_html=True,
    )

    pred_compare = state_predictions[
        state_predictions["LocationDesc"].isin([state_a, state_b])
    ].copy()

    if pred_compare.empty:
        st.warning("No prediction comparison data available for these states.")

    else:
        pred_compare = pred_compare[
            [
                "LocationDesc",
                "YearStart",
                "diabetes_adults",
                "diabetes_next_year",
                "high_risk_diabetes_next_year",
                "predicted_high_risk",
                "predicted_probability_high_risk",
            ]
        ].sort_values(["LocationDesc", "YearStart"])

        pred_compare["Model High-Risk Score"] = (
            pred_compare["predicted_probability_high_risk"] * 100
        ).round(1)

        pred_display = pred_compare.rename(
            columns={
                "LocationDesc": "State",
                "YearStart": "Year",
                "diabetes_adults": "Current Diabetes %",
                "diabetes_next_year": "Next-Year Diabetes %",
                "high_risk_diabetes_next_year": "Actual High-Risk Label",
                "predicted_high_risk": "Predicted High-Risk Label",
            }
        )

        pred_display = pred_display[
            [
                "State",
                "Year",
                "Current Diabetes %",
                "Next-Year Diabetes %",
                "Actual High-Risk Label",
                "Predicted High-Risk Label",
                "Model High-Risk Score",
            ]
        ]

        st.dataframe(pred_display, width="stretch")

        score_chart = pred_display[["State", "Model High-Risk Score"]].set_index("State")
        st.bar_chart(score_chart)

    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0;">How to interpret this page</h3>
            <p style="font-size:16px; color:#475569;">
                The <b>model high-risk score</b> is not a real-world probability that someone will get diabetes.
                It shows how strongly the model classifies a state as belonging to the higher diabetes-burden group.
            </p>
            <p style="font-size:16px; color:#475569;">
                This page is best used to compare public-health patterns across states and explain model behavior.
                It should not be used as a medical diagnosis or official government forecast.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )