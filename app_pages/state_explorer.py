import streamlit as st

from helpers import (
    get_state_prediction_for_selected_state,
    format_feature_name,
    format_model_score,
)

from ui_components import (
    build_share_text,
    create_downloadable_report,
    create_pdf_report,
    render_share_box,
)


# ============================================================
# app_pages/state_explorer.py
# State Diabetes Early Warning Explorer page
# ============================================================


def render_state_explorer_page(data):
    state_df = data["state_df"]
    state_predictions = data["state_predictions"]
    state_coefficients = data["state_coefficients"]

    st.title("🗺️ State Diabetes Early Warning Explorer")

    st.markdown(
        """
        Explore how a selected state compares with other U.S. states based on the model's
        diabetes-burden warning signal. This page is meant for **public-health awareness**,
        not individual diagnosis.
        """
    )

    states = sorted(state_predictions["LocationDesc"].unique())
    selected_state = st.selectbox("Choose a state", states)

    state_result = get_state_prediction_for_selected_state(
        state_predictions=state_predictions,
        state_name=selected_state,
    )

    if state_result is None:
        st.warning("No prediction found for this state.")
        return

    state_row, state_score, state_signal, state_icon = state_result

    ranking_df = state_predictions.copy()
    ranking_df = ranking_df.sort_values(
        "predicted_probability_high_risk",
        ascending=False,
    ).reset_index(drop=True)

    ranking_df["rank"] = ranking_df.index + 1

    selected_rank_row = ranking_df[ranking_df["LocationDesc"] == selected_state]

    if not selected_rank_row.empty:
        selected_rank = int(selected_rank_row.iloc[0]["rank"])
        total_states = int(ranking_df["LocationDesc"].nunique())
    else:
        selected_rank = None
        total_states = int(ranking_df["LocationDesc"].nunique())

    st.markdown("---")
    st.markdown(
        '<div class="section-header">🗺️ State Risk Snapshot</div>',
        unsafe_allow_html=True,
    )

    top_col1, top_col2, top_col3, top_col4 = st.columns(4)

    with top_col1:
        st.metric("Selected state", selected_state)

    with top_col2:
        st.metric("Risk signal", f"{state_icon} {state_signal}")

    with top_col3:
        st.metric("Model High-Risk Score", format_model_score(state_score))

    with top_col4:
        if selected_rank is not None:
            st.metric("Risk rank", f"#{selected_rank} of {total_states}")
        else:
            st.metric("Risk rank", "N/A")

    st.info(
        "The Model High-Risk Score shows how strongly the model classifies this state "
        "as belonging to the higher diabetes-burden group. It is not a real-world disease probability."
    )

    if state_signal == "Very Strong Signal":
        st.error(
            f"{selected_state} has a very strong model signal for belonging to the higher diabetes-burden group."
        )
    elif state_signal == "High Signal":
        st.warning(
            f"{selected_state} has a high model signal. Public-health indicators should be monitored closely."
        )
    elif state_signal == "Moderate Signal":
        st.warning(
            f"{selected_state} is in a moderate monitoring range. Public-health indicators should be watched."
        )
    else:
        st.success(
            f"{selected_state} has a low model signal for belonging to the higher diabetes-burden group."
        )

    st.markdown(
        '<div class="section-header">📊 How this state compares</div>',
        unsafe_allow_html=True,
    )

    compare_left, compare_right = st.columns([1, 1])

    with compare_left:
        st.markdown(
            f"""
            <div class="glass-card">
                <h3 style="margin-top:0;">{selected_state} public-health context</h3>
                <p style="font-size:16px; color:#475569;">
                    The model estimates whether a state may fall into a higher diabetes-burden group
                    in the next available year. A low score does not mean there is no diabetes burden.
                    It means the model does not strongly classify the state as part of the highest-risk group.
                </p>
                <p style="font-size:16px; color:#475569;">
                    <b>Prediction year:</b> {int(state_row["YearStart"])}<br>
                    <b>Current adult diabetes prevalence:</b> {float(state_row["diabetes_adults"]):.1f}%<br>
                    <b>Next-year diabetes prevalence:</b> {float(state_row["diabetes_next_year"]):.1f}%<br>
                    <b>Model High-Risk Score:</b> {format_model_score(state_score)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with compare_right:
        top_10 = ranking_df.head(10).copy()
        top_10["Model High-Risk Score"] = (
            top_10["predicted_probability_high_risk"] * 100
        ).round(1)

        top_10_display = top_10[
            [
                "rank",
                "LocationDesc",
                "Model High-Risk Score",
                "diabetes_adults",
                "diabetes_next_year",
            ]
        ].rename(
            columns={
                "rank": "Rank",
                "LocationDesc": "State",
                "diabetes_adults": "Current Diabetes %",
                "diabetes_next_year": "Next-Year Diabetes %",
            }
        )

        st.markdown("#### Top model high-risk states")
        st.dataframe(top_10_display, width="stretch")

    st.markdown(
        '<div class="section-header">📈 State diabetes trend</div>',
        unsafe_allow_html=True,
    )

    trend_df = state_df[state_df["LocationDesc"] == selected_state].sort_values("YearStart")

    if not trend_df.empty:
        chart_data = trend_df[["YearStart", "diabetes_adults"]].set_index("YearStart")
        st.line_chart(chart_data)

        display_cols = [
            "YearStart",
            "diabetes_adults",
            "obesity_adults",
            "current_smoking_adults",
            "no_leisure_physical_activity_adults",
            "lack_health_insurance_adults_18_64",
            "routine_checkup_past_year_adults",
        ]

        available_display_cols = [col for col in display_cols if col in trend_df.columns]

        state_table = trend_df[available_display_cols].sort_values(
            "YearStart",
            ascending=False,
        ).copy()

        state_table = state_table.rename(
            columns={
                "YearStart": "Year",
                "diabetes_adults": "Adult Diabetes %",
                "obesity_adults": "Adult Obesity %",
                "current_smoking_adults": "Current Smoking %",
                "no_leisure_physical_activity_adults": "No Leisure Physical Activity %",
                "lack_health_insurance_adults_18_64": "Lack Health Insurance %",
                "routine_checkup_past_year_adults": "Routine Checkup %",
            }
        )

        st.dataframe(state_table, width="stretch")

    st.markdown(
        '<div class="section-header">🧠 Top state-level model drivers</div>',
        unsafe_allow_html=True,
    )

    coef_show = state_coefficients.copy()
    coef_show["feature_readable"] = coef_show["feature"].apply(format_feature_name)
    coef_show = coef_show.sort_values("absolute_coefficient", ascending=False).head(8)

    st.bar_chart(coef_show.set_index("feature_readable")["absolute_coefficient"])

    share_text = build_share_text(
        state_name=selected_state,
        state_band=state_signal,
        state_probability=state_score,
    )

    render_share_box(share_text)

    report_text = create_downloadable_report(
        report_title="Chronic Risk Watch State Risk Report",
        state_name=selected_state,
        state_band=state_signal,
        state_probability=state_score,
    )

    pdf_report = create_pdf_report(report_text)

    st.download_button(
        label="📄 Download PDF report",
        data=pdf_report,
        file_name="chronic_risk_watch_state_report.pdf",
        mime="application/pdf",
    )