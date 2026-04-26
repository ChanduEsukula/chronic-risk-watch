import streamlit as st

from ui_components import set_page, render_disclaimer_card


# ============================================================
# pages/home.py
# Home page for Chronic Risk Watch
# ============================================================


def render_home_page(data):
    state_predictions = data["state_predictions"]

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-badge">🚀 AI + Public Health + Personal Risk Awareness</div>
            <div class="main-title">Chronic Risk Watch</div>
            <div class="subtitle">
                A smart, interactive diabetes risk-awareness dashboard that combines your personal health profile
                with state-level public-health intelligence.
            </div>
            <div class="hero-mini">
                <div class="pill">👤 Personal Risk Check</div>
                <div class="pill">🗺️ State Early Warning</div>
                <div class="pill">🔗 Combined Awareness Score</div>
                <div class="pill">📤 Shareable Mini Reports</div>
                <div class="pill">🧪 What-if Simulator</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_disclaimer_card()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <h3>👤 Check My Risk</h3>
                <p>
                Enter basic health and lifestyle information like height, weight, age group, blood pressure, cholesterol,
                physical activity, and smoking history. Get a simple risk-awareness level in seconds.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🗺️ Explore My State</h3>
                <p>
                See whether your state is flagged as high-risk for future adult diabetes burden using CDC Chronic Disease Indicators.
                Compare trends and top state-level drivers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🔗 Combined View</h3>
                <p>
                Combine your personal profile with state-level public-health context and generate a shareable mini report.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main navigation buttons
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("👤 Start Personal Risk Check", width="stretch"):
            set_page("Personal Risk Check")

    with b2:
        if st.button("🗺️ Explore State Risk", width="stretch"):
            set_page("State Risk Explorer")

    with b3:
        if st.button("🔗 Open Combined View", width="stretch"):
            set_page("Combined Risk View")

    # Extra navigation buttons
    b4, b5 = st.columns(2)

    with b4:
        if st.button("⚖️ Compare States", width="stretch"):
            set_page("Compare States")

    with b5:
        if st.button("📊 View Model Insights", width="stretch"):
            set_page("Model Insights")

    st.markdown(
        '<div class="section-header">✨ Why this app feels useful</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>⚡ 60-second check</h4>
                <p>No medical jargon. Answer simple questions and instantly see a risk-awareness level.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🧪 What-if simulator</h4>
                <p>Change BMI, activity, or smoking inputs and see how the model output changes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🏆 State leaderboard</h4>
                <p>Discover which states are predicted to have the highest diabetes burden risk.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f4:
        st.markdown(
            """
            <div class="feature-card">
                <h4>📄 Mini reports</h4>
                <p>Download or copy a clean summary to share with classmates, friends, or reviewers.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-header">📊 Project Snapshot</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Personal records", "253,680")

    with m2:
        st.metric("Personal model recall", "76.1%")

    with m3:
        st.metric("State model accuracy", "86.0%")

    with m4:
        st.metric("State model ROC-AUC", "93.8%")

    st.markdown(
        '<div class="section-header">🔥 Highest state risk predictions</div>',
        unsafe_allow_html=True,
    )

    top_states_home = state_predictions.sort_values(
        "predicted_probability_high_risk",
        ascending=False,
    ).head(10).copy()

    top_states_home["Predicted Probability"] = (
        top_states_home["predicted_probability_high_risk"] * 100
    ).round(1)

    leaderboard_display = top_states_home[
        ["LocationDesc", "Predicted Probability", "diabetes_adults", "diabetes_next_year"]
    ].rename(
        columns={
            "LocationDesc": "State",
            "diabetes_adults": "Current Diabetes %",
            "diabetes_next_year": "Next-Year Diabetes %",
        }
    )

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown(
            """
            <div class="leaderboard-card">
                <h3>Top 10 predicted high-risk states</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(leaderboard_display, width="stretch")

    with right_col:
        chart_df = top_states_home.set_index("LocationDesc")[
            "predicted_probability_high_risk"
        ]
        st.bar_chart(chart_df)

    st.markdown(
        """
        <div class="share-card">
            <h3>💡 How to use this app</h3>
            <p>
            Start with <b>Personal Risk Check</b> to get your own risk-awareness result.
            Then open <b>Combined Risk View</b> to connect your personal profile with your state-level public-health context.
            Use <b>Compare States</b> to see how two states differ, and <b>Model Insights</b> to explain the machine learning work.
            Finally, copy or download your mini report and share it with someone.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )