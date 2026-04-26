import streamlit as st
from app_pages.home import render_home_page
from app_pages.personal_risk import render_personal_risk_page
from app_pages.state_explorer import render_state_explorer_page
from app_pages.combined_view import render_combined_view_page
from app_pages.compare_states import render_compare_states_page
from app_pages.model_insights import render_model_insights_page

from app_config import PAGES
from app_styles import apply_styles
from data_loader import load_models, load_data
from ui_components import set_page


# ============================================================
# streamlit_app.py
# Main app runner for Chronic Risk Watch
# ============================================================


st.set_page_config(
    page_title="Chronic Risk Watch",
    page_icon="🩺",
    layout="wide",
)

apply_styles()

try:
    models = load_models()
    data = load_data()
except Exception as e:
    st.error("Some required files were not found or could not be loaded.")
    st.write("Error details:")
    st.exception(e)
    st.stop()


if "page" not in st.session_state:
    st.session_state.page = "Home"


st.sidebar.title("🩺 Chronic Risk Watch")

selected_page = st.sidebar.radio(
    "Choose a section",
    PAGES,
    index=PAGES.index(st.session_state.page),
)

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

page = st.session_state.page

st.sidebar.markdown("---")
st.sidebar.warning("Educational use only. This app is not a medical diagnosis tool.")


if page != "Home":
    if st.button("⬅️ Back to Home", width="content"):
        set_page("Home")


if page == "Home":
    render_home_page(data)

elif page == "Personal Risk Check":
    render_personal_risk_page(models)

elif page == "State Risk Explorer":
    render_state_explorer_page(data)

elif page == "Combined Risk View":
    render_combined_view_page(models, data)

elif page == "Compare States":
    render_compare_states_page(data)

elif page == "Model Insights":
    render_model_insights_page(data)