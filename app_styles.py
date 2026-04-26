import streamlit as st


# ============================================================
# app_styles.py
# Premium visual styling for Chronic Risk Watch
# ============================================================


def apply_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 34%),
                radial-gradient(circle at top right, rgba(124, 58, 237, 0.14), transparent 32%),
                linear-gradient(135deg, #f8fbff 0%, #eef6ff 45%, #fdfefe 100%);
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 45%, #1e1b4b 100%);
            border-right: 1px solid rgba(255,255,255,0.12);
        }

        section[data-testid="stSidebar"] * {
            color: white;
        }

        .hero-wrap {
            background:
                linear-gradient(135deg, rgba(255,255,255,0.92), rgba(239,246,255,0.86)),
                radial-gradient(circle at top right, rgba(124,58,237,0.18), transparent 30%);
            border: 1px solid rgba(255,255,255,0.85);
            border-radius: 34px;
            padding: 42px 42px 34px 42px;
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.12);
            margin-bottom: 24px;
            overflow: hidden;
        }

        .hero-badge {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 800;
            color: #1d4ed8;
            background: rgba(219,234,254,0.85);
            border: 1px solid rgba(147,197,253,0.8);
            margin-bottom: 16px;
        }

        .main-title {
            font-size: 64px;
            line-height: 1.02;
            font-weight: 950;
            background: linear-gradient(90deg, #0f766e, #2563eb, #7c3aed, #db2777);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
            letter-spacing: -2px;
        }

        .subtitle {
            font-size: 22px;
            color: #475569;
            max-width: 900px;
            margin-top: 0px;
            margin-bottom: 26px;
            line-height: 1.5;
        }

        .hero-mini {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 10px;
        }

        .pill {
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.8);
            border: 1px solid #dbeafe;
            box-shadow: 0 6px 18px rgba(15,23,42,0.06);
            color: #334155;
            font-weight: 750;
            font-size: 14px;
        }

        .metric-card {
            background: white;
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
            border: 1px solid #e2e8f0;
            border-left: 8px solid #2563eb;
            margin-bottom: 18px;
            min-height: 170px;
            transition: all 0.22s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px) scale(1.01);
            box-shadow: 0 18px 42px rgba(37, 99, 235, 0.16);
        }

        .metric-card h3 {
            margin-top: 0px;
            margin-bottom: 10px;
            font-size: 24px;
            color: #0f172a;
        }

        .metric-card p {
            color: #475569;
            line-height: 1.55;
            font-size: 15.8px;
        }

        .feature-card {
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            border-radius: 22px;
            padding: 22px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
            border: 1px solid #e2e8f0;
            min-height: 145px;
            margin-bottom: 16px;
        }

        .feature-card h4 {
            font-size: 20px;
            margin-bottom: 8px;
            color: #0f172a;
        }

        .feature-card p {
            color: #64748b;
            font-size: 15px;
            line-height: 1.45;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.86);
            border-radius: 26px;
            padding: 28px;
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.85);
            margin-bottom: 20px;
        }

        .risk-low {
            background: linear-gradient(135deg, #dcfce7, #f0fdf4);
            border-left: 8px solid #22c55e;
            border-radius: 24px;
            padding: 26px;
            box-shadow: 0 12px 28px rgba(34, 197, 94, 0.18);
            margin-bottom: 18px;
        }

        .risk-moderate {
            background: linear-gradient(135deg, #fef3c7, #fff7ed);
            border-left: 8px solid #f59e0b;
            border-radius: 24px;
            padding: 26px;
            box-shadow: 0 12px 28px rgba(245, 158, 11, 0.18);
            margin-bottom: 18px;
        }

        .risk-high {
            background: linear-gradient(135deg, #fee2e2, #fff1f2);
            border-left: 8px solid #ef4444;
            border-radius: 24px;
            padding: 26px;
            box-shadow: 0 12px 28px rgba(239, 68, 68, 0.18);
            margin-bottom: 18px;
        }

        .big-score {
            font-size: 62px;
            font-weight: 950;
            margin-bottom: 0px;
            letter-spacing: -1.5px;
            color: #0f172a;
        }

        .risk-label {
            font-size: 24px;
            font-weight: 900;
            color: #0f172a;
        }

        .small-muted {
            color: #64748b;
            font-size: 15px;
            line-height: 1.5;
        }

        .factor-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 10px;
            box-shadow: 0 7px 20px rgba(15, 23, 42, 0.07);
            border: 1px solid #e2e8f0;
        }

        .section-header {
            font-size: 31px;
            font-weight: 900;
            color: #0f172a;
            margin-top: 24px;
            margin-bottom: 12px;
            letter-spacing: -0.5px;
        }

        .share-card {
            background: linear-gradient(135deg, #111827, #1e293b, #312e81);
            border-radius: 26px;
            padding: 26px;
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
            margin-top: 18px;
            margin-bottom: 18px;
        }

        .share-card h3 {
            color: white;
            margin-top: 0px;
            font-size: 26px;
        }

        .share-card p {
            color: #e5e7eb;
            font-size: 16px;
            line-height: 1.55;
        }

        .leaderboard-card {
            background: white;
            border-radius: 22px;
            padding: 22px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 10px 24px rgba(15,23,42,0.08);
        }

        .warning-card {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-left: 7px solid #f97316;
            padding: 18px 20px;
            border-radius: 20px;
            color: #7c2d12;
            font-weight: 650;
            margin-bottom: 18px;
        }

        .stButton > button {
            border-radius: 16px;
            padding: 13px 24px;
            font-weight: 850;
            border: none;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            color: white;
            box-shadow: 0 9px 20px rgba(37, 99, 235, 0.25);
            transition: all 0.18s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 26px rgba(37, 99, 235, 0.35);
            color: white;
        }

        [data-testid="stMetricValue"] {
            font-size: 30px;
            font-weight: 900;
            color: #0f172a;
        }

        [data-testid="stMetricLabel"] {
            color: #475569;
            font-weight: 700;
        }

        [data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(15,23,42,0.07);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )