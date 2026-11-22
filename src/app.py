from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import HousingFeatureEngineer  # noqa: F401

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide",
)

ARTIFACTS_DIR = Path("artifacts")
PIPELINE_PATH = ARTIFACTS_DIR / "best_model_pipeline.joblib"
MODEL_NAME_PATH = ARTIFACTS_DIR / "best_model_name.joblib"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# Unique colorful bright palette: warm sunset coral, amber gold, indigo & emerald
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 12%, rgba(251, 146, 60, 0.22), transparent 36%),
                radial-gradient(circle at 92% 10%, rgba(99, 102, 241, 0.20), transparent 38%),
                radial-gradient(circle at 85% 88%, rgba(20, 184, 166, 0.20), transparent 40%),
                linear-gradient(155deg, #fffaf5 0%, #f4f7ff 52%, #f0fdfa 100%);
            color: #0f172a;
        }

        .block-container {
            padding-top: 1.75rem;
            padding-bottom: 3rem;
            max-width: 1080px;
        }

        .login-banner {
            background: linear-gradient(125deg, #ffedd5 0%, #fef9c3 48%, #e0e7ff 100%);
            border: 2px solid #fdba74;
            border-radius: 24px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.35rem;
            box-shadow: 0 14px 28px -10px rgba(234, 88, 12, 0.16);
        }

        .login-chip {
            display: inline-block;
            background: linear-gradient(90deg, #ea580c 0%, #4f46e5 100%);
            color: #ffffff;
            font-size: 0.76rem;
            font-weight: 700;
            padding: 0.28rem 0.8rem;
            border-radius: 999px;
            margin-bottom: 0.6rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .login-title {
            font-size: 1.95rem;
            font-weight: 800;
            color: #1e1b4b;
            margin: 0 0 0.35rem 0;
        }

        .login-sub {
            font-size: 0.96rem;
            color: #334155;
            margin: 0;
        }

        .hero-card {
            background: linear-gradient(125deg, #ffffff 0%, #fff7ed 45%, #eef2ff 100%);
            border: 1.5px solid #c7d2fe;
            border-radius: 24px;
            padding: 1.65rem 2rem;
            box-shadow: 0 14px 30px -10px rgba(79, 70, 229, 0.14);
            margin-bottom: 1.4rem;
        }

        .hero-badge {
            display: inline-block;
            background: linear-gradient(90deg, #0d9488 0%, #4f46e5 100%);
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            margin-bottom: 0.65rem;
        }

        .user-pill {
            display: inline-block;
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.28rem 0.8rem;
            border-radius: 999px;
            margin-left: 0.5rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 0.35rem 0;
        }

        .hero-subtitle {
            font-size: 0.96rem;
            color: #475569;
            margin: 0;
        }

        .section-heading-coral {
            font-size: 1.06rem;
            font-weight: 700;
            color: #c2410c;
            margin: 0.15rem 0 0.9rem 0;
            padding-bottom: 0.45rem;
            border-bottom: 2px solid #fed7aa;
        }

        .section-heading-indigo {
            font-size: 1.06rem;
            font-weight: 700;
            color: #3730a3;
            margin: 0.15rem 0 0.9rem 0;
            padding-bottom: 0.45rem;
            border-bottom: 2px solid #c7d2fe;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: rgba(255, 255, 255, 0.94);
            border: 1.5px solid #e2e8f0 !important;
            border-radius: 20px !important;
            padding: 1.15rem 1.3rem !important;
            box-shadow: 0 8px 22px -8px rgba(15, 23, 42, 0.08) !important;
        }

        .result-card {
            background: linear-gradient(135deg, #ffffff 0%, #fef3c7 50%, #ccfbf1 100%);
            border: 1.5px solid #5eead4;
            border-left: 7px solid #0d9488;
            border-radius: 20px;
            padding: 1.5rem 1.8rem;
            box-shadow: 0 14px 26px -10px rgba(13, 148, 136, 0.22);
            margin-top: 1.15rem;
        }

        .result-label {
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #0f766e;
            margin-bottom: 0.2rem;
        }

        .result-price {
            font-size: 2.3rem;
            font-weight: 800;
            color: #1e1b4b;
            margin: 0.15rem 0;
        }

        .result-sub {
            font-size: 0.92rem;
            color: #334155;
            margin: 0;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #f97316 0%, #ec4899 50%, #4f46e5 100%);
            color: #ffffff;
            border: none;
            border-radius: 14px;
            padding: 0.68rem 1.6rem;
            font-weight: 700;
            font-size: 0.98rem;
            box-shadow: 0 10px 20px -8px rgba(79, 70, 229, 0.45);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px -8px rgba(236, 72, 153, 0.5);
        }

        .disclaimer-box {
            background: linear-gradient(90deg, #ffffff 0%, #f8fafc 100%);
            border: 1.5px dashed #94a3b8;
            border-radius: 16px;
            padding: 0.95rem 1.2rem;
            color: #475569;
            font-size: 0.84rem;
            margin-top: 1.5rem;
        }

        [data-testid="stSlider"] label,
        [data-testid="stSlider"] label p,
        [data-testid="stSlider"] [data-testid="stWidgetLabel"],
        [data-testid="stSlider"] [data-testid="stWidgetLabel"] p,
        [data-testid="stNumberInput"] label,
        [data-testid="stNumberInput"] label p,
        [data-testid="stNumberInput"] [data-testid="stWidgetLabel"],
        [data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p,
        [data-testid="stTextInput"] label,
        [data-testid="stTextInput"] label p,
        [data-testid="stTextInput"] [data-testid="stWidgetLabel"],
        [data-testid="stTextInput"] [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span {
            color: #0f172a !important;
        }

        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            color: #0f172a !important;
            background-color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_login_screen():
    st.markdown(
        """
        <div class="login-banner">
            <div class="login-chip">Welcome • Member Access</div>
            <h1 class="login-title">🏡 House Price Prediction Portal</h1>
            <p class="login-sub">
                Sign in to launch the interactive regression valuation studio and explore block-group
                property price estimates.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.5, 1])
    with center_col:
        with st.container(border=True):
            st.markdown('<div class="section-heading-indigo">Sign In to Continue</div>', unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                full_name = st.text_input("Full Name", value="", placeholder="Enter your name")
                email = st.text_input("Email Address", placeholder="name@example.com")
                access_code = st.text_input("Access Passcode", type="password", placeholder="Enter any passcode to sign in")
                submitted = st.form_submit_button("Enter Prediction Studio", type="primary", use_container_width=True)

                if submitted:
                    if not full_name.strip() or not email.strip() or not access_code.strip():
                        st.error("Please enter your name, email address, and access passcode.")
                    elif "@" not in email or "." not in email.split("@")[-1]:
                        st.error("Please enter a valid email address.")
                    else:
                        st.session_state.authenticated = True
                        st.session_state.user_name = full_name.strip()
                        st.session_state.user_email = email.strip()
                        st.rerun()


def render_predictor_app():
    if not PIPELINE_PATH.exists():
        st.error("Trained model not found. Run `python -m src.train_evaluate` first.")
        st.stop()

    pipeline = joblib.load(PIPELINE_PATH)
    best_model_name = joblib.load(MODEL_NAME_PATH) if MODEL_NAME_PATH.exists() else "Best Model"

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown(
            f"""
            <div class="hero-card">
                <div>
                    <span class="hero-badge">Active Model: {best_model_name}</span>
                    <span class="user-pill">Signed in: {st.session_state.user_name}</span>
                </div>
                <h1 class="hero-title">House Price Prediction System</h1>
                <p class="hero-subtitle">
                    Comparative Regression Study — Interactive block-group valuation powered by engineered
                    structural, demographic, and spatial features.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown('<div class="section-heading-coral">Property & Household Attributes</div>', unsafe_allow_html=True)
            med_inc = st.slider("Median Income ($10k)", min_value=0.5, max_value=15.0, value=3.87, step=0.05)
            house_age = st.slider("House Age (years)", min_value=1.0, max_value=52.0, value=28.0, step=1.0)
            ave_rooms = st.slider("Average Rooms per Household", min_value=1.0, max_value=15.0, value=5.43, step=0.1)
            ave_bedrms = st.slider("Average Bedrooms per Household", min_value=0.5, max_value=6.0, value=1.10, step=0.05)

    with col2:
        with st.container(border=True):
            st.markdown('<div class="section-heading-indigo">Population & Location Coordinates</div>', unsafe_allow_html=True)
            population = st.number_input("Block Group Population", min_value=3.0, max_value=36000.0, value=1425.0, step=25.0)
            ave_occup = st.slider("Average Occupancy (members/household)", min_value=0.5, max_value=15.0, value=3.07, step=0.1)
            latitude = st.slider("Latitude (°N)", min_value=32.5, max_value=42.0, value=34.05, step=0.01)
            longitude = st.slider("Longitude (°W, negative)", min_value=-124.4, max_value=-114.3, value=-118.24, step=0.01)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Median House Value", type="primary")

    if predict_clicked:
        input_df = pd.DataFrame(
            [
                {
                    "MedInc": med_inc,
                    "HouseAge": house_age,
                    "AveRooms": ave_rooms,
                    "AveBedrms": ave_bedrms,
                    "Population": population,
                    "AveOccup": ave_occup,
                    "Latitude": latitude,
                    "Longitude": longitude,
                }
            ]
        )

        pred_100k = float(pipeline.predict(input_df)[0])
        estimated_usd = pred_100k * 100_000.0

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated Median House Value</div>
                <div class="result-price">${estimated_usd:,.2f}</div>
                <p class="result-sub">
                    Model output: <strong>{pred_100k:.2f}</strong> ($100,000 units) using <strong>{best_model_name}</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if pred_100k >= 5.0:
            st.warning("Prediction is near or at the USD 500,000 cap present in the California Housing dataset.")

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Disclaimer:</strong> Estimates are indicative and derived from the 1990 US Census
            California Housing block-group dataset (target capped at USD 500,000). This tool is for
            academic demonstration and is not a certified appraisal or financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


if not st.session_state.authenticated:
    render_login_screen()
else:
    render_predictor_app()