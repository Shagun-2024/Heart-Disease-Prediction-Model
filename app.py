import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(
    page_title="Heart Disease Risk Check | Shagun",
    page_icon="❤️",
    layout="centered",
)

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("Linear_Regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns


model, scaler, expected_columns = load_artifacts()

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Manrope:wght@400;500;600;700&display=swap');

    :root {
        --ink: #14213D;
        --muted: #5B6578;
        --pulse: #E5383B;
        --pulse-dark: #BA2D30;
        --ok: #1B8A6B;
        --paper: #F5F7FA;
        --card: #FFFFFF;
        --line: #E2E7EF;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--paper); }

    /* hide default Streamlit chrome */
    #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

    .block-container {
        max-width: 760px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;   /* room for the fixed footer */
    }

    /* ---------- Hero ---------- */
    .hero {
        background: var(--ink);
        border-radius: 20px;
        padding: 2.2rem 2rem 1.4rem;
        color: #fff;
        margin-bottom: 1.5rem;
        overflow: hidden;
    }
    .hero h1 {
        font-family: 'DM Serif Display', serif;
        font-weight: 400;
        font-size: 2.4rem;
        line-height: 1.15;
        margin: 0 0 .5rem;
        color: #fff;
    }
    .hero p {
        margin: 0;
        color: #B9C3D6;
        font-size: 1rem;
        max-width: 480px;
    }
    .ecg { width: 100%; height: 70px; margin-top: 1.2rem; display: block; }
    .ecg path {
        fill: none;
        stroke: var(--pulse);
        stroke-width: 2.5;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-dasharray: 900;
        stroke-dashoffset: 900;
        animation: draw 2.4s ease-out forwards;
    }
    @keyframes draw { to { stroke-dashoffset: 0; } }
    @media (prefers-reduced-motion: reduce) {
        .ecg path { animation: none; stroke-dashoffset: 0; }
    }

    /* ---------- Form sections ---------- */
    [data-testid="stForm"] {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.5rem 1.5rem 1rem;
        box-shadow: 0 1px 2px rgba(20, 33, 61, .04);
    }
    .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin: .2rem 0 .8rem;
        color: var(--ink);
    }
    .section-note {
        color: var(--muted);
        font-size: .85rem;
        margin: -.5rem 0 1rem;
    }
    label, [data-testid="stWidgetLabel"] p {
        font-weight: 600 !important;
        color: var(--ink) !important;
    }

    /* ---------- Button ---------- */
    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background: var(--pulse);
        color: #fff;
        border: none;
        border-radius: 12px;
        padding: .8rem 1rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: background .15s ease, transform .1s ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
        background: var(--pulse-dark);
        color: #fff;
        border: none;
    }
    .stButton > button:active, [data-testid="stFormSubmitButton"] > button:active {
        transform: scale(.98);
    }
    .stButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible {
        outline: 3px solid #14213D55;
        outline-offset: 2px;
    }

    /* ---------- Result card ---------- */
    .result {
        border-radius: 16px;
        padding: 1.6rem 1.6rem 1.4rem;
        margin-top: 1.5rem;
        border: 1px solid;
        background: var(--card);
    }
    .result.high { border-color: #F3B4B5; background: #FFF4F4; }
    .result.low  { border-color: #A9DCCB; background: #F1FBF7; }
    .result h2 {
        font-family: 'DM Serif Display', serif;
        font-weight: 400;
        font-size: 1.8rem;
        margin: 0 0 .3rem;
    }
    .result.high h2 { color: var(--pulse-dark); }
    .result.low h2  { color: var(--ok); }
    .result p { margin: 0; color: var(--muted); font-size: .95rem; }
    .meter {
        height: 10px;
        background: #E7EBF2;
        border-radius: 999px;
        margin: 1rem 0 .4rem;
        overflow: hidden;
    }
    .meter > div { height: 100%; border-radius: 999px; }
    .high .meter > div { background: var(--pulse); }
    .low  .meter > div { background: var(--ok); }
    .meter-label { font-size: .8rem; color: var(--muted); }

    .disclaimer {
        margin-top: 1rem;
        font-size: .8rem;
        color: var(--muted);
        text-align: center;
    }

    /* ---------- Footer ---------- */
    .app-footer {
        position: fixed;
        left: 0; right: 0; bottom: 0;
        background: var(--ink);
        color: #B9C3D6;
        text-align: center;
        padding: .75rem 1rem;
        font-size: .88rem;
        z-index: 999;
    }
    .app-footer b { color: #fff; }
    .app-footer .heart { color: var(--pulse); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
        <h1>Heart Disease Risk Check</h1>
        <p>Answer a few questions about your health and get an instant risk estimate.</p>
        <svg class="ecg" viewBox="0 0 700 70" preserveAspectRatio="none" aria-hidden="true">
            <path d="M0 40 H120 L140 40 L155 12 L175 62 L195 28 L210 40 H340
                     L360 40 L375 8 L395 64 L415 26 L430 40 H560 L580 40 L595 16
                     L612 58 L628 32 L640 40 H700"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Input form ----------
with st.form("risk_form"):
    st.markdown('<div class="section-title">About you</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 100, 40)
    with c2:
        sex = st.selectbox(
            "Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female"
        )

    st.markdown('<div class="section-title">Vitals and blood tests</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        resting_bp = st.number_input("Resting blood pressure (mm Hg)", 80, 200, 120)
        fasting_bs = st.selectbox(
            "Fasting blood sugar above 120 mg/dL",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
    with c2:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        max_hr = st.slider("Max heart rate", 60, 220, 150)

    st.markdown('<div class="section-title">Heart tests and symptoms</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        chest_pain = st.selectbox(
            "Chest pain type",
            ["ATA", "NAP", "TA", "ASY"],
            format_func=lambda x: {
                "ATA": "Atypical angina (ATA)",
                "NAP": "Non-anginal pain (NAP)",
                "TA": "Typical angina (TA)",
                "ASY": "No symptoms (ASY)",
            }[x],
        )
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            format_func=lambda x: {
                "Normal": "Normal",
                "ST": "ST-T wave abnormality",
                "LVH": "Left ventricular hypertrophy",
            }[x],
        )
        exercise_angina = st.selectbox(
            "Chest pain during exercise",
            ["N", "Y"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
        )
    with c2:
        oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Check my risk")

# ---------- Prediction ----------
if submitted:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    # Add any missing one-hot columns as 0, then match training order
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    # Linear Regression returns a continuous score, so threshold it at 0.5
    score = float(model.predict(scaled_input)[0])
    is_high = score >= 0.5
    pct = int(max(0.0, min(1.0, score)) * 100)

    if is_high:
        st.markdown(
            f"""
            <div class="result high">
                <h2>⚠️ High risk of heart disease</h2>
                <p>Your answers match patterns linked to heart disease. Please see a doctor for a proper check-up.</p>
                <div class="meter"><div style="width:{pct}%"></div></div>
                <span class="meter-label">Risk score: {pct}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result low">
                <h2>✅ Low risk of heart disease</h2>
                <p>Your answers don't match the patterns linked to heart disease. Keep up regular check-ups and healthy habits.</p>
                <div class="meter"><div style="width:{pct}%"></div></div>
                <span class="meter-label">Risk score: {pct}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="disclaimer">This tool is for learning and awareness only. It is not a medical diagnosis.</div>',
        unsafe_allow_html=True,
    )

# ---------- Footer ----------
st.markdown(
    """
    <div class="app-footer">
        Made with <span class="heart">❤</span> by <b>Shagun</b> &nbsp;|&nbsp; © 2026 Heart Disease Risk Check
    </div>
    """,
    unsafe_allow_html=True,
)