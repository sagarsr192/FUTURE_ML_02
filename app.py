from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Support Ticket AI",
    page_icon="🎫",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --ink: #e7eef7;
        --muted: #9db0c3;
        --panel: rgba(13, 24, 36, 0.82);
        --accent: #ffb703;
        --accent-2: #8ecae6;
        --high: #d90429;
        --medium: #f77f00;
        --low: #2a9d8f;
    }

    .stApp {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        background:
            radial-gradient(circle at 5% 8%, rgba(255, 183, 3, 0.14) 0, transparent 22%),
            radial-gradient(circle at 88% 0%, rgba(142, 202, 230, 0.16) 0, transparent 28%),
            linear-gradient(125deg, #05090f 0%, #0a1220 52%, #111d2f 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(6, 12, 20, 0.62);
        backdrop-filter: blur(8px);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 18, 30, 0.97), rgba(14, 24, 38, 0.97));
        border-right: 1px solid rgba(157, 176, 195, 0.2);
    }

    section[data-testid="stSidebar"] * {
        color: #dfeaf6;
    }

    .block-container {
        padding-top: 1.3rem;
        max-width: 1180px;
    }

    .hero {
        border: 1px solid rgba(157, 176, 195, 0.24);
        border-radius: 24px;
        padding: 1.5rem 1.7rem;
        background: linear-gradient(120deg, rgba(16,28,42,0.9), rgba(9,18,30,0.84));
        box-shadow: 0 14px 38px rgba(0, 0, 0, 0.35);
        animation: rise 420ms ease-out;
        margin-bottom: 1rem;
    }

    .chip {
        display: inline-block;
        padding: 0.28rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(157, 176, 195, 0.28);
        margin-right: 0.5rem;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(11, 22, 34, 0.92);
    }

    .pane {
        border: 1px solid rgba(157, 176, 195, 0.2);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        background: var(--panel);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.32);
        animation: rise 500ms ease-out;
    }

    .kpi {
        border-radius: 16px;
        border: 1px solid rgba(157, 176, 195, 0.18);
        background: rgba(8, 16, 26, 0.86);
        padding: 0.9rem 1rem;
        margin-top: 0.35rem;
        min-height: 100px;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.84rem;
        margin-bottom: 0.35rem;
    }

    .kpi-value {
        font-weight: 800;
        font-size: 1.3rem;
        line-height: 1.2;
    }

    .badge-priority {
        display: inline-block;
        margin-top: 0.45rem;
        padding: 0.18rem 0.62rem;
        border-radius: 999px;
        color: #fff;
        font-size: 0.76rem;
        font-weight: 700;
    }

    .badge-high { background: var(--high); }
    .badge-medium { background: var(--medium); }
    .badge-low { background: var(--low); }

    .mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.83rem;
        color: #c8d7e6;
        background: rgba(9, 18, 30, 0.95);
        border-radius: 10px;
        border: 1px solid rgba(157, 176, 195, 0.18);
        padding: 0.65rem 0.8rem;
    }

    .stTextArea label,
    .stSelectbox label,
    .stCaption {
        color: #b7c8d9 !important;
    }

    .stTextArea textarea {
        background: #152538 !important;
        color: #edf3fb !important;
        border: 1px solid rgba(157, 176, 195, 0.24) !important;
        border-radius: 12px !important;
    }

    .stTextArea textarea::placeholder {
        color: #9fb2c7 !important;
    }

    div[data-baseweb="select"] > div {
        background: #16273a !important;
        color: #edf3fb !important;
        border: 1px solid rgba(157, 176, 195, 0.24) !important;
        border-radius: 12px !important;
    }

    div[role="listbox"] {
        background: #152538 !important;
        color: #edf3fb !important;
        border: 1px solid rgba(157, 176, 195, 0.24) !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(157, 176, 195, 0.24) !important;
        background: linear-gradient(120deg, #1b2e43, #223953) !important;
        color: #edf3fb !important;
        font-weight: 700 !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #ef4444, #f97316) !important;
        border: none !important;
        color: #fff !important;
    }

    .stButton > button:disabled {
        background: #243548 !important;
        color: #94a9be !important;
        border: 1px solid rgba(157, 176, 195, 0.2) !important;
        opacity: 1 !important;
    }

    .stCodeBlock pre,
    .stCode {
        background: #152538 !important;
        color: #e8f1fb !important;
        border: 1px solid rgba(157, 176, 195, 0.22) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stDataFrame"] div[role="table"] {
        background: #1b2a3a;
        color: #eaf2fb;
        border: 1px solid rgba(157, 176, 195, 0.22);
        border-radius: 12px;
    }

    div[data-testid="stDataFrame"] [role="columnheader"] {
        background: #223447;
        color: #eaf2fb;
        font-weight: 700;
        border-bottom: 1px solid rgba(157, 176, 195, 0.24);
    }

    div[data-testid="stDataFrame"] [role="gridcell"] {
        background: #1b2a3a;
        color: #eaf2fb;
        border-top: 1px solid rgba(157, 176, 195, 0.14);
    }

    @keyframes rise {
        from { transform: translateY(8px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    @media (max-width: 850px) {
        .hero { padding: 1.1rem 1rem; }
        .kpi-value { font-size: 1.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = Path("models")
CATEGORY_MODEL_PATH = MODEL_DIR / "ticket_category_model.joblib"
PRIORITY_MODEL_PATH = MODEL_DIR / "ticket_priority_model.joblib"

SAMPLE_TICKETS = [
    "Payment deducted twice but invoice still shows pending.",
    "Production dashboard shows 500 error for all users.",
    "Unable to log in after enabling two-factor authentication.",
    "Need documentation for integrating with Salesforce.",
]


@st.cache_resource
def load_models():
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    priority_model = joblib.load(PRIORITY_MODEL_PATH)
    return category_model, priority_model


def get_priority_badge(priority_value: str) -> str:
    lowered = priority_value.lower()
    if lowered == "high":
        return '<span class="badge-priority badge-high">HIGH PRIORITY</span>'
    if lowered == "medium":
        return '<span class="badge-priority badge-medium">MEDIUM PRIORITY</span>'
    return '<span class="badge-priority badge-low">LOW PRIORITY</span>'


def confidence_table(model, text: str) -> pd.DataFrame:
    if not hasattr(model, "predict_proba"):
        return pd.DataFrame()

    probs = model.predict_proba([text])[0]
    labels = model.classes_
    frame = pd.DataFrame({"Label": labels, "Confidence": probs})
    frame = frame.sort_values("Confidence", ascending=False).reset_index(drop=True)
    return frame


def confidence_table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""

    rows = []
    for _, row in frame.iterrows():
        label = str(row["Label"])
        confidence = f"{float(row['Confidence']):.2%}"
        rows.append(f"<tr><td>{label}</td><td>{confidence}</td></tr>")

    return (
        '<table class="dark-table">'
        '<thead><tr><th>Label</th><th>Confidence</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


st.markdown(
        """
        <section class="hero">
            <h1 style="margin:0; font-size:2.2rem; letter-spacing:-0.03em;">Support Ticket Command Center</h1>
            <p style="margin:0.4rem 0 0.65rem 0; color:#395066;">Classify issue type and urgency instantly for faster triage.</p>
            <span class="chip">NLP Classification</span>
            <span class="chip">TF-IDF</span>
            <span class="chip">Scikit-learn</span>
            <span class="chip">Ops Analytics</span>
        </section>
        """,
        unsafe_allow_html=True,
)

if not CATEGORY_MODEL_PATH.exists() or not PRIORITY_MODEL_PATH.exists():
    st.error(
        "Models not found. Train first using: python src/train.py --data data/support_tickets.csv --model_dir models"
    )
    st.stop()

category_model, priority_model = load_models()

with st.sidebar:
    st.header("Quick Input")
    sample_text = st.selectbox("Choose a sample ticket", SAMPLE_TICKETS, index=0)
    use_sample = st.button("Use Sample", use_container_width=True)
    st.markdown("---")
    st.caption("Model files")
    st.markdown(
        '<div class="mono">models/ticket_category_model.joblib<br>models/ticket_priority_model.joblib</div>',
        unsafe_allow_html=True,
    )
    st.caption("Run command")
    st.code("streamlit run app.py", language="bash")

if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = "Payment was deducted twice and dashboard is unavailable for my team."

if use_sample:
    st.session_state.ticket_text = sample_text

col_left, col_right = st.columns([1.3, 1.0], gap="large")

with col_left:
    st.markdown('<div class="pane">', unsafe_allow_html=True)
    st.subheader("Ticket Analyzer")
    ticket_text = st.text_area(
        "Paste support ticket text",
        key="ticket_text",
        height=220,
        help="Include symptom, impact, and urgency for better classification.",
    )

    predict_clicked = st.button("Analyze Ticket", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        if not ticket_text.strip():
            st.warning("Please enter ticket text.")
        else:
            category_pred = category_model.predict([ticket_text])[0]
            priority_pred = priority_model.predict([ticket_text])[0]

            category_frame = confidence_table(category_model, ticket_text)
            priority_frame = confidence_table(priority_model, ticket_text)

            category_conf = float(category_frame.iloc[0]["Confidence"]) if not category_frame.empty else None
            priority_conf = float(priority_frame.iloc[0]["Confidence"]) if not priority_frame.empty else None

            st.markdown("### Prediction Summary")
            kpi_col_1, kpi_col_2 = st.columns(2)
            with kpi_col_1:
                st.markdown(
                    f'''
                    <div class="kpi">
                        <div class="kpi-label">Predicted Category</div>
                        <div class="kpi-value">{category_pred}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
            with kpi_col_2:
                st.markdown(
                    f'''
                    <div class="kpi">
                        <div class="kpi-label">Predicted Priority</div>
                        <div class="kpi-value">{priority_pred}</div>
                        {get_priority_badge(str(priority_pred))}
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

            st.markdown("### Confidence")
            conf_1, conf_2 = st.columns(2)
            with conf_1:
                if category_conf is not None:
                    st.caption(f"Category confidence: {category_conf:.2%}")
                    st.progress(min(max(category_conf, 0.0), 1.0))
                if not category_frame.empty:
                    st.markdown(confidence_table_html(category_frame), unsafe_allow_html=True)
            with conf_2:
                if priority_conf is not None:
                    st.caption(f"Priority confidence: {priority_conf:.2%}")
                    st.progress(min(max(priority_conf, 0.0), 1.0))
                if not priority_frame.empty:
                    st.markdown(confidence_table_html(priority_frame), unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="pane">', unsafe_allow_html=True)
    st.subheader("Triage Guidance")
    st.markdown("- Mention business impact, such as outage or revenue loss.")
    st.markdown("- Include platform context: web, mobile, API, or billing portal.")
    st.markdown("- Add urgency indicators: blocked users, deadlines, or repeated failures.")

    st.subheader("Interpreting Output")
    st.markdown("- Use category to route ticket ownership.")
    st.markdown("- Use priority to set SLA and queue ordering.")
    st.markdown("- Confidence scores indicate model certainty level.")
    st.markdown('</div>', unsafe_allow_html=True)
