from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    APP_ROOT,
    ARTIFACT_PATH,
    EDA_IMAGE_PATHS,
    EDA_REPORT_PATH,
    EDA_SUMMARY_PATH,
    FEATURE_IMPORTANCE_PATH,
    MODEL_IMAGE_PATHS,
    MODEL_METRICS_PATH,
    SAMPLE_BATCH_PATH,
)
from src.data.loaders import load_dataset, read_json, read_text
from src.features.schema import (
    FEATURE_GROUPS,
    FIELD_HELP,
    FIELD_LABELS,
    RISK_BAND_RECOMMENDATIONS,
    get_default_record,
    get_feature_options,
)
from src.inference.predict import classify_risk_band, explain_prediction, load_artifact, predict_dataframe, top_global_drivers


st.set_page_config(
    page_title="10Alytics Churn Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND_LOGO_RASTER_PATH = APP_ROOT / "assets" / "10alytics_logo.png"
BRAND_LOGO_VECTOR_PATH = APP_ROOT / "assets" / "10alytics_wordmark.svg"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --brand-navy: #4d4d4d;
            --brand-gold: #db6800;
            --brand-cream: #fff8f2;
            --brand-mist: #f4efea;
            --brand-panel: #ffffff;
            --brand-ink: #2f2f2f;
            --brand-muted: #6b6b6b;
            --brand-line: rgba(77, 77, 77, 0.10);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(219,104,0,0.12), transparent 22%),
                radial-gradient(circle at top right, rgba(77,77,77,0.08), transparent 30%),
                linear-gradient(180deg, #fffdfa 0%, #f8f2ec 100%);
        }
        body, .stMarkdown, .stText, .stAlert, .stDataFrame, .stMetric {
            color: var(--brand-ink);
        }
        h1, h2, h3, h4 {
            color: var(--brand-navy);
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: -0.01em;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .brand-hero {
            padding: 1.5rem 1.6rem;
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(49,49,49,0.98) 0%, rgba(78,78,78,0.98) 54%, rgba(219,104,0,0.96) 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(77, 77, 77, 0.16);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .brand-hero h1 {
            color: white;
            margin: 0;
            font-size: 2.25rem;
            line-height: 1.1;
        }
        .brand-hero p {
            margin: 0.85rem 0 0 0;
            font-size: 1rem;
            max-width: 760px;
            color: rgba(255,255,255,0.88);
        }
        .hero-kicker {
            color: rgba(255,255,255,0.78);
            font-size: 0.85rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .metric-card {
            background: var(--brand-panel);
            border: 1px solid var(--brand-line);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 28px rgba(77, 77, 77, 0.06);
            min-height: 120px;
        }
        .metric-label {
            color: var(--brand-muted);
            font-size: 0.8rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .metric-value {
            color: var(--brand-navy);
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }
        .metric-help {
            color: var(--brand-muted);
            font-size: 0.92rem;
            margin-top: 0.35rem;
        }
        .insight-card {
            background: rgba(255,255,255,0.94);
            border: 1px solid var(--brand-line);
            border-left: 5px solid var(--brand-gold);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 10px 24px rgba(77, 77, 77, 0.05);
        }
        .section-kicker {
            color: var(--brand-gold);
            font-size: 0.82rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .soft-panel {
            background: rgba(255,255,255,0.78);
            border: 1px solid var(--brand-line);
            border-radius: 20px;
            padding: 1rem 1rem 0.8rem 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.7);
            border: 1px solid var(--brand-line);
            border-radius: 999px;
            padding: 0.45rem 0.95rem;
        }
        .stTabs [aria-selected="true"] {
            background: var(--brand-gold) !important;
            color: white !important;
            border-color: var(--brand-gold) !important;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(52,52,52,0.99) 0%, rgba(35,35,35,0.99) 100%);
        }
        [data-testid="stSidebar"] * {
            color: white;
        }
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stCaption {
            color: rgba(255,255,255,0.82) !important;
        }
        .eda-callout {
            background: rgba(219,104,0,0.09);
            border: 1px solid rgba(219,104,0,0.22);
            border-radius: 18px;
            padding: 0.95rem 1rem 0.8rem 1rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(219,104,0,0.16);
            background: linear-gradient(135deg, #db6800 0%, #f08a1b 100%);
            color: white;
            font-weight: 600;
            box-shadow: 0 10px 20px rgba(219,104,0,0.16);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(219,104,0,0.24);
            color: white;
        }
        .stSelectbox > div > div, .stNumberInput > div > div > input, .stFileUploader {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource(show_spinner=False)
def get_artifact():
    return load_artifact(ARTIFACT_PATH)


@st.cache_data(show_spinner=False)
def get_eda_summary() -> dict[str, object]:
    return read_json(EDA_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def get_model_metrics() -> dict[str, object]:
    return read_json(MODEL_METRICS_PATH)


@st.cache_data(show_spinner=False)
def get_feature_importance() -> pd.DataFrame:
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_brand_logo(max_width: int = 320) -> None:
    if BRAND_LOGO_RASTER_PATH.exists():
        st.image(str(BRAND_LOGO_RASTER_PATH), width=max_width)
        return

    if BRAND_LOGO_VECTOR_PATH.exists():
        svg_markup = BRAND_LOGO_VECTOR_PATH.read_text(encoding="utf-8")
        st.markdown(
            f'<div style="max-width:{max_width}px;">{svg_markup}</div>',
            unsafe_allow_html=True,
        )


def info_card(title: str, body: str, kicker: str = "Insight") -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="section-kicker">{kicker}</div>
            <h4 style="margin:0.4rem 0 0.35rem 0;">{title}</h4>
            <p style="margin:0; color:var(--brand-ink); line-height:1.55;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_brand_hero(title: str, subtitle: str, kicker: str) -> None:
    left, right = st.columns([1.0, 2.4], vertical_alignment="center")
    with left:
        render_brand_logo(max_width=340)
    with right:
        st.markdown(
            f"""
            <div class="brand-hero">
                <div class="hero-kicker">{kicker}</div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_overview() -> None:
    dataframe = get_dataset()
    eda_summary = get_eda_summary()
    metrics = get_model_metrics()

    render_brand_hero(
        "Customer Churn Studio",
        "A visually refined Streamlit app prepared for Streamlit Cloud deployment, with explainable scoring, batch inference, and an EDA storytelling layer.",
        "10Alytics Branded Delivery",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Customers", f"{len(dataframe):,}", "Training rows available in the production dataset.")
    with col2:
        metric_card("Churn Rate", f"{eda_summary['churn_rate']:.1%}", "Share of customers that churned in the historical data.")
    with col3:
        metric_card("Recall", f"{metrics['recall']:.3f}", "The model favors catching churn cases before they are missed.")
    with col4:
        metric_card("F1 Score", f"{metrics['f1_score']:.3f}", "Balanced quality measure for the churn class.")

    st.markdown("### What this production app now does")
    col_a, col_b = st.columns([1.2, 1.0], vertical_alignment="top")
    with col_a:
        info_card(
            "Operational scoring",
            "The app supports both single-customer scoring and batch CSV scoring with downloadable outputs, so the project can move beyond notebooks into a usable decision-support tool.",
            "Production",
        )
        info_card(
            "Explainable modelling",
            "The decision tree stays readable and exposes both global feature importance and record-level decision steps, which keeps the work useful for business stakeholders and teaching.",
            "Interpretability",
        )
    with col_b:
        st.image(str(EDA_IMAGE_PATHS["churn_distribution"]), use_container_width=True)

    st.markdown("### Deployment posture")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            """
            <div class="soft-panel">
                <div class="section-kicker">Cloud ready</div>
                <p style="margin:0.5rem 0 0 0;">
                    The project already contains the saved artifact, generated visuals, runtime dependencies,
                    sample batch template, and Streamlit configuration required for Streamlit Community Cloud.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            """
            <div class="soft-panel">
                <div class="section-kicker">Suggested next step</div>
                <p style="margin:0.5rem 0 0 0;">
                    Push this folder to GitHub, point Streamlit Cloud at <code>app.py</code>, and deploy with Python 3.12.
                    No external secrets are required for the current version.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_single_record_input(dataframe: pd.DataFrame) -> pd.DataFrame:
    defaults = get_default_record(dataframe)
    options = get_feature_options(dataframe)
    record: dict[str, object] = {}

    tab_names = list(FEATURE_GROUPS.keys())
    tabs = st.tabs(tab_names)

    for tab, group_name in zip(tabs, tab_names, strict=False):
        with tab:
            columns = st.columns(2)
            for index, field in enumerate(FEATURE_GROUPS[group_name]):
                with columns[index % 2]:
                    label = FIELD_LABELS.get(field, field)
                    help_text = FIELD_HELP.get(field)

                    if field == "SeniorCitizen":
                        choices = options["SeniorCitizen"]
                        default = defaults[field]
                        selected = st.selectbox(label, choices, index=choices.index(default), help=help_text)
                        record[field] = 1 if selected == "Yes" else 0
                    elif field in options:
                        choices = options[field]
                        default = defaults[field]
                        default_index = choices.index(default) if default in choices else 0
                        record[field] = st.selectbox(label, choices, index=default_index, help=help_text)
                    else:
                        is_integer = field == "tenure"
                        value = int(round(float(defaults[field]))) if is_integer else float(defaults[field])
                        step = 1 if is_integer else 0.01
                        record[field] = st.number_input(
                            label,
                            min_value=0 if field in {"tenure", "MonthlyCharges", "TotalCharges"} else None,
                            value=value,
                            step=step,
                            help=help_text,
                        )

    return pd.DataFrame([record])


def render_single_customer_scoring() -> None:
    dataframe = get_dataset()
    artifact = get_artifact()

    render_brand_hero(
        "Single-customer scoring",
        "Score one customer profile at a time, inspect the estimated churn probability, and read the path the decision tree followed.",
        "Prediction workflow",
    )

    with st.form("single_customer_form"):
        record = build_single_record_input(dataframe)
        submitted = st.form_submit_button("Score customer", use_container_width=True)

    if not submitted:
        return

    predictions = predict_dataframe(record, artifact)
    result = predictions.iloc[0]
    decision_steps = explain_prediction(record, artifact)
    risk_band = classify_risk_band(float(result["churn_probability"]))

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Predicted label", str(result["prediction"]), "Binary churn outcome from the production pipeline.")
    with col2:
        metric_card("Churn probability", f"{float(result['churn_probability']):.1%}", "Positive-class probability for the current record.")
    with col3:
        metric_card("Risk band", risk_band, "Business-friendly interpretation of the raw churn score.")

    st.markdown("### Decision path explanation")
    if decision_steps:
        for step_number, step_text in enumerate(decision_steps, start=1):
            st.write(f"{step_number}. {step_text}")
    else:
        st.info("No detailed decision path was available for this record.")

    st.markdown("### Recommended next actions")
    for recommendation in RISK_BAND_RECOMMENDATIONS[risk_band]:
        st.write(f"- {recommendation}")


def render_batch_scoring() -> None:
    artifact = get_artifact()

    render_brand_hero(
        "Batch CSV scoring",
        "Upload multiple customer records, score them with the saved pipeline, and download a prediction file ready for downstream action.",
        "Operations",
    )

    template_bytes = SAMPLE_BATCH_PATH.read_bytes()
    st.download_button(
        label="Download sample batch template",
        data=template_bytes,
        file_name=SAMPLE_BATCH_PATH.name,
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        return

    batch_frame = pd.read_csv(uploaded)
    predictions = predict_dataframe(batch_frame, artifact)

    st.success(f"Scored {len(predictions):,} records successfully.")
    st.dataframe(predictions.head(20), use_container_width=True)

    csv_bytes = predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download scored predictions",
        data=csv_bytes,
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
    )


def render_model_insights() -> None:
    artifact = get_artifact()
    metrics = get_model_metrics()
    feature_importance = get_feature_importance()
    top_drivers = top_global_drivers(artifact, limit=6)

    render_brand_hero(
        "Model insights",
        "Review how the deployed decision tree performs, what it learned from the churn data, and why its outputs remain explainable.",
        "Diagnostics",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Baseline", f"{metrics['baseline_accuracy']:.3f}", "Accuracy from always predicting the majority class.")
    with col2:
        metric_card("Accuracy", f"{metrics['accuracy']:.3f}", "Held-out test accuracy for the chosen tree.")
    with col3:
        metric_card("Precision", f"{metrics['precision']:.3f}", "How often churn predictions are correct.")
    with col4:
        metric_card("Recall", f"{metrics['recall']:.3f}", "How many churn cases the model catches.")

    perf_tab, interpret_tab = st.tabs(["Performance visuals", "Interpretation"])
    with perf_tab:
        a, b = st.columns(2)
        with a:
            st.image(str(MODEL_IMAGE_PATHS["confusion_matrix"]), use_container_width=True)
        with b:
            st.image(str(MODEL_IMAGE_PATHS["feature_importance"]), use_container_width=True)
    with interpret_tab:
        left, right = st.columns([1.4, 1.0], vertical_alignment="top")
        with left:
            st.image(str(MODEL_IMAGE_PATHS["decision_tree"]), use_container_width=True)
        with right:
            info_card(
                "Top global drivers",
                "The strongest signals remain contract commitment, internet-service type, tenure, and billing behavior. That keeps the model aligned with the EDA story rather than behaving like a black box.",
                "Interpretation",
            )
            st.dataframe(pd.DataFrame(top_drivers), use_container_width=True)
            st.dataframe(feature_importance.head(15), use_container_width=True)


def render_eda_story() -> None:
    dataframe = get_dataset()
    summary = get_eda_summary()
    report_text = read_text(EDA_REPORT_PATH)

    render_brand_hero(
        "EDA story",
        "This page turns the raw charts into a guided story about who churns, why those customers are at risk, and which signals matter most before modelling.",
        "Exploratory analysis",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows", f"{summary['rows']:,}", "Customer records used in the case study.")
    with c2:
        metric_card("Feature columns", str(summary["columns"]), "Variables available before feature engineering.")
    with c3:
        metric_card("Missing TotalCharges", str(summary["missing_total_charges"]), "Very small missingness concentrated among new customers.")
    with c4:
        metric_card("Churn rate", f"{summary['churn_rate']:.1%}", "The problem is imbalanced but still manageable.")

    target_tab, segment_tab, lifecycle_tab, report_tab = st.tabs(
        ["Target overview", "Segment drivers", "Lifecycle & revenue", "Report & raw data"]
    )

    with target_tab:
        left, right = st.columns([1.2, 1.0], vertical_alignment="center")
        with left:
            st.image(str(EDA_IMAGE_PATHS["churn_distribution"]), use_container_width=True)
        with right:
            info_card(
                "Why class balance matters",
                "About one quarter of customers churned. That means plain accuracy can be misleading, so the app and training workflow both surface churn-focused metrics like recall and F1 score.",
                "EDA insight",
            )
            st.markdown(
                """
                <div class="eda-callout">
                    <strong>Teaching angle:</strong> this is a clean example of why we inspect the target variable before any modelling choices are made.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with segment_tab:
        top, bottom = st.columns(2)
        with top:
            st.image(str(EDA_IMAGE_PATHS["contract_churn"]), use_container_width=True)
        with bottom:
            st.image(str(EDA_IMAGE_PATHS["service_billing_churn"]), use_container_width=True)
        st.write(
            f"Month-to-month customers show the sharpest churn risk, and customers using electronic checks remain a higher-risk billing segment. "
            f"Fiber-optic customers also churn more often than the DSL and no-internet groups."
        )

    with lifecycle_tab:
        left, right = st.columns([1.15, 1.0], vertical_alignment="top")
        with left:
            st.image(str(EDA_IMAGE_PATHS["numeric_boxplots"]), use_container_width=True)
        with right:
            info_card(
                "Retention patterns",
                "Customers who churn tend to be newer and pay more monthly. That combination of short tenure plus higher recurring charges is one of the clearest patterns in the case study.",
                "EDA insight",
            )
            st.dataframe(
                dataframe.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]].mean().round(2),
                use_container_width=True,
            )

    with report_tab:
        st.markdown(report_text)
        st.dataframe(dataframe.head(50), use_container_width=True)


def render_sidebar() -> str:
    with st.sidebar:
        render_brand_logo(max_width=240)
        st.markdown("### Navigation")
        view = st.radio(
            "Select a page",
            [
                "Overview",
                "EDA story",
                "Single customer scoring",
                "Batch scoring",
                "Model insights",
            ],
        )

        st.markdown("---")
        st.caption("Prepared for Streamlit Community Cloud deployment")
        st.caption(f"Project root: `{APP_ROOT.name}`")
        st.caption(f"Artifact: `{Path(ARTIFACT_PATH).name}`")
        return view


def main() -> None:
    inject_styles()

    missing_outputs = [
        path
        for path in [
            ARTIFACT_PATH,
            EDA_SUMMARY_PATH,
            MODEL_METRICS_PATH,
            SAMPLE_BATCH_PATH,
            *EDA_IMAGE_PATHS.values(),
            *MODEL_IMAGE_PATHS.values(),
        ]
        if not Path(path).exists()
    ]
    if missing_outputs:
        st.error(
            "This app is missing generated assets. Run `python eda.py` and `python train.py` before launching Streamlit."
        )
        st.stop()

    selected_view = render_sidebar()

    if selected_view == "Overview":
        render_overview()
    elif selected_view == "EDA story":
        render_eda_story()
    elif selected_view == "Single customer scoring":
        render_single_customer_scoring()
    elif selected_view == "Batch scoring":
        render_batch_scoring()
    else:
        render_model_insights()


if __name__ == "__main__":
    main()
