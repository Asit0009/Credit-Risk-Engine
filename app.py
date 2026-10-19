import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, List

from core.model_service import (
    load_model_artifacts,
    validate_applicant_inputs,
    predict_risk,
    explain_prediction,
    build_decision_summary,
    DEFAULT_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_EXPLAINER_PATH,
    DEFAULT_METRICS_PATH,
    EMPLOYMENT_STR_TO_INT
)
from core.database import (
    init_db,
    save_assessment,
    get_all_assessments,
    get_assessment_by_id,
    delete_assessment,
    clear_all_assessments
)

# page setup
st.set_page_config(
    page_title="Credit Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# dark theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* remove top white header bar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* main app background */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    /* sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d1322 !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #cbd5e1 !important;
        font-size: 0.92rem;
        font-weight: 500;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        color: #60a5fa !important;
        background-color: rgba(59, 130, 246, 0.08);
    }

    /* input fields (remove white backgrounds and make dark fintech) */
    div[data-baseweb="input"] {
        background-color: #131b2e !important;
        border: 1px solid #27354f !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    div[data-baseweb="input"] input {
        color: #f8fafc !important;
        font-weight: 500 !important;
        background-color: transparent !important;
    }
    button[data-testid="stNumberInputStepDown"], 
    button[data-testid="stNumberInputStepUp"] {
        background-color: #1a243a !important;
        color: #cbd5e1 !important;
        border-color: #27354f !important;
    }
    button[data-testid="stNumberInputStepDown"]:hover, 
    button[data-testid="stNumberInputStepUp"]:hover {
        background-color: #27354f !important;
        color: #f8fafc !important;
    }

    /* select dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #131b2e !important;
        border: 1px solid #27354f !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }
    div[data-baseweb="select"] * {
        color: #f8fafc !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #131b2e !important;
        border: 1px solid #27354f !important;
    }
    div[data-baseweb="popover"] li {
        color: #e2e8f0 !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #1e293b !important;
    }

    /* metrics styling (crisp readable cards) */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #131b2e 0%, #0e1628 100%) !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        padding: 0.9rem 1.15rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    div[data-testid="stMetric"] label[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #334155 !important;
        color: #f8fafc !important;
    }

    /* dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    .fintech-card {
        background: linear-gradient(145deg, #131b2e 0%, #0e1628 100%);
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .fintech-card-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .fintech-card-value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .verdict-badge {
        display: inline-block;
        padding: 0.65rem 1.4rem;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-align: center;
        width: 100%;
        margin-bottom: 0.75rem;
    }

    .verdict-approved {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #10b981;
    }

    .verdict-rejected {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
    }

    .driver-item {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }

    .driver-tag-risk {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .driver-tag-safe {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .driver-text {
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.4;
    }

    .disclaimer-box {
        background-color: #0f172a;
        border-left: 3px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)



# cache model resources so streamlit runs faster
@st.cache_resource(show_spinner=False)
def get_cached_artifacts():
    return load_model_artifacts(
        model_path=DEFAULT_MODEL_PATH,
        explainer_path=DEFAULT_EXPLAINER_PATH,
        metrics_path=DEFAULT_METRICS_PATH
    )


init_db()  # init local database


# sidebar navigation
with st.sidebar:
    st.markdown("### Credit Risk Engine")
    st.caption("Loan Underwriting Decision Support")
    st.markdown("---")

    selected_view = st.radio(
        "Navigation",
        options=[
            "New Assessment",
            "Decision History",
            "Model Performance",
            "About Project"
        ],
        index=0
    )

    st.markdown("---")
    st.caption("Configuration:")
    st.caption("Model: LightGBM (Cost-Weighted)")
    st.caption("Threshold: 10% Default Risk Cutoff")
    st.caption("Explainability: TreeSHAP")


# new assessment view
if selected_view == "New Assessment":
    st.title("Credit Risk Assessment")
    st.markdown("Enter applicant details to estimate default risk and see contributing factors.")

    # load model and check availability
    try:
        model, explainer, metrics = get_cached_artifacts()
        artifacts_ready = True
    except Exception as e:
        artifacts_ready = False
        st.error(f"Model artifacts not found: {e}. Please run python train.py first.")

    if artifacts_ready:
        st.markdown("### Applicant Profile & Loan Details")

        col_left, col_right = st.columns(2)  # two column layout

        with col_left:
            st.markdown("#### Financial Profile")
            cibil_input = st.number_input(
                "CIBIL Score (300 to 900)",
                min_value=300,
                max_value=900,
                value=720,
                step=5,
                help="Credit score between 300 and 900."
            )

            income_input = st.number_input(
                "Annual Gross Income (₹)",
                min_value=100000,
                max_value=50000000,
                value=850000,
                step=50000,
                format="%d",
                help="Total annual gross income in INR."
            )

            age_input = st.number_input(
                "Age",
                min_value=21,
                max_value=70,
                value=32,
                step=1
            )

            emp_status_input = st.selectbox(
                "Employment Status",
                options=["Salaried", "Self-Employed", "Unemployed"],
                index=0
            )

            prior_def_input = st.selectbox(
                "Past Defaults",
                options=["No history of default", "Has prior default record"],
                index=0
            )
            has_prior_default = 1 if "Has prior default" in prior_def_input else 0

        with col_right:
            st.markdown("#### Loan Obligations")
            limit_input = st.number_input(
                "Requested Limit (₹)",
                min_value=15000,
                max_value=2000000,
                value=150000,
                step=10000,
                format="%d"
            )

            emi_input = st.number_input(
                "Monthly Existing EMIs (₹)",
                min_value=0,
                max_value=500000,
                value=15000,
                step=2500,
                format="%d"
            )

            # quick preview of ratios
            st.markdown("#### Key Indicators")
            calc_dti = (emi_input * 12.0) / income_input if income_input > 0 else 0.0
            calc_limit_ratio = limit_input / income_input if income_input > 0 else 0.0

            dti_col, ltr_col = st.columns(2)
            with dti_col:
                st.metric(label="Debt to Income (DTI)", value=f"{calc_dti:.1%}")
            with ltr_col:
                st.metric(label="Limit to Income", value=f"{calc_limit_ratio:.1%}")

        st.markdown("---")
        btn_col, reset_col, _ = st.columns([2, 1, 3])
        with btn_col:
            assess_clicked = st.button("Assess Credit Risk", type="primary", use_container_width=True)
        with reset_col:
            reset_clicked = st.button("Reset Form", use_container_width=True)

        if reset_clicked:
            st.rerun()

        # on submit
        if assess_clicked:
            validation_error = None
            try:
                applicant_dict, applicant_df = validate_applicant_inputs(
                    cibil_score=cibil_input,
                    annual_income=income_input,
                    age=age_input,
                    employment_type=emp_status_input,
                    requested_limit=limit_input,
                    monthly_emi=emi_input,
                    prior_defaults=has_prior_default
                )
            except ValueError as ve:
                validation_error = str(ve)

            if validation_error:
                st.error(f"Validation error: {validation_error}")
            else:
                with st.spinner("Evaluating risk and computing explanations..."):
                    threshold = metrics.get("decision_threshold", DEFAULT_THRESHOLD)
                    risk_prob, verdict = predict_risk(model, applicant_df, threshold=threshold)  # model prediction

                    # compute shap explanations
                    top_factors, all_attributions = explain_prediction(
                        explainer=explainer,
                        applicant_df=applicant_df,
                        applicant_dict=applicant_dict,
                        verdict=verdict,
                        top_k=3
                    )

                    # text summary
                    summary_text = build_decision_summary(
                        prob=risk_prob,
                        verdict=verdict,
                        threshold=threshold,
                        top_factors=top_factors
                    )

                    # save to sqlite database
                    save_data = {
                        **applicant_dict,
                        "risk_probability": risk_prob,
                        "verdict": verdict,
                        "threshold": threshold,
                        "model_version": metrics.get("model_type", "LightGBM-v1.0"),
                        "top_factors": top_factors,
                        "summary": summary_text
                    }
                    try:
                        record_id = save_assessment(save_data)  # save record
                    except Exception as db_err:
                        record_id = None
                        st.warning(f"Could not save to history: {db_err}")

                # result panel
                st.markdown("---")
                st.markdown("### Assessment Result")

                res_left, res_right = st.columns([1.1, 1.9])

                with res_left:
                    badge_class = "verdict-approved" if verdict == "APPROVED" else "verdict-rejected"
                    st.markdown(
                        f"<div class='verdict-badge {badge_class}'>{verdict}</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown('<div class="fintech-card">', unsafe_allow_html=True)
                    st.markdown('<div class="fintech-card-title">Estimated Chance of Default</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="fintech-card-value">{risk_prob * 100:.2f}%</div>', unsafe_allow_html=True)

                    progress_val = min(1.0, max(0.0, risk_prob / (threshold * 2.0)))
                    st.progress(progress_val)
                    st.caption(f"Rejection threshold is set at {threshold * 100:.1f}%.")
                    st.markdown('</div>', unsafe_allow_html=True)

                    if record_id:
                        st.success(f"Assessment saved to local history (Record #{record_id}).")

                with res_right:
                    st.markdown("#### Key Influencing Factors")
                    for factor in top_factors:
                        is_risk = factor["direction"] == "increases_risk"
                        tag_class = "driver-tag-risk" if is_risk else "driver-tag-safe"
                        tag_text = "Increases Risk" if is_risk else "Reduces Risk"

                        st.markdown(
                            f"""
                            <div class="driver-item">
                                <span class="{tag_class}">{tag_text}</span>
                                <span class="driver-text">{factor['description']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown("##### Summary")
                    st.info(summary_text)

                # feature attribution chart
                st.markdown("---")
                st.markdown("### Feature Attribution Breakdown")
                st.caption("Red bars push toward default risk, green bars push toward repayment.")

                fig, ax = plt.subplots(figsize=(10, 4.2))
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")

                features_sorted = sorted(all_attributions, key=lambda x: x["shap_value"])
                bar_names = [f["display_name"] for f in features_sorted]
                bar_values = [f["shap_value"] for f in features_sorted]
                bar_colors = ["#ef4444" if v >= 0 else "#10b981" for v in bar_values]

                y_pos = np.arange(len(bar_names))
                bars = ax.barh(y_pos, bar_values, color=bar_colors, height=0.55, edgecolor="none")

                ax.set_yticks(y_pos)
                ax.set_yticklabels(bar_names, color="#e2e8f0", fontsize=9.5)
                ax.axvline(0, color="#475569", linewidth=1.2, linestyle="-")

                ax.xaxis.grid(True, color="#1e293b", linestyle="--", linewidth=0.7)
                ax.set_axisbelow(True)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#334155")
                ax.spines["bottom"].set_color("#334155")
                ax.tick_params(axis="x", colors="#94a3b8", labelsize=8.5)

                for bar in bars:
                    w = bar.get_width()
                    offset = 0.015 if w >= 0 else -0.015
                    ha = "left" if w >= 0 else "right"
                    col = "#fca5a5" if w >= 0 else "#6ee7b7"
                    ax.text(
                        w + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{w:+.3f}",
                        va="center",
                        ha=ha,
                        color=col,
                        fontsize=8.5,
                        fontweight="bold"
                    )

                ax.set_xlabel("SHAP Impact on Default Log-Odds", color="#94a3b8", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # technical details
                with st.expander("Raw Applicant Details"):
                    t1, t2, t3, t4 = st.columns(4)
                    with t1:
                        st.write(f"Income: ₹{applicant_dict['annual_income']:,.0f}")
                        st.write(f"EMI: ₹{applicant_dict['monthly_emi']:,.0f}")
                    with t2:
                        st.write(f"DTI: {applicant_dict['dti_ratio']:.2%}")
                        st.write(f"Limit Ratio: {applicant_dict['limit_ratio']:.2%}")
                    with t3:
                        st.write(f"Employment: {applicant_dict['employment_label']}")
                        st.write(f"Prior Defaults: {'Yes' if applicant_dict['prior_defaults'] else 'No'}")
                    with t4:
                        st.write(f"Model: {metrics.get('model_type', 'LightGBM')}")
                        st.write(f"Threshold: {threshold * 100:.1f}%")

                st.markdown(
                    """
                    <div class="disclaimer-box">
                        Note: This is a student decision-support project built for learning and demonstration. 
                        It uses simulated Indian credit profiles to showcase LightGBM and SHAP explainability.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# decision history view
elif selected_view == "Decision History":
    st.title("Decision History")
    st.markdown("Saved assessment records stored locally in SQLite database.")

    records = get_all_assessments()

    if not records:
        st.info("No saved records found yet. Run an assessment to save one.")
    else:
        tot_count = len(records)
        app_count = sum(1 for r in records if r["verdict"] == "APPROVED")
        rej_count = sum(1 for r in records if r["verdict"] == "REJECTED")
        avg_risk = sum(r["risk_probability"] for r in records) / tot_count

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Assessments", tot_count)
        with k2:
            st.metric("Approved", app_count)
        with k3:
            st.metric("Rejected", rej_count)
        with k4:
            st.metric("Average Risk", f"{avg_risk * 100:.2f}%")

        st.markdown("---")

        df_records = pd.DataFrame([
            {
                "ID": r["id"],
                "Timestamp": r["timestamp"][:19].replace("T", " "),
                "CIBIL": r["cibil_score"],
                "Income": f"₹{r['annual_income']:,.0f}",
                "EMI": f"₹{r['monthly_emi']:,.0f}",
                "Requested Limit": f"₹{r['requested_limit']:,.0f}",
                "Risk Score": f"{r['risk_probability'] * 100:.2f}%",
                "Verdict": r["verdict"]
            }
            for r in records
        ])
        st.dataframe(df_records, use_container_width=True)

        st.markdown("---")
        st.markdown("### View Details")

        col_sel, col_del = st.columns([3, 1])
        with col_sel:
            record_options = {f"Record #{r['id']} - {r['verdict']} ({r['risk_probability']*100:.1f}%) - CIBIL {r['cibil_score']}": r["id"] for r in records}
            selected_label = st.selectbox("Select record", options=list(record_options.keys()))
            selected_id = record_options[selected_label]

        with col_del:
            st.write("&nbsp;")
            if st.button("Delete Record"):
                if delete_assessment(selected_id):
                    st.success(f"Record #{selected_id} deleted.")
                    st.rerun()

        record_detail = get_assessment_by_id(selected_id)
        if record_detail:
            st.markdown(f"#### Record #{record_detail['id']} Details")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.write(f"Verdict: {record_detail['verdict']}")
                st.write(f"Risk: {record_detail['risk_probability'] * 100:.2f}%")
                st.write(f"Threshold: {record_detail['threshold'] * 100:.1f}%")
            with d2:
                st.write(f"CIBIL: {record_detail['cibil_score']}")
                st.write(f"Income: ₹{record_detail['annual_income']:,.0f}")
                st.write(f"EMI: ₹{record_detail['monthly_emi']:,.0f}")
            with d3:
                st.write(f"DTI: {record_detail['dti_ratio']:.1%}")
                st.write(f"Limit Ratio: {record_detail['limit_ratio']:.1%}")
                st.write(f"Employment: {record_detail['employment_type']}")

            st.markdown("Summary:")
            st.info(record_detail["summary"])

            st.markdown("Key Factors:")
            for f in record_detail.get("top_factors", []):
                st.markdown(f"- {f.get('description', '')}")

        st.markdown("---")
        with st.expander("Clear Database History"):
            st.warning("This deletes all records in the local database.")
            confirm_clear = st.checkbox("Confirm delete all history")
            if st.button("Erase All Records", disabled=not confirm_clear):
                deleted_n = clear_all_assessments()
                st.success(f"Deleted {deleted_n} records.")
                st.rerun()


# model performance view
elif selected_view == "Model Performance":
    st.title("Model Performance")
    st.markdown("Saved evaluation metrics on the test dataset.")

    try:
        model, explainer, metrics = get_cached_artifacts()
    except Exception as e:
        metrics = None
        st.error(f"Could not load metrics: {e}")

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}")
        with m2:
            st.metric("PR-AUC", f"{metrics.get('pr_auc', 0.0):.4f}")
        with m3:
            st.metric("Decision Threshold", f"{metrics.get('decision_threshold', 0.10) * 100:.1f}%")
        with m4:
            st.metric("Recall at 10% Cutoff", f"{metrics.get('recall', 0.0):.1%}")

        st.markdown("---")

        col_cm, col_exp = st.columns([1.2, 1.8])

        with col_cm:
            st.markdown("#### Test Set Confusion Matrix")
            cm = metrics.get("confusion_matrix")
            if cm and len(cm) == 2:
                tn, fp = cm[0]
                fn, tp = cm[1]

                st.markdown(f"""
                <div style="background-color: #131b2e; border: 1px solid #1e293b; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; text-align: center;">
                        <thead>
                            <tr style="border-bottom: 1px solid #27354f; color: #94a3b8;">
                                <th style="padding: 0.6rem 0.5rem; text-align: left;">Actual Status</th>
                                <th style="padding: 0.6rem 0.5rem;">Predicted: Approved</th>
                                <th style="padding: 0.6rem 0.5rem;">Predicted: Rejected</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 0.75rem 0.5rem; text-align: left; font-weight: 600; color: #e2e8f0;">Repaid (Safe)</td>
                                <td style="padding: 0.75rem 0.5rem; background: rgba(16, 185, 129, 0.12); color: #34d399; font-weight: 700; border-radius: 4px;">{tn:,} <span style="font-size: 0.75rem; color: #94a3b8; font-weight: normal;">(TN)</span></td>
                                <td style="padding: 0.75rem 0.5rem; background: rgba(245, 158, 11, 0.12); color: #fbbf24; font-weight: 700; border-radius: 4px;">{fp:,} <span style="font-size: 0.75rem; color: #94a3b8; font-weight: normal;">(FP)</span></td>
                            </tr>
                            <tr>
                                <td style="padding: 0.75rem 0.5rem; text-align: left; font-weight: 600; color: #e2e8f0;">Defaulted (Risk)</td>
                                <td style="padding: 0.75rem 0.5rem; background: rgba(239, 68, 68, 0.15); color: #f87171; font-weight: 700; border-radius: 4px;">{fn:,} <span style="font-size: 0.75rem; color: #94a3b8; font-weight: normal;">(FN - Missed)</span></td>
                                <td style="padding: 0.75rem 0.5rem; background: rgba(16, 185, 129, 0.15); color: #34d399; font-weight: 700; border-radius: 4px;">{tp:,} <span style="font-size: 0.75rem; color: #94a3b8; font-weight: normal;">(TP - Caught)</span></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    f"""
                    - Correctly Approved (TN): `{tn}`
                    - Cautiously Rejected (FP): `{fp}`
                    - Missed Defaults (FN): `{fn}`
                    - Caught Defaults (TP): `{tp}`
                    """
                )

        with col_exp:
            st.markdown("#### Why PR-AUC and Recall?")
            st.markdown(
                """
                - **Class Imbalance:** 97.5% of borrowers pay on time, only 2.5% default. A dummy model predicting all safe gets 97.5% accuracy but catches zero defaulters.
                - **Cost Asymmetry:** Missing a defaulter loses 100% of the loan amount, whereas declining a good applicant only loses a small fee margin.
                - **10% Threshold:** Dropping the decision threshold from 50% down to 10% catches 96.7% of defaulters in our test set.
                """
            )

        st.markdown("---")
        st.markdown("#### Dataset Summary")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.write(f"- Total records: `{metrics.get('total_count', 6000):,}`")
        with s2:
            st.write(f"- Default rate: `{metrics.get('defaults_count', 150) / metrics.get('total_count', 6000):.2%}`")
        with s3:
            st.write(f"- Split: 70% train / 10% val / 20% test")
        with s4:
            st.write(f"- Loss weight: `scale_pos_weight = 39.0`")


# about view
elif selected_view == "About Project":
    st.title("About Project")
    st.markdown("Credit Risk Scoring and Underwriting Decision Engine.")

    st.markdown("### How It Works")
    st.markdown(
        """
        1. **Inputs:** CIBIL score (300-900), annual income, age, employment, requested limit, monthly EMI, and past defaults.
        2. **Ratios:** Calculates Debt-to-Income (DTI) and Limit-to-Income ratios.
        3. **Model:** LightGBM gradient boosting trained with `scale_pos_weight = 39` to handle the 2.5% default rate.
        4. **Decision:** If estimated default probability is 10% or more, the applicant is rejected; otherwise approved.
        5. **Explanation:** TreeSHAP calculates the exact contribution of each factor to show why the decision was made.
        6. **History:** Decisions are stored in a local SQLite database for review.
        """
    )

    st.markdown("### Notes on the Data")
    st.markdown(
        """
        To protect real customer privacy and model Indian retail lending rules, this project uses simulated Indian credit profiles. 
        It models Indian CIBIL brackets (300-900), INR salary distributions, and an authentic 2.5% default rate.
        """
    )
