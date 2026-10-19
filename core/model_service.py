import os
import math
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

# features expected by lightgbm model
FEATURE_NAMES: List[str] = [
    "cibil_score",
    "annual_income",
    "age",
    "employment_type",
    "requested_limit",
    "monthly_emi",
    "prior_defaults"
]

# employment mapping
EMPLOYMENT_MAP: Dict[int, str] = {
    0: "Salaried",
    1: "Self-Employed",
    2: "Unemployed"
}

EMPLOYMENT_STR_TO_INT: Dict[str, int] = {
    "Salaried": 0,
    "Self-Employed": 1,
    "Unemployed": 2
}

DEFAULT_MODEL_PATH = os.path.join("models", "indian_credit_model.pkl")  # model path
DEFAULT_EXPLAINER_PATH = os.path.join("models", "indian_shap_explainer.pkl")  # shap path
DEFAULT_METRICS_PATH = os.path.join("models", "indian_metrics.pkl")  # metrics path
DEFAULT_THRESHOLD: float = 0.10  # 10% risk threshold

# templates for plain english explanations
EXPLANATION_TEMPLATES = {
    "cibil_score": {
        "positive": "Lower CIBIL score ({value}) increases the model's estimated risk.",
        "negative": "Healthy CIBIL score ({value}) strongly reduces estimated risk."
    },
    "annual_income": {
        "positive": "Annual income of ₹{value:,.0f} is relatively low compared to requested obligations.",
        "negative": "Solid annual income of ₹{value:,.0f} provides strong repayment support."
    },
    "age": {
        "positive": "Applicant age of {value} years slightly increases risk under the model's distribution.",
        "negative": "Applicant age of {value} years indicates an established financial stage."
    },
    "employment_type": {
        "positive": "Employment status ({type_str}) has higher income variability than salaried roles.",
        "negative": "Salaried employment provides consistent, predictable monthly cash flows."
    },
    "requested_limit": {
        "positive": "Requested credit limit of ₹{value:,.0f} is high relative to income.",
        "negative": "Requested credit limit of ₹{value:,.0f} is conservative relative to income."
    },
    "monthly_emi": {
        "positive": "Existing monthly EMI load of ₹{value:,.0f} narrows available cash margin.",
        "negative": "Low monthly EMI commitments of ₹{value:,.0f} leave ample repayment bandwidth."
    },
    "prior_defaults": {
        "positive": "Record of past credit defaults significantly raises predicted default risk.",
        "negative": "Zero past credit defaults indicates a clean borrowing track record."
    }
}


def validate_applicant_inputs(
    cibil_score: Any,
    annual_income: Any,
    age: Any,
    employment_type: Any,
    requested_limit: Any,
    monthly_emi: Any,
    prior_defaults: Any
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    # check numeric types
    numeric_fields = {
        "cibil_score": cibil_score,
        "annual_income": annual_income,
        "age": age,
        "requested_limit": requested_limit,
        "monthly_emi": monthly_emi
    }

    for name, val in numeric_fields.items():
        if val is None:
            raise ValueError(f"Field '{name}' is required and cannot be empty.")
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"Field '{name}' must be numeric, got {type(val).__name__}.")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Field '{name}' cannot be NaN or infinite.")

    # map employment string to int code
    if isinstance(employment_type, str):
        if employment_type not in EMPLOYMENT_STR_TO_INT:
            raise ValueError(
                f"Invalid employment type '{employment_type}'. Must be one of: {list(EMPLOYMENT_STR_TO_INT.keys())}."
            )
        emp_int = EMPLOYMENT_STR_TO_INT[employment_type]
    elif isinstance(employment_type, (int, np.integer)):
        if employment_type not in EMPLOYMENT_MAP:
            raise ValueError(
                f"Invalid employment type code {employment_type}. Must be 0 (Salaried), 1 (Self-Employed), or 2 (Unemployed)."
            )
        emp_int = int(employment_type)
    else:
        raise ValueError("Employment type must be a valid category string or integer code.")

    # check default flag
    if isinstance(prior_defaults, bool):
        prior_def_int = 1 if prior_defaults else 0
    elif isinstance(prior_defaults, (int, np.integer)):
        if prior_defaults not in (0, 1):
            raise ValueError("Prior defaults must be 0 (no history) or 1 (history of default).")
        prior_def_int = int(prior_defaults)
    else:
        raise ValueError("Prior defaults must be boolean or integer 0/1.")

    # cibil range check
    cibil_int = int(round(cibil_score))
    if not (300 <= cibil_int <= 900):
        raise ValueError(f"CIBIL score must be between 300 and 900, received {cibil_int}.")

    # income range check
    income_val = float(annual_income)
    if income_val <= 0:
        raise ValueError(f"Annual gross income must be strictly greater than zero, received ₹{income_val:,.0f}.")
    if income_val < 50000:
        raise ValueError(f"Annual gross income must be at least ₹50,000 for assessment, received ₹{income_val:,.0f}.")
    if income_val > 500000000:
        raise ValueError("Annual gross income exceeds practical evaluation limit of ₹50 Crores.")

    # age check
    age_int = int(round(age))
    if not (18 <= age_int <= 75):
        raise ValueError(f"Applicant age must be between 18 and 75 years, received {age_int}.")

    # requested limit check
    limit_val = float(requested_limit)
    if limit_val <= 0:
        raise ValueError(f"Requested credit limit must be positive, received ₹{limit_val:,.0f}.")

    # emi check
    emi_val = float(monthly_emi)
    if emi_val < 0:
        raise ValueError(f"Monthly EMI obligations cannot be negative, received ₹{emi_val:,.0f}.")

    # calculate ratios safely
    annual_emi = emi_val * 12.0
    dti_ratio = annual_emi / income_val  # debt to income
    limit_ratio = limit_val / income_val  # limit to income

    clean_dict = {
        "cibil_score": cibil_int,
        "annual_income": income_val,
        "age": age_int,
        "employment_type": emp_int,
        "requested_limit": limit_val,
        "monthly_emi": emi_val,
        "prior_defaults": prior_def_int,
        "dti_ratio": dti_ratio,
        "limit_ratio": limit_ratio,
        "employment_label": EMPLOYMENT_MAP[emp_int]
    }

    # dataframe with exact column ordering
    df = pd.DataFrame([{col: clean_dict[col] for col in FEATURE_NAMES}])
    return clean_dict, df


def load_model_artifacts(
    model_path: str = DEFAULT_MODEL_PATH,
    explainer_path: str = DEFAULT_EXPLAINER_PATH,
    metrics_path: str = DEFAULT_METRICS_PATH
) -> Tuple[Any, Any, Dict[str, Any]]:
    # verify file paths
    for path, name in [
        (model_path, "Model"),
        (explainer_path, "SHAP explainer"),
        (metrics_path, "Evaluation metrics")
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} artifact not found at '{path}'.")

    # load pickle files
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # lightgbm model
    except Exception as e:
        raise RuntimeError(f"Failed to load LightGBM model from '{model_path}': {e}")

    try:
        with open(explainer_path, "rb") as f:
            explainer = pickle.load(f)  # tree explainer
    except Exception as e:
        raise RuntimeError(f"Failed to load SHAP explainer from '{explainer_path}': {e}")

    try:
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)  # test metrics
    except Exception as e:
        raise RuntimeError(f"Failed to load metrics from '{metrics_path}': {e}")

    # check feature names match
    if hasattr(model, "feature_name"):
        model_features = list(model.feature_name())
        if model_features != FEATURE_NAMES:
            raise RuntimeError(
                f"Model feature mismatch. Expected: {FEATURE_NAMES}, found: {model_features}"
            )

    return model, explainer, metrics


def predict_risk(
    model: Any,
    applicant_df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD
) -> Tuple[float, str]:
    # order features
    ordered_df = applicant_df[FEATURE_NAMES]
    raw_pred = model.predict(ordered_df)  # predict default probability

    if isinstance(raw_pred, (list, np.ndarray)):
        prob = float(raw_pred[0])
    else:
        prob = float(raw_pred)

    prob = max(0.0, min(1.0, prob))  # clip between 0 and 1

    # 10% cutoff rule
    verdict = "REJECTED" if prob >= threshold else "APPROVED"
    return prob, verdict


def extract_shap_attributions(
    explainer: Any,
    applicant_df: pd.DataFrame
) -> np.ndarray:
    ordered_df = applicant_df[FEATURE_NAMES]
    raw_shap = explainer.shap_values(ordered_df)  # raw shap values

    if isinstance(raw_shap, list):
        target_shap = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
        vals = np.array(target_shap[0], dtype=float)
    elif isinstance(raw_shap, np.ndarray):
        if raw_shap.ndim == 3:
            vals = raw_shap[0, :, 1].astype(float)
        elif raw_shap.ndim == 2:
            vals = raw_shap[0].astype(float)
        elif raw_shap.ndim == 1:
            vals = raw_shap.astype(float)
        else:
            raise ValueError(f"Unexpected SHAP array dimensions: {raw_shap.shape}")
    else:
        vals = np.array(raw_shap, dtype=float)

    if len(vals) != len(FEATURE_NAMES):
        raise ValueError("SHAP output length mismatch.")

    return vals


def explain_prediction(
    explainer: Any,
    applicant_df: pd.DataFrame,
    applicant_dict: Dict[str, Any],
    verdict: str,
    top_k: int = 3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shap_scores = extract_shap_attributions(explainer, applicant_df)
    all_attributions: List[Dict[str, Any]] = []

    for idx, feature_name in enumerate(FEATURE_NAMES):
        s_val = float(shap_scores[idx])
        raw_val = applicant_dict[feature_name]
        direction = "increases_risk" if s_val >= 0 else "reduces_risk"

        template_dir = "positive" if s_val >= 0 else "negative"
        template = EXPLANATION_TEMPLATES[feature_name][template_dir]

        if feature_name == "prior_defaults":
            desc = template
        elif feature_name == "employment_type":
            desc = template.format(type_str=applicant_dict.get("employment_label", "Applicant Profile"))
        else:
            desc = template.format(value=raw_val)

        all_attributions.append({
            "feature": feature_name,
            "display_name": {
                "cibil_score": "CIBIL Score",
                "annual_income": "Annual Income",
                "age": "Borrower Age",
                "employment_type": "Employment Status",
                "requested_limit": "Requested Card Limit",
                "monthly_emi": "Monthly EMIs",
                "prior_defaults": "Prior Default History"
            }.get(feature_name, feature_name),
            "shap_value": s_val,
            "direction": direction,
            "description": desc,
            "raw_value": raw_val
        })

    # sort top drivers based on verdict
    if verdict == "REJECTED":
        sorted_by_risk = sorted(all_attributions, key=lambda x: x["shap_value"], reverse=True)
        top_factors = sorted_by_risk[:top_k]
    else:
        sorted_by_safety = sorted(all_attributions, key=lambda x: x["shap_value"])
        top_factors = sorted_by_safety[:top_k]

    return top_factors, all_attributions


def build_decision_summary(
    prob: float,
    verdict: str,
    threshold: float,
    top_factors: List[Dict[str, Any]]
) -> str:
    prob_pct = prob * 100.0
    thresh_pct = threshold * 100.0

    if verdict == "REJECTED":
        top_reasons_text = "; ".join([f["description"] for f in top_factors])
        return (
            f"The decision engine estimated an applicant default probability of {prob_pct:.2f}%, "
            f"which meets or exceeds the underwriting threshold of {thresh_pct:.1f}%. "
            f"Key factors driving elevated risk: {top_reasons_text}"
        )
    else:
        top_strengths_text = "; ".join([f["description"] for f in top_factors])
        return (
            f"The decision engine estimated an applicant default probability of {prob_pct:.2f}%, "
            f"which remains safely below the underwriting threshold of {thresh_pct:.1f}%. "
            f"Key factors supporting approval: {top_strengths_text}"
        )
