import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from core.model_service import (
    load_model_artifacts,
    extract_shap_attributions,
    explain_prediction,
    FEATURE_NAMES
)


def test_shap_attributions_with_mock_explainer():
    # test shap attribution handling for 2d and list shapes
    sample_df = pd.DataFrame([{col: 100 for col in FEATURE_NAMES}])
    mock_explainer = MagicMock()

    # 2d array output: shape (1, 7)
    mock_explainer.shap_values.return_value = np.array([[-0.5, 0.2, -0.1, 0.4, -0.2, 0.8, 1.2]])
    vals_2d = extract_shap_attributions(mock_explainer, sample_df)
    assert len(vals_2d) == 7
    assert vals_2d[0] == -0.5
    assert vals_2d[6] == 1.2

    # list of arrays [class_0, class_1]
    mock_explainer.shap_values.return_value = [
        np.array([[0.5, -0.2, 0.1, -0.4, 0.2, -0.8, -1.2]]),
        np.array([[-0.5, 0.2, -0.1, 0.4, -0.2, 0.8, 1.2]])
    ]
    vals_list = extract_shap_attributions(mock_explainer, sample_df)
    assert len(vals_list) == 7
    assert vals_list[6] == 1.2


def test_explain_prediction_direction_and_top_k():
    # test top 3 factors for rejected borrower
    _, explainer, _ = load_model_artifacts()

    applicant_dict = {
        "cibil_score": 550,
        "annual_income": 300000,
        "age": 25,
        "employment_type": 1,
        "requested_limit": 150000,
        "monthly_emi": 20000,
        "prior_defaults": 1,
        "employment_label": "Self-Employed"
    }
    applicant_df = pd.DataFrame([{col: applicant_dict[col] for col in FEATURE_NAMES}])

    top_factors, all_attributions = explain_prediction(
        explainer=explainer,
        applicant_df=applicant_df,
        applicant_dict=applicant_dict,
        verdict="REJECTED",
        top_k=3
    )

    # check 3 top drivers and 7 full attributions
    assert len(top_factors) == 3
    assert len(all_attributions) == 7

    for f in top_factors:
        assert "description" in f
        assert "shap_value" in f
        assert "direction" in f
        assert f["direction"] in ("increases_risk", "reduces_risk")
        assert "{" not in f["description"]
        assert "}" not in f["description"]


def test_approved_applicant_shap_explanation():
    # test top mitigating factors for approved borrower
    _, explainer, _ = load_model_artifacts()

    prime_dict = {
        "cibil_score": 820,
        "annual_income": 1800000,
        "age": 35,
        "employment_type": 0,
        "requested_limit": 100000,
        "monthly_emi": 10000,
        "prior_defaults": 0,
        "employment_label": "Salaried"
    }
    prime_df = pd.DataFrame([{col: prime_dict[col] for col in FEATURE_NAMES}])

    top_factors, _ = explain_prediction(
        explainer=explainer,
        applicant_df=prime_df,
        applicant_dict=prime_dict,
        verdict="APPROVED",
        top_k=3
    )

    assert len(top_factors) == 3
    # first factor should be strongest safety factor
    assert top_factors[0]["shap_value"] <= top_factors[1]["shap_value"]
