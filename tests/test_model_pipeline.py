import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from core.model_service import (
    load_model_artifacts,
    predict_risk,
    FEATURE_NAMES,
    DEFAULT_THRESHOLD
)


def test_load_real_artifacts():
    model, explainer, metrics = load_model_artifacts()  # load saved model files

    assert model is not None
    assert explainer is not None
    assert isinstance(metrics, dict)
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "decision_threshold" in metrics

    # check feature schema matches
    assert hasattr(model, "feature_name")
    assert list(model.feature_name()) == FEATURE_NAMES


def test_missing_artifacts_raises_file_not_found():
    # test error raised when model file missing
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model_artifacts(model_path="non_existent_path.pkl")


def test_threshold_decisions_boundary():
    # test 10% underwriting boundary rule
    mock_model = MagicMock()
    sample_df = pd.DataFrame([{col: 1 for col in FEATURE_NAMES}])

    # 1. just below 10% cutoff
    mock_model.predict.return_value = np.array([0.0999])
    prob, verdict = predict_risk(mock_model, sample_df, threshold=0.10)
    assert abs(prob - 0.0999) < 1e-6
    assert verdict == "APPROVED"

    # 2. exactly at 10% cutoff
    mock_model.predict.return_value = np.array([0.1000])
    prob, verdict = predict_risk(mock_model, sample_df, threshold=0.10)
    assert abs(prob - 0.1000) < 1e-6
    assert verdict == "REJECTED"

    # 3. above cutoff
    mock_model.predict.return_value = np.array([0.1001])
    prob, verdict = predict_risk(mock_model, sample_df, threshold=0.10)
    assert abs(prob - 0.1001) < 1e-6
    assert verdict == "REJECTED"


def test_real_model_inference_known_profiles():
    # evaluate real model on two contrasting profiles
    model, _, metrics = load_model_artifacts()
    threshold = metrics.get("decision_threshold", DEFAULT_THRESHOLD)

    # strong prime profile: 800 cibil, 25l income, no defaults
    prime_df = pd.DataFrame([{
        "cibil_score": 800,
        "annual_income": 2500000,
        "age": 38,
        "employment_type": 0,
        "requested_limit": 200000,
        "monthly_emi": 20000,
        "prior_defaults": 0
    }])
    prime_prob, prime_verdict = predict_risk(model, prime_df, threshold=threshold)
    assert 0.0 <= prime_prob <= 1.0
    assert prime_prob < threshold
    assert prime_verdict == "APPROVED"

    # high risk profile: 480 cibil, 2.5l income, prior default
    risky_df = pd.DataFrame([{
        "cibil_score": 480,
        "annual_income": 250000,
        "age": 28,
        "employment_type": 2,
        "requested_limit": 100000,
        "monthly_emi": 40000,
        "prior_defaults": 1
    }])
    risky_prob, risky_verdict = predict_risk(model, risky_df, threshold=threshold)
    assert 0.0 <= risky_prob <= 1.0
    assert risky_prob >= threshold
    assert risky_verdict == "REJECTED"
