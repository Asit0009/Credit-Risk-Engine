import pytest
import math
import numpy as np
from core.model_service import validate_applicant_inputs, FEATURE_NAMES


def test_valid_applicant_inputs():
    clean_dict, df = validate_applicant_inputs(
        cibil_score=750,
        annual_income=1200000,
        age=32,
        employment_type="Salaried",
        requested_limit=200000,
        monthly_emi=25000,
        prior_defaults=False
    )

    # verify parsed fields
    assert clean_dict["cibil_score"] == 750
    assert clean_dict["annual_income"] == 1200000.0
    assert clean_dict["age"] == 32
    assert clean_dict["employment_type"] == 0
    assert clean_dict["prior_defaults"] == 0
    assert list(df.columns) == FEATURE_NAMES
    assert len(df) == 1

    # check calculated ratios
    assert math.isclose(clean_dict["dti_ratio"], 0.25, rel_tol=1e-5)  # (25000 * 12) / 1200000 = 0.25
    assert math.isclose(clean_dict["limit_ratio"], 200000 / 1200000, rel_tol=1e-5)


def test_cibil_score_boundaries():
    # min boundary 300
    _, df_min = validate_applicant_inputs(300, 500000, 30, "Salaried", 50000, 5000, 0)
    assert df_min["cibil_score"].iloc[0] == 300

    # max boundary 900
    _, df_max = validate_applicant_inputs(900, 500000, 30, "Salaried", 50000, 5000, 0)
    assert df_max["cibil_score"].iloc[0] == 900

    # below 300 should fail
    with pytest.raises(ValueError, match="CIBIL score must be between 300 and 900"):
        validate_applicant_inputs(299, 500000, 30, "Salaried", 50000, 5000, 0)

    # above 900 should fail
    with pytest.raises(ValueError, match="CIBIL score must be between 300 and 900"):
        validate_applicant_inputs(901, 500000, 30, "Salaried", 50000, 5000, 0)


def test_zero_and_negative_income_handling():
    # prevent division by zero in dti calculation
    with pytest.raises(ValueError, match="Annual gross income must be strictly greater than zero"):
        validate_applicant_inputs(700, 0, 30, "Salaried", 50000, 5000, 0)

    with pytest.raises(ValueError, match="Annual gross income must be strictly greater than zero"):
        validate_applicant_inputs(700, -100000, 30, "Salaried", 50000, 5000, 0)


def test_nan_and_infinity_rejection():
    # reject invalid float values
    with pytest.raises(ValueError, match="cannot be NaN or infinite"):
        validate_applicant_inputs(float("nan"), 500000, 30, "Salaried", 50000, 5000, 0)

    with pytest.raises(ValueError, match="cannot be NaN or infinite"):
        validate_applicant_inputs(700, float("inf"), 30, "Salaried", 50000, 5000, 0)

    with pytest.raises(ValueError, match="cannot be NaN or infinite"):
        validate_applicant_inputs(700, 500000, 30, "Salaried", float("nan"), 5000, 0)


def test_negative_emi_and_negative_limit():
    # emi and requested limit cannot be negative
    with pytest.raises(ValueError, match="Monthly EMI obligations cannot be negative"):
        validate_applicant_inputs(700, 500000, 30, "Salaried", 50000, -500, 0)

    with pytest.raises(ValueError, match="Requested credit limit must be positive"):
        validate_applicant_inputs(700, 500000, 30, "Salaried", -1000, 5000, 0)


def test_invalid_employment_and_defaults():
    # unrecognized employment string
    with pytest.raises(ValueError, match="Invalid employment type"):
        validate_applicant_inputs(700, 500000, 30, "Freelancer", 50000, 5000, 0)

    with pytest.raises(ValueError, match="Prior defaults must be"):
        validate_applicant_inputs(700, 500000, 30, "Salaried", 50000, 5000, 99)
