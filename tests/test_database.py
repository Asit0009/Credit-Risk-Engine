import os
import pytest
from core.database import (
    init_db,
    save_assessment,
    get_all_assessments,
    get_assessment_by_id,
    delete_assessment,
    clear_all_assessments
)


@pytest.fixture
def test_db_path(tmp_path):
    # isolated temp sqlite database for tests
    db_file = tmp_path / "test_credit_history.db"
    return str(db_file)


def test_db_init_and_empty_retrieval(test_db_path):
    # init db and check empty list
    init_db(test_db_path)
    records = get_all_assessments(test_db_path)
    assert records == []


def test_save_and_retrieve_assessment(test_db_path):
    # insert one record and retrieve it
    sample_record = {
        "cibil_score": 760,
        "annual_income": 950000.0,
        "age": 31,
        "employment_type": "Salaried",
        "employment_label": "Salaried",
        "requested_limit": 120000.0,
        "monthly_emi": 18000.0,
        "prior_defaults": 0,
        "dti_ratio": 0.227,
        "limit_ratio": 0.126,
        "risk_probability": 0.034,
        "verdict": "APPROVED",
        "threshold": 0.10,
        "model_version": "LightGBM-v1.0",
        "top_factors": [
            {"feature": "cibil_score", "description": "Healthy CIBIL score reduces risk."}
        ],
        "summary": "Applicant risk probability is 3.4%."
    }

    inserted_id = save_assessment(sample_record, test_db_path)
    assert inserted_id > 0

    records = get_all_assessments(test_db_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["id"] == inserted_id
    assert rec["cibil_score"] == 760
    assert rec["annual_income"] == 950000.0
    assert rec["verdict"] == "APPROVED"
    assert len(rec["top_factors"]) == 1
    assert rec["top_factors"][0]["feature"] == "cibil_score"


def test_persistence_across_connection_reopen(test_db_path):
    # check persistence across separate connections
    sample = {
        "cibil_score": 620,
        "annual_income": 400000.0,
        "age": 45,
        "employment_type": "Self-Employed",
        "requested_limit": 150000.0,
        "monthly_emi": 35000.0,
        "prior_defaults": 1,
        "dti_ratio": 1.05,
        "limit_ratio": 0.375,
        "risk_probability": 0.42,
        "verdict": "REJECTED",
        "threshold": 0.10,
        "model_version": "LightGBM-v1.0",
        "top_factors": [],
        "summary": "Rejected"
    }

    inserted_id = save_assessment(sample, test_db_path)

    # retrieve in fresh connection
    retrieved = get_assessment_by_id(inserted_id, test_db_path)
    assert retrieved is not None
    assert retrieved["id"] == inserted_id
    assert retrieved["verdict"] == "REJECTED"
    assert retrieved["prior_defaults"] == 1


def test_delete_individual_assessment(test_db_path):
    # delete single record by id
    sample = {
        "cibil_score": 700,
        "annual_income": 600000.0,
        "age": 29,
        "employment_type": "Salaried",
        "requested_limit": 50000.0,
        "monthly_emi": 5000.0,
        "prior_defaults": 0,
        "risk_probability": 0.02,
        "verdict": "APPROVED",
        "top_factors": []
    }
    id1 = save_assessment(sample, test_db_path)
    id2 = save_assessment(sample, test_db_path)

    assert len(get_all_assessments(test_db_path)) == 2

    deleted = delete_assessment(id1, test_db_path)
    assert deleted is True

    remaining = get_all_assessments(test_db_path)
    assert len(remaining) == 1
    assert remaining[0]["id"] == id2

    # deleting missing id returns false
    assert delete_assessment(9999, test_db_path) is False


def test_clear_all_assessments(test_db_path):
    # clear all records
    sample = {
        "cibil_score": 700,
        "annual_income": 600000.0,
        "age": 29,
        "employment_type": "Salaried",
        "requested_limit": 50000.0,
        "monthly_emi": 5000.0,
        "prior_defaults": 0,
        "risk_probability": 0.02,
        "verdict": "APPROVED",
        "top_factors": []
    }
    save_assessment(sample, test_db_path)
    save_assessment(sample, test_db_path)
    assert len(get_all_assessments(test_db_path)) == 2

    count = clear_all_assessments(test_db_path)
    assert count == 2
    assert len(get_all_assessments(test_db_path)) == 0
