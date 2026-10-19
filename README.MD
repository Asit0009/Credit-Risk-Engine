# Credit Risk Decision Engine

An automated loan and credit card underwriting web app for the Indian banking market. It predicts whether an applicant is likely to default and explains the key factors behind the decision using LightGBM and SHAP.

---

## What This Project Does

In retail banking, about 97.5% of borrowers repay on time, while only 2.5% default. Because defaults are rare and expensive, standard models that guess "approved" for everyone get high accuracy but fail to catch bad borrowers. 

This project solves that problem by:
- Using **LightGBM with cost-sensitive weighting (`scale_pos_weight = 39`)** to penalize missed defaults heavily.
- Setting a strict **10% risk threshold** instead of the standard 50% cutoff, catching over 96% of test defaulters.
- Explaining every decision with **TreeSHAP**, showing the top 3 factors that influenced the approval or rejection.
- Saving every assessment into a local **SQLite database** so past decisions can be reviewed.

---

## Project Structure

```text
├── app.py                     # Streamlit web dashboard
├── train.py                   # Script to train model and save artifacts
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
│
├── core/
│   ├── model_service.py       # Input validation, inference, and SHAP logic
│   └── database.py            # SQLite storage functions
│
├── models/
│   ├── indian_credit_model.pkl
│   ├── indian_shap_explainer.pkl
│   └── indian_metrics.pkl
│
├── tests/
│   ├── test_validation.py
│   ├── test_model_pipeline.py
│   ├── test_shap_explainer.py
│   ├── test_database.py
│   └── test_streamlit_app.py
│
└── data/
    └── decision_history.db    # Local database (created on first run)
```

---

## Dataset

Real bank customer data is confidential under Indian privacy laws. To model real Indian banking behavior accurately, the dataset simulates 6,000 Indian applicant profiles:
- **CIBIL Score:** 300 to 900
- **Annual Income:** ₹1.5 Lakhs to ₹45 Lakhs
- **Age:** 21 to 65
- **Employment:** Salaried, Self-Employed, Unemployed
- **Monthly EMIs & Requested Limit**
- **Default rate:** 2.5% (150 defaults out of 6,000 records)

Two key financial ratios are also calculated:
- **Debt-to-Income (DTI):** `(Monthly EMI * 12) / Annual Income`
- **Limit-to-Income:** `Requested Limit / Annual Income`

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 3. (Optional) Re-train the Model
Pre-trained model artifacts are already included in `models/`. To train from scratch:
```bash
python train.py
```

### 4. Run the Tests
```bash
pytest -v
```

---

## Tech Stack
- **Python** (Pandas, NumPy, Scikit-Learn)
- **LightGBM** (Gradient boosted trees)
- **SHAP** (Explainable AI / TreeExplainer)
- **Streamlit** (Web interface)
- **SQLite** (Audit trail storage)
- **Pytest** (Automated unit and integration testing)