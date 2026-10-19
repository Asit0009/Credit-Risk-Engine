import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import lightgbm as lgb
import shap

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "indian_credit_model.pkl")  # trained model
EXPLAINER_PATH = os.path.join(MODELS_DIR, "indian_shap_explainer.pkl")  # shap explainer
METRICS_PATH = os.path.join(MODELS_DIR, "indian_metrics.pkl")  # test evaluation


def generate_synthetic_indian_credit_data(num_samples: int = 6000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # cibil score bounded 300 to 900
    cibil_score = np.random.normal(loc=680, scale=85, size=num_samples).astype(int)
    cibil_score = np.clip(cibil_score, 300, 900)

    # income in inr with right skew
    annual_income = np.random.lognormal(mean=13.6, sigma=0.5, size=num_samples).astype(int)
    annual_income = np.clip(annual_income, 150000, 4500000)

    # age and employment type
    age = np.random.randint(21, 65, size=num_samples)
    employment_type = np.random.choice([0, 1, 2], size=num_samples, p=[0.65, 0.30, 0.05])

    # requested limit based on income
    requested_limit = []
    for inc in annual_income:
        pct = np.random.uniform(0.08, 0.30)
        requested_limit.append(int(np.clip(inc * pct, 15000, 1000000)))
    requested_limit = np.array(requested_limit)

    # monthly emi around 25% of monthly salary
    monthly_emi = []
    for cs, inc in zip(cibil_score, annual_income):
        base_emi_pct = np.random.beta(a=2, b=6)
        monthly_inc = inc / 12.0
        monthly_emi.append(int(np.clip(monthly_inc * base_emi_pct, 0, 150000)))
    monthly_emi = np.array(monthly_emi)

    # past default history depends on cibil
    prior_defaults = []
    for cs in cibil_score:
        prob = 0.35 if cs < 550 else (0.10 if cs < 700 else (0.02 if cs < 750 else 0.001))
        prior_defaults.append(1 if np.random.random() < prob else 0)
    prior_defaults = np.array(prior_defaults)

    # domain financial ratios
    dti = (monthly_emi * 12.0) / annual_income  # debt to income
    limit_ratio = requested_limit / annual_income  # limit ratio

    # calculate ground truth risk score
    risk_score = (
        -0.018 * (cibil_score - 650)
        + 4.5 * (dti - 0.25)
        + 1.8 * (limit_ratio - 0.15)
        + 0.8 * (employment_type == 1)
        + 2.8 * (employment_type == 2)
        + 3.5 * prior_defaults
        - 0.015 * (age - 35)
        + np.random.normal(scale=0.5, size=num_samples)
    )

    # calibrate top 2.5% as defaulters
    target_default_pct = 0.025
    threshold_idx = int(num_samples * (1 - target_default_pct))
    threshold = np.sort(risk_score)[threshold_idx]
    defaults = (risk_score >= threshold).astype(int)

    df = pd.DataFrame({
        "cibil_score": cibil_score,
        "annual_income": annual_income,
        "age": age,
        "employment_type": employment_type,
        "requested_limit": requested_limit,
        "monthly_emi": monthly_emi,
        "prior_defaults": prior_defaults,
        "default": defaults
    })
    return df


def train_pipeline(num_samples: int = 6000, seed: int = 42) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = generate_synthetic_indian_credit_data(num_samples=num_samples, seed=seed)

    X = df.drop(columns=["default"])
    y = df["default"]

    # 80% train (4,800), 10% validation (600), 10% test (600)
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.50, random_state=seed, stratify=y_val_test
    )

    # calculate scale pos weight for class imbalance
    neg_count = len(y_train) - y_train.sum()
    pos_count = y_train.sum()
    scale_pos_weight = float(neg_count / pos_count)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1,
        "random_state": seed
    }

    # train lightgbm model with early stopping on validation set only
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )

    # evaluate final model once on the untouched holdout test set
    y_pred_prob = model.predict(X_test)
    roc_auc = float(roc_auc_score(y_test, y_pred_prob))

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = float(auc(recall_curve, precision_curve))

    decision_threshold = 0.10  # 10% underwriting cutoff
    y_pred_binary = (y_pred_prob >= decision_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred_binary).tolist()
    prec = float(precision_score(y_test, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_test, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_test, y_pred_binary, zero_division=0))

    # create shap tree explainer
    explainer = shap.TreeExplainer(model)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "decision_threshold": decision_threshold,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "defaults_count": int(y.sum()),
        "total_count": int(len(y)),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "test_defaults_count": int(y_test.sum()),
        "scale_pos_weight": scale_pos_weight,
        "model_type": "LightGBM Classifier",
        "features": list(X.columns)
    }

    # save artifacts to models folder
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(EXPLAINER_PATH, "wb") as f:
        pickle.dump(explainer, f)
    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)

    print("Training finished successfully.")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"At {decision_threshold*100:.0f}% cutoff: Recall={rec:.4f}, Precision={prec:.4f}, F1={f1:.4f}")
    return metrics


if __name__ == "__main__":
    train_pipeline()
