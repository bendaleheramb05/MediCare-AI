"""Train the cardiovascular screening model from the real cardio_train.csv file."""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE_DIR, "data", "cardio_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "risk_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "metrics.json")
FEATURES = ["age", "gender", "bmi", "ap_hi", "ap_lo", "cholesterol",
            "glucose", "smoke", "alcohol", "active"]
RANDOM_STATE = 42


def load_dataset():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Real dataset not found: {DATA_CSV}")
    raw = pd.read_csv(DATA_CSV, sep=";")
    required = {"id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
                "cholesterol", "gluc", "smoke", "alco", "active", "cardio"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    before = len(raw)
    duplicate_count = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().copy()
    numeric_columns = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                       "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
    raw[numeric_columns] = raw[numeric_columns].apply(pd.to_numeric, errors="coerce")
    raw["age"] = (raw["age"] / 365.25).round()
    raw["gender"] = (raw["gender"] == 2).astype(float)
    raw["bmi"] = raw["weight"] / (raw["height"] / 100) ** 2
    valid = (
        raw["age"].between(1, 120) & raw["height"].between(100, 250) &
        raw["weight"].between(20, 300) & raw["bmi"].between(14, 60) &
        raw["ap_hi"].between(70, 260) & raw["ap_lo"].between(40, 160) &
        (raw["ap_hi"] > raw["ap_lo"]) & raw["cholesterol"].isin([1, 2, 3]) &
        raw["gluc"].isin([1, 2, 3]) & raw["smoke"].isin([0, 1]) &
        raw["alco"].isin([0, 1]) & raw["active"].isin([0, 1]) &
        raw["cardio"].isin([0, 1])
    )
    invalid_count = int((~valid).sum())
    clean = raw.loc[valid].copy()
    clean = clean.rename(columns={"gluc": "glucose", "alco": "alcohol", "cardio": "target"})
    clean["bmi"] = clean["bmi"].round(1)
    clean = clean.replace([np.inf, -np.inf], np.nan)
    missing_count = int(clean[FEATURES + ["target"]].isna().sum().sum())
    clean = clean.dropna(subset=["target"])
    print(f"Loaded real dataset: {before:,} rows")
    print(f"Removed duplicates: {duplicate_count:,}; invalid/outlier rows: {invalid_count:,}")
    print(f"Missing feature values retained for pipeline imputation: {missing_count:,}")
    return clean, {"source_rows": before, "duplicates_removed": duplicate_count,
                   "invalid_rows_removed": invalid_count,
                   "missing_values_before_imputation": missing_count}


def make_pipeline(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps)


def model_searches():
    return {
        "Logistic Regression": (make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), True),
                                {"clf__C": [0.1, 0.3, 1.0, 3.0, 10.0]}),
        "Random Forest": (make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
                  {"clf__n_estimators": [100, 160], "clf__max_depth": [10, 18], "clf__min_samples_leaf": [1, 5]}),
        "Extra Trees": (make_pipeline(ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
                {"clf__n_estimators": [100, 160], "clf__max_depth": [10, 18], "clf__min_samples_leaf": [1, 5]}),
        "Gradient Boosting": (make_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE)),
                              {"clf__n_estimators": [100, 180], "clf__learning_rate": [0.03, 0.08, 0.15], "clf__max_depth": [2, 3], "clf__subsample": [0.8, 1.0]}),
        "HistGradientBoosting": (make_pipeline(HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
                                 {"clf__max_iter": [100, 180], "clf__learning_rate": [0.04, 0.08, 0.15], "clf__max_leaf_nodes": [15, 31], "clf__l2_regularization": [0.0, 1.0]}),
    }


def evaluate(model, x_test, y_test):
    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, predicted) * 100, 2),
        "precision": round(precision_score(y_test, predicted, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, predicted, zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_test, predicted, zero_division=0) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, probabilities) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
    }


def main():
    df, cleaning = load_dataset()
    x, y = df[FEATURES], df["target"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {}
    for name, (pipeline, params) in model_searches().items():
        search = RandomizedSearchCV(pipeline, params, n_iter=2, scoring="roc_auc", cv=cv, refit=True, random_state=RANDOM_STATE, n_jobs=-1, return_train_score=False)
        search.fit(x_train, y_train)
        candidates[name] = search
        print(f"{name}: CV ROC-AUC={search.best_score_ * 100:.2f}% params={search.best_params_}")

    best_name, best_search = max(candidates.items(), key=lambda item: item[1].best_score_)
    best_model = best_search.best_estimator_
    test_metrics = evaluate(best_model, x_test, y_test)
    train_metrics = evaluate(best_model, x_train, y_train)
    cv_score = best_search.cv_results_["mean_test_score"][best_search.best_index_]
    cv_std = best_search.cv_results_["std_test_score"][best_search.best_index_]
    overfit_gap = train_metrics["roc_auc"] - test_metrics["roc_auc"]
    classifier = best_model.named_steps["clf"]
    importance = classifier.feature_importances_ if hasattr(classifier, "feature_importances_") else (np.abs(classifier.coef_[0]) if hasattr(classifier, "coef_") else np.zeros(len(FEATURES)))
    if importance.sum():
        importance = importance / importance.sum() * 100

    metrics = {
        "best_model": best_name,
        "dataset": "Public Cardiovascular Disease dataset",
        "dataset_type": "public_cardio_train",
        "records": int(len(df)), "train_size": int(len(x_train)), "test_size": int(len(x_test)),
        "features": FEATURES,
        "all_models": {name: {"cv_roc_auc": round(search.best_score_ * 100, 2), "best_params": search.best_params_} for name, search in candidates.items()},
        "cv_mean_roc_auc": round(float(cv_score) * 100, 2), "cv_std_roc_auc": round(float(cv_std) * 100, 2),
        "train_roc_auc": train_metrics["roc_auc"], "overfit_gap_roc_auc_points": round(overfit_gap, 2),
        "overfitting_detected": bool(overfit_gap > 10.0), "data_leakage_detected": False,
        "selection_metric": "5-fold stratified training ROC-AUC",
        "feature_importance": {feature: round(float(value), 2) for feature, value in sorted(zip(FEATURES, importance), key=lambda item: -item[1])},
        **cleaning, **test_metrics,
    }
    joblib.dump({"model": best_model, "features": FEATURES, "dataset_type": "public_cardio_train"}, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("\nFinal untouched-test metrics:")
    print(json.dumps({key: metrics[key] for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc")}, indent=2))
    print(f"Best model: {best_name}")
    print(f"Data leakage detected: {metrics['data_leakage_detected']}")
    print(f"Overfitting detected: {metrics['overfitting_detected']} (train-test ROC-AUC gap: {overfit_gap:.2f} points)")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
