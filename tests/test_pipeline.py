# Unit Tests for AutoML Modules including Tuning & Evaluation

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from automl.preprocessing import TelcoPreprocessor
from automl.feature_engineering import FeatureSelector
from automl.model_selector import ModelSelector
from automl.pipeline import run_pipeline
from automl.tuner import tune_xgboost
from automl.evaluator import evaluate_classification_model
from xgboost import XGBClassifier

@pytest.fixture
def sample_data():
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X = df.drop(columns="Churn")
    y = df["Churn"]
    return X, y


def test_preprocessing(sample_data):
    X, y = sample_data
    preprocessor = TelcoPreprocessor()
    X_proc = preprocessor.fit_transform(X)
    y_proc = preprocessor.transform_target(y)
    assert X_proc.shape[0] == X.shape[0]
    assert len(y_proc) == len(y)


def test_feature_selection(sample_data):
    X, y = sample_data
    preprocessor = TelcoPreprocessor()
    X_proc = preprocessor.fit_transform(X)
    y_proc = preprocessor.transform_target(y)

    selector = FeatureSelector(k=10)
    X_sel = selector.fit_transform(X_proc, y_proc)
    assert X_sel.shape[1] == 10


def test_model_selector(sample_data):
    X, y = sample_data
    preprocessor = TelcoPreprocessor()
    X_proc = preprocessor.fit_transform(X)
    y_proc = preprocessor.transform_target(y)
    selector = FeatureSelector(k=10)
    X_sel = selector.fit_transform(X_proc, y_proc)

    model_selector = ModelSelector(cv=3, scoring='f1')
    model_selector.fit(X_sel, y_proc)
    best_name, best_model = model_selector.get_best_model()
    assert best_model is not None


def test_pipeline():
    results = run_pipeline(
        data_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        target_col="Churn",
        k_features=10,
        scoring='f1',
        cv=3,
        do_tuning=False
    )
    assert 'best_model' in results
    assert results['best_model'] is not None


def test_tuner(sample_data):
    X, y = sample_data
    preprocessor = TelcoPreprocessor()
    X_proc = preprocessor.fit_transform(X)
    y_proc = preprocessor.transform_target(y)
    selector = FeatureSelector(k=5)
    X_sel = selector.fit_transform(X_proc, y_proc)

    best_params = tune_xgboost(X_sel, y_proc, n_trials=5, cv=2)
    assert isinstance(best_params, dict)
    assert 'n_estimators' in best_params


def test_evaluator(sample_data):
    X, y = sample_data
    preprocessor = TelcoPreprocessor()
    X_proc = preprocessor.fit_transform(X)
    y_proc = preprocessor.transform_target(y)
    selector = FeatureSelector(k=5)
    X_sel = selector.fit_transform(X_proc, y_proc)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_sel, y_proc)

    metrics = evaluate_classification_model(model, X_sel, y_proc)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert metrics['accuracy'] >= 0.0
