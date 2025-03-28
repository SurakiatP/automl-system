# AutoML Pipeline - Combine preprocessing, feature selection, model selection, and optional tuning

import pandas as pd
from automl.preprocessing import TelcoPreprocessor
from automl.feature_engineering import FeatureSelector
from automl.model_selector import ModelSelector
from automl.tuner import tune_xgboost
from xgboost import XGBClassifier


def run_pipeline(data_path, target_col='Churn', k_features=15, scoring='f1', cv=5, do_tuning=False):
    # Load raw data
    df = pd.read_csv(data_path)
    X = df.drop(columns=target_col)
    y = df[target_col]

    # Preprocessing
    preprocessor = TelcoPreprocessor()
    X_processed = preprocessor.fit_transform(X)
    y_processed = preprocessor.transform_target(y)

    pd.DataFrame(X_processed, columns=preprocessor.get_feature_names()).to_csv("data/processed/X.csv", index=False)
    pd.DataFrame({target_col: y_processed}).to_csv("data/processed/y.csv", index=False)

    # Feature Selection
    selector = FeatureSelector(k=k_features)
    X_selected = selector.fit_transform(X_processed, y_processed)

    if do_tuning:
        print("\n Running hyperparameter tuning for XGBoost...")
        best_params = tune_xgboost(X_selected, y_processed, cv=cv, scoring=scoring)
        best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        best_model.fit(X_selected, y_processed)
        print("\n Best model trained with tuned hyperparameters.")

        # Wrap model selector to match return structure
        class DummySelector:
            def __init__(self, model):
                self.results = {'XGBoost (Tuned)': {'mean_score': None, 'std_score': None}}
                self.model = model

            def rank_models(self):
                return pd.DataFrame.from_dict(self.results, orient='index')

            def get_best_model(self):
                return 'XGBoost (Tuned)', self.model

        model_selector = DummySelector(best_model)
    else:
        #  Model Selection
        model_selector = ModelSelector(cv=cv, scoring=scoring)
        model_selector.fit(X_selected, y_processed)
        best_name, best_model = model_selector.get_best_model()
        print("\nModel Performance:")
        print(model_selector.rank_models())
        print(f"\n Best Model: {best_name}")

    return {
        'preprocessor': preprocessor,
        'feature_selector': selector,
        'model_selector': model_selector,
        'best_model': model_selector.get_best_model()[1]
    }


# Example run
if __name__ == '__main__':
    results = run_pipeline(
        data_path='../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        target_col='Churn',
        k_features=15,
        scoring='f1',
        cv=5,
        do_tuning=True
    )