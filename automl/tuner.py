# Hyperparameter Tuning with Optuna

import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import numpy as np


def tune_xgboost(X, y, cv=5, scoring='f1', n_trials=30, random_state=42):
    """Run Optuna to tune XGBoost hyperparameters."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': random_state,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }

        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("\n Best Trial:")
    print(study.best_trial)

    return study.best_params