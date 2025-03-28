from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd


class ModelSelector:
    def __init__(self, cv=5, scoring='f1'):
        self.cv = cv
        self.scoring = scoring
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.results = {}

    def evaluate_model(self, model, X, y):
        scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
        return scores

    def fit(self, X, y):
        for name, model in self.models.items():
            scores = self.evaluate_model(model, X, y)
            self.results[name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }
        return self

    def rank_models(self):
        return pd.DataFrame(self.results).T.sort_values(by='mean_score', ascending=False)

    def get_best_model(self):
        best_name = max(self.results, key=lambda name: self.results[name]['mean_score'])
        return best_name, self.models[best_name]