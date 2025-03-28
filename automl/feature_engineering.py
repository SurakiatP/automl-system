# 🛠 Feature Engineering module for Telco Customer Churn

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k
        self.selector = None

    def fit(self, X, y):
        self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self):
        return self.selector.get_support()


class PCAFeatureReducer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X, y=None):
        return self.pca.fit(X).transform(X)
