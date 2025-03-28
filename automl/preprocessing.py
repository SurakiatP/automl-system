import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class TelcoPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.cat_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        self.label_col = 'Churn'
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.pipeline = None

    def fit(self, X, y=None):
        X = X.copy()
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

        num_imputer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', self.scaler)
        ])

        cat_imputer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', self.encoder)
        ])

        self.pipeline = ColumnTransformer([
            ('num', num_imputer, self.num_cols),
            ('cat', cat_imputer, self.cat_cols)
        ])

        self.pipeline.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        X = X.drop(columns=['customerID'])
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        X_transformed = self.fit(X).transform(X)

        try:
            columns = self.get_feature_names()
            df_out = pd.DataFrame(X_transformed, columns=columns)
            df_out.to_csv("data/processed/preprocessed_data.csv", index=False)
            print(" Saved preprocessed data to data/processed/preprocessed_data.csv")
        except Exception as e:
            print(" Failed to save preprocessed data:", e)

        return X_transformed

    def transform_target(self, y):
        return y.map({'No': 0, 'Yes': 1}).astype(int)
    
    def get_feature_names(self):
        try:
            cat_features = self.pipeline.named_transformers_['cat']['encoder'].get_feature_names_out(self.cat_cols)
            all_features = np.concatenate([self.num_cols, cat_features])
            return all_features
        except:
            return [f'feature_{i}' for i in range(self.pipeline.transformers_[0][2] + self.pipeline.transformers_[1][2])]
