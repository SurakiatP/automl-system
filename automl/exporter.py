import joblib
import json
import os
from pathlib import Path
import pandas as pd


def save_model(model, path="models/best_model.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f" Model saved to: {path}")


def save_metrics(metrics_dict, path="reports/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f" Metrics saved to: {path}")


def save_report_dataframe(report_df, path="reports/run_summary.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(path, index=False)
    print(f" Report saved to: {path}")