# CLI Entry Point for AutoML Pipeline using config.yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from automl.pipeline import run_pipeline
from automl.config import load_config
from automl.exporter import save_model, save_metrics, save_report_dataframe
from pathlib import Path
import joblib 

def main():
    parser = argparse.ArgumentParser(description="Run the AutoML pipeline")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n Running AutoML Pipeline with config.yaml...")
    results = run_pipeline(
        data_path=config['data_path'],
        target_col=config['target_col'],
        k_features=config['k_features'],
        scoring=config['scoring'],
        cv=config['cv'],
        do_tuning=config.get('do_tuning', False)
    )

    # Save outputs
    save_model(results['best_model'], config['model_output'])
    save_metrics(results['model_selector'].results, config['metrics_output'])
    ranked_df = results['model_selector'].rank_models()
    save_report_dataframe(ranked_df, config['report_output'])


if __name__ == '__main__':
    main()