import os
import sys
import argparse
import pandas as pd
from engine import execute_engine, run_stress_test_suite
from data_utils import clean_data
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from analytics import calculate_stability_metrics, generate_stability_dashboard

def parse_args():
    """Parse command-line arguments for generic dataset support. 🎀"""
    parser = argparse.ArgumentParser(
        description='Robustness-First Model Stability Analyzer - Works with any CSV dataset'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='telecom_churn.csv',
        help='Path to the CSV dataset file (default: telecom_churn.csv)'
    )
    parser.add_argument(
        '--target', '-t',
        type=str,
        default=None,
        help='Name of the target column. If not provided, uses the last column.'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='.',
        help='Directory to save output files (default: current directory)'
    )
    parser.add_argument(
        '--repeats', '-r',
        type=int,
        default=10,
        help='Number of repeats for cross-validation (default: 10)'
    )
    parser.add_argument(
        '--folds', '-f',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    parser.add_argument(
        '--models', '-m',
        type=str,
        default='all',
        choices=['all', 'lr', 'knn', 'rf', 'xgb'],
        help='Models to evaluate: all, lr (Logistic Regression), knn, rf (Random Forest), xgb (default: all)'
    )
    return parser.parse_args()

def get_model_suite(task_type='binary', models_choice='all'):
    """Define the Model Suite based on user selection. 🎀"""
    all_models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, solver='lbfgs'),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }

    if models_choice == 'all':
        return all_models

    model_map = {
        'lr': ["Logistic Regression"],
        'knn': ["KNN"],
        'rf': ["Random Forest"],
        'xgb': ["XGBoost"]
    }

    selected = {}
    for key in model_map.get(models_choice, []):
        selected[key] = all_models[key]
    return selected

def main():
    args = parse_args()

    dataset_path = args.dataset
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset_prefix = os.path.splitext(os.path.basename(dataset_path))[0]
    vault_file = os.path.join(output_dir, f'{dataset_prefix}_metric_vault.csv')
    dashboard_file = os.path.join(output_dir, f'{dataset_prefix}_stability_dashboard.png')
    certificate_file = os.path.join(output_dir, f'{dataset_prefix}_stability_certificate_data.csv')
    report_file = os.path.join(output_dir, f'{dataset_prefix}_stability_analysis_report.txt')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Required dataset file '{dataset_path}' not found.")

    rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=42)
    total_splits = rskf.get_n_splits() * rskf.n_repeats
    print(f"Engine configured for {rskf.get_n_splits()} folds x {rskf.n_repeats} repeats = {total_splits} total splits per model.")

    X, y, meta = clean_data(dataset_path, target_col=args.target)
    task_type = meta['task_type']
    dataset_name = meta['dataset_name']

    models = get_model_suite(task_type=task_type, models_choice=args.models)
    print(f"Models selected: {list(models.keys())}")

    print("\n=== PHASE 1: BASELINE STABILITY ANALYSIS ===")
    raw_vault = execute_engine(models, X, y, rskf, task_type=task_type)

    raw_vault.to_csv(vault_file, index=False)
    print(f"Phase 1 Complete: {len(raw_vault)} simulations stored in Metric Vault.")

    raw_vault = pd.read_csv(vault_file)

    leaderboard = calculate_stability_metrics(raw_vault)
    print("\n--- Reliability Leaderboard (Stability Index) ---")
    print(leaderboard)

    print("\n[*] Generating stability dashboard...")
    generate_stability_dashboard(raw_vault, output_path=dashboard_file)

    print("\n=== PHASE 2: STRESS TESTING ===")
    stress_vault = run_stress_test_suite(models, X, y, rskf, task_type=task_type)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE for dataset: {dataset_name}")
    print(f"Outputs saved to: {os.path.abspath(output_dir)}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

