import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from data_utils import inject_noise, flip_labels

class ModelWrapper:
    def __init__(self, model_name, model_obj, task_type='binary'):
        self.model_name = model_name
        self.model = model_obj
        self.task_type = task_type

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        y_train_pred = self.model.predict(X_train_scaled)

        f1_avg = 'binary' if self.task_type == 'binary' else 'weighted'

        return {
            'Model': self.model_name,
            'Train_Acc': accuracy_score(y_train, y_train_pred),
            'Test_Acc': accuracy_score(y_test, y_pred),
            'F1_Score': f1_score(y_test, y_pred, average=f1_avg),
            'Generalization_Gap': accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_pred)
        }

def run_single_simulation(model_wrapper, X, y, train_idx, test_idx):
    """Execute a single fold/repeat in parallel. 🎀"""
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return model_wrapper.fit_and_evaluate(X_train, y_train, X_test, y_test)

def execute_engine(models, X, y, rskf, task_type='binary'):
    """Main execution loop iterating through models and parallelizing folds. 🎀"""
    metric_vault = []
    for name, model_obj in models.items():
        wrapper = ModelWrapper(name, model_obj, task_type=task_type)
        results = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(wrapper, X, y, train_idx, test_idx)
            for train_idx, test_idx in rskf.split(X, y)
        )
        metric_vault.extend(results)
    return pd.DataFrame(metric_vault)

def run_stress_test_suite(models, X, y, rskf, task_type='binary'):
    """Run stress tests at multiple noise levels. 🎀"""
    stress_results = []
    noise_levels = [0.0, 0.05, 0.10, 0.15]

    for level in noise_levels:
        print(f"Executing Stress Test at {level*100}% corruption...")

        if level > 0:
            X_noisy = inject_noise(X, noise_level=level)
            y_noisy = flip_labels(y, flip_rate=level)
        else:
            X_noisy = X.copy()
            y_noisy = y.copy()

        vault_at_level = execute_engine(models, X_noisy, y_noisy, rskf, task_type=task_type)
        vault_at_level['Noise_Level'] = level
        stress_results.append(vault_at_level)

    return pd.concat(stress_results)

