import warnings
warnings.filterwarnings('ignore', message='Could not infer format', category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATASET_CONFIGS = {
    'telecom_churn.csv': {'target': 'churn', 'type': 'classification'},
    'creditcard.csv': {'target': 'Class', 'type': 'classification'},
    'online_shoppers_intention.csv': {'target': 'Revenue', 'type': 'classification'},
    'energydata_complete.csv': {'target': 'Appliances', 'type': 'regression', 'binarize_threshold': 'median'}
}

def clean_dataset(filepath, config=None):
    """Load and clean any supported dataset. 🎀"""
    import os
    filename = os.path.basename(filepath)

    if config is None:
        config = DATASET_CONFIGS.get(filename, DATASET_CONFIGS['telecom_churn.csv'])

    df = pd.read_csv(filepath)
    df = df.copy()
    df = df.dropna()

    object_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in object_cols:
        try:
            parsed = pd.to_datetime(df[col], errors='raise')
            if parsed.notna().sum() > len(parsed) * 0.5:
                df = df.drop(columns=[col])
                continue
        except Exception:
            pass

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    target_col = config['target']
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    if y.dtype == bool:
        y = y.astype(int)

    if config['type'] == 'regression':
        threshold = config.get('binarize_threshold', 'median')
        if threshold == 'median':
            threshold = y.median()
        y = (y > threshold).astype(int)

    return X, y

def clean_data(filepath, target_col=None):
    """Generic data loader with auto target detection and metadata. 🎀"""
    import os
    filename = os.path.basename(filepath)
    config = DATASET_CONFIGS.get(filename)

    if config is None and target_col is None:
        df = pd.read_csv(filepath)
        target_col = df.columns[-1]
        config = {'target': target_col, 'type': 'classification'}
    elif config is None:
        config = {'target': target_col, 'type': 'classification'}
    elif target_col is not None:
        config = dict(config)
        config['target'] = target_col

    X, y = clean_dataset(filepath, config)

    n_classes = y.nunique()
    task_type = 'binary' if n_classes == 2 else 'multiclass'

    meta = {
        'dataset_name': filename,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': n_classes,
        'task_type': task_type
    }

    return X, y, meta

def inject_noise(X, noise_level=0.05):
    """Adds Gaussian noise to numerical features. 🎀"""
    X_noisy = X.copy()
    numerical_cols = X_noisy.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        sigma = X_noisy[col].std() * noise_level
        if sigma > 0:
            noise = np.random.normal(0, sigma, X_noisy[col].shape)
            X_noisy[col] = X_noisy[col] + noise
    return X_noisy

def flip_labels(y, flip_rate=0.05):
    """Randomly flips binary labels. 🎀"""
    y_flipped = y.copy().astype(int)
    n_flip = int(len(y_flipped) * flip_rate)
    if n_flip > 0:
        flip_indices = np.random.choice(y_flipped.index, size=n_flip, replace=False)
        y_flipped.loc[flip_indices] = (1 - y_flipped.loc[flip_indices]).astype(int)
    return y_flipped

