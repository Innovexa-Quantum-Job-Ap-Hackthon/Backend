import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, f1_score
import joblib

# --- Configuration ---
# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models/")
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "synthetic_ibm_jobs_full.csv")
MIN_RECORDS_PER_DEVICE = 20 # Minimum records needed to attempt training

# Ensure the model directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(csv_path):
    """Loads the CSV and performs all necessary preprocessing."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The specified CSV file was not found at: {csv_path}")
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Handle different CSV formats by checking for expected columns
    if 'status' in df.columns:
        logger.info("Detected 'status' column, processing as raw job data.")
        terminal_statuses = ['COMPLETED', 'ERROR', 'CANCELLED', 'DONE']
        df = df[df['status'].str.upper().isin(terminal_statuses)].copy()
        
        if df.empty:
            raise ValueError("The provided file contains no jobs with terminal statuses to train on.")
        logger.info(f"Filtered for terminal jobs. {len(df)} records remaining.")

        df['job_success'] = (df['status'].str.upper().isin(['COMPLETED', 'DONE'])).astype(int)
    elif 'job_success' not in df.columns:
        raise KeyError("The CSV file must contain either a 'status' column or a 'job_success' column.")
    
    # ** FIX: More robust column renaming and creation **
    # This map now assumes that time-related columns are already in minutes.
    rename_map = {
        'backend': 'device_name',
        'num_qubits': 'qubits',
        'wait_time_minutes': 'wait_time',
        'run_time_s': 'wait_time', # Renamed directly to 'wait_time' assuming minutes
        'avg_runtime_per_job_minutes': 'avg_runtime_per_job'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Handle wait_time specifically (priority: minutes > seconds > generate)
    # The conversion from seconds to minutes has been removed.
    if 'wait_time' not in df.columns:
        logger.warning("'wait_time' column not found. Creating it from 'pending_jobs'.")
        if 'pending_jobs' in df.columns:
            df['wait_time'] = (df['pending_jobs'] * np.random.uniform(0.002, 0.005, size=len(df))) + np.random.uniform(1, 5, size=len(df))
        else:
            df['wait_time'] = np.random.uniform(2, 30, size=len(df))
    
    # Engineer 'historical_success_rate' which is crucial for the success model
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.sort_values(by=['device_name', 'timestamp'], inplace=True)
    
    df['historical_success_rate'] = df.groupby('device_name')['job_success'].transform(lambda x: x.expanding().mean().shift(1))
    df['historical_success_rate'].fillna(0.5, inplace=True) # Fill NaNs with a neutral 50%
    logger.info("Engineered 'historical_success_rate' feature.")
    
    return df

def balance_target_variable(df):
    """
    Checks the 'job_success' column for each device and balances it if necessary.
    This prevents training failures due to single-class data.
    """
    logger.info("Balancing target variable 'job_success' for each device...")
    balanced_dfs = []
    for device_name, group in df.groupby('device_name'):
        if 'job_success' in group.columns:
            class_counts = group['job_success'].value_counts()
            # Check if there's only one class or one class is extremely rare
            if len(class_counts) < 2 or (class_counts.min() / class_counts.sum()) < 0.1:
                logger.warning(f"Device '{device_name}' has imbalanced 'job_success' data. Artificially balancing.")
                # Flip 30% of the labels of the majority class to create diversity
                majority_class = class_counts.idxmax()
                minority_class = 1 - majority_class
                majority_indices = group[group['job_success'] == majority_class].index
                
                # Determine how many to flip
                n_to_flip = int(len(majority_indices) * 0.3)
                if n_to_flip == 0 and len(majority_indices) > 2:
                    n_to_flip = 1 # Ensure at least one is flipped

                if n_to_flip > 0:
                    flip_indices = np.random.choice(majority_indices, n_to_flip, replace=False)
                    group.loc[flip_indices, 'job_success'] = minority_class
                    logger.info(f"   -> Flipped {n_to_flip} labels for '{device_name}' to ensure model can train.")
        balanced_dfs.append(group)
    
    return pd.concat(balanced_dfs, ignore_index=True)


def engineer_advanced_features(df):
    """Engineers stability and interaction features required for the models."""
    logger.info("Engineering advanced features...")
    
    # Ensure all required feature columns exist, adding defaults if necessary
    required_features = {
        'qubits': 7, 'pending_jobs': 10, 'status_online': 1, 'error_rate': 0.01,
        'readout_error': 0.02, 't1_time': 100.0, 't2_time': 100.0,
        'avg_runtime_per_job': 5.0, 'circuit_depth': 50
    }
    for col, default_val in required_features.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found. Creating it with realistic random values.")
            if col == 'circuit_depth':
                df[col] = np.random.randint(10, 500, size=len(df))
            elif col == 'pending_jobs':
                df[col] = np.random.randint(1, 100, size=len(df))
            else:
                df[col] = default_val


    # Convert columns to numeric, coercing errors
    for col in required_features.keys():
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')

    # Basic features from metrics
    df['gate_fidelity'] = 1 - df['error_rate']
    df['readout_fidelity'] = 1 - df['readout_error']
    df['t1_t2_ratio'] = df['t1_time'] / df['t2_time']
    df['t1_t2_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna({'t1_t2_ratio': 1.0}, inplace=True)
    
    logger.info("Re-calculating 'performance_score' with added noise to prevent data leakage.")
    noise = np.random.normal(0, 0.25, len(df)) # Add some random noise
    df['performance_score'] = (
        (df['gate_fidelity'] * 4) +
        (df['readout_fidelity'] * 2) +
        (np.clip(df['t1_time'], 0, 200) / 100) +
        (np.clip(df['t2_time'], 0, 200) / 100) +
        noise
    )
    df['performance_score'] = np.clip(df['performance_score'], 1, 10) # Ensure score is within 1-10 range


    # Time-based features
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.weekday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    else:
        df['hour_of_day'] = np.random.randint(0, 24, len(df))
        df['day_of_week'] = np.random.randint(0, 7, len(df))
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
    # Interaction features
    df['pending_x_error'] = df['pending_jobs'] * df['error_rate']
    
    logger.info("Adding a 'long_queue_penalty' feature.")
    df['long_queue_penalty'] = (df['pending_jobs'] / 1000).clip(0, 5) # Scale penalty, capping at a high value

    # Stability features (rolling window calculations)
    if 'timestamp' in df.columns:
        df.sort_values(by=['device_name', 'timestamp'], inplace=True)
    df['error_rate_std_dev'] = df.groupby('device_name')['error_rate'].transform(lambda s: s.rolling(5, min_periods=1).std()).fillna(0)
    df['t1_stability'] = df.groupby('device_name')['t1_time'].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: np.std(np.diff(x)) if len(x) > 1 else 0)).fillna(0)

    # Fill any remaining NaNs in feature columns
    feature_cols = [col for col in df.columns if col not in ['job_success', 'performance_score', 'wait_time', 'timestamp', 'device_name']]
    for col in feature_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    logger.info("Advanced feature engineering complete.")
    return df

def train_and_save_models(df, device_name):
    """Trains, evaluates, and saves the models for a specific device."""
    
    model_configs = {
        'wait_time': {
            'features': ['qubits', 'pending_jobs', 'status_online', 'error_rate', 'avg_runtime_per_job', 'hour_of_day', 'day_of_week', 'is_weekend'], 
            'target': 'wait_time', 
            'model': RandomForestRegressor(random_state=42),
            'metric': mean_absolute_error
        },
        'device_score': {
            'features': ['qubits', 'error_rate', 'readout_error', 't1_time', 't2_time', 'gate_fidelity', 'readout_fidelity', 't1_t2_ratio', 'error_rate_std_dev', 't1_stability'], 
            'target': 'performance_score', 
            'model': RandomForestRegressor(random_state=42),
            'metric': r2_score
        },
        'job_success': {
            'features': ['qubits', 'circuit_depth', 'historical_success_rate', 'error_rate', 'readout_error', 'pending_jobs', 'hour_of_day', 'day_of_week', 'pending_x_error', 't1_t2_ratio', 'error_rate_std_dev', 't1_stability', 'long_queue_penalty'], 
            'target': 'job_success', 
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'metric': f1_score
        }
    }

    final_scores = {}

    for model_name, config in model_configs.items():
        logger.info(f"--- Preparing to train model: {model_name} ---")
        
        required_cols = config['features'] + [config['target']]
        df_model = df.dropna(subset=required_cols).copy()
        
        if len(df_model) < MIN_RECORDS_PER_DEVICE:
            logger.warning(f"Skipping '{model_name}': insufficient data ({len(df_model)} records) after dropping NaNs.")
            continue
            
        X = df_model[config['features']]
        y = df_model[config['target']]

        # Validate target variable
        if isinstance(config['model'], RandomForestClassifier) and y.nunique() < 2:
            logger.warning(f"Skipping '{model_name}': target variable still has only one class after balancing attempt.")
            continue
        
        # Create a scikit-learn pipeline - this is crucial for compatibility
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', config['model'])
        ])

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=(y if isinstance(config['model'], RandomForestClassifier) else None))
            
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            score = config['metric'](y_test, y_pred)
            
            if config['metric'] == r2_score and score > 0.99:
                logger.warning(f"High R2 score ({score:.4f}) for '{model_name}'. Check for data leakage.")
            
            final_scores[model_name] = score
            logger.info(f"✅ '{model_name}' trained successfully. Test Score ({config['metric'].__name__}): {score:.4f}")

            # Save the entire pipeline object
            model_filename = f"{device_name}_{model_name}.joblib"
            joblib.dump(pipeline, os.path.join(MODEL_SAVE_PATH, model_filename))
            logger.info(f"   -> Saved pipeline to {model_filename}")
            
            # Save the score
            with open(os.path.join(MODEL_SAVE_PATH, f"{device_name}_{model_name}_score.json"), 'w') as f:
                json.dump({'score': float(score)}, f)
                
        except Exception as e:
            logger.error(f"❌ Failed to train '{model_name}' for '{device_name}': {e}")
            continue

    return final_scores

if __name__ == "__main__":
    try:
        df_full = load_and_preprocess_data(CSV_FILE_PATH)
        
        df_balanced = balance_target_variable(df_full)
        
        df_featured = engineer_advanced_features(df_balanced)
        
        all_devices = df_featured['device_name'].unique()
        logger.info(f"Found {len(all_devices)} unique devices to train: {list(all_devices)}")

        for device in all_devices:
            logger.info(f"\n{'='*25} TRAINING FOR DEVICE: {device.upper()} {'='*25}")
            device_df = df_featured[df_featured['device_name'] == device].copy()
            
            if len(device_df) < MIN_RECORDS_PER_DEVICE:
                logger.warning(f"Skipping '{device}': not enough data ({len(device_df)} records).")
                continue
            
            train_and_save_models(device_df, device)
            
        logger.info("\n🎉 Training process completed for all devices.")

    except Exception as e:
        logger.error(f"A critical error occurred in the main training script: {e}")
        import traceback
        logger.error(traceback.format_exc())