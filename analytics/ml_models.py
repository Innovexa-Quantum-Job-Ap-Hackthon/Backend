import os
import re
import json
import logging
import base64
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import SHAP and LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. SHAP explanations will not be available.")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not installed. LIME explanations will not be available.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_SAVE_PATH = "analytics/models/"
MIN_SAMPLES_FOR_RETRAIN = 20
RETRAIN_TRIGGER_COUNT = 10

# --- Global State ---
ml_models: Dict[str, Dict[str, Pipeline]] = {}
model_performance_scores: Dict[str, Dict[str, float]] = {}
historical_job_data: List[Dict[str, Any]] = []
new_data_counter = 0


def ensure_model_dir():
    """Create the model directory if it doesn't exist."""
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)


def get_simulated_training_data(n_samples: int = 200) -> pd.DataFrame:
    """
    Generates diverse FAKE data for bootstrapping and testing across multiple fake devices.
    """
    np.random.seed(42)
    timestamps = [datetime.now() - timedelta(hours=np.random.randint(0, 72), minutes=np.random.randint(0, 59)) for _ in range(n_samples)]
    devices = [f"fake_device_{i}" for i in np.random.randint(1, 4, size=n_samples)]

    df = pd.DataFrame({
        'timestamp': timestamps,
        'device_name': devices,
        'qubits': np.random.randint(5, 127, size=n_samples),
        'pending_jobs': np.random.randint(1, 8000, size=n_samples),
        'status_online': 1,
        'circuit_depth': np.random.randint(10, 500, size=n_samples),
        'avg_runtime_per_job': np.random.uniform(1, 10, size=n_samples)
    })

    # Generate varied device properties based on device name hash
    for device in df['device_name'].unique():
        device_hash = hash(device)
        mask = df['device_name'] == device
        df.loc[mask, 'error_rate'] = np.random.uniform(0.001, 0.05) * (0.9 + 0.2 * (device_hash % 10) / 10)
        df.loc[mask, 'readout_error'] = np.random.uniform(0.01, 0.08) * (0.8 + 0.4 * (device_hash % 10) / 10)
        df.loc[mask, 't1_time'] = np.random.uniform(80, 250) * (0.85 + 0.3 * (device_hash % 10) / 10)
        df.loc[mask, 't2_time'] = np.random.uniform(50, 200) * (0.85 + 0.3 * (device_hash % 10) / 10)

    # Feature Engineering
    df['gate_fidelity'] = 1 - df['error_rate']
    df['readout_fidelity'] = 1 - df['readout_error']
    df['t1_t2_ratio'] = df['t1_time'] / df['t2_time']
    df['t1_t2_ratio'].fillna(1.0, inplace=True)
    df['error_rate_std_dev'] = df.groupby('device_name')['error_rate'].transform(lambda s: s.rolling(5, min_periods=1).std()).fillna(0)
    df['t1_stability'] = df.groupby('device_name')['t1_time'].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: np.std(np.diff(x)) if len(x) > 1 else 0)).fillna(0)

    # Time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.weekday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['pending_x_error'] = df['pending_jobs'] * df['error_rate']

    # Generate Target Variables
    df['wait_time'] = df['pending_jobs'] * np.random.uniform(0.002, 0.005, size=n_samples)
    stability_penalty = (df['error_rate_std_dev'] * 500) + (df['t1_stability'] * 5)
    df['performance_score'] = np.clip((df['gate_fidelity'] * 4 + (df['t1_time'] / 200) * 2.0 + (df['t2_time'] / 200) * 2.0) - stability_penalty, 1, 10)

    # Ensure diverse job_success outcomes
    success_threshold = 0.95 - (df['error_rate'] * 5) - ((df['error_rate_std_dev'] * 100) + (df['t1_stability'] * 10))
    df['job_success'] = (np.random.rand(n_samples) < np.clip(success_threshold, 0.15, 0.95)).astype(int)
    if df['job_success'].nunique() < 2:
        num_to_flip = max(1, int(0.2 * n_samples))
        flip_indices = df.sample(n=num_to_flip).index
        df.loc[flip_indices, 'job_success'] = 1 - df.loc[flip_indices, 'job_success']
        
    df['historical_success_rate'] = df.groupby('device_name')['job_success'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0.85)
    return df


def save_ml_models():
    """Saves ALL trained models and scores with device-specific names."""
    ensure_model_dir()
    for device_name, models in ml_models.items():
        for model_name, model in models.items():
            if _is_model_fitted(model_name, device_name):
                try:
                    joblib.dump(model, os.path.join(MODEL_SAVE_PATH, f"{device_name}_{model_name}.joblib"))
                    score = model_performance_scores.get(device_name, {}).get(model_name)
                    with open(os.path.join(MODEL_SAVE_PATH, f"{device_name}_{model_name}_score.json"), 'w') as f:
                        json.dump({'score': score}, f)
                except Exception as e:
                    logger.error(f"Failed to save {model_name} for {device_name}: {e}")
    logger.info("Device-specific models and scores saved.")


def load_ml_models() -> bool:
    """Loads ALL device-specific models from disk into the nested dictionary."""
    global ml_models, model_performance_scores
    loaded_any = False
    ensure_model_dir()
    
    # CORRECTED Regex to correctly parse 'device_name' and 'model_name'
    # It specifically looks for your model names and is non-greedy.
    model_file_pattern = re.compile(r'(.+?)_(device_score|job_success|wait_time)\.joblib')
    
    for filename in os.listdir(MODEL_SAVE_PATH):
        if filename.endswith('.joblib'):
            match = model_file_pattern.match(filename)
            if not match: 
                # This helps in debugging if some files are being skipped
                logger.debug(f"Skipping file with non-matching pattern: {filename}")
                continue
            
            device_name, model_name = match.groups()
            try:
                if device_name not in ml_models: ml_models[device_name] = {}
                if device_name not in model_performance_scores: model_performance_scores[device_name] = {}
                
                model_path = os.path.join(MODEL_SAVE_PATH, filename)
                ml_models[device_name][model_name] = joblib.load(model_path)
                
                score_filename = filename.replace('.joblib', '_score.json')
                score_path = os.path.join(MODEL_SAVE_PATH, score_filename)
                
                if os.path.exists(score_path):
                    with open(score_path, 'r') as f:
                        score_data = json.load(f)
                        model_performance_scores[device_name][model_name] = score_data.get('score')
                        # More descriptive logging
                        logger.info(f"✅ Loaded model '{model_name}' for device '{device_name}' with score {score_data.get('score'):.4f}.")
                else:
                    logger.info(f"✅ Loaded model '{model_name}' for device '{device_name}' (no score file found).")
                
                loaded_any = True
            except Exception as e:
                logger.warning(f"Could not load model from {filename}: {e}")
    
    if not loaded_any:
        logger.warning("⚠️ No device-specific ML models were loaded from the 'analytics/models/' directory.")

    return loaded_any


def _is_model_fitted(model_name: str, device_name: str) -> bool:
    """Checks if a device-specific model is fitted."""
    model = ml_models.get(device_name, {}).get(model_name)
    if not model:
        return False
    
    # Check if it's a pipeline and the final estimator is fitted
    if hasattr(model, 'steps') and len(model.steps) > 0:
        final_estimator = model.steps[-1][1]
        if hasattr(final_estimator, 'estimators_') or hasattr(final_estimator, 'coef_') or hasattr(final_estimator, 'feature_importances_'):
            return True
    
    return False


def extract_advanced_features(current_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates advanced features for real-time prediction."""
    features = current_data.copy()
    features['t1_t2_ratio'] = features.get('t1_time', 1) / features.get('t2_time', 1)
    # Assume stability for real-time prediction, as we don't have historical points
    features['error_rate_std_dev'] = 0.0
    features['t1_stability'] = 0.0
    return features


def predict_wait_time(device_name: str, device_data: Dict[str, Any], use_ml: bool = True) -> Dict[str, Any]:
    """Predicts wait time using a device-specific model or fallback."""
    pipeline = ml_models.get(device_name, {}).get('wait_time')
    if use_ml and pipeline and _is_model_fitted('wait_time', device_name):
        try:
            now = datetime.now()
            features = {**device_data, 'hour_of_day': now.hour, 'day_of_week': now.weekday(), 'is_weekend': now.weekday() >= 5}
            # The scaler in the pipeline needs all features used during training
            # We need to get the feature names from the scaler itself
            scaler_features = pipeline.steps[0][1].get_feature_names_out()
            features_df = pd.DataFrame([features])[scaler_features]
            return max(1, int(pipeline.predict(features_df)[0]))
        except Exception as e:
            logger.warning(f"ML wait time prediction for '{device_name}' failed: {e}. Falling back.")
    
    fallback = max(1, int(device_data.get("pending_jobs", 5) * 0.18))
    return fallback


def calculate_device_score(device_name: str, device_data: Dict[str, Any], use_ml: bool = True) -> Dict[str, Any]:
    """Calculates device score using a device-specific model or fallback."""
    pipeline = ml_models.get(device_name, {}).get('device_score')
    if use_ml and pipeline and _is_model_fitted('device_score', device_name):
        try:
            features_dict = extract_advanced_features(device_data)
            scaler_features = pipeline.steps[0][1].get_feature_names_out()
            features_df = pd.DataFrame([features_dict])[scaler_features]
            return float(np.clip(pipeline.predict(features_df)[0], 1, 10))
        except Exception as e:
            logger.warning(f"ML score prediction for '{device_name}' failed: {e}. Falling back.")

    fidelity = (device_data.get("gate_fidelity", 0.99) * 0.7) + (device_data.get("readout_fidelity", 0.98) * 0.3)
    coherence = (device_data.get("t1_time", 150) / 200 * 0.5) + (device_data.get("t2_time", 120) / 200 * 0.5)
    fallback = float(np.clip((fidelity * 5) + (coherence * 5), 1, 10))
    return fallback


def predict_job_success_ml(device_name: str, device_data: Dict[str, Any], use_ml: bool = True) -> Dict[str, Any]:
    """Predicts job success using a device-specific model or fallback."""
    pipeline = ml_models.get(device_name, {}).get('job_success')
    if use_ml and pipeline and _is_model_fitted('job_success', device_name):
        try:
            now = datetime.now()
            features_dict = extract_advanced_features(device_data)
            # Add missing real-time features
            features_dict.update({
                'hour_of_day': now.hour, 
                'day_of_week': now.weekday(), 
                'pending_x_error': device_data.get("pending_jobs", 10) * device_data.get("error_rate", 0.01),
                'circuit_depth': 50, # Assume an average circuit depth for general prediction
                'historical_success_rate': 0.9 # Assume a high prior success rate
            })
            scaler_features = pipeline.steps[0][1].get_feature_names_out()
            features_df = pd.DataFrame([features_dict])[scaler_features]
            pred_proba = pipeline.predict_proba(features_df)
            success_prob = pred_proba[0][1] if pred_proba.shape[1] > 1 else (1.0 if pipeline.classes_[0] == 1 else 0.0)
            return float(np.clip(success_prob, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"ML success prediction for '{device_name}' failed: {e}. Falling back.")
    
    fallback = float(np.clip(0.95 - (device_data.get("error_rate", 0.01) * 8), 0.0, 1.0))
    return fallback


def predict_real_run_time(device_name: str, circuit_details: Dict[str, Any], use_ml: bool = True) -> float:
    """Predicts execution time in seconds for a real job."""
    pipeline = ml_models.get(device_name, {}).get('real_run_time')
    if use_ml and pipeline and _is_model_fitted('real_run_time', device_name):
        try:
            scaler_features = pipeline.steps[0][1].get_feature_names_out()
            features_df = pd.DataFrame([circuit_details])[scaler_features]
            return max(0.1, float(pipeline.predict(features_df)[0]))
        except Exception as e:
            logger.warning(f"ML real run time prediction for '{device_name}' failed: {e}.")
    return float(circuit_details.get('depth', 10) * 0.1)


def update_ml_models_with_history(job_details: Dict[str, Any]):
    """Adds completed job data to the historical log for future retraining."""
    global new_data_counter, historical_job_data
    historical_job_data.append(job_details)
    new_data_counter += 1
    if new_data_counter >= RETRAIN_TRIGGER_COUNT:
        retrain_ml_models()
        new_data_counter = 0


def retrain_ml_models(force_retrain: bool = False):
    """Retrains all device-specific models with accumulated historical data."""
    if not force_retrain and len(historical_job_data) < MIN_SAMPLES_FOR_RETRAIN: return
    logger.info(f"🚀 Starting retraining with {len(historical_job_data)} data points.")
    df = pd.DataFrame(historical_job_data)
    
    # Feature Engineering for the entire dataset
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.weekday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['pending_x_error'] = df['pending_jobs'] * df['error_rate']
    df['t1_t2_ratio'] = df['t1_time'] / df['t2_time']
    df['error_rate_std_dev'] = df.groupby('device_name')['error_rate'].transform(lambda s: s.rolling(5, min_periods=1).std()).fillna(0)
    df['t1_stability'] = df.groupby('device_name')['t1_time'].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: np.std(np.diff(x)) if len(x)>1 else 0)).fillna(0)
    
    model_configs = {
        'wait_time': {'features': ['qubits', 'pending_jobs', 'status_online', 'error_rate', 'avg_runtime_per_job', 'hour_of_day', 'day_of_week', 'is_weekend'], 'target': 'wait_time', 'model': RandomForestRegressor, 'metric': mean_absolute_error, 'goal': 'minimize'},
        'device_score': {'features': ['qubits', 'error_rate', 'readout_error', 't1_time', 't2_time', 'gate_fidelity', 'readout_fidelity', 't1_t2_ratio', 'error_rate_std_dev', 't1_stability'], 'target': 'performance_score', 'model': RandomForestRegressor, 'metric': r2_score, 'goal': 'maximize'},
        'job_success': {'features': ['qubits', 'circuit_depth', 'historical_success_rate', 'error_rate', 'readout_error', 'pending_jobs', 'hour_of_day', 'day_of_week', 'pending_x_error', 't1_t2_ratio', 'error_rate_std_dev', 't1_stability'], 'target': 'job_success', 'model': RandomForestClassifier, 'metric': f1_score, 'goal': 'maximize'}
    }

    models_updated = False
    for device_name, device_df in df.groupby('device_name'):
        for model_name, config in model_configs.items():
            if not all(col in device_df.columns for col in config['features'] + [config['target']]): continue
            X = device_df[config['features']]; y = device_df[config['target']].dropna()
            # For classification, we need at least one sample of each class
            if model_name == 'job_success' and y.nunique() < 2: continue
            if y.empty: continue
            
            X_train, X_test, y_train, y_test = train_test_split(X.loc[y.index], y, test_size=0.2, random_state=42, stratify=y if model_name=='job_success' else None)
            
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', config['model'](random_state=42))])
            pipeline.fit(X_train, y_train)
            pipeline.feature_names = config['features']
            new_score = config['metric'](y_test, pipeline.predict(X_test))
            
            current_score = model_performance_scores.get(device_name, {}).get(model_name)
            if current_score is None or (config['goal'] == 'maximize' and new_score > current_score) or (config['goal'] == 'minimize' and new_score < current_score):
                if device_name not in ml_models: ml_models[device_name] = {}
                if device_name not in model_performance_scores: model_performance_scores[device_name] = {}
                ml_models[device_name][model_name] = pipeline
                model_performance_scores[device_name][model_name] = new_score
                models_updated = True
                logger.info(f"Updated {model_name} for {device_name} (New Score: {new_score:.4f}).")

    if models_updated:
        save_ml_models()


def analyze_historical_trends(days: int, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Analyzes historical data to provide trends and insights."""
    if not historical_job_data: return {"data": [], "overall_stats": {}, "device_performance": {}, "has_user_data": False}
    df = pd.DataFrame(historical_job_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df_filtered = df
    has_user_data = True
    if user_id:
        user_df = df[df['user_id'] == user_id]
        if user_df.empty:
            has_user_data = False
        else:
            df_filtered = user_df
            
    recent_df = df_filtered[df_filtered['timestamp'] >= datetime.now() - timedelta(days=days)]
    if recent_df.empty: return {"data": [], "overall_stats": {}, "device_performance": {}, "has_user_data": has_user_data}

    for col in ['performance_score', 'wait_time', 'job_success']:
        if col in recent_df.columns:
            recent_df[col].fillna(recent_df[col].median(), inplace=True)

    time_series_data = recent_df.set_index('timestamp').resample('D').agg(totalJobs=('device_name', 'count'), completed=('job_success', lambda x: (x == 1).sum())).reset_index()
    time_series_data['date'] = time_series_data['timestamp'].dt.strftime('%Y-%m-%d')
    
    device_performance = recent_df.groupby('device_name').agg(total_jobs=('timestamp', 'count'), success_rate=('job_success', 'mean'), avg_score=('performance_score', 'mean')).to_dict('index')
    overall_stats = {"total_jobs": int(recent_df.shape[0]), "overall_success_rate": float(recent_df['job_success'].mean())}
    
    return {"data": time_series_data.to_dict('records'), "overall_stats": overall_stats, "device_performance": device_performance, "has_user_data": has_user_data}


def get_ai_recommendations() -> List[str]:
    """Generates AI-driven text recommendations based on model performance."""
    if not model_performance_scores: return ["Submit more jobs to enable personalized recommendations."]
    all_f1_scores = [scores.get('job_success', 0) for scores in model_performance_scores.values()]
    avg_f1_score = np.mean(all_f1_scores) if all_f1_scores else 0
    recs = [f"Job success models are performing with an average F1 Score of {avg_f1_score:.2f}."]
    if len(historical_job_data) > 50: recs.append("Models are trained on your recent job history for better accuracy.")
    return recs

def historical_job_data_for_ml(device_name: str) -> Dict[str, float]:
    """
    Analyzes historical data for a specific device to provide ML features.
    """
    if not historical_job_data:
        # Return reasonable defaults if no historical data is available
        return {"success_rate_avg": 0.85, "wait_time_avg": 10.0}

    # Use pandas for efficient filtering and calculation
    df = pd.DataFrame(historical_job_data)
    device_df = df[df['device_name'] == device_name]

    if device_df.empty:
        # Return defaults if there's no data for this specific device
        return {"success_rate_avg": 0.85, "wait_time_avg": 10.0}

    # Calculate and return the aggregated stats
    success_rate = device_df['job_success'].mean()
    avg_wait_time = device_df['wait_time'].mean()

    return {
        "success_rate_avg": float(success_rate),
        "wait_time_avg": float(avg_wait_time)
    }

def ai_noise_characterization(metrics: Dict[str, Any]) -> Dict[str, str]:
    t1, t2, error = metrics.get('t1_time', 100), metrics.get('t2_time', 100), metrics.get('error_rate', 0.01)
    if error > 0.02 and t1 < 100: return {"noise_profile": "High Decoherence & Gate Error", "recommendation": "Best for shallow circuits."}
    if t1 > 180 and t2 > 150 and error < 0.005: return {"noise_profile": "Low Noise, High Coherence", "recommendation": "Excellent for complex circuits."}
    return {"noise_profile": "Balanced", "recommendation": "Standard error mitigation should be effective."}
    



def initialize_ml_models_with_options(train_with_simulated: bool = False) -> bool:
    """
    Loads all device-specific models or trains with simulated data as a fallback.
    
    Args:
        train_with_simulated: If True, train models with simulated data if no real models exist.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        logger.info("🚀 Initializing ML models...")

        # Try to load existing models first
        models_loaded = load_ml_models()

        if not models_loaded or train_with_simulated:
            logger.info("No existing models found or training with simulated data requested. Training new models...")

            # Generate simulated training data
            simulated_df = get_simulated_training_data(n_samples=200)

            # Convert to historical job data format
            global historical_job_data
            historical_job_data.extend(simulated_df.to_dict('records'))

            # Train models with the simulated data
            retrain_ml_models(force_retrain=True)

            logger.info("✅ ML models trained successfully with simulated data")
        else:
            logger.info("✅ ML models loaded successfully from disk")

        # Verify that we have at least one model fitted
        has_models = any(
            _is_model_fitted(model_name, device_name)
            for device_name in ml_models
            for model_name in ['wait_time', 'device_score', 'job_success']
        )

        if has_models:
            logger.info("✅ ML model initialization completed successfully")
            return True
        else:
            logger.warning("⚠️ No ML models are fitted after initialization")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to initialize ML models: {e}")
        return False

        
if __name__ == '__main__':
                ensure_model_dir()
                # When running this file directly, load models or train with simulated data
                if not load_ml_models():
                    logger.info("No models found, training new ones with simulated data for testing.")
                    simulated_df = get_simulated_training_data()
                    historical_job_data.extend(simulated_df.to_dict('records'))
                    retrain_ml_models(force_retrain=True)

                device_name = "fake_device_1"
                device_data = {"qubits": 127, "pending_jobs": 4500, "status": "online", "error_rate": 0.008, "readout_error": 0.015, "t1_time": 210, "t2_time": 180, "avg_runtime_per_job": 5, "gate_fidelity": 0.992, "readout_fidelity": 0.985, "historical_success_rate": 0.9}
                
                print(f"\n--- Predictions for {device_name} ---")
                if _is_model_fitted("wait_time", device_name):
                    print(f"Wait Time: {predict_wait_time(device_name, device_data)}")
                    print(f"Device Score: {calculate_device_score(device_name, device_data)}")
                    print(f"Success Prob: {predict_job_success_ml(device_name, device_data)}")
                else:
                    print(f"Models for {device_name} are not fitted. Check data and retraining process.")
