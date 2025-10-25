import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import shap
import lime.lime_tabular

from .ml_models import ml_models, _is_model_fitted, extract_advanced_features, historical_job_data

logger = logging.getLogger(__name__)

def explain_prediction(device_name: str, model_name: str, input_data: Dict[str, Any], method: str = "shap") -> Dict[str, Any]:
    """
    Explain the prediction for a given device and model using SHAP or LIME.

    Args:
        device_name: Name of the device.
        model_name: Name of the model ('wait_time', 'device_score', 'job_success').
        input_data: Dictionary of input features.
        method: 'shap' or 'lime'.

    Returns:
        Dict with 'prediction', 'contributions', 'type'.

    Raises:
        ValueError: If model not fitted or invalid method.
    """
    pipeline = ml_models.get(device_name, {}).get(model_name)
    if not pipeline or not _is_model_fitted(model_name, device_name):
        raise ValueError(f"Model {model_name} for device {device_name} is not trained or fitted.")

    # Get feature names from scaler
    scaler_features = pipeline.steps[0][1].get_feature_names_out()

    # Prepare input data with necessary features
    now = datetime.now()
    input_data_full = {**input_data, 'hour_of_day': now.hour, 'day_of_week': now.weekday(), 'is_weekend': now.weekday() >= 5}

    if model_name == 'wait_time':
        input_data_full.update({
            'status_online': input_data.get('status_online', 1),
            'avg_runtime_per_job': input_data.get('avg_runtime_per_job', 5)
        })
    elif model_name == 'device_score':
        input_data_full = extract_advanced_features(input_data_full)
        input_data_full.update({
            'gate_fidelity': input_data.get('gate_fidelity', 0.99),
            'readout_fidelity': input_data.get('readout_fidelity', 0.98),
            'readout_error': input_data.get('readout_error', 0.01),
            't1_time': input_data.get('t1_time', 150),
            't2_time': input_data.get('t2_time', 120)
        })
    elif model_name == 'job_success':
        input_data_full.update({
            'pending_x_error': input_data.get("pending_jobs", 10) * input_data.get("error_rate", 0.01),
            'circuit_depth': 50,
            'historical_success_rate': 0.9
        })
        input_data_full = extract_advanced_features(input_data_full)

    # Create DataFrame
    X = pd.DataFrame([input_data_full])[scaler_features]

    # Get prediction
    if model_name == 'job_success':
        pred_proba = pipeline.predict_proba(X)
        prediction = float(pred_proba[0][1])
    else:
        prediction = float(pipeline.predict(X)[0])

    contributions = {}

    if method == "shap":
        # Use TreeExplainer for RandomForest
        if hasattr(pipeline.steps[1][1], 'estimators_'):  # RandomForest
            explainer = shap.TreeExplainer(pipeline.steps[1][1])
            X_scaled = pipeline.steps[0][1].transform(X)
            shap_values = explainer.shap_values(X_scaled)
            if model_name == 'job_success':
                contributions = dict(zip(scaler_features, shap_values[1][0]))
            else:
                contributions = dict(zip(scaler_features, shap_values[0]))
        else:
            # General explainer
            explainer = shap.Explainer(pipeline)
            shap_values = explainer(X)
            contributions = dict(zip(scaler_features, shap_values.values[0]))

    elif method == "lime":
        # Use historical data for LIME
        if not historical_job_data:
            raise ValueError("No historical data available for LIME explanation.")

        df = pd.DataFrame(historical_job_data)
        device_df = df[df['device_name'] == device_name]
        if device_df.empty:
            raise ValueError(f"No historical data for device {device_name} for LIME explanation.")

        train_X = device_df[scaler_features].values
        if model_name == 'job_success':
            train_y = device_df['job_success'].values
            explainer = lime.lime_tabular.LimeTabularExplainer(train_X, feature_names=scaler_features, class_names=['fail', 'success'], discretize_continuous=True)
            exp = explainer.explain_instance(X.values[0], pipeline.predict_proba, num_features=len(scaler_features))
        else:
            explainer = lime.lime_tabular.LimeTabularExplainer(train_X, feature_names=scaler_features, discretize_continuous=True)
            exp = explainer.explain_instance(X.values[0], pipeline.predict, num_features=len(scaler_features))

        contributions = dict(exp.as_list())
    else:
        raise ValueError(f"Invalid method: {method}. Use 'shap' or 'lime'.")

    return {
        "prediction": prediction,
        "contributions": contributions,
        "type": method
    }
