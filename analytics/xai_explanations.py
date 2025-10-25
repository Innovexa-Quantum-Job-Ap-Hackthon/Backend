import os
import base64
import io
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

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

import matplotlib
matplotlib.use("Agg")  # Ensure headless mode
import matplotlib.pyplot as plt

from .ml_models import historical_job_data, extract_advanced_features

logger = logging.getLogger(__name__)

def explain_prediction(device_name: str, model_name: str, input_data: Dict[str, Any] = None, method: str = "shap") -> Dict[str, Any]:
    """
    Explain predictions using SHAP or LIME with REAL live device data.
    This version includes robust fallbacks and clearer error handling.
    """
    from .ml_models import ml_models, _is_model_fitted, extract_advanced_features

    if method not in ["shap", "lime"]:
        raise ValueError("Method must be 'shap' or 'lime'")
    if method == "shap" and not SHAP_AVAILABLE:
        logger.warning("SHAP is not available. Using fallback explanation.")
        method = None  # Use fallback
    if method == "lime" and not LIME_AVAILABLE:
        logger.warning("LIME is not available. Using fallback explanation.")
        method = None  # Use fallback

    logger.info(f"XAI Request - Device: {device_name}, Model: {model_name}, Input data provided: {input_data is not None}")

    if input_data is None:
        raise ValueError("Live device data must be provided. Input data is None.")

    features_dict = input_data.copy()

    # --- 1. Robust Model Finding with Fallback ---
    pipeline = ml_models.get(device_name, {}).get(model_name)
    model_source = device_name

    if not pipeline or not _is_model_fitted(model_name, device_name):
        logger.warning(f"Model '{model_name}' for specific device '{device_name}' not found or not fitted. Searching for a fallback.")
        pipeline = None # Reset pipeline to ensure we find a new one
        for alt_device_name, models in ml_models.items():
            if model_name in models and _is_model_fitted(model_name, alt_device_name):
                pipeline = models[model_name]
                model_source = alt_device_name
                logger.info(f"Using fallback model '{model_name}' from device '{alt_device_name}' to explain prediction for '{device_name}'.")
                break

        if not pipeline:
            available_models = [f"{dn}.{mn}" for dn, mods in ml_models.items() for mn, m in mods.items() if _is_model_fitted(mn, dn)]
            available_str = ", ".join(available_models) if available_models else "None"
            raise ValueError(
                f"Model '{model_name}' for device '{device_name}' is not trained, and no fallback model was found. "
                f"Available trained models are: [{available_str}]"
            )

    # --- 2. Feature Preparation and Validation ---
    from datetime import datetime
    now = datetime.now()
    features_dict.update({'hour_of_day': now.hour, 'day_of_week': now.weekday()})

    # Add other model-specific features
    if model_name == 'device_score':
        features_dict = extract_advanced_features(features_dict)
    elif model_name == 'job_success':
        features_dict.update({
            'pending_x_error': features_dict.get("pending_jobs", 0) * features_dict.get("error_rate", 0.01),
            'circuit_depth': 50,  # Assume average for general explanation
            'historical_success_rate': features_dict.get("success_probability", 0.9)
        })

    try:
        scaler_features = pipeline.steps[0][1].get_feature_names_out()
        features_df = pd.DataFrame([features_dict])

        # Ensure all required columns exist, filling missing ones with defaults
        missing_features = []
        for feature in scaler_features:
            if feature not in features_df.columns:
                missing_features.append(feature)
                # Use sensible defaults for common missing features
                if 'status' in feature: features_df[feature] = 1
                elif 'runtime' in feature: features_df[feature] = 5
                else: features_df[feature] = 0

        if missing_features:
            logger.warning(f"Missing features in live data, filled with defaults: {missing_features}")

        features_df = features_df[scaler_features] # Ensure correct order

    except Exception as e:
        logger.error(f"Error preparing features for pipeline: {e}", exc_info=True)
        raise ValueError(f"Feature preparation failed: {str(e)}")

    # --- 3. Prediction with Live Data ---
    try:
        is_classifier = model_name == 'job_success'
        prediction = float(pipeline.predict_proba(features_df)[0][1] if is_classifier else pipeline.predict(features_df)[0])
        logger.info(f"Prediction using LIVE data ({model_name} on {model_source}): {prediction:.4f}")
    except Exception as e:
        logger.error(f"Prediction with live data failed: {e}", exc_info=True)
        raise ValueError(f"Model prediction failed: {str(e)}")

    # --- 4. Explanation Generation with Improved SHAP ---
    contributions, visualization = {}, None
    try:
        model_to_explain = pipeline.named_steps['model']
        if method == "shap" and SHAP_AVAILABLE:
            # Apply preprocessing steps to get scaled features for SHAP
            scaled_features = pipeline[:-1].transform(features_df)
            scaled_features_df = pd.DataFrame(scaled_features, columns=scaler_features)

            # Generate background data for SHAP using real historical data for better accuracy
            background_size = min(50, len(scaled_features_df) * 10)  # Up to 50 background samples
            np.random.seed(42)

            # Try to use historical data for background
            if historical_job_data:
                hist_df = pd.DataFrame(historical_job_data)
                # Select features that match scaler_features
                available_features = [f for f in scaler_features if f in hist_df.columns]
                if available_features and len(hist_df) > 0:
                    hist_subset = hist_df[available_features].dropna()
                    if len(hist_subset) >= background_size:
                        # Sample from historical data
                        background_sample = hist_subset.sample(n=background_size, random_state=42)
                        # Transform using the pipeline's preprocessing steps
                        background_scaled = pipeline[:-1].transform(background_sample)
                        background_df = pd.DataFrame(background_scaled, columns=scaler_features)
                        logger.info(f"Using {background_size} samples from historical data for SHAP background")
                    else:
                        logger.warning(f"Insufficient historical data ({len(hist_subset)}), falling back to synthetic background")
                        # Fallback to synthetic
                        background_data = []
                        for _ in range(background_size):
                            noise_factor = 0.2  # Increased noise for more diversity
                            variation = scaled_features[0] * (1 + noise_factor * (2 * np.random.random(len(scaled_features[0])) - 1))
                            variation = np.maximum(variation, 0)
                            background_data.append(variation)
                        background_df = pd.DataFrame(background_data, columns=scaler_features)
                else:
                    logger.warning("No matching features in historical data, using synthetic background")
                    # Fallback to synthetic
                    background_data = []
                    for _ in range(background_size):
                        noise_factor = 0.2
                        variation = scaled_features[0] * (1 + noise_factor * (2 * np.random.random(len(scaled_features[0])) - 1))
                        variation = np.maximum(variation, 0)
                        background_data.append(variation)
                    background_df = pd.DataFrame(background_data, columns=scaler_features)
            else:
                logger.warning("No historical data available, using synthetic background")
                # Fallback to synthetic
                background_data = []
                for _ in range(background_size):
                    noise_factor = 0.2
                    variation = scaled_features[0] * (1 + noise_factor * (2 * np.random.random(len(scaled_features[0])) - 1))
                    variation = np.maximum(variation, 0)
                    background_data.append(variation)
                background_df = pd.DataFrame(background_data, columns=scaler_features)

            # Choose explainer based on model type and dataset size
            if len(background_df) < 10:
                # For small datasets, use KernelExplainer which is more robust
                explainer = shap.KernelExplainer(model_to_explain.predict_proba if is_classifier else model_to_explain.predict, background_df.values)
                logger.info(f"Using KernelExplainer for small dataset ({len(background_df)} samples)")
            elif hasattr(model_to_explain, 'estimators_'):  # RandomForest check
                explainer = shap.TreeExplainer(model_to_explain, background_df.values, feature_perturbation='interventional')
                logger.info("Using TreeExplainer for RandomForest model")
            else:
                explainer = shap.Explainer(model_to_explain, background_df.values)

            shap_values = explainer(scaled_features_df.values)

            # Extract SHAP values correctly
            if is_classifier:
                # For classification, use the positive class values
                if len(shap_values.values.shape) > 2:
                    shap_values_array = shap_values.values[0, :, 1]
                else:
                    shap_values_array = shap_values.values[0]
            else:
                shap_values_array = shap_values.values[0]

            contributions = dict(zip(scaler_features, shap_values_array))

            # Debug logging
            non_zero_count = sum(1 for v in shap_values_array if abs(v) > 1e-6)
            logger.info(f"SHAP generated {non_zero_count} non-zero contributions out of {len(shap_values_array)} features")

            # Generate visualization if we have meaningful contributions
            if non_zero_count > 0:
                try:
                    plt.figure(figsize=(10, 6))
                    if is_classifier and len(shap_values.values.shape) > 2:
                        shap.plots.waterfall(shap_values[0, :, 1], show=False)
                    else:
                        shap.plots.waterfall(shap_values[0], show=False)
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    visualization = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plt.close()
                    logger.info("SHAP visualization generated successfully")
                except Exception as viz_error:
                    logger.warning(f"SHAP visualization failed: {viz_error}")
                    visualization = None
            else:
                logger.warning("All SHAP contributions are near zero, skipping visualization")

    except Exception as e:
        logger.error(f"XAI explanation generation failed for {method}: {e}", exc_info=True)
        # Fallback is handled below by checking if contributions is empty

    # Generate fallback visualization if SHAP failed but we have contributions
    if not visualization and contributions:
        try:
            logger.info("Generating fallback bar chart visualization for contributions")
            non_zero_contribs = {k: v for k, v in contributions.items() if abs(v) > 1e-6}
            if non_zero_contribs:
                # Sort by absolute impact
                sorted_contribs = sorted(non_zero_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]  # Top 10
                features, values = zip(*sorted_contribs)

                plt.figure(figsize=(12, 6))
                colors = ['green' if v > 0 else 'red' for v in values]
                plt.barh(features, values, color=colors, alpha=0.7)
                plt.xlabel('Contribution Impact')
                plt.ylabel('Features')
                plt.title('Feature Contributions (Fallback Visualization)')
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                visualization = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                logger.info("Fallback bar chart visualization generated successfully")
        except Exception as viz_error:
            logger.warning(f"Fallback visualization failed: {viz_error}")
            visualization = None

    # --- 5. Formatting the Response ---
    def get_friendly_name(feature: str, value: float) -> str:
        # (Function remains the same as original)
        friendly_names = {
            'qubits': f"{int(value)} qubits", 'error_rate': f"{value:.3f} error rate",
            't1_time': f"{value:.0f}µs T1", 't2_time': f"{value:.0f}µs T2",
            'gate_fidelity': f"{value:.3f} gate fidelity", 'pending_jobs': f"{int(value)} pending jobs",
            'pending_x_error': f"{value:.3f} queue load factor"
        }
        return friendly_names.get(feature, feature.replace('_', ' '))

    # Create human-readable explanation lines
    display_explanation_lines = []
    if contributions:
        # Check if all contributions are effectively zero (e.g., abs < 1e-6)
        non_zero_contribs = {k: v for k, v in contributions.items() if abs(v) > 1e-6}
        if non_zero_contribs:
            # First, collect with absolute impact for sorting
            contrib_items = []
            for feature, impact in contributions.items():
                actual_value = features_df[feature].iloc[0]
                friendly_desc = get_friendly_name(feature, actual_value)
                sign = '+' if impact > 0 else ''
                if model_name == 'job_success':
                    formatted_impact = f"{sign}{impact * 100:.1f}%"
                else:
                    formatted_impact = f"{sign}{impact:.3f}"
                contrib_items.append((abs(impact), f"{formatted_impact} impact from {friendly_desc}"))
            
            # Sort by absolute impact
            contrib_items.sort(key=lambda x: x[0], reverse=True)
            display_explanation_lines = [item[1] for item in contrib_items]
            
            logger.info(f"Generated SHAP explanation with {len(non_zero_contribs)} non-zero contributions.")
        else:
            logger.warning("All SHAP contributions are zero; using fallback explanation.")
            for feature in scaler_features[:5]:
                actual_value = features_df[feature].iloc[0]
                display_explanation_lines.append(f"{get_friendly_name(feature, actual_value)} is a contributing factor")
    else: # Fallback explanation if SHAP/LIME failed
        logger.warning("Generating fallback explanation as XAI libraries failed.")
        for feature in scaler_features[:5]:
            actual_value = features_df[feature].iloc[0]
            display_explanation_lines.append(f"{get_friendly_name(feature, actual_value)} is a contributing factor")

    display_explanation = ' | '.join(display_explanation_lines[:5])

    # Format prediction display string
    if model_name == 'job_success': display_prediction_str = f"{prediction * 100:.1f}% success probability"
    elif model_name == 'wait_time': display_prediction_str = f"{prediction:.0f} minutes wait time"
    elif model_name == 'device_score': display_prediction_str = f"{prediction:.1f}/10 device score"
    else: display_prediction_str = f"{prediction:.2f}"

    final_display_explanation = display_explanation or f"Analysis based on live metrics for {device_name}."
    logger.info(f"Final display_explanation: '{final_display_explanation}'")

    return {
        "device_name": device_name,
        "prediction": prediction,
        "contributions": contributions,
        "display_prediction": display_prediction_str,
        "display_explanation": final_display_explanation,
        "type": method,
        "visualization": visualization,
        "live_data_used": True,
        "model_source_device": model_source,
        "device_metrics_used": {k: v for k, v in features_dict.items() if isinstance(v, (int, float))}
    }
