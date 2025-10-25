from analytics.ml_models import load_ml_models, _is_model_fitted
from analytics.explainability import explain_prediction
load_ml_models()
print('ibm_brisbane wait_time fitted:', _is_model_fitted('wait_time', 'ibm_brisbane'))
print('ibm_brisbane device_score fitted:', _is_model_fitted('device_score', 'ibm_brisbane'))
print('ibm_brisbane job_success fitted:', _is_model_fitted('job_success', 'ibm_brisbane'))

# Test explain_prediction
input_data = {
    "qubits": 127,
    "pending_jobs": 4500,
    "status_online": 1,
    "error_rate": 0.008,
    "readout_error": 0.015,
    "t1_time": 210,
    "t2_time": 180,
    "avg_runtime_per_job": 5,
    "gate_fidelity": 0.992,
    "readout_fidelity": 0.985
}

try:
    result = explain_prediction('ibm_brisbane', 'wait_time', input_data, 'shap')
    print('SHAP result:', result)
except Exception as e:
    print('SHAP Error:', e)

try:
    result = explain_prediction('ibm_brisbane', 'wait_time', input_data, 'lime')
    print('LIME result:', result)
except Exception as e:
    print('LIME Error:', e)
