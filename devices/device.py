from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

# Constants
COST_PER_MINUTE = 0.5  # USD per minute
POWER_CONSUMPTION_KW = 0.1  # kW
CARBON_PER_KWH = 0.5  # kg CO2 per kWh
PREDICTION_WEIGHTS = {
    "qubits": 0.3,
    "pending_jobs": 0.2,
    "error_rate": 0.3,
    "gate_fidelity": 0.2
}

# Pydantic Models
class DeviceProperties(BaseModel):
    name: str
    qubits: int
    status: str
    pending_jobs: int
    error_rate: float
    readout_error: float
    t1_time: float
    t2_time: float
    gate_fidelity: float
    readout_fidelity: float

class JobRequest(BaseModel):
    backend_name: str
    shots: int
    circuit_data: Dict[str, Any]

class UserJob(BaseModel):
    id: str
    backend: str
    status: str
    shots: int
    submitted_at: datetime
    completed_at: Optional[datetime]
    results: Optional[Dict[str, Any]]
    actual_wait_time: Optional[int]
    performance_score: Optional[float]

class SubmitJobRequest(BaseModel):
    backend_name: str
    shots: int
    circuit_data: Dict[str, Any]

class JobResult(BaseModel):
    job_id: str
    status: str
    results: Optional[Dict[str, Any]]
    completed_at: Optional[datetime]

class AnalyticsRequest(BaseModel):
    days: int = 30

class JobSuccessPrediction(BaseModel):
    device_name: str
    success_probability: float
    confidence: float

class JobOptimizerRequest(BaseModel):
    qubits_required: int
    gates_required: int
    circuit_depth: int
    priority: str = "balanced"

class JobOptimizerResponse(BaseModel):
    recommended_device: str
    success_probability: float
    estimated_wait_time: int
    estimated_cost: float
    estimated_carbon: float
    explanation: str
    alternatives: List[Dict[str, Any]]

class CircuitCompilationRequest(BaseModel):
    circuit_code: str
    circuit_format: str = "qasm"

class CircuitCompilationResult(BaseModel):
    device_name: str
    can_execute: bool
    additional_swaps: Optional[int]
    circuit_depth: Optional[int]
    estimated_fidelity: Optional[float]
    compilation_time: Optional[float]
    required_qubits: int
    available_qubits: int
    error_message: Optional[str]

# Utility Functions
def extract_device_metrics(backend, status, properties):
    """Extract device metrics from backend objects."""
    return {
        "error_rate": getattr(properties, 'avg_gate_error', 0.05) if properties else 0.05,
        "readout_error": getattr(properties, 'avg_readout_error', 0.05) if properties else 0.05,
        "t1_time": getattr(properties, 'avg_t1', 100.0) if properties else 100.0,
        "t2_time": getattr(properties, 'avg_t2', 100.0) if properties else 100.0,
        "gate_fidelity": getattr(properties, 'avg_gate_fidelity', 0.95) if properties else 0.95,
        "readout_fidelity": getattr(properties, 'avg_readout_fidelity', 0.94) if properties else 0.94,
        "avg_runtime": getattr(properties, 'avg_runtime', 3.0) if properties else 3.0
    }

def estimate_runtime_and_cost(device, gates_required=0, circuit_depth=0):
    """Estimate runtime and cost for a job."""
    base_runtime = device.get("avg_runtime", 3.0)
    estimated_runtime = max(base_runtime, gates_required * 0.01, circuit_depth * 0.05)
    cost_usd = estimated_runtime * COST_PER_MINUTE
    energy_kwh = (estimated_runtime / 60) * POWER_CONSUMPTION_KW
    carbon_kg = energy_kwh * CARBON_PER_KWH

    return {
        "runtime_minutes": estimated_runtime,
        "cost_usd": cost_usd,
        "energy_kwh": energy_kwh,
        "carbon_kg": carbon_kg
    }

def calculate_swap_gates(circuit):
    """Calculate the number of swap gates in a circuit."""
    # Placeholder implementation
    return 0

def estimate_fidelity(circuit, backend, properties):
    """Estimate the fidelity of a circuit on a backend."""
    # Placeholder implementation
    return 0.95

def estimate_single_qubit_gate_fidelity(device):
    """Estimate single qubit gate fidelity."""
    return device.get("gate_fidelity", 0.95)

def estimate_two_qubit_gate_fidelity(device):
    """Estimate two qubit gate fidelity."""
    return device.get("gate_fidelity", 0.95) * 0.9

def estimate_parameterized_gate_fidelity(device):
    """Estimate parameterized gate fidelity."""
    return device.get("gate_fidelity", 0.95) * 0.8

def estimate_three_qubit_gate_fidelity(device):
    """Estimate three qubit gate fidelity."""
    return device.get("gate_fidelity", 0.95) * 0.7

def predict_device_fitness_qsvm(device_data):
    """Predict device fitness using QSVM."""
    # Placeholder implementation
    return 0.8

def calculate_job_specific_score(device, job_request):
    """Calculate job-specific score for a device."""
    # Placeholder implementation
    return 7.5

def generate_explanation(device, job_request):
    """Generate explanation for device recommendation."""
    return f"Device {device['name']} is recommended due to its {device['qubits']} qubits and low error rate of {device['error_rate']}."
