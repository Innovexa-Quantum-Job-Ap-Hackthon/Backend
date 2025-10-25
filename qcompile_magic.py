"""
Step 3: Device Integration
Enhanced Jupyter magic word with IBM Quantum device integration and multi-format support.
"""

import sys
import os
import re
import requests
from IPython.core.magic import register_cell_magic
from IPython.display import display, HTML
from typing import Dict, List, Tuple 
import html 

# --- QISKIT IMPORTS (Attempt for Python detection) ---
try:
    from qiskit import QuantumCircuit
    # CRITICAL FIX: Import qasm2 for dumping QASM string
    from qiskit import qasm2
    from qiskit.circuit.library import UGate, CXGate
    import numpy as np
    
    IBM_AVAILABLE = True
except ImportError:
    class QuantumCircuit:
        def __init__(self, *args, **kwargs): pass
    # Define dummy placeholders if imports failed
    class qasm2:
        @staticmethod
        def dumps(circuit): return "OPENQASM 2.0; // Qiskit not found"
    class UGate: pass
    class CXGate: pass
    class np:
        pi = 3.14159
    
    IBM_AVAILABLE = False
    print("Warning: Qiskit not fully available.")


# --- CORE LOGIC FOR CODE FORMAT DETECTION ---

def detect_circuit_format_and_code(cell: str) -> Tuple[str, str, str]:
    """
    Analyzes the cell content to determine format.
    Returns: (circuit_format, circuit_code_for_compilation, initial_circuit_name)
    """
    stripped_cell = cell.strip()
    
    # 1. Detect QASM Headers
    if stripped_cell.startswith("OPENQASM 3.0"):
        return "qasm3", stripped_cell, "qasm_code"
    if stripped_cell.startswith("OPENQASM 2.0"):
        return "qasm", stripped_cell, "qasm_code"
    
    # 2. Assume Qiskit Python Code
    return "qiskit-python", cell, "qc" 

def extract_qasm_from_python(cell: str) -> str:
    """
    Executes Python cell code in a safe, injected scope to get the QuantumCircuit object 
    and returns its QASM 2.0 representation.
    
    FIX: Prioritizes modern qasm2.dumps() but uses a more robust error handling
    to ensure success when running exec().
    """
    
    # Define the execution scope, injecting essential Qiskit and utility objects
    exec_scope = {
        '__builtins__': __builtins__,
        'np': np,
        'pi': np.pi,
        'QuantumCircuit': QuantumCircuit,
        'UGate': UGate, 
        'CXGate': CXGate,
    }
    
    local_ns = {}
    
    try:
        # Execute the Python code using the defined scope
        exec(cell, exec_scope, local_ns)
    except Exception as e:
        # If the code fails execution (e.g., syntax, missing local variable), report it.
        raise ValueError(f"Execution Error in Python circuit: {str(e)}")
        
    # Search for a QuantumCircuit object in local_ns first
    circuit_object = None
    for value in local_ns.values():
        if isinstance(value, QuantumCircuit):
            circuit_object = value
            break
    
    if not circuit_object:
        raise ValueError("No QuantumCircuit object found after execution.")

    # CRITICAL CONVERSION LOGIC
    try:
        # 1. Try modern QASM 2.0 method (most compatible with Qiskit 0.46+)
        return qasm2.dumps(circuit_object)
    except Exception as e:
        # 2. Fallback to deprecated .qasm() method if dumps() fails (for older Qiskit versions)
        try:
            if hasattr(circuit_object, 'qasm') and callable(circuit_object.qasm):
                 return circuit_object.qasm()
            else:
                raise AttributeError("Circuit object has no standard QASM export methods.")
        except Exception as fallback_error:
            # If both fail, raise a fatal error describing the conversion failure
            raise RuntimeError(f"QASM conversion failed using both dumps() and .qasm(). Original error: {str(e)}. Fallback error: {str(fallback_error)}")


@register_cell_magic
def qcompile(line, cell):
    """
    Quantum compilation magic word that analyzes circuit code against IBM backends.
    """
    print("🔬 Quantum Circuit Analysis and Compilation")

    # --- 1. CODE FORMAT DETECTION ---
    detected_format, circuit_code, circuit_name = detect_circuit_format_and_code(cell)
    
    # Initialize the code for the JS submit button (must be QASM 2.0). 
    # It defaults to the input code if it's not Python.
    qasm_code_for_submission = circuit_code 

    # --- CRITICAL FIX: CONVERT PYTHON TO QASM FOR SUBMISSION & API ANALYSIS ---
    if detected_format == "qiskit-python":
        try:
            # 1. Convert Python circuit to QASM 2.0
            converted_qasm = extract_qasm_from_python(circuit_code)

            # 2. Update the variables used for the API call and the submission button
            
            # CRITICAL FIX 1: Send the QASM string to the compiler API
            circuit_code = converted_qasm
            # CRITICAL FIX 2: Tell the compiler API the format is now QASM
            detected_format = "qasm" 
            
            # Prepare for JS submission (always QASM)
            qasm_code_for_submission = converted_qasm
            
            print("✅ Successfully converted Qiskit Python to QASM 2.0 for submission and analysis.")
        except Exception as e:
            # If conversion fails, circuit_code and detected_format remain the raw Python code/format,
            # which might cause the compilation API call to fail.
            print(f"Error during Qiskit-to-QASM conversion (Analysis API will receive raw Python): {e}")

    # --- 2. AUTHENTICATION (Placeholder) ---
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9kb2dhbWluZzRAZ21haWwuY29tIiwiZXhwIjozNTYwMDMwMDg5fQ.yhNUIV_PJB-HVDwmBPT5L8igLGwMntNpgKdAGvtt1eU"
    if not token:
        print("Error: Authorization token is missing.")
        return

    # --- 3. PREPARE API PAYLOAD (Includes explicit defaults for Pydantic compliance) ---
    payload = {
        # CRITICAL FIX: These now use the QASM string and 'qasm' format if the input was Python
        "circuit_code": circuit_code, 
        "circuit_format": detected_format,
        "shots": 1024,                      
        "optimization_level": 3             
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "Authorization": f"Bearer {token}"
    }

    print(f"Format Detected: {detected_format.upper()}. Submitting to compiler service...")

    # --- 4. CALL BACKEND API to compile circuits (with retry logic) ---
    try:
        max_attempts = 3
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                # The payload now uses the converted QASM code and 'qasm' format if original was Python
                response = requests.post("http://localhost:8000/api/circuit-compiler", json=payload, headers=headers)
                response.raise_for_status() 
                break 
            except requests.exceptions.RequestException as e:
                if attempt == max_attempts:
                    raise Exception(f"Failed to call compilation API after {max_attempts} attempts: {e}")
                print(f"HTTP error on attempt {attempt}/{max_attempts}: {e}")

        compilation_results = response.json()
        
        if response.status_code != 200:
            error_data = response.json()
            if 'validation_errors' in error_data.get('detail', {}):
                print(f"Compilation Failed (Validation): {error_data['detail'].get('error', 'Unknown Validation Error')}")
                for err in error_data['detail']['validation_errors']:
                    print(f"  Line {err.get('line')}: {err.get('message')} (Suggestion: {err.get('suggestion')})")
            else:
                # This block is now more likely to catch the backend failure if the Python code was sent unconverted.
                print(f"API call failed with status {response.status_code}: {response.text}")
            return # IMPORTANT: Exit if the API call failed

    except requests.exceptions.ConnectionError:
        print("Error: Failed to connect to the compilation API. Ensure your backend service is running on http://localhost:8000")
        return
    except Exception as e:
        print(f"Failed to call compilation API: {e}")
        return

    # --- 5. DISPLAY RESULTS (Styled HTML and WORKING SUBMIT BUTTON) ---
    table_html = """
    <style>
    .qc-table {
        border-collapse: collapse;
        width: 100%;
        font-family: 'Inter', Arial, sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    .qc-table th, .qc-table td {
        border: 1px solid #e0e0e0;
        padding: 10px 15px;
        text-align: center;
    }
    .qc-table th {
        background-color: #007bff;
        color: white;
        font-weight: 600;
        white-space: nowrap;
    }
    .qc-table tr:nth-child(even) {background-color: #f7f7f7;}
    .qc-table tr:hover {background-color: #e9ecef;}
    .status-ok { color: #28a745; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .submit-btn {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 8px 15px;
        cursor: pointer;
        border-radius: 4px;
        transition: background-color 0.2s;
    }
    .submit-btn:hover { background-color: #0056b3; }
    .submit-btn:disabled { background-color: #6c757d; cursor: not-allowed; }
    .job-id { color: #28a745; font-weight: bold; }
    </style>
    <h3>Compilation Results Summary</h3>
    <table class="qc-table" id="compilation-table">
        <thead>
            <tr>
                <th>Device</th>
                <th>Status</th>
                <th>Req. Qubits</th>
                <th>Add. SWAPs</th>
                <th>Depth</th>
                <th>Est. Fidelity</th>
                <th>Time (s)</th>
                <th>Cost (USD)</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # 1. Prepare Arguments for JavaScript 
    # This correctly uses the qasm_code_for_submission (which is now the converted QASM string on success)
    code_for_js = qasm_code_for_submission.replace("'", "&#39;").replace('\n', '\\n').replace('\r', '')
    
    # Final HTML escape ensures safety
    escaped_code = html.escape(code_for_js)
    escaped_token = html.escape(token)
    
    # For submission, we always use the QASM format type since we converted it.
    escaped_format_for_sub = html.escape("qasm") 
    
    for result in compilation_results:

        can_execute = result.get("can_execute")
        status = f'<span class="status-ok">Executable</span>' if can_execute else f'<span class="status-error">Not Executable</span>'
        device_name = result.get('device_name')
        
        
        if can_execute:
            # 2. Call JS function with the converted QASM code and explicitly set 'qasm' format
            action = f"<button class=\"submit-btn\" onclick=\"submitJob('{device_name}', '{escaped_code}', '{escaped_format_for_sub}', '{escaped_token}')\">Submit Job</button>"
        else:
            action = f'<span title="{result.get("error_message") or "Incompatible"}">Incompatible</span>'
        
        est_fidelity = f"{result.get('estimated_fidelity', 0) * 100:.2f}%" if result.get("estimated_fidelity") else "N/A"
        cost_usd = f"${result.get('cost_estimation', {}).get('estimated_cost_usd', 0):.6f}" if result.get('cost_estimation') else "N/A"

        table_html += f"""
            <tr>
                <td>{device_name}</td>
                <td>{status}</td>
                <td>{result.get('required_qubits')}</td>
                <td>{result.get('additional_swaps')}</td>
                <td>{result.get('circuit_depth')}</td>
                <td>{est_fidelity}</td>
                <td>{result.get('compilation_time', 0):.3f}</td>
                <td>{cost_usd}</td>
                <td id="action-{device_name}">{action}</td>
            </tr>
        """
    table_html += """
        </tbody>
    </table>
    <script>
    async function submitJob(deviceName, circuitCode, circuitFormat, token) {
        const actionCell = document.getElementById(`action-${deviceName}`);
        const button = actionCell.querySelector('.submit-btn');
        
        if (button) {
            button.disabled = true;
            button.textContent = 'Submitting...';
        }

        try {
            // Restore newlines and prepare the payload
            const codeToSubmit = circuitCode.replace(/\\n/g, '\\n');

            const response = await fetch('http://localhost:8000/api/submit-circuit-to-ibm', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}` 
                },
                body: JSON.stringify({
                    // CRITICAL: Submitting the QASM string generated by Python, 
                    // and explicitly labeling the format as 'qasm'.
                    circuit_code: codeToSubmit, 
                    device_name: deviceName,
                    circuit_format: 'qasm', 
                    shots: 1024
                })
            });

            if (response.ok) {
                const data = await response.json();
                actionCell.innerHTML = `<span class="job-id">Job ID: ${data.job_id}</span>`;
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Submission failed');
            }
        } catch (error) {
            console.error('Error submitting job:', error);
            actionCell.innerHTML = `<span style="color: red; font-size: 0.9em;">Error: ${error.message.substring(0, 50)}...</span>`;
            if (button) {
                button.disabled = false;
                button.textContent = 'Submit Job';
            }
        }
    }
    </script>
    """
    display(HTML(table_html))
    print("\nStatus: success")
    print("Message: Compilation and device integration analysis completed")
    print("===============================")

def load_ipython_extension(ipython):
    ipython.register_magic_function(qcompile, 'cell', 'qcompile')
    setup_info = """
<div style="border: 2px solid #007bff; padding: 15px; border-radius: 5px; background-color: #f0f8ff; margin: 10px 0;">
    <h3 style="color: #007bff; margin-top: 0;">✅ Quantum Compiler Magic Loaded</h3>
    <p>Use the <code>%%qcompile</code> cell magic to analyze your quantum circuit code.</p>
    <p><strong>Supported Formats:</strong> Qiskit Python, OpenQASM 2.0, OpenQASM 3.0 (auto-detected).</p>
</div>
"""
    display(HTML(setup_info))

def unload_ipython_extension(ipython):
    if 'qcompile' in ipython.magics_manager.magics['cell']:
        del ipython.magics_manager.magics['cell']['qcompile']
    print("qcompile magic unloaded.")

# Auto-load when imported in an IPython environment
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython:
        load_ipython_extension(ipython)
except:
    pass
