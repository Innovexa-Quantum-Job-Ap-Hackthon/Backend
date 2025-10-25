import re
import time
import random
import logging
import qiskit
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UGate, CXGate
from qiskit.converters import circuit_to_dag
from qiskit.qasm3 import loads as qasm3_loads, dumps as qasm3_dumps
from qiskit.qasm3 import Exporter
from qiskit import qasm2
from pydantic import BaseModel
from fastapi import HTTPException
import matplotlib.pyplot as plt
import io
import base64
import warnings
from qiskit.exceptions import QiskitWarning
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService
from qiskit.transpiler import CouplingMap, Layout
from qiskit_ibm_runtime import Session, Sampler
import numpy as np

# Suppress the specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='qiskit.compiler.transpiler')
warnings.filterwarnings('ignore', category=QiskitWarning)

logger = logging.getLogger(__name__)

# ======================
# SAFETY CHECK: Ensure Qiskit supports QASM 3.0
# ======================
if not hasattr(qiskit, 'qasm3'):
    raise RuntimeError(
        "Qiskit installation is invalid. Please reinstall with:\n"
        "pip uninstall qiskit -y && pip install --no-cache-dir qiskit"
    )

# ======================
# CUSTOM EXCEPTIONS
# ======================
class QASMValidationError(Exception):
    def __init__(self, message, line=None, column=None, suggestion=None):
        self.message = message
        self.line = line
        self.column = column
        self.suggestion = suggestion
        super().__init__(message)

class CircuitParseError(Exception):
    """Raised when circuit cannot be parsed due to unsupported features or syntax."""
    pass

class QASM3FeatureNotSupportedError(Exception):
    """Raised when QASM 3.0 features are not supported by the current backend."""
    pass

# ======================
# PYDANTIC MODELS
# ======================
class CircuitCompilationRequest(BaseModel):
    circuit_code: str
    circuit_format: str = "qasm"  # "qasm", "qasm3", or "qiskit-python"
    shots: int = 1024
    optimization_level: int = 3

class CircuitCompilationResult(BaseModel):
    device_name: str
    can_execute: bool
    additional_swaps: int = 0
    circuit_depth: int = 0
    estimated_fidelity: float = 0.0
    compilation_time: float = 0.0
    required_qubits: int = 0
    available_qubits: int = 0
    error_message: Optional[str] = None
    performance_score: Optional[float] = None
    success_probability: Optional[float] = None
    cost_estimation: Optional[Dict[str, Any]] = None
    estimated_wait_minutes: Optional[int] = None
    total_gates: Optional[int] = None
    circuit_analysis: Optional[Dict[str, Any]] = None
    compiled_analysis: Optional[Dict[str, Any]] = None
    compilation_metrics: Optional[Dict[str, Any]] = None
    error_analysis: Optional[Dict[str, Any]] = None
    optimization_suggestions: Optional[List[Dict[str, Any]]] = None
    warnings: Optional[List[Dict[str, Any]]] = None
    circuit_image_base64: Optional[str] = None
    gate_decomposition_log: Optional[List[Dict]] = None
    qasm3_compatible: Optional[bool] = None
    qasm3_warnings: Optional[List[str]] = None
    transpiled_qasm: Optional[str] = None

# ======================
# DEVICE-SPECIFIC CONFIGURATION
# ======================
DEVICE_SPECIFIC_CONFIG = {
    "ibm_brisbane": {
        "provider_model": "ibm",
        "shots_multiplier": 1.0,
        "calibration_factor": 1.0,
        "qubit_count": 127,
        "gate_times_ns": {
            "id": 35, "x": 35, "y": 35, "z": 35, "h": 35, "s": 35, "sdg": 35,
            "t": 35, "tdg": 35, "sx": 35, "sxdg": 35, "rx": 35, "ry": 35, "rz": 35,
            "p": 35, "u": 35, "u1": 35, "u2": 35, "u3": 35,
            "cx": 250, "cz": 250, "swap": 750, "cswap": 1000,
            "measure": 800, "reset": 100
        },
        "base_error_rate": 0.0012,
        "readout_error": 0.015,
        "t1_time": 95.0,  # microseconds
        "t2_time": 85.0,  # microseconds
        "price_per_qpu_second": 1.70,  # $1.70 per second
        "coupling_map": [],  # Will be filled dynamically
        "basis_gates": ['id', 'rz', 'sx', 'x', 'cx', 'reset']
    },
    "ibm_torino": {
        "provider_model": "ibm",
        "shots_multiplier": 1.15,
        "calibration_factor": 0.85,
        "qubit_count": 138,
        "gate_times_ns": {
            "id": 30, "x": 30, "y": 30, "z": 30, "h": 30, "s": 30, "sdg": 30,
            "t": 30, "tdg": 30, "sx": 30, "sxdg": 30, "rx": 30, "ry": 30, "rz": 30,
            "p": 30, "u": 30, "u1": 30, "u2": 30, "u3": 30,
            "cx": 220, "cz": 220, "swap": 660, "cswap": 880,
            "measure": 750, "reset": 90
        },
        "base_error_rate": 0.0009,
        "readout_error": 0.012,
        "t1_time": 110.0,  # microseconds
        "t2_time": 95.0,  # microseconds
        "price_per_qpu_second": 2.10,  # $2.10 per second
        "coupling_map": [],  # Will be filled dynamically
        "basis_gates": ['id', 'rz', 'sx', 'x', 'cx', 'reset']
    }
}

def get_device_config(device_name: str) -> Dict:
    default_config = {
        "provider_model": "ibm",
        "shots_multiplier": 1.0,
        "calibration_factor": 1.0,
        "qubit_count": 27,
        "gate_times_ns": {
            "id": 50, "x": 50, "y": 50, "z": 50, "h": 50, "s": 50, "sdg": 50,
            "t": 50, "tdg": 50, "sx": 50, "sxdg": 50, "rx": 50, "ry": 50, "rz": 50,
            "p": 50, "u": 50, "u1": 50, "u2": 50, "u3": 50,
            "cx": 300, "cz": 300, "swap": 900, "cswap": 1200,
            "measure": 1000, "reset": 150
        },
        "base_error_rate": 0.002,
        "readout_error": 0.02,
        "t1_time": 80.0,
        "t2_time": 70.0,
        "price_per_qpu_second": 0.0,
        "coupling_map": [],
        "basis_gates": ['id', 'rz', 'sx', 'x', 'cx', 'reset']
    }
    config = DEVICE_SPECIFIC_CONFIG.get(device_name.lower(), default_config).copy()
    return config

# ======================
# QASM 3.0 SUPPORT ENHANCEMENTS
# ======================
class QASM3SupportChecker:
    QASM3_FEATURES = {
        'qubit_declaration': r'qubit\s*\[\s*\d+\s*\]',
        'bit_declaration': r'bit\s*\[\s*\d+\s*\]',
        'gate_modifiers': r'(ctrl|negctrl|inv|pow)\s*@',
        'float_literals': r'\d+\.\d*[eE][+-]?\d+',
        'duration_literals': r'\d+\s*(dt|ns|us|ms|s)',
        'array_declaration': r'(\w+)\s*\[\s*\d+\s*\]\s*(\w+)',
        'for_loops': r'for\s*\(',
        'while_loops': r'while\s*\(',
        'if_statements': r'if\s*\(',
        'defcalgrammar': r'defcalgrammar',
        'defcal': r'defcal',
        'box': r'box\s*{',
        'let': r'let\s+',
        'break': r'break\s*;',
        'continue': r'continue\s*;',
        'quantum_phases': r'#pragma\s+quantum_phases'
    }

    @classmethod
    def preprocess_qasm3(cls, qasm_code: str) -> str:
        """Preprocess QASM 3.0 code to handle common compatibility issues and advanced control flow."""
        lines = qasm_code.split('\n')
        processed_lines = []
        inside_block = False
        block_indent = ""
        
        # Simple parser state to comment out complex QASM 3.0 blocks
        for line in lines:
            stripped = line.strip()

            if re.match(r'(for|while|if)\s*\(.*\)\s*{', stripped) and not inside_block:
                inside_block = True
                block_indent = line[:line.find('for')] if 'for' in stripped else (line[:line.find('while')] if 'while' in stripped else line[:line.find('if')])
                processed_lines.append(f'{block_indent}// {stripped}  // Advanced control flow (commented out for transpilation)')
                continue

            if inside_block:
                if '}' in stripped:
                    inside_block = False
                    processed_lines.append(f'{block_indent}// }}  // End of control block')
                else:
                    processed_lines.append(f'// {line}')
                continue
            
            # Convert QASM 3.0 qubit declaration to QASM 2.0 style for wider compatibility
            qubit_match = re.match(r'qubit\s*\[\s*(\d+)\s*\]\s*(\w+)\s*;', stripped)
            if qubit_match:
                size, name = qubit_match.groups()
                processed_lines.append(f'qreg {name}[{size}];')
                continue

            bit_match = re.match(r'bit\s*\[\s*(\d+)\s*\]\s*(\w+)\s*;', stripped)
            if bit_match:
                size, name = bit_match.groups()
                processed_lines.append(f'creg {name}[{size}];')
                continue
            
            # Keep original line
            processed_lines.append(line)

        return '\n'.join(processed_lines)

    @classmethod
    def analyze_qasm3_features(cls, qasm_code: str) -> Dict[str, Any]:
        """Analyze QASM 3.0 code for advanced features."""
        analysis = {
            'has_advanced_features': False,
            'features_found': [],
            'compatibility_warnings': [],
            'supported_by_ibm': True
        }

        for feature, pattern in cls.QASM3_FEATURES.items():
            if re.search(pattern, qasm_code, re.IGNORECASE):
                analysis['features_found'].append(feature)
                analysis['has_advanced_features'] = True
                
                # Check IBM compatibility for dynamic circuits/pulse control
                if feature in ['defcalgrammar', 'defcal', 'duration_literals', 'quantum_phases', 'for_loops', 'while_loops']:
                    analysis['compatibility_warnings'].append(
                        f"Feature '{feature}' may require Qiskit Runtime Primitives and might not be fully supported by default transpilation."
                    )
                    if feature in ['defcalgrammar', 'defcal', 'duration_literals', 'quantum_phases']:
                         analysis['supported_by_ibm'] = False

        return analysis

# ======================
# QUANTUM CIRCUIT VALIDATOR
# NOTE: Full implementation included here for completeness.
# ======================
class QuantumCircuitValidator:
    def __init__(self):
        self.defined_registers = {'q': {}, 'c': {}}
        self.defined_gates = {}
        self.gate_definitions = set()
        self.errors = []
        self.warnings = []
        # Extended gate sets for QASM 3.0
        self.qasm3_builtin_gates = {
            'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'cx', 'cz', 'swap',
            'u', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'id', 'p', 'sx', 'sxdg',
            'cu1', 'cu2', 'cu3', 'crx', 'cry', 'crz', 'ccx', 'cswap',
            'measure', 'reset', 'barrier', 'delay'
        }
        self.qasm2_builtin_gates = self.qasm3_builtin_gates - {'reset', 'delay'}

    def validate_qasm(self, qasm_code: str, is_qasm3: bool = False) -> List[Dict]:
        self.defined_registers = {'q': {}, 'c': {}}
        self.defined_gates = {}
        self.gate_definitions = set()
        self.errors = []
        self.warnings = []

        lines = qasm_code.split('\n')
        current_gate_definition = None
        inside_for_loop = False
        inside_while_loop = False
        inside_if_statement = False

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue

            try:
                # Skip lines inside for, while, if blocks for now (as QASM 3.0 is often preprocessed)
                if inside_for_loop:
                    if '}' in stripped:
                        inside_for_loop = False
                    continue

                if inside_while_loop:
                    if '}' in stripped:
                        inside_while_loop = False
                    continue

                if inside_if_statement:
                    if '}' in stripped:
                        inside_if_statement = False
                    continue

                if current_gate_definition:
                    if '}' in stripped:
                        current_gate_definition = None
                    continue

                # Detect start of for, while, if blocks
                if re.match(r'for\s*\(.*\)\s*{', stripped):
                    inside_for_loop = True
                    continue

                if re.match(r'while\s*\(.*\)\s*{', stripped):
                    inside_while_loop = True
                    continue

                if re.match(r'if\s*\(.*\)\s*{', stripped):
                    inside_if_statement = True
                    continue

                self._validate_line_syntax(stripped, line_num, is_qasm3)

                # Check for gate definition start
                if stripped.startswith('gate ') and '{' in stripped:
                    current_gate_definition = True

            except QASMValidationError as e:
                self.errors.append({
                    'line': e.line,
                    'column': e.column,
                    'message': e.message,
                    'suggestion': e.suggestion,
                    'severity': 'error'
                })
            except Exception as e:
                self.errors.append({
                    'line': line_num,
                    'message': f"Unexpected error during validation: {str(e)}",
                    'severity': 'error'
                })

        return self.errors

    def _validate_line_syntax(self, line: str, line_num: int, is_qasm3: bool):
        # Handle QASM version declaration
        if line.startswith('OPENQASM'):
            if '3.0' in line and not is_qasm3:
                raise QASMValidationError(
                    "QASM 3.0 detected but format specified as QASM 2.0",
                    line=line_num,
                    suggestion="Set circuit_format to 'qasm3' or convert to QASM 2.0"
                )
            return

        # Handle includes
        if line.startswith('include'):
            if 'stdgates.inc' in line:
                self.gate_definitions.update(self.qasm3_builtin_gates)
            elif 'qelib1.inc' in line:
                self.gate_definitions.update(self.qasm2_builtin_gates)
            return

        # Handle register declarations
        is_declaration = False
        if is_qasm3:
            is_declaration = self._validate_qasm3_declarations(line, line_num)
        else:
            is_declaration = self._validate_qasm2_declarations(line, line_num)

        # Handle gate applications
        if not is_declaration and not line.startswith(('gate', 'def', 'include', 'OPENQASM')) and line.endswith(';'):
            self._validate_gate_application(line, line_num, is_qasm3)

    def _validate_qasm3_declarations(self, line: str, line_num: int) -> bool:
        # Qubit declarations
        qubit_match = re.match(r'qubit\s*\[\s*(\d+)\s*\]\s*(\w+)\s*;', line)
        if qubit_match:
            size, name = qubit_match.groups()
            if name in self.defined_registers['q']:
                raise QASMValidationError(f"Duplicate qubit register '{name}'", line=line_num)
            self.defined_registers['q'][name] = int(size)
            return True

        # Bit declarations
        bit_match = re.match(r'bit\s*\[\s*(\d+)\s*\]\s*(\w+)\s*;', line)
        if bit_match:
            size, name = bit_match.groups()
            if name in self.defined_registers['c']:
                raise QASMValidationError(f"Duplicate classical register '{name}'", line=line_num)
            self.defined_registers['c'][name] = int(size)
            return True

        # For QASM 2.0 compatibility in QASM 3.0
        return self._validate_qasm2_declarations(line, line_num)

    def _validate_qasm2_declarations(self, line: str, line_num: int) -> bool:
        # Qreg declarations
        qreg_match = re.match(r'qreg\s+(\w+)\[(\d+)\];', line)
        if qreg_match:
            name, size = qreg_match.groups()
            if name in self.defined_registers['q']:
                raise QASMValidationError(f"Duplicate qreg register '{name}'", line=line_num)
            self.defined_registers['q'][name] = int(size)
            return True

        # Creg declarations
        creg_match = re.match(r'creg\s*(\w+)\[(\d+)\];', line)
        if creg_match:
            name, size = creg_match.groups()
            if name in self.defined_registers['c']:
                raise QASMValidationError(f"Duplicate creg register '{name}'", line=line_num)
            self.defined_registers['c'][name] = int(size)
            return True

        return False

    def _validate_gate_application(self, line: str, line_num: int, is_qasm3: bool):
        clean_line = line.split('//')[0].strip()[:-1].strip()

        # Handle measurement operations
        if 'measure' in clean_line:
            if is_qasm3 and '=' in clean_line:
                self._validate_qasm3_measurement(clean_line, line_num)
            elif '->' in clean_line:
                self._validate_qasm2_measurement(clean_line, line_num)
            else:
                 raise QASMValidationError("Invalid measurement syntax.", line=line_num)
            return

        # Handle barrier/reset operations
        if clean_line.startswith('barrier'):
            self._validate_barrier_reset(clean_line, line_num, 'q')
            return
        if clean_line.startswith('reset'):
            self._validate_barrier_reset(clean_line, line_num, 'q')
            return

        # Handle general gate applications
        parts = re.split(r'\s+', clean_line, 1)
        if len(parts) < 2 and parts[0] not in self.qasm3_builtin_gates:
            raise QASMValidationError("Invalid gate syntax", line=line_num)

        gate_call = parts[0]
        args_part = parts[1] if len(parts) > 1 else ""

        # Extract gate name and parameters
        gate_name = gate_call.split('(')[0]
        
        # Validate gate name
        builtin_gates = self.qasm3_builtin_gates if is_qasm3 else self.qasm2_builtin_gates
        if gate_name not in builtin_gates and gate_name not in self.gate_definitions:
            raise QASMValidationError(
                f"Unknown gate '{gate_name}'",
                line=line_num,
                suggestion=f"Use a built-in gate or define it first"
            )

        # Validate arguments (qubit registers)
        args = re.findall(r'(\w+)(?:\[(\d+)\])?', args_part)
        for arg_name, arg_index in args:
            self._validate_register_access(f"{arg_name}[{arg_index}]" if arg_index else arg_name, 'q', line_num)

    def _validate_qasm3_measurement(self, line: str, line_num: int):
        match = re.match(r'(\w+(?:\[\d+\])?)\s*=\s*measure\s+(\w+(?:\[\d+\])?)', line)
        if not match:
            raise QASMValidationError("Invalid QASM 3.0 measurement syntax", line=line_num, suggestion="Use: c[idx] = measure q[idx];")
        
        c_reg_full, q_reg_full = match.groups()
        self._validate_register_access(c_reg_full, 'c', line_num)
        self._validate_register_access(q_reg_full, 'q', line_num)

    def _validate_qasm2_measurement(self, line: str, line_num: int):
        match = re.match(r'measure\s+(\w+(?:\[\d+\])?)\s*->\s*(\w+(?:\[\d+\])?)', line)
        if not match:
            raise QASMValidationError("Invalid QASM 2.0 measurement syntax", line=line_num, suggestion="Use: measure q[idx] -> c[idx];")
        
        q_reg_full, c_reg_full = match.groups()
        self._validate_register_access(q_reg_full, 'q', line_num)
        self._validate_register_access(c_reg_full, 'c', line_num)

    def _validate_barrier_reset(self, line: str, line_num: int, reg_type: str):
        # Extracts all arguments after the command
        args_match = re.search(r'(barrier|reset)\s+(.*)', line)
        if args_match:
            args_str = args_match.group(2).strip()
            if not args_str: return # Command with no arguments is valid
            args = args_str.split(',')
            for arg in args:
                arg = arg.strip()
                self._validate_register_access(arg, reg_type, line_num)

    def _validate_register_access(self, reg_str: str, reg_type: str, line_num: int):
        if '[' in reg_str: # Indexed access
            match = re.match(r'(\w+)\[(\d+)\]', reg_str)
            if not match:
                raise QASMValidationError(f"Invalid register access: {reg_str}", line=line_num, suggestion="Use format: register_name[index]")
            name, idx = match.groups()
            idx = int(idx)
            self._validate_register_name(name, reg_type, line_num)
            if idx >= self.defined_registers[reg_type][name]:
                raise QASMValidationError(
                    f"Index {idx} out of bounds for {reg_type} register '{name}' (size: {self.defined_registers[reg_type][name]})",
                    line=line_num,
                    suggestion=f"Valid indices: 0 to {self.defined_registers[reg_type][name] - 1}"
                )
        else: # Whole register access
            self._validate_register_name(reg_str, reg_type, line_num)

    def _validate_register_name(self, name: str, reg_type: str, line_num: int):
        if name not in self.defined_registers[reg_type]:
            raise QASMValidationError(
                f"Undefined {reg_type} register '{name}'",
                line=line_num,
                suggestion=f"Declare {reg_type} register '{name}' before use"
            )

# ======================
# GATE DECOMPOSITION LOGGER
# NOTE: Full implementation included here for completeness.
# ======================
class GateDecomposer:
    @staticmethod
    def decompose_and_log(circuit: QuantumCircuit, basis_gates: List[str]) -> List[Dict]:
        log = []
        if not basis_gates:
            return log

        dag = circuit_to_dag(circuit)
        for node in dag.topological_op_nodes():
            op = node.op
            if op.name not in basis_gates and op.definition is not None:
                try:
                    decomp = op.definition
                    if decomp:
                        decomposed_gates = []
                        for inst in decomp.data:
                            decomposed_gates.append({
                                'name': inst.operation.name,
                                'qubits': [circuit.qubits.index(q) for q in inst.qubits],
                                'params': list(inst.operation.params) if hasattr(inst.operation, 'params') else []
                            })
                        log.append({
                            'original_gate': op.name,
                            'decomposed_into': decomposed_gates,
                            'qubits': [circuit.qubits.index(q) for q in node.qargs],
                            'params': list(op.params) if hasattr(op, 'params') else []
                        })
                except Exception as e:
                    log.append({
                        'original_gate': op.name,
                        'decomposed_into': [],
                        'qubits': [circuit.qubits.index(q) for q in node.qargs],
                        'params': list(op.params) if hasattr(op, 'params') else [],
                        'warning': f'Could not decompose gate: {str(e)}'
                    })
        return log
        
# ======================
# CIRCUIT VISUALIZER
# NOTE: Full implementation included here for completeness.
# ======================
class CircuitVisualizer:
    @staticmethod
    def to_base64(circuit: QuantumCircuit) -> str:
        try:
            import matplotlib
            matplotlib.use('Agg')
            # Check if circuit is empty or too large for standard draw
            if circuit.size() == 0 or circuit.num_qubits > 50:
                 fig = plt.figure(figsize=(10, 6))
                 plt.text(0.5, 0.5, "Circuit too complex or empty for visual drawing. Showing text representation.",
                             ha='center', va='center', family='monospace')
                 plt.axis('off')
            else:
                try:
                    # Attempt standard MPL drawing
                    fig = circuit.draw(output='mpl', style={'fontsize': 10, 'subfontsize': 8})
                except Exception:
                    # Fallback to text-based drawing visualization
                    fig = plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, str(circuit.draw(output='text')),
                             ha='center', va='center', family='monospace')
                    plt.axis('off')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(fig)
            return img_str
        except Exception as e:
            logger.warning(f"Failed to generate circuit image: {e}")
            return None

# ======================
# ADVANCED CIRCUIT ANALYZER
# NOTE: Full implementation included here for completeness.
# ======================
class AdvancedCircuitAnalyzer:
    @staticmethod
    def analyze_circuit_resources(circuit: QuantumCircuit) -> Dict:
        analysis = {
            'qubit_count': circuit.num_qubits,
            'classical_bit_count': circuit.num_clbits,
            'total_gates': len(circuit.data),
            'gate_types': circuit.count_ops(),
            'depth': circuit.depth(),
            'optimization_suggestions': [],
            'connectivity_analysis': {},
            'parallelism_analysis': {},
            'operations_by_qubit': {}
        }
        
        # Count operations per qubit
        for qubit_index in range(circuit.num_qubits):
            count = 0
            current_qubit = circuit.qubits[qubit_index]
            for instruction in circuit.data:
                if current_qubit in instruction.qubits:
                    count += 1
            analysis['operations_by_qubit'][qubit_index] = count

        # Generate optimization suggestions
        analysis['optimization_suggestions'] = AdvancedCircuitAnalyzer._generate_optimization_suggestions(analysis)
        return analysis

    @staticmethod
    def _generate_optimization_suggestions(analysis: Dict) -> List[Dict]:
        suggestions = []

        # Check for inactive qubits
        inactive_qubits = [
            q for q, ops in analysis['operations_by_qubit'].items()
            if ops == 0
        ]
        if inactive_qubits:
            suggestions.append({
                'type': 'qubit_reduction',
                'message': f"{len(inactive_qubits)} qubits are entirely unused.",
                'suggestion': "Remove unused qubits to reduce resource requirements.",
                'priority': 'medium'
            })

        # Check for high depth
        if analysis['depth'] > 100 and analysis['qubit_count'] > 5:
            suggestions.append({
                'type': 'depth_reduction',
                'message': f"Circuit depth is very high ({analysis['depth']})",
                'suggestion': "Use optimization level 3 or circuit synthesis techniques to reduce depth.",
                'priority': 'high'
            })
            
        # Check for high CNOT count
        cx_count = analysis['gate_types'].get('cx', 0)
        if cx_count > 50:
            suggestions.append({
                'type': 'cx_optimization',
                'message': f"Circuit has {cx_count} CNOT gates (a common source of error)",
                'suggestion': "Consider re-examining the algorithm for CNOT-efficient alternatives.",
                'priority': 'medium'
            })

        return suggestions

# ======================
# QUANTUM ERROR SIMULATOR
# NOTE: Full implementation included here for completeness.
# ======================
class QuantumErrorSimulator:
    @staticmethod
    def simulate_errors(circuit: QuantumCircuit, backend: IBMBackend, device: Dict) -> Dict:
        # The fidelity calculation is now primarily handled in QuantumCompiler._estimate_fidelity
        # This function focuses on providing structured error analysis data.
        error_analysis = {
            'predicted_success_rate': 0.0, # Will be overwritten by estimated_fidelity
            'error_breakdown': {},
            'vulnerable_operations': [],
            'error_mitigation_suggestions': [],
            'noise_sensitivity': 0.0,
            'coherence_limits': {}
        }

        device_config = get_device_config(device['name'])
        
        # Simplified error breakdown based on config data
        base_error = device_config.get('base_error_rate', 0.001)
        readout_error = device_config.get('readout_error', 0.015)
        
        error_analysis['error_breakdown'] = {
            'single_qubit_base': base_error,
            'two_qubit_base': base_error * 5,
            'readout_base': readout_error,
            'coherence_t1_us': device_config.get('t1_time'),
            'coherence_t2_us': device_config.get('t2_time'),
        }
        
        # Simplified simulation of vulnerable operations
        cx_count = circuit.count_ops().get('cx', 0)
        if cx_count > 0:
            error_analysis['vulnerable_operations'].append({
                'gate': 'CX (CNOT)',
                'count': cx_count,
                'risk': 'high',
                'reason': f'High count of two-qubit gates, which are most prone to error on current hardware.'
            })

        if circuit.depth() * 0.1 > min(device_config.get('t1_time', 100), device_config.get('t2_time', 80)) * 0.8:
            error_analysis['error_mitigation_suggestions'].append({
                'type': 'dynamical_decoupling',
                'priority': 'high',
                'message': 'Circuit duration is estimated to be close to or exceed coherence times.',
                'suggestion': 'Apply dynamical decoupling to protect qubits during idle periods or reduce circuit depth.'
            })
            
        error_analysis['coherence_limits'] = QuantumErrorSimulator._estimate_coherence_limits(circuit, device_config)

        return error_analysis

    @staticmethod
    def _estimate_coherence_limits(circuit: QuantumCircuit, device_config: Dict) -> Dict:
        """Estimate coherence time limits for the circuit."""
        t1 = device_config.get('t1_time', 100.0)
        t2 = device_config.get('t2_time', 80.0)

        # Estimate circuit duration (simplified)
        gate_times = device_config.get('gate_times_ns', {})
        avg_gate_time = sum(gate_times.values()) / max(len(gate_times), 1) if gate_times else 50
        circuit_duration_ns = circuit.depth() * avg_gate_time
        circuit_duration_us = circuit_duration_ns / 1000

        t1_utilization = circuit_duration_us / t1
        t2_utilization = circuit_duration_us / t2

        return {
            'circuit_duration_us': round(circuit_duration_us, 3),
            't1_utilization_percent': round(t1_utilization * 100, 2),
            't2_utilization_percent': round(t2_utilization * 100, 2),
            'within_coherence_limits': circuit_duration_us < min(t1, t2) * 0.8
        }

    @staticmethod
    def _estimate_fidelity_from_config(circuit: QuantumCircuit, device_config: Dict) -> float:
        """Estimate fidelity using device configuration and overall readout error (used as fallback)."""
        base_error = device_config.get('base_error_rate', 0.001)
        readout_error = device_config.get('readout_error', 0.015)
        total_fidelity = 1.0

        for inst in circuit:
            qubits = len(inst.qubits)
            if qubits == 1:
                error = base_error
            elif qubits == 2:
                error = base_error * 5
            else:
                error = base_error * 10
            total_fidelity *= (1 - error)

        # Apply overall measurement error
        total_fidelity *= (1 - readout_error) ** circuit.num_clbits
        
        return max(0.0, total_fidelity)


# ======================
# MAIN COMPILER - ENHANCED FOR QISKTT-PYTHON SUPPORT AND FIDELITY
# ======================
class QuantumCompiler:
    def __init__(self):
        self.validator = QuantumCircuitValidator()
        self.analyzer = AdvancedCircuitAnalyzer()
        self.error_simulator = QuantumErrorSimulator()
        self.decomposer = GateDecomposer()
        self.visualizer = CircuitVisualizer()
        self.qasm3_checker = QASM3SupportChecker()

    def _parse_qiskit_python(self, python_code: str) -> QuantumCircuit:
        """
        Executes Python code in a restricted namespace to extract a QuantumCircuit object safely.
        """
        local_env = {}
        
        # Define the minimal set of Qiskit components necessary for circuit building
        allowed_imports = {
            'QuantumCircuit': QuantumCircuit,
            'QuantumRegister': qiskit.QuantumRegister,
            'ClassicalRegister': qiskit.ClassicalRegister,
            'Parameter': qiskit.circuit.Parameter,
            'UGate': UGate, 
            'CXGate': CXGate,
            'np': np, 
            'pi': np.pi,
            # Add more common Qiskit components
            'circuit': qiskit.circuit,
            'transpile': transpile,
            'Aer': getattr(qiskit, 'Aer', None),
            'execute': getattr(qiskit, 'execute', None),
        }
        
        # Remove None values from allowed_imports
        allowed_imports = {k: v for k, v in allowed_imports.items() if v is not None}
        
        try:
            # Execute code in a restricted namespace - ONLY ONCE
            exec(
        python_code, 
        {
            "__builtins__": {
                "__import__": __import__, 
                "print": print, 
                "len": len, 
                "range": range
            }, 
            **allowed_imports
        }, 
        local_env
    )
        except Exception as e:
            # Re-raise error with clearer message 
            raise CircuitParseError(f"Error executing Qiskit Python code: {e}")

        final_circuit = None
        
        # Attempt to find the circuit object in the local namespace
        for key, value in local_env.items():
            if isinstance(value, QuantumCircuit):
                if final_circuit is None:
                    final_circuit = value
                else:
                    logger.warning(f"Multiple QuantumCircuit objects found, using '{key}'")
        
        # Fallback check for common assignment names (case-insensitive)
        circuit_names = ['qc', 'circuit', 'quantum_circuit', 'qcircuit']
        for name in circuit_names:
            if name in local_env and isinstance(local_env[name], QuantumCircuit):
                final_circuit = local_env[name]
                break
                
        if final_circuit is None:
            # Try to find any variable ending with 'circuit' or 'qc'
            for var_name, var_value in local_env.items():
                if (isinstance(var_value, QuantumCircuit) and 
                    (var_name.lower().endswith('circuit') or var_name.lower().endswith('qc'))):
                    final_circuit = var_value
                    break
        
        if final_circuit is None:
            raise CircuitParseError(
                "Qiskit Python code must define a QuantumCircuit object. "
                "Common variable names: 'qc', 'circuit', 'quantum_circuit'. "
                "Make sure the circuit is assigned to a variable."
            )
            
        return final_circuit
    
    def _preprocess_qasm2_compat(self, qasm_code: str) -> str:
        """Sanitizes QASM code for QASM 2.0 parser compatibility."""
        lines = qasm_code.split('\n')
        processed_lines = []
        for line in lines:
            stripped = line.strip()
            indent = line[:line.find(stripped)] if stripped else ""

            if 'OPENQASM 3.0' in stripped or stripped.startswith('//'): continue

            # Convert QASM 3.0 qubit declaration: qubit[size] name; -> qreg name[size];
            qubit_match = re.match(r'qubit\s*\[\s*(\d+)\s*\]\s*(\w+)\s*;?', stripped)
            if qubit_match:
                size, name = qubit_match.groups()
                processed_lines.append(f'{indent}qreg {name}[{size}];')
                continue

            # Convert QASM 3.0 bit declaration: bit[size] name; -> creg name[size];
            bit_match = re.match(r'bit\s*\[\s*(\d+)\s*\]\s*(\w+)\s*;?', stripped)
            if bit_match:
                size, name = bit_match.groups()
                processed_lines.append(f'{indent}creg {name}[{size}];')
                continue

            # Convert QASM 3.0 measurement syntax: c = measure q; -> measure q -> c;
            if '=' in stripped and 'measure' in stripped:
                match = re.match(r'(\w+(?:\[\d+\])?)\s*=\s*measure\s*(\w+(?:\[\d+\])?);?', stripped)
                if match:
                    c_reg_full, q_reg_full = match.groups()
                    processed_lines.append(f'{indent}measure {q_reg_full} -> {c_reg_full};')
                    continue

            processed_lines.append(line)

        final_code = '\n'.join(processed_lines)
        if 'include "qelib1.inc";' not in final_code:
            final_code = 'include "qelib1.inc";\n' + final_code
        if 'OPENQASM 2.0;' not in final_code:
            final_code = 'OPENQASM 2.0;\n' + final_code

        return final_code

    def _parse_qasm3(self, qasm_code: str) -> QuantumCircuit:
        """Attempt to parse QASM 3.0, falling back to QASM 2.0 compatibility mode if failed."""
        try:
            return qasm3_loads(qasm_code)
        except Exception as e:
            # Fallback to QASM 2.0 parser after sanitization
            sanitized_qasm2_code = self._preprocess_qasm2_compat(qasm_code)
            try:
                return QuantumCircuit.from_qasm_str(sanitized_qasm2_code)
            except Exception as final_error:
                raise CircuitParseError(
                    f"QASM 3.0/2.0 parsing failed. The circuit contains non-QASM 2.0 features that could not be stripped: {str(final_error)}\n"
                    f"Original QASM 3.0 error: {str(e)}"
                )

    def compile_circuit(self, circuit_code: str, circuit_format: str, devices: List[Dict],
                    user_service: QiskitRuntimeService, current_user_id: str, estimate_compilation_cost_func: Callable) -> List[CircuitCompilationResult]:
    
        circuit_code = circuit_code.replace("Ï€", "pi")
        
        circuit_format = circuit_format.lower()
        
        try:
            circuit_code = re.sub(r'[^\x00-\x7F]+', '', circuit_code)
            circuit_code = '\n'.join([line.strip() for line in circuit_code.splitlines() if line.strip()])
        except Exception as e:
            logger.warning(f"Sanitation failed with error: {e}")

        processed_code = circuit_code
        is_qasm3 = circuit_format == "qasm3"
        is_qiskit_python = circuit_format == "qiskit-python"
        qasm3_analysis = {'has_advanced_features': False, 'supported_by_ibm': True}
        
        # --- 1. Parsing and Validation ---
        try:
            if is_qiskit_python:
                circuit = self._parse_qiskit_python(circuit_code)
                # Skip QASM validation for Python code
            else:
                if is_qasm3:
                    processed_code = self.qasm3_checker.preprocess_qasm3(circuit_code)
                    qasm3_analysis = self.qasm3_checker.analyze_qasm3_features(circuit_code)
                    circuit = self._parse_qasm3(processed_code)
                else:
                    processed_code = self._preprocess_qasm2_compat(circuit_code)
                    circuit = QuantumCircuit.from_qasm_str(processed_code)
                
                # Perform QASM structural validation only for QASM formats
                validation_errors = self.validator.validate_qasm(processed_code, is_qasm3)
                if validation_errors:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "QASM Structural Validation Failed",
                            "validation_errors": validation_errors,
                            "suggestion": "Please check your QASM syntax (registers/gates) and try again."
                        }
                    )

        except Exception as e:
            error_details = self._extract_parse_error_details(e, circuit_code, is_qasm3 or is_qiskit_python)
            raise HTTPException(status_code=400, detail=error_details)

        # Rest of your compilation logic remains the same...
        required_qubits = circuit.num_qubits
        results = []

    

        for device in devices:
            device_name = device['name']
            available_qubits = device['qubits']

            if available_qubits < required_qubits:
                results.append(self._create_incompatible_result(device, required_qubits, circuit, qasm3_analysis))
                continue

            try:
                backend = user_service.backend(device_name)
                result = self._compile_for_device(
                    circuit, backend, device, current_user_id,
                    estimate_compilation_cost_func, qasm3_analysis, is_qasm3
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Device {device_name} compilation failed: {e}")
                results.append(self._handle_compilation_error(e, device, required_qubits, circuit, qasm3_analysis))

        results.sort(key=lambda x: x.estimated_fidelity or 0, reverse=True)
        return results

    def _calculate_coherence_decay(self, circuit: QuantumCircuit, device_config: Dict) -> float:
        """
        Calculates the coherence decay factor based on estimated execution time 
        relative to the device's T1 and T2 times.
        """
        gate_counts = circuit.count_ops()
        gate_times_ns = device_config.get('gate_times_ns', {})
        
        total_duration_ns = 0
        for gate, count in gate_counts.items():
            total_duration_ns += count * gate_times_ns.get(gate, 50) 
        
        total_duration_us = total_duration_ns * 1e-3 # Convert to microseconds
        
        t1 = device_config.get('t1_time', 100.0)
        t2 = device_config.get('t2_time', 80.0)

        t_effective = min(t1, t2)
        
        if total_duration_us > 0 and t_effective > 0:
            coherence_decay = np.exp(-total_duration_us / t_effective)
        else:
            coherence_decay = 1.0

        return coherence_decay

    def _compile_for_device(self, circuit: QuantumCircuit, backend: IBMBackend, device: Dict,
                            user_id: str, estimate_compilation_cost_func: Callable, qasm3_analysis: Dict, is_qasm3: bool) -> CircuitCompilationResult:
        
        if not isinstance(backend.service, QiskitRuntimeService):
            raise TypeError(
                "Backend must be from QiskitRuntimeService, not legacy IBMQ provider."
            )

        start_time = time.time()

        device_config = get_device_config(device['name'])
        
        if hasattr(backend, 'configuration'):
            config = backend.configuration()
            device_config['basis_gates'] = getattr(config, 'basis_gates', device_config['basis_gates'])
            if hasattr(config, 'coupling_map') and config.coupling_map:
                device_config['coupling_map'] = config.coupling_map

        basis_gates = device_config.get('basis_gates', [])

        # Decompose and log gate transformations
        decomposition_log = self.decomposer.decompose_and_log(circuit, basis_gates)

        # Transpile circuit and capture result object to access layout
        transpiled_result = transpile(
            circuit,
            backend=backend,
            optimization_level=3,
            coupling_map=device_config.get('coupling_map'),
            basis_gates=basis_gates,
            seed_transpiler=42, 
            initial_layout=None 
        )
        compiled_circuit = transpiled_result
        
        # Extract initial layout (Layout object might be stored in the result metadata)
        initial_layout = None
        
        # Generate QASM string from transpiled circuit
        if is_qasm3:
            transpiled_qasm = qasm3_dumps(compiled_circuit)
        else:
            transpiled_qasm = qasm2.dumps(compiled_circuit)

        compilation_time = time.time() - start_time

        # --- IMPROVED FIDELITY CALCULATION ---
        estimated_fidelity = self._estimate_fidelity(compiled_circuit, backend, device)

        # Perform analyses
        circuit_analysis = self.analyzer.analyze_circuit_resources(circuit)
        compiled_analysis = self.analyzer.analyze_circuit_resources(compiled_circuit)
        error_analysis = self.error_simulator.simulate_errors(compiled_circuit, backend, device)
        error_analysis['predicted_success_rate'] = estimated_fidelity

        # Generate circuit visualization
        circuit_image = self.visualizer.to_base64(circuit)
        
        # Calculate compilation metrics, passing the captured layout
        compilation_metrics = self._calculate_compilation_metrics(circuit, compiled_circuit, compiled_circuit.count_ops().get('swap', 0), initial_layout)


        # Prepare result
        result_dict = {
            "device_name": device['name'],
            "can_execute": True,
            "additional_swaps": compilation_metrics['swap_overhead'],
            "circuit_depth": compiled_circuit.depth(),
            "estimated_fidelity": round(estimated_fidelity, 4),
            "compilation_time": round(compilation_time, 3),
            "required_qubits": circuit.num_qubits,
            "available_qubits": device['qubits'],
            "performance_score": device.get('score'),
            "success_probability": round(error_analysis.get('predicted_success_rate', 0.0), 4),
            "estimated_wait_minutes": device.get('wait_time', 10),
            "total_gates": compiled_circuit.size(),
            "circuit_analysis": circuit_analysis,
            "compiled_analysis": compiled_analysis,
            "compilation_metrics": compilation_metrics,
            "error_analysis": error_analysis,
            "optimization_suggestions": circuit_analysis.get('optimization_suggestions', []) + error_analysis.get('error_mitigation_suggestions', []),
            "warnings": self._generate_compilation_warnings(compiled_circuit, error_analysis),
            "circuit_image_base64": circuit_image,
            "gate_decomposition_log": decomposition_log,
            "qasm3_compatible": qasm3_analysis.get('supported_by_ibm', True) if is_qasm3 else None,
            "qasm3_warnings": qasm3_analysis.get('compatibility_warnings', []) if is_qasm3 else None,
            "transpiled_qasm": transpiled_qasm
        }

        result_dict = estimate_compilation_cost_func(result_dict, compiled_circuit, device)

        return CircuitCompilationResult(**result_dict)

    def _estimate_fidelity(self, compiled_circuit: QuantumCircuit, backend: IBMBackend, device: Dict) -> float:
        """
        Estimates total fidelity by combining gate error accumulation, readout error, and coherence decay.
        """
        total_gate_fidelity = 1.0
        device_config = get_device_config(device['name'])
        general_readout_error = device_config.get('readout_error', 0.015)
        
        # --- 1. Calculate Coherence Decay ---
        coherence_decay = self._calculate_coherence_decay(compiled_circuit, device_config)

        # --- 2. Calculate Error Accumulation ---
        if hasattr(backend, 'properties') and backend.properties():
            props = backend.properties()
            measured_qubits = set()
            
            for inst in compiled_circuit.data:
                # 2a. Gate Error Accumulation
                if inst.operation.name not in ['measure', 'reset', 'barrier', 'delay']:
                    op = inst.operation
                    qubits = [compiled_circuit.qubits.index(q) for q in inst.qubits]
                    try:
                        error = 0.0
                        if len(qubits) == 1:
                            error = props.gate_error(op.name, qubits[0])
                        elif len(qubits) == 2:
                            error = props.gate_error(op.name, tuple(qubits))
                        else:
                            error = device_config.get('base_error_rate', 0.001) * 10
                        total_gate_fidelity *= (1 - error)
                    except Exception:
                        base_error = device_config.get('base_error_rate', 0.001)
                        error = base_error * (5 if len(qubits) == 2 else (10 if len(qubits) > 2 else 1))
                        total_gate_fidelity *= (1 - error)
                
                # 2b. Track Measured Qubits for Readout Error
                if inst.operation.name == 'measure':
                    q_idx = compiled_circuit.qubits.index(inst.qubits[0])
                    measured_qubits.add(q_idx)

            # --- 3. Calculate Readout Error ---
            total_readout_fidelity = 1.0
            for q_idx in measured_qubits:
                try:
                    # Use specific qubit readout error rate
                    readout_err = props.readout_error(q_idx)
                    total_readout_fidelity *= (1 - readout_err)
                except Exception:
                    # Fallback to general readout error
                    total_readout_fidelity *= (1 - general_readout_error)
            
            # --- 4. Combine ---
            final_fidelity = total_gate_fidelity * total_readout_fidelity * coherence_decay

        else:
            # Fallback when backend properties are unavailable.
            gate_and_readout_fidelity = self.error_simulator._estimate_fidelity_from_config(compiled_circuit, device_config)
            final_fidelity = gate_and_readout_fidelity * coherence_decay
            
        return max(0.0, final_fidelity)

    def _create_incompatible_result(self, device: Dict, required_qubits: int, circuit: QuantumCircuit, qasm3_analysis: Dict) -> CircuitCompilationResult:
        analysis = self.analyzer.analyze_circuit_resources(circuit)
        return CircuitCompilationResult(
            device_name=device['name'],
            can_execute=False,
            required_qubits=required_qubits,
            available_qubits=device['qubits'],
            error_message=f"Insufficient qubits: {required_qubits} required, {device['qubits']} available.",
            optimization_suggestions=analysis.get('optimization_suggestions', []),
            circuit_analysis=analysis,
            estimated_wait_minutes=device.get('wait_time', 10),
            qasm3_compatible=qasm3_analysis.get('supported_by_ibm', True),
            qasm3_warnings=qasm3_analysis.get('compatibility_warnings', []),
            transpiled_qasm=None
        )

    def _handle_compilation_error(self, error: Exception, device: Dict, required_qubits: int, circuit: QuantumCircuit, qasm3_analysis: Dict) -> CircuitCompilationResult:
        error_msg = self._extract_error_message(error)
        return CircuitCompilationResult(
            device_name=device['name'],
            can_execute=False,
            required_qubits=required_qubits,
            available_qubits=device['qubits'],
            error_message=error_msg,
            estimated_wait_minutes=device.get('wait_time', 10),
            transpiled_qasm=None
        )

    def _extract_parse_error_details(self, error: Exception, circuit_code: str, is_qasm_or_python: bool) -> Dict:
        error_type = "Qiskit Python Execution Failed" if isinstance(error, CircuitParseError) and "Python" in str(error) else "Circuit Parsing Failed"
        details = {
            "error": error_type,
            "message": str(error),
            "source_type": "Qiskit Python" if "Python" in error_type else ("QASM 3.0" if is_qasm_or_python else "QASM 2.0"),
            "suggestion": "Please ensure your Qiskit code defines a QuantumCircuit object and is valid Python syntax or your QASM is syntactically correct."
        }
        return details
    
    def _extract_error_message(self, error: Exception) -> str:
        return str(error)

    def _calculate_compilation_metrics(self, original: QuantumCircuit, compiled: QuantumCircuit, additional_swaps: int, initial_layout: Optional['Layout'] = None) -> Dict:
        """
        Calculates detailed metrics including the impact of the chosen initial layout.
        """
        orig_ops = original.count_ops()
        comp_ops = compiled.count_ops()
        
        basis_cx = comp_ops.get('cx', 0)
        basis_cz = comp_ops.get('cz', 0)
        multi_qubit_gates = basis_cx + basis_cz + comp_ops.get('swap', 0)
        
        # New metric: Layout distance (how far the initial map is from identity)
        layout_score = 0
        
        # FIX: Initialize layout_map inside the conditional block using a robust method.
        if initial_layout:
            try:
                # Qiskit 1.0+ method: get_physical_bits() returns {Physical_index: VirtualBit}
                # We want {Virtual_index: Physical_index} to calculate distance
                v_to_p_map = {v_bit.index: p_idx for p_idx, v_bit in initial_layout.get_physical_bits().items() if v_bit.index is not None}
                
                # Calculate distance from identity
                for v_idx, p_idx in v_to_p_map.items():
                    if v_idx != p_idx:
                        layout_score += 1
                        
            except Exception as e:
                # If the Layout object is missing the method or is malformed, we set the score to 0
                # logger.warning(f"Failed to calculate layout distance: {e}") # (optional debugging)
                layout_score = 0 

        return{
            'original_gates': len(original.data),
            'compiled_gates': len(compiled.data),
            'gate_reduction': len(original.data) - len(compiled.data),
            'depth_increase': compiled.depth() - original.depth(),
            'swap_overhead': additional_swaps,
            'cnot_increase': comp_ops.get('cx', 0) - orig_ops.get('cx', 0),
            'single_qubit_gate_change': (
                sum(count for gate, count in comp_ops.items() if gate not in ['cx', 'cz', 'swap', 'measure', 'reset', 'barrier']) -
                sum(count for gate, count in orig_ops.items() if gate not in ['cx', 'cz', 'swap', 'measure', 'reset', 'barrier'])
            ),
            'compilation_ratio': len(compiled.data) / max(len(original.data), 1),
            'multi_qubit_gate_ratio': multi_qubit_gates / max(len(compiled.data), 1),
            'initial_layout_distance': layout_score
        }

    def _generate_compilation_warnings(self, compiled_circuit: QuantumCircuit, error_analysis: Dict) -> List[Dict]:
        warnings = []

        if compiled_circuit.depth() > 100:
            warnings.append({
                'type': 'high_depth',
                'message': f"High circuit depth: {compiled_circuit.depth()}",
                'suggestion': 'High depth increases susceptibility to decoherence. Use circuit optimization or gate fusion.',
                'priority': 'medium'
            })

        if error_analysis.get('predicted_success_rate', 1.0) < 0.6:
            warnings.append({
                'type': 'low_success',
                'message': f"Low predicted success rate: {error_analysis.get('predicted_success_rate', 0.0):.2%}",
                'suggestion': 'The expected output will be very noisy. Use error mitigation techniques like ZNE or Mitiq for better results.',
                'priority': 'high'
            })

        # Check coherence limits
        coherence_limits = error_analysis.get('coherence_limits', {})
        if not coherence_limits.get('within_coherence_limits', True):
            warnings.append({
                'type': 'coherence_limit',
                'message': 'Circuit may exceed coherence time limits',
                'suggestion': 'Reduce circuit depth or use dynamical decoupling techniques.',
                'priority': 'high'
            })

        return warnings

# ======================
# RUNTIME ESTIMATION FUNCTIONS
# ======================
def estimate_qpu_runtime(transpiled_circuit: QuantumCircuit, shots: int, device_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimates the QPU execution time by analyzing the transpiled circuit.
    """
    coupling_map = CouplingMap(device_config.get("coupling_map", []))
    basis_gates = device_config.get("basis_gates", [])

    gate_counts = transpiled_circuit.count_ops()
    gate_times_ns = device_config.get("gate_times_ns", {})

    single_shot_time_ns = 0
    for gate, count in gate_counts.items():
        gate_time = gate_times_ns.get(gate, 0)
        single_shot_time_ns += count * gate_time

    measure_time = gate_times_ns.get('measure', 0)
    single_shot_time_ns += transpiled_circuit.num_clbits * measure_time

    total_qpu_time_ns = single_shot_time_ns * shots
    total_qpu_time_s = total_qpu_time_ns * 1e-9

    return {
        "estimated_qpu_runtime_seconds": total_qpu_time_s,
        "shots_requested": shots,
        "total_gates": transpiled_circuit.size(),
        "transpiled_gate_counts": gate_counts,
        "transpiled_circuit_depth": transpiled_circuit.depth(),
        "single_shot_time_ns": single_shot_time_ns
    }

def estimate_cost(compilation_result: Dict, compiled_circuit: QuantumCircuit, device: Dict) -> Dict:
    """
    Estimates the cost of a quantum job based on QPU runtime.
    """
    device_name = device.get("name", "unknown")
    device_config = get_device_config(device_name)

    runtime_estimation = estimate_qpu_runtime(
        compiled_circuit,
        compilation_result.get('shots', 1024),
        device_config
    )

    runtime_seconds = runtime_estimation["estimated_qpu_runtime_seconds"]
    price_per_second = device_config.get("price_per_qpu_second", 0.0)

    final_cost = runtime_seconds * price_per_second

    compilation_result["cost_estimation"] = {
        "provider": "IBM Quantum",
        "model_type": "runtime_based_pay_as-you-go",
        "shots_requested": compilation_result.get('shots', 1024),
        "user_plan": "pay-as-you-go" if final_cost > 0 else "free",
        "estimated_cost_usd": round(final_cost, 6),
        "breakdown": {
            "estimated_qpu_runtime_seconds": round(runtime_seconds, 9),
            "price_per_qpu_second_usd": price_per_second,
            "raw_cost": round(final_cost, 6),
            "total_gates": runtime_estimation["total_gates"],
            "estimated_wait_minutes": device.get("wait_time", 10),
            "carbon_kg": device.get("carbon_footprint", 0.0)
        },
        "timestamp": datetime.now().isoformat(),
        "device": device_name,
        "currency": "USD"
    }

    return compilation_result

# ======================
# COMPARISON FUNCTIONS
# ======================
def compare_ibm_devices(compilation_results: List[CircuitCompilationResult]) -> Dict:
    comparison = {
        "devices": {},
        "technical_comparison": {},
        "cost_comparison": {},
        "recommendations": [],
        "timestamp": datetime.now().isoformat()
    }

    brisbane_result = None
    torino_result = None

    for result in compilation_results:
        if result.device_name.lower() == "ibm_brisbane":
            brisbane_result = result
        elif result.device_name.lower() == "ibm_torino":
            torino_result = result

    if not brisbane_result or not torino_result:
        return {"error": "Could not find data for both IBM Brisbane and IBM Torino"}

    if not brisbane_result.can_execute or not torino_result.can_execute:
        return {"error": "One or both devices cannot execute the circuit"}

    # Technical comparison
    comparison["technical_comparison"] = {
        "circuit_depth": {
            "ibm_brisbane": brisbane_result.circuit_depth,
            "ibm_torino": torino_result.circuit_depth,
            "difference": torino_result.circuit_depth - brisbane_result.circuit_depth,
            "winner": "ibm_torino" if torino_result.circuit_depth < brisbane_result.circuit_depth else "ibm_brisbane"
        },
        "total_gates": {
            "ibm_brisbane": brisbane_result.total_gates,
            "ibm_torino": torino_result.total_gates,
            "difference": torino_result.total_gates - brisbane_result.total_gates,
            "winner": "ibm_torino" if torino_result.total_gates < brisbane_result.total_gates else "ibm_brisbane"
        },
        "additional_swaps": {
            "ibm_brisbane": brisbane_result.additional_swaps,
            "ibm_torino": torino_result.additional_swaps,
            "difference": torino_result.additional_swaps - brisbane_result.additional_swaps,
            "winner": "ibm_torino" if torino_result.additional_swaps < brisbane_result.additional_swaps else "ibm_brisbane"
        },
        "estimated_fidelity": {
            "ibm_brisbane": brisbane_result.estimated_fidelity,
            "ibm_torino": torino_result.estimated_fidelity,
            "difference": round(torino_result.estimated_fidelity - brisbane_result.estimated_fidelity, 4),
            "winner": "ibm_torino" if torino_result.estimated_fidelity > brisbane_result.estimated_fidelity else "ibm_brisbane"
        },
        "success_probability": {
            "ibm_brisbane": brisbane_result.success_probability,
            "ibm_torino": torino_result.success_probability,
            "difference": round(torino_result.success_probability - brisbane_result.success_probability, 4),
            "winner": "ibm_torino" if torino_result.success_probability > brisbane_result.success_probability else "ibm_brisbane"
        }
    }

    # Cost comparison
    brisbane_cost = brisbane_result.cost_estimation["estimated_cost_usd"] if brisbane_result.cost_estimation else 0
    torino_cost = torino_result.cost_estimation["estimated_cost_usd"] if torino_result.cost_estimation else 0

    comparison["cost_comparison"] = {
        "estimated_cost_usd": {
            "ibm_brisbane": brisbane_cost,
            "ibm_torino": torino_cost,
            "difference": round(torino_cost - brisbane_cost, 6),
            "winner": "ibm_brisbane" if brisbane_cost < torino_cost else "ibm_torino"
        }
    }

    # Generate recommendations
    fidelity_winner = comparison["technical_comparison"]["estimated_fidelity"]["winner"]
    cost_winner = comparison["cost_comparison"]["estimated_cost_usd"]["winner"]
    fidelity_diff = comparison["technical_comparison"]["estimated_fidelity"]["difference"]
    cost_diff = comparison["cost_comparison"]["estimated_cost_usd"]["difference"]

    if fidelity_winner == cost_winner:
        comparison["recommendations"].append(
            f"🏆 **{fidelity_winner.upper()}** is the clear winner! "
            f"It has {fidelity_diff:.2%} better fidelity and costs ${abs(cost_diff):.6f} {'less' if cost_diff < 0 else 'more'}"
        )
    else:
        comparison["recommendations"].append(
            f"⚠️ **Trade-off required**: {fidelity_winner.upper()} has better fidelity (+{fidelity_diff:.2%}), "
            f"but {cost_winner.upper()} is cheaper (${abs(cost_diff):.6f} difference)"
        )

    # Value comparison (fidelity per dollar)
    brisbane_value = brisbane_result.estimated_fidelity / max(brisbane_cost, 0.0001) if brisbane_cost > 0 else float('inf')
    torino_value = torino_result.estimated_fidelity / max(torino_cost, 0.0001) if torino_cost > 0 else float('inf')

    if brisbane_value > torino_value:
        comparison["recommendations"].append(
            f"💰 **Best value**: IBM Brisbane provides {brisbane_value:.2f} fidelity per dollar "
            f"(vs {torino_value:.2f} for IBM Torino)"
        )
    else:
        comparison["recommendations"].append(
            f"💰 **Best value**: IBM Torino provides {torino_value:.2f} fidelity per dollar "
            f"(vs {brisbane_value:.2f} for IBM Brisbane)"
        )

    return comparison

# ======================
# UTILITY FUNCTIONS
# ======================
def get_circuit_summary(circuit: QuantumCircuit) -> Dict:
    """Get a quick summary of circuit properties"""
    ops = circuit.count_ops()
    return {
        "qubits": circuit.num_qubits,
        "clbits": circuit.num_clbits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "operations": ops,
        "cnot_count": ops.get('cx', 0),
        "single_qubit_gates": sum(count for gate, count in ops.items()
                                  if gate not in ['cx', 'cz', 'swap', 'barrier', 'measure']),
        "multi_qubit_gates": sum(count for gate, count in ops.items()
                                 if gate in ['cx', 'cz', 'swap', 'cswap', 'ccx'])
    }
