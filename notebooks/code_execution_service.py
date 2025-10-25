from typing import Dict, Any, Optional
import logging
from jupyter_client import KernelManager

logger = logging.getLogger(__name__)

class CodeExecutionService:
    def __init__(self):
        self.timeout = 30  # seconds
        self.max_output_size = 1024 * 1024  # 1MB

    def execute_code(self, code: str, cell_id: str = None) -> Dict[str, Any]:
        """
        Execute Python/Qiskit code using Jupyter kernel and return results asynchronously.
        """
        try:
            km = KernelManager(kernel_name='python3')
            km.start_kernel()
            kc = km.client()
            kc.start_channels()

            # Execute code
            msg_id = kc.execute(code)

            output = ""
            error = ""
            plots = []
            circuit_diagrams = []

            while True:
                msg = kc.get_iopub_msg(timeout=5)
                msg_type = msg['header']['msg_type']

                if msg_type == 'stream':
                    output += msg['content']['text']
                elif msg_type == 'error':
                    error += '\n'.join(msg['content']['traceback'])
                elif msg_type == 'execute_result':
                    data = msg['content']['data']
                    if 'text/plain' in data:
                        output += data['text/plain']
                    if 'image/png' in data:
                        img_data = data['image/png']
                        plots.append(f"data:image/png;base64,{img_data}")
                elif msg_type == 'display_data':
                    data = msg['content']['data']
                    if 'image/png' in data:
                        img_data = data['image/png']
                        plots.append(f"data:image/png;base64,{img_data}")

                if msg['parent_header'].get('msg_id') == msg_id and msg_type == 'status' and msg['content']['execution_state'] == 'idle':
                    break

            kc.stop_channels()
            km.shutdown_kernel()

            return {
                "success": error == "",
                "output": output.strip(),
                "error": error.strip(),
                "plots": plots,
                "circuit_diagrams": circuit_diagrams
            }

        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "plots": [],
                "circuit_diagrams": []
            }



    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Basic validation of Python code before execution
        """
        issues = []

        # Check for dangerous imports
        dangerous_imports = [
            'os', 'subprocess', 'sys', 'importlib', 'builtins',
            'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3',
            'socket', 'ssl', 'http', 'urllib', 'ftplib', 'poplib', 'imaplib', 'smtplib'
        ]

        for dangerous in dangerous_imports:
            if f"import {dangerous}" in code or f"from {dangerous}" in code:
                issues.append(f"Import of '{dangerous}' module is not allowed")

        # Check for file operations
        file_operations = ['open(', 'read(', 'write(', 'close(']
        for op in file_operations:
            if op in code:
                issues.append(f"File operation '{op[:-1]}' is not allowed")

        # Check for system calls
        if 'exec(' in code or 'eval(' in code or '__import__(' in code:
            issues.append("Code execution functions are not allowed")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
