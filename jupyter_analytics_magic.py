import os
import requests
import json
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.display import display, HTML
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import html

# --- Configuration & Global State ---
# Assuming FastAPI is running locally on port 8000 as per main.py
API_BASE_URL = "http://localhost:8000/api"

# Mock credentials storage for the notebook session
# NOTE: In a real environment, the user would securely authenticate.
GLOBAL_CREDENTIALS = {
    "user_id": "1", # Mock User ID (assuming '1' is a valid user ID in your DB)
    "auth_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9kb2dhbWluZzRAZ21haWwuY29tIiwiZXhwIjozNTYwMDMwMDg5fQ.yhNUIV_PJB-HVDwmBPT5L8igLGwMntNpgKdAGvtt1eU", # Mock Token
}

@magics_class
class QuantumAnalyticsMagics(Magics):

    def _make_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict[str, Any]] = None) -> Any:
        """Helper to make authenticated requests to the FastAPI backend."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GLOBAL_CREDENTIALS['auth_token']}"
        }
        url = f"{API_BASE_URL}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=data)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data)
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            error_msg = f"API Error ({response.status_code}): {response.text}"
            display(HTML(f"<div style='color: red; padding: 10px; border: 1px solid red; border-radius: 5px;'>{error_msg}</div>"))
            return None
        except requests.exceptions.ConnectionError:
            display(HTML("<div style='color: red; padding: 10px; border: 1px solid red; border-radius: 5px;'>Connection Error: Ensure your FastAPI backend is running on http://localhost:8000.</div>"))
            return None
        except Exception as e:
            display(HTML(f"<div style='color: red; padding: 10px; border: 1px solid red; border-radius: 5px;'>An unexpected error occurred: {e}</div>"))
            return None

    @line_magic
    def analytics_setup(self, line: str):
        """%analytics_setup - Sets up necessary mock user credentials."""
        # This simulates logging in and obtaining necessary credentials
        # In a real setup, this would call an authentication endpoint
        
        # NOTE: For simplicity, we are hardcoding a mock user ID and token.
        # This user ID must match a user in your local FastAPI database (e.g., ID 1).
        
        display(HTML(f"""
        <div style="background-color: #e0f7fa; border: 1px solid #00bcd4; padding: 15px; border-radius: 8px;">
            <h3>🔑 Analytics Setup Complete</h3>
            <p>Mock user ID <code>{GLOBAL_CREDENTIALS['user_id']}</code> initialized for API calls.</p>
            <p>You can now run device and job magic commands.</p>
        </div>
        """))
        

    @line_magic
    def device_details(self, line: str):
        """%device_details <device_name> - Shows full device metrics."""
        device_name = line.strip()
        if not device_name:
            display(HTML("<div style='color: orange;'>Usage: %device_details &lt;device_name&gt;</div>"))
            return
        
        # Call the endpoint: /api/devices/{device_name}
        data = self._make_request(f"/devices/{device_name}", method='GET')
        
        if data:
            # Format detailed device information into an attractive HTML structure
            html_output = f"""
            <style>
                .detail-card {{
                    font-family: 'Inter', sans-serif;
                    border: 2px solid #007bff;
                    border-radius: 12px;
                    padding: 20px;
                    margin: 15px 0;
                    background-color: #f8fafd;
                    box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15);
                }}
                .detail-card h3 {{ color: #007bff; margin-top: 0; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px; }}
                .metric-item {{ padding: 10px; border-left: 3px solid #00bcd4; background-color: white; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
                .metric-label {{ font-weight: 600; color: #555; font-size: 0.9em; }}
                .metric-value {{ font-size: 1.1em; color: #333; }}
                .status-badge {{ display: inline-block; padding: 4px 10px; border-radius: 15px; font-weight: bold; font-size: 0.9em; }}
                .status-online {{ background-color: #e6ffed; color: #008000; }}
                .status-offline {{ background-color: #ffeaea; color: #dc3545; }}
            </style>
            <div class="detail-card">
                <h3>{html.escape(data.get('name', 'N/A'))} Details 
                    <span class="status-badge status-{html.escape(data.get('status', 'offline'))}">
                        {html.escape(data.get('status', 'N/A').upper())}
                    </span>
                </h3>
                <div class="metric-grid">
                    <div class="metric-item"><div class="metric-label">Qubits</div><div class="metric-value">{data.get('qubits', 'N/A')}</div></div>
                    <div class="metric-item"><div class="metric-label">Queue Length</div><div class="metric-value">{data.get('queue_length', 'N/A')} jobs</div></div>
                    <div class="metric-item"><div class="metric-label">Performance Score</div><div class="metric-value">{data.get('score', 0):.2f} / 10.0</div></div>
                    <div class="metric-item"><div class="metric-label">Success Probability (ML)</div><div class="metric-value">{(data.get('success_probability', 0) * 100):.2f}%</div></div>
                    <div class="metric-item"><div class="metric-label">Est. Wait Time (ML)</div><div class="metric-value">{data.get('wait_time', 'N/A')} min</div></div>
                    <div class="metric-item"><div class="metric-label">Gate Fidelity</div><div class="metric-value">{(data.get('gate_fidelity', 0) * 100):.2f}%</div></div>
                    <div class="metric-item"><div class="metric-label">Error Rate (CX)</div><div class="metric-value">{(data.get('error_rate', 0) * 100):.3f}%</div></div>
                    <div class="metric-item"><div class="metric-label">T1 Time</div><div class="metric-value">{data.get('t1_time', 0):.2f} µs</div></div>
                    <div class="metric-item"><div class="metric-label">T2 Time</div><div class="metric-value">{data.get('t2_time', 0):.2f} µs</div></div>
                    <div class="metric-item"><div class="metric-label">Est. Cost (USD)</div><div class="metric-value">${data.get('cost_estimate', 0):.4f}</div></div>
                    <div class="metric-item"><div class="metric-label">Noise Profile</div><div class="metric-value">{html.escape(data.get('noise_profile', 'N/A'))}</div></div>
                    <div class="metric-item"><div class="metric-label">Noise Rec.</div><div class="metric-value">{html.escape(data.get('noise_recommendation', 'N/A'))}</div></div>
                </div>
                <p style="margin-top: 20px; font-size: 0.85em; color: #6c757d;">Last Updated: {data.get('last_updated', 'N/A')}</p>
            </div>
            """
            display(HTML(html_output))
            
    @line_magic
    def device_summary(self, line: str):
        """%device_summary - Shows a summary table of all devices."""
        # Call the endpoint: /api/devices
        data = self._make_request(f"/devices", method='GET', data={"user_id": GLOBAL_CREDENTIALS['user_id']})
        
        if data and data.get('devices'):
            devices = data['devices']
            
            # Use pandas for easy tabular formatting
            df = pd.DataFrame(devices)
            
            # Select and rename columns for display
            df_display = df[[
                'name', 'status', 'qubits', 'queue_length', 'score', 'success_probability', 'wait_time'
            ]].copy()
            
            df_display.rename(columns={
                'name': 'Device',
                'status': 'Status',
                'qubits': 'Qubits',
                'queue_length': 'Queue',
                'score': 'ML Score',
                'success_probability': 'Success Prob.',
                'wait_time': 'Est. Wait (min)'
            }, inplace=True)
            
            # Format columns
            df_display['ML Score'] = df_display['ML Score'].apply(lambda x: f"{x:.2f}")
            df_display['Success Prob.'] = df_display['Success Prob.'].apply(lambda x: f"{x * 100:.1f}%")
            
            # Convert to HTML and add styling
            html_table = df_display.to_html(index=False, classes=['dataframe', 'device-summary-table'])

            # --- CORRECTED CSS BLOCK USING F-STRING ---
            css = f"""
            <style>
                .device-summary-table {{
                    border-collapse: collapse;
                    width: 100%;
                    font-family: 'Inter', sans-serif;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }}
                .device-summary-table th, .device-summary-table td {{
                    border: 1px solid #e9ecef;
                    padding: 10px 15px;
                    text-align: left;
                }}
                .device-summary-table th {{
                    background-color: #007bff;
                    color: white;
                    font-weight: 600;
                    white-space: nowrap;
                }}
                .device-summary-table tr:nth-child(even) {{background-color: #f7f7f7;}}
                .device-summary-table tr:hover {{background-color: #e9ecef;}}
                .status {{ font-weight: bold; }}
                .online {{ color: #008000; }}
                .offline {{ color: #dc3545; }}
            </style>
            <h3>Quantum Device Summary ({len(devices)} Devices)</h3>
            <p style="font-size: 0.9em; color: #6c757d;">Last Updated: {data.get('last_updated', 'N/A')} from {data.get('source', 'Cache')}</p>
            """ 
            # --- END CORRECTED CSS BLOCK ---

            # Apply status color based on content (simple string replacement)
            html_table = html_table.replace('>online<', '><span class="status online">ONLINE</span><')
            html_table = html_table.replace('>offline<', '><span class="status offline">OFFLINE</span><')

            display(HTML(css + html_table))

    @line_magic
    def device_score(self, line: str):
        """%device_score <device_name> - Shows ML-specific metrics."""
        device_name = line.strip()
        if not device_name:
            display(HTML("<div style='color: orange;'>Usage: %device_score &lt;device_name&gt;</div>"))
            return
        
        # Call the endpoint: /api/device/{device_name}/success-probability
        data = self._make_request(f"/device/{device_name}/success-probability", method='GET')
        
        if data:
            html_output = f"""
            <style>
                .score-card {{
                    font-family: 'Inter', sans-serif;
                    border: 1px solid #28a745;
                    border-left: 5px solid #28a745;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 15px 0;
                    background-color: #e6ffed;
                }}
                .score-card h4 {{ color: #28a745; margin-top: 0; }}
                .score-grid {{ display: flex; justify-content: space-around; text-align: center; }}
                .score-metric {{ padding: 10px; }}
                .score-value {{ font-size: 1.8em; font-weight: bold; color: #1e7e34; }}
                .score-label {{ font-weight: 500; color: #555; font-size: 0.9em; }}
            </style>
            <div class="score-card">
                <h4>ML Predictions for {html.escape(data.get('device_name', 'N/A'))}</h4>
                <div class="score-grid">
                    <div class="score-metric">
                        <div class="score-value">{(data.get('success_probability', 0) * 100):.2f}%</div>
                        <div class="score-label">Success Probability</div>
                    </div>
                    <div class="score-metric">
                        <div class="score-value">{data.get('wait_time', 0)} min</div>
                        <div class="score-label">Est. Wait Time</div>
                    </div>
                    <div class="score-metric">
                        <div class="score-value">{(1 - data.get('error_rate', 0)) * 100:.2f}%</div>
                        <div class="score-label">Est. Gate Fidelity</div>
                    </div>
                </div>
                <p style="margin-top: 15px; font-size: 0.8em; color: #1e7e34;">
                    <strong>Noise Profile:</strong> {html.escape(data.get('noise_profile', 'N/A'))} - {html.escape(data.get('noise_recommendation', 'N/A'))}
                </p>
            </div>
            """
            display(HTML(html_output))

    @line_magic
    def job_tracker(self, line: str):
        """%job_tracker <job_id> - Tracks a job by its unique ID."""
        job_id = line.strip()
        if not job_id:
            display(HTML("<div style='color: orange;'>Usage: %job_tracker &lt;job_id&gt;</div>"))
            return
        
        # Call the endpoint: /api/job/{job_id}?user_id={user_id}
        # Note: We include user_id in the query params as required by your main.py
        data = self._make_request(f"/job/{job_id}", method='GET', data={"user_id": GLOBAL_CREDENTIALS['user_id']})
        
        if data:
            status = data.get('Status', data.get('status', 'UNKNOWN')).upper()
            
            if status in ['COMPLETED', 'DONE']:
                status_color = '#28a745'
                bg_color = '#e6ffed'
            elif status in ['FAILED', 'ERROR']:
                status_color = '#dc3545'
                bg_color = '#ffeaea'
            elif status in ['QUEUED', 'RUNNING', 'SUBMITTED', 'RETRYING']:
                status_color = '#ffc107'
                bg_color = '#fffbe6'
            else:
                status_color = '#6c757d'
                bg_color = '#f8f9fa'

            html_output = f"""
            <style>
                .job-card {{
                    font-family: 'Inter', sans-serif;
                    border: 1px solid {status_color};
                    border-left: 5px solid {status_color};
                    border-radius: 8px;
                    padding: 20px;
                    margin: 15px 0;
                    background-color: {bg_color};
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                }}
                .job-card h4 {{ color: {status_color}; margin-top: 0; }}
                .job-detail {{ margin-bottom: 5px; font-size: 0.95em; }}
                .job-label {{ font-weight: 600; color: #555; display: inline-block; width: 150px; }}
            </style>
            <div class="job-card">
                <h4>Job Tracker - ID: {html.escape(job_id)}</h4>
                <div class="job-detail"><span class="job-label">Status:</span> <strong><span style="color: {status_color};">{status}</span></strong></div>
                <div class="job-detail"><span class="job-label">Device:</span> {html.escape(data.get('Device') or data.get('backend', 'N/A'))}</div>
                <div class="job-detail"><span class="job-label">Shots:</span> {data.get('Shots') or data.get('shots', 'N/A')}</div>
                <div class="job-detail"><span class="job-label">Submitted:</span> {data.get('JobRasied') or data.get('submitted_at', 'N/A').split('.')[0]}</div>
                <div class="job-detail"><span class="job-label">Completed:</span> {data.get('JobCompletion') or data.get('completed_at', 'N/A').split('.')[0]}</div>
                <div class="job-detail"><span class="job-label">Mode:</span> {data.get('mode', 'N/A').upper()}</div>
            </div>
            """
            display(HTML(html_output))

    @cell_magic
    def job_optimizer(self, line: str, cell: str):
        """
        %%job_optimizer
        qubits_required=5
        gates_required=100
        circuit_depth=50
        priority=speed
        
        Optimizes device selection based on circuit requirements.
        """
        try:
            # Parse cell content into a dictionary
            params = {}
            for line in cell.split('\n'):
                if '=' in line and line.strip():
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Type conversion based on expected types in JobOptimizerRequest
                    if key in ['qubits_required', 'gates_required', 'circuit_depth']:
                        params[key] = int(value)
                    elif key == 'priority':
                        params[key] = value.lower()
                    else:
                        params[key] = value
            
            # Default values if not provided (matching JobOptimizerRequest defaults)
            payload = {
                "qubits_required": params.get('qubits_required', 5),
                "gates_required": params.get('gates_required', 0),
                "circuit_depth": params.get('circuit_depth', 0),
                "priority": params.get('priority', 'balanced')
            }
            
            # Call the endpoint: /api/job-optimizer
            data = self._make_request(f"/job-optimizer", method='POST', data=payload)
            
            if data:
                # Format the optimization results
                recommended = data['recommended_device']
                
                html_output = f"""
                <style>
                    .opt-card {{
                        font-family: 'Inter', sans-serif;
                        border: 2px solid #28a745;
                        border-radius: 12px;
                        padding: 25px;
                        margin: 15px 0;
                        background-color: #f7fff7;
                        box-shadow: 0 6px 15px rgba(40, 167, 69, 0.1);
                    }}
                    .opt-card h4 {{ color: #28a745; margin-top: 0; border-bottom: 2px solid #28a745; padding-bottom: 10px; }}
                    .rec-device {{ font-size: 1.5em; font-weight: bold; color: #007bff; margin-top: 10px; }}
                    .explanation {{ background-color: white; padding: 10px; border-radius: 6px; border: 1px solid #ddd; margin-top: 10px; font-size: 0.95em; }}
                    .alternatives-list {{ margin-top: 20px; border-top: 1px dashed #ddd; padding-top: 15px; }}
                    .alt-item {{ margin-bottom: 10px; padding: 10px; background-color: #f0f8ff; border-radius: 4px; border-left: 3px solid #00bcd4; font-size: 0.9em; }}
                </style>
                <div class="opt-card">
                    <h4>⚙️ Job Optimization Results</h4>
                    <p style="font-weight: 500;">Job Requirements: {payload['qubits_required']} Qubits, {payload['circuit_depth']} Depth, Priority: {payload['priority'].upper()}</p>
                    
                    <div style="text-align: center;">
                        <span style="font-size: 1.1em; color: #555;">Best Recommended Device:</span>
                        <div class="rec-device">{html.escape(recommended)}</div>
                    </div>

                    <div class="explanation">
                        <p><strong>Rationale:</strong> {html.escape(data['explanation'])}</p>
                        <p><strong>Predicted Success:</strong> {(data['success_probability'] * 100):.2f}%</p>
                        <p><strong>Est. Wait Time:</strong> {data['estimated_wait_time']} minutes</p>
                        <p><strong>Est. Cost:</strong> ${data['estimated_cost']:.4f}</p>
                    </div>

                    <div class="alternatives-list">
                        <p style="font-weight: bold; color: #007bff;">Top Alternatives:</p>
                        {''.join([
                            f"""
                            <div class="alt-item">
                                <strong>{html.escape(alt['device_name'])}</strong> 
                                (Success: {(alt['success_probability'] * 100):.1f}%, Wait: {alt['estimated_wait_time']} min)
                                <span style="display: block; font-style: italic; color: #6c757d;">{html.escape(alt['reason'])}</span>
                            </div>
                            """
                            for alt in data['alternatives']
                        ])}
                    </div>
                </div>
                """
                display(HTML(html_output))

        except Exception as e:
            display(HTML(f"<div style='color: red; padding: 10px; border: 1px solid red; border-radius: 5px;'>Parsing Error or Invalid Input: {e}. Please check the format in the cell.</div>"))


def load_ipython_extension(ipython):
    """Function to call to register the magic commands."""
    ipython.register_magics(QuantumAnalyticsMagics)
    setup_info = """
<div style="border: 2px solid #007bff; padding: 15px; border-radius: 5px; background-color: #f0f8ff; margin: 10px 0;">
    <h3 style="color: #007bff; margin-top: 0;">✅ Quantum Analytics Magics Loaded</h3>
    <p>Run <code>%analytics_setup</code> first, then use the following commands:</p>
    <ul>
        <li><code>%device_summary</code>: Show performance metrics for all devices.</li>
        <li><code>%device_details &lt;device_name&gt;</code>: Show full details for one device.</li>
        <li><code>%device_score &lt;device_name&gt;</code>: Show ML score and success probability.</li>
        <li><code>%job_tracker &lt;job_id&gt;</code>: Check job status and details.</li>
        <li><code>%%job_optimizer</code>: Use requirements in the cell to find the best device.</li>
    </ul>
</div>
"""
    display(HTML(setup_info))

def unload_ipython_extension(ipython):
    """Function to call to unregister the magic commands."""
    # Simple check to prevent errors on unload
    if 'QuantumAnalyticsMagics' in ipython.magics_manager.magics['line']:
         del ipython.magics_manager.magics['line']['QuantumAnalyticsMagics']
    print("Quantum Analytics Magics unloaded.")

# Auto-load when imported in an IPython environment
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython:
        load_ipython_extension(ipython)
except:
    pass
