import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_records=1000):
    """Generate synthetic IBM Quantum job data for training ML models."""

    # Define device names
    devices = ['ibmq_qasm_simulator', 'ibmq_lagos', 'ibmq_perth', 'ibmq_jakarta', 'ibmq_manila']

    # Generate base data
    np.random.seed(42)  # For reproducibility

    data = {
        'job_id': ['job_' + str(i).zfill(6) for i in range(num_records)],
        'backend': np.random.choice(devices, num_records),
        'status': np.random.choice(['COMPLETED', 'ERROR', 'CANCELLED', 'DONE'], num_records, p=[0.7, 0.15, 0.1, 0.05]),
        'num_qubits': np.random.randint(1, 128, num_records),
        'pending_jobs': np.random.randint(0, 50, num_records),
        'error_rate': np.random.uniform(0.001, 0.05, num_records),
        'readout_error': np.random.uniform(0.005, 0.1, num_records),
        't1_time': np.random.uniform(50, 200, num_records),
        't2_time': np.random.uniform(30, 150, num_records),
        'circuit_depth': np.random.randint(10, 500, num_records),
        'wait_time_minutes': np.random.exponential(5, num_records),  # Exponential distribution for wait times
        'run_time_s': np.random.exponential(300, num_records),  # Runtime in seconds
        'avg_runtime_per_job_minutes': np.random.uniform(2, 15, num_records),
        'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 24*30)) for _ in range(num_records)]  # Last 30 days
    }

    df = pd.DataFrame(data)

    # Add some realistic correlations
    # Devices with more qubits tend to have higher error rates
    for device in devices:
        mask = df['backend'] == device
        if device == 'ibmq_qasm_simulator':
            # Simulator has very low error rates
            df.loc[mask, 'error_rate'] = np.random.uniform(0.0001, 0.001, mask.sum())
            df.loc[mask, 'readout_error'] = np.random.uniform(0.0001, 0.001, mask.sum())
        else:
            # Real devices have higher error rates
            df.loc[mask, 'error_rate'] = np.random.uniform(0.01, 0.05, mask.sum())
            df.loc[mask, 'readout_error'] = np.random.uniform(0.02, 0.1, mask.sum())

    # Wait time correlates with pending jobs
    df['wait_time_minutes'] = df['pending_jobs'] * np.random.uniform(0.5, 2, num_records) + np.random.exponential(2, num_records)

    # Add some noise to make it more realistic
    df['wait_time_minutes'] = df['wait_time_minutes'] + np.random.normal(0, 2, num_records)

    return df

def save_data(df, filename):
    """Save the generated data to CSV."""
    df.to_csv(filename, index=False)
    print(f"Generated {len(df)} records and saved to {filename}")

if __name__ == "__main__":
    # Generate sample data
    print("Generating synthetic IBM Quantum job data...")
    df = generate_synthetic_data(2000)  # Generate 2000 records

    # Save to the expected location
    csv_path = "synthetic_ibm_jobs_full.csv"
    save_data(df, csv_path)

    # Also save a copy with .csv extension for compatibility
    csv_path_with_extension = "synthetic_ibm_jobs_full.csv.csv"
    save_data(df, csv_path_with_extension)

    print("\nData generation complete!")
    print(f"Records generated: {len(df)}")
    print(f"Devices included: {list(df['backend'].unique())}")
    success_rate = df['status'].isin(['COMPLETED', 'DONE']).mean()
    print(f"Success rate: {success_rate".3f"}")
    avg_wait = df['wait_time_minutes'].mean()
    print(f"Average wait time: {avg_wait".2f"} minutes")
