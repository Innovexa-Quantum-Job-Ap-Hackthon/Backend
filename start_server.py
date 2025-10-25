#!/usr/bin/env python3
"""
Startup script for the Quantum Job Optimizer API server.
This script starts the FastAPI server on port 8000.
"""

import os
import sys
import subprocess
import time

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import qiskit
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting Quantum Job Optimizer API server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📍 API documentation at: http://localhost:8000/docs")
    print("📍 Admin interface at: http://localhost:8000/admin")
    print("Press Ctrl+C to stop the server\n")

    try:
        # Start the server using uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], cwd=os.path.dirname(os.path.abspath(__file__)))
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    # Change to the Backend-Innovexa directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Start the server
    start_server()
