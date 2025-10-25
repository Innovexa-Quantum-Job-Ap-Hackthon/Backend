# Quantum Job Optimizer API

A FastAPI-based backend server for quantum job optimization and compilation.

## Issues Fixed

✅ **Connection Refused Error**: Added missing server startup code to `main.py`
✅ **Missing Model File**: The `qsvm_fitness_model.pkl` issue is handled gracefully with fallback to classical ML models

## Quick Start

### Option 1: Using the Startup Script (Recommended)

```bash
cd Backend-Innovexa
python start_server.py
```

### Option 2: Direct Command

```bash
cd Backend-Innovexa
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Using main.py directly

```bash
cd Backend-Innovexa
python main.py
```

## Server Endpoints

Once the server is running, you'll have access to:

- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Admin Interface**: http://localhost:8000/admin
- **Circuit Compiler**: http://localhost:8000/api/circuit-compiler

## Jupyter Integration

The Jupyter magic command `%%qcompile` will now work properly as it connects to:
- http://localhost:8000/api/circuit-compiler

## Troubleshooting

### Port 8000 Already in Use
If port 8000 is busy, use a different port:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Missing Dependencies
Install requirements:
```bash
pip install -r requirements.txt
```

### Database Issues
The server will automatically create the SQLite database (`quantum.db`) on first run.

## Features

- Quantum circuit compilation and optimization
- IBM Quantum device integration
- Real-time job monitoring
- ML-based device recommendations
- Admin interface for database management
- WebSocket support for notebook collaboration
