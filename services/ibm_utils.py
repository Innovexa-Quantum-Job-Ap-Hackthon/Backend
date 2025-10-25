import time
import logging
from typing import Optional
from qiskit_ibm_runtime import QiskitRuntimeService
from sqlalchemy.orm import Session
import Database.DBmodels.database_models as models

logger = logging.getLogger(__name__)

def get_ibm_service_for_user(current_user: models.User, skip_validation: bool = False):
    """Get IBM Quantum service for a user with caching."""
    from main import _QISKIT_AVAILABLE, service_cache, SERVICE_CACHE_TIMEOUT

    if not _QISKIT_AVAILABLE:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Qiskit IBM Runtime not available on server.")

    if not current_user.ibm_api_key_encrypted or not current_user.ibm_instance_key_encrypted:
        logger.warning(f"IBM keys missing for user {current_user.email}")
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="IBM API key or instance key not set for user.")

    cache_key = f"{current_user.id}_{current_user.ibm_api_key_encrypted}_{current_user.ibm_instance_key_encrypted}"
    current_time = time.time()

    if cache_key in service_cache:
        cached_entry = service_cache[cache_key]
        if current_time - cached_entry['timestamp'] < SERVICE_CACHE_TIMEOUT:
            logger.info(f"Using cached IBM service for user {current_user.email}")
            return cached_entry['service']
        else:
            del service_cache[cache_key]

    try:
        logger.info(f"Initializing IBM service for user {current_user.email} with database keys")
        user_service = QiskitRuntimeService(
            channel="ibm_cloud",
            token=current_user.ibm_api_key_encrypted,
            instance=current_user.ibm_instance_key_encrypted
        )
        if not skip_validation:
            user_service.backends()
            logger.info(f"Successfully initialized and validated IBM service for user {current_user.email}")
        else:
            logger.info(f"Initialized IBM service for user {current_user.email} without validation (skip_validation=True)")

        service_cache[cache_key] = {
            'service': user_service,
            'timestamp': current_time
        }

        return user_service
    except Exception as e:
        logger.error(f"Failed to initialize IBM Quantum service for user {current_user.email}: {e}")
        if "401" in str(e) or "Unauthorized" in str(e) or "Invalid API key" in str(e):
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="Invalid IBM API key or instance. Please check your credentials.")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to connect to IBM Quantum service: {e}")

def fetch_ibm_jobs_with_timeout(user_service, user_id):
    """Fetch IBM jobs with timeout handling."""
    from main import extract_completion_time_from_job

    try:
        recent_jobs = user_service.jobs(limit=50)
        ibm_jobs = []

        for job in recent_jobs:
            try:
                job_status = job.status()
                status_name = job_status.name if hasattr(job_status, 'name') else str(job_status)
                mapped_status = status_name.lower()
                if mapped_status == 'done': mapped_status = 'completed'

                backend_name = getattr(job.backend(), 'name', 'unknown') if hasattr(job, 'backend') and callable(job.backend) else 'unknown'

                meta = {}
                raw_metadata = getattr(job, 'metadata', {})
                meta = raw_metadata() if callable(raw_metadata) else raw_metadata

                ibm_jobs.append({
                    "job_id": job.job_id(),
                    "id": job.job_id(),
                    "backend": backend_name,
                    "status": mapped_status,
                    "timestamp": job.creation_date.isoformat(),
                    "user_id": str(user_id),
                    "activity": f"Job {mapped_status} on {backend_name}",
                    "shots": meta.get("shots", 1024),
                    "submitted_at": job.creation_date.isoformat(),
                    "completed_at": extract_completion_time_from_job(job, meta),
                    "mode": "ibm_quantum"
                })
            except Exception as e:
                logger.warning(f"Error processing IBM job: {e}")
                continue

        return ibm_jobs
    except Exception as e:
        logger.error(f"Error in fetch_ibm_jobs_with_timeout: {e}")
        return []
