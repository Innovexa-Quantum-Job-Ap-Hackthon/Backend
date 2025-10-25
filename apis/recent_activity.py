from datetime import datetime
from sqlalchemy.orm import Session
from Database.DBConfiguration.database import SessionLocal
from  Database.DBmodels.database_models import JobLogs

def log_job_activity(job_id: str, user_id: str, status: str, backend_name: str):
    """Log job activity to the database"""
    db = SessionLocal()
    try:
        # Check if job log already exists
        existing_log = db.query(JobLogs).filter(JobLogs.jobId == job_id).first()

        if existing_log:
            # Update existing log
            existing_log.Status = status
            existing_log.Device = backend_name
            existing_log.JobCompletion = datetime.utcnow()
        else:
            # Create new log entry
            job_log = JobLogs(
                jobId=job_id,
                JobRasied=datetime.utcnow(),
                Device=backend_name,
                Status=status,
                Shots=1024,  # Default value, can be updated later
                JobCompletion=datetime.utcnow(),
                user_id=int(user_id) if user_id and user_id.isdigit() else None
            )
            db.add(job_log)

        db.commit()
    except Exception as e:
        print(f"Error logging job activity: {e}")
        db.rollback()
    finally:
        db.close()

def get_recent_activity(limit: int = 50):
    """Get recent job activity from the database"""
    db = SessionLocal()
    try:
        # Get recent job logs ordered by creation time (most recent first)
        recent_logs = db.query(JobLogs).order_by(JobLogs.JobRasied.desc()).limit(limit).all()

        # Convert to the expected format for the frontend (matching LiveJobsSection expected props)
        activity_list = []
        for log in recent_logs:
            # Ensure proper date formatting to avoid "Invalid Date" errors
            try:
                timestamp = log.JobRasied.isoformat() if log.JobRasied else datetime.utcnow().isoformat()
            except Exception as e:
                print(f"Error formatting timestamp for job {log.jobId}: {e}")
                timestamp = datetime.utcnow().isoformat()

            activity_list.append({
                "job_id": log.jobId,  # Use job_id consistently
                "id": log.jobId,      # Also include id for compatibility
                "user_id": str(log.user_id) if log.user_id else "anonymous",
                "status": log.Status,
                "backend": log.Device,
                "timestamp": timestamp,  # Use timestamp field expected by frontend
                "submitted_at": timestamp,  # Keep submitted_at for compatibility
                "completed_at": log.JobCompletion.isoformat() if log.JobCompletion else None,
                "shots": getattr(log, "Shots", None),
                "results": None,  # Could be extended to fetch results if stored
                "actual_wait_time": None,  # Could be calculated if timestamps available
                "performance_score": None,  # Could be added if stored/calculated
                "estimated_cost_on_submission": None,
                "estimated_carbon_on_submission": None,
                "mode": "database",
                "activity": f"Job {log.Status.lower()} on {log.Device}"  # Add activity description
            })

        return activity_list
    except Exception as e:
        print(f"Error fetching recent activity: {e}")
        return []
    finally:
        db.close()

def get_live_jobs(user_id: str = None, limit: int = 50):
    """Get live jobs (running, queued, recently completed) from the database"""
    db = SessionLocal()
    try:
        # Get jobs that are likely still "live" - running, queued, or recently completed
        from datetime import timedelta
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)  # Last 24 hours

        # Build query without project_id to avoid column not found error
        query = db.query(
            JobLogs.jobId,
            JobLogs.JobRasied,
            JobLogs.Device,
            JobLogs.Status,
            JobLogs.Shots,
            JobLogs.JobCompletion,
            JobLogs.user_id
        ).filter(JobLogs.JobRasied >= recent_cutoff)

        # If user_id is provided, filter by user
        if user_id:
            query = query.filter(JobLogs.user_id == int(user_id) if user_id.isdigit() else None)

        # Apply ordering and limit after all filters
        query = query.order_by(JobLogs.JobRasied.desc()).limit(limit)

        live_logs = query.all()

        # Convert to the format expected by LiveJobsSection
        live_jobs = []
        for log in live_logs:
            # Unpack tuple: (jobId, JobRasied, Device, Status, Shots, JobCompletion, user_id)
            job_id, job_raised, device, status, shots, job_completion, user_id = log

            # Map status to expected format
            status_lower = status.lower() if status else 'running'
            if status_lower in ['running', 'queued', 'pending']:
                mapped_status = status_lower
            elif status_lower in ['done', 'completed', 'success']:
                mapped_status = 'completed'
            elif status_lower in ['error', 'failed', 'cancelled']:
                mapped_status = 'failed'
            else:
                mapped_status = 'running'  # Default to running for unknown statuses

            # Ensure proper date formatting to avoid "Invalid Date" errors
            try:
                timestamp = job_raised.isoformat() if job_raised else datetime.utcnow().isoformat()
            except Exception as e:
                print(f"Error formatting timestamp for live job {job_id}: {e}")
                timestamp = datetime.utcnow().isoformat()

            live_jobs.append({
                "job_id": job_id,  # Ensure job_id is included
                "id": job_id,      # Also include id for compatibility
                "backend": device,
                "status": mapped_status,
                "timestamp": timestamp,  # Use timestamp field expected by frontend
                "user_id": str(user_id) if user_id else "anonymous",
                "activity": f"Job {mapped_status} on {device}",
                "shots": shots
            })

        return live_jobs
    except Exception as e:
        print(f"Error fetching live jobs: {e}")
        return []
    finally:
        db.close()
