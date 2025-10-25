from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import asyncio
import logging
from typing import List, Dict, Any, Optional

from Database.DBConfiguration.database import SessionLocal
from Authorization.auth import get_current_user
import Database.DBmodels.database_models as models
from services.ibm_utils import get_ibm_service_for_user, fetch_ibm_jobs_with_timeout

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/api/analytics/live")
async def get_live_analytics(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        logger.debug(f"📊 Running live analytics for user {current_user.email}")

        # Get live jobs data
        from apis import recent_activity
        live_jobs_data = recent_activity.get_live_jobs(str(current_user.id) if current_user else None)

        # Get IBM jobs if user has credentials
        ibm_jobs = []
        if current_user and current_user.ibm_api_key_encrypted and current_user.ibm_instance_key_encrypted:
            try:
                user_service = await asyncio.to_thread(get_ibm_service_for_user, current_user, True)
                if user_service:
                    ibm_jobs = await asyncio.to_thread(fetch_ibm_jobs_with_timeout, user_service, current_user.id)
            except Exception as e:
                logger.warning(f"Error fetching IBM jobs for analytics: {e}")

        # Combine and deduplicate jobs
        all_jobs = live_jobs_data + ibm_jobs
        seen_job_ids = set()
        unique_jobs = []

        for job in all_jobs:
            job_id = job.get('job_id') or job.get('id')
            if job_id and job_id not in seen_job_ids:
                seen_job_ids.add(job_id)
                unique_jobs.append(job)

        # Sort by timestamp (most recent first)
        unique_jobs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Calculate live analytics
        total_jobs = len(unique_jobs)

        # Status distribution
        status_counts = {
            'completed': 0,
            'running': 0,
            'queued': 0,
            'failed': 0,
            'error': 0
        }

        # Backend distribution
        backend_counts = {}
        recent_activity = []
        success_rate_data = []

        for job in unique_jobs[:100]:  # Analyze last 100 jobs for performance
            status = job.get('status', '').lower()
            if status in status_counts:
                status_counts[status] += 1
            elif status in ['done', 'completed']:
                status_counts['completed'] += 1
            elif status in ['error', 'failed']:
                status_counts['failed'] += 1

            backend = job.get('backend', 'unknown')
            backend_counts[backend] = backend_counts.get(backend, 0) + 1

            # Recent activity (last 24 hours)
            timestamp = job.get('timestamp', '')
            if timestamp:
                try:
                    job_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if job_time > datetime.now() - timedelta(hours=24):
                        recent_activity.append({
                            'timestamp': timestamp,
                            'status': status,
                            'backend': backend,
                            'job_id': job.get('job_id', job.get('id', 'unknown'))
                        })
                except:
                    pass

        # Calculate success rate
        completed_jobs = status_counts['completed']
        total_completed_jobs = completed_jobs + status_counts['failed'] + status_counts['error']
        success_rate = (completed_jobs / total_completed_jobs * 100) if total_completed_jobs > 0 else 0

        # Calculate performance improvement (comparing recent vs older jobs)
        recent_jobs = [j for j in unique_jobs[:50]]  # Last 50 jobs
        older_jobs = [j for j in unique_jobs[50:100]] if len(unique_jobs) > 50 else []

        recent_success_rate = 0
        older_success_rate = 0

        if recent_jobs:
            recent_completed = len([j for j in recent_jobs if j.get('status', '').lower() in ['completed', 'done']])
            recent_failed = len([j for j in recent_jobs if j.get('status', '').lower() in ['failed', 'error']])
            if recent_completed + recent_failed > 0:
                recent_success_rate = (recent_completed / (recent_completed + recent_failed)) * 100

        if older_jobs:
            older_completed = len([j for j in older_jobs if j.get('status', '').lower() in ['completed', 'done']])
            older_failed = len([j for j in older_jobs if j.get('status', '').lower() in ['failed', 'error']])
            if older_completed + older_failed > 0:
                older_success_rate = (older_completed / (older_completed + older_failed)) * 100

        performance_improvement = recent_success_rate - older_success_rate if older_jobs else recent_success_rate

        # Prepare time-series data for charts (last 7 days)
        time_series_data = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            day_jobs = [j for j in unique_jobs if j.get('timestamp', '').startswith(date)]

            time_series_data.append({
                'date': date,
                'totalJobs': len(day_jobs),
                'completed': len([j for j in day_jobs if j.get('status', '').lower() in ['completed', 'done']]),
                'running': len([j for j in day_jobs if j.get('status', '').lower() == 'running']),
                'failed': len([j for j in day_jobs if j.get('status', '').lower() in ['failed', 'error']]),
                'queued': len([j for j in day_jobs if j.get('status', '').lower() == 'queued']),
                'avgPerformance': success_rate,  # Use actual success rate
                'avgCost': 0.15,  # Could be enhanced with actual cost data
                'carbonFootprint': 0.005  # Could be enhanced with actual carbon data
            })

        # Sort time series by date
        time_series_data.sort(key=lambda x: x['date'])

        result = {
            'data': time_series_data,
            'overall_stats': {
                'total_jobs': total_jobs,
                'success_rate': round(success_rate, 1),
                'active_jobs': status_counts['running'] + status_counts['queued'],
                'completed_today': len([j for j in unique_jobs if j.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d')) and j.get('status', '').lower() in ['completed', 'done']]),
                'performance_improvement': round(performance_improvement, 1)
            },
            'device_performance': {
                'backend_distribution': backend_counts,
                'status_distribution': status_counts
            },
            'has_user_data': total_jobs > 0,
            'recent_activity': recent_activity[:20]  # Last 20 activities
        }

        return {"status": "success", "data": result}

    except Exception as e:
        logger.exception("❌ Error in /api/analytics/live")
        return {"status": "error", "message": f"Internal error: {str(e)}"}
