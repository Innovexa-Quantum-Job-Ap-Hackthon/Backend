from fastapi import APIRouter, Query
from . import recent_activity

router = APIRouter()

@router.get("/api/recent-activity")
async def recent_activity():
    """
    Returns the recent activity log of job status updates.
    """
    return {"status": "success", "data": recent_activity.get_recent_activity()}

# Removed conflicting /api/live-jobs endpoint
# The main IBM Quantum endpoint in main.py will be used instead
