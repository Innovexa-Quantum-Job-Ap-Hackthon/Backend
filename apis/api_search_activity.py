from fastapi import APIRouter, Request
from . import recent_activity

router = APIRouter()

@router.post("/api/log-search")
async def log_search_activity(request: Request):
    """
    Logs a job tracer search action to the recent activity log.
    Expects JSON body with 'user_id' and 'search_query' fields.
    """
    data = await request.json()
    user_id = data.get("user_id")
    search_query = data.get("search_query", "")
    if not user_id:
        return {"status": "error", "message": "user_id is required"}

    # Log the search action as a recent activity entry
    recent_activity.log_job_activity(
        job_id="search_action",
        user_id=user_id,
        status=f"Search performed: {search_query}",
        backend_name="job_tracer"
    )
    return {"status": "success", "message": "Search activity logged"}
