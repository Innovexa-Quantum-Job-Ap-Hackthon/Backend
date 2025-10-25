from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import json
import secrets
import logging

from Database.DBConfiguration.database import get_db
from Authorization.auth import get_current_user
import Database.DBmodels.schemas as schemas
import Database.DBmodels.database_models as models
from collaboration.collaboration import CollaborationService
from services.redis_cache import RedisCache

router = APIRouter(prefix="/collaboration", tags=["Collaboration"])

# Initialize Redis cache
redis_cache = RedisCache()
CACHE_KEY_PREFIX = "team_jobs"
CACHE_EXPIRY = 60  # 5 minutes in seconds

@router.post("/teams", response_model=schemas.TeamOut)
async def create_team(
    team: schemas.TeamCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new team"""
    service = CollaborationService(db)
    db_team = service.create_team(team, current_user.id)
    # Create TeamOut with member_count
    team_dict = {
        "id": db_team.id,
        "name": db_team.name,
        "description": db_team.description,
        "created_at": db_team.created_at,
        "created_by_id": db_team.created_by_id,
        "member_count": 1  # New team has 1 member (creator)
    }
    return schemas.TeamOut(**team_dict)

@router.post("/teams/{team_id}/invite")
async def invite_to_team(
    team_id: int,
    invitation: schemas.TeamInvitationCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Invite a user to a team"""
    service = CollaborationService(db)
    try:
        invitation_result = service.invite_to_team(team_id, invitation.email, current_user.id)
        # After invitation is created, create a notification for the invited user
        from Database.DBmodels.database_models import TeamInvitationNotification
        notification = TeamInvitationNotification(
            invitation_id=invitation_result.id,
            user_email=invitation.email,
            team_id=team_id,
            created_at=datetime.utcnow(),
            status="unread"
        )
        db.add(notification)
        db.commit()

        # Invalidate cache for this team's jobs immediately after invitation
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔄 [INVITE] Starting cache invalidation for team {team_id}")
        try:
            cache_key = f"{CACHE_KEY_PREFIX}:{team_id}:30"  # Default 30 days
            logger.info(f"🔄 [INVITE] Cache key: {cache_key}")
            cache_delete_result = redis_cache.delete(cache_key)
            logger.info(f"✅ [INVITE] Cache deletion result: {cache_delete_result} for team {team_id}")
        except Exception as e:
            logger.error(f"❌ [INVITE] Failed to invalidate cache after invitation: {e}")
            logger.error(f"❌ [INVITE] Exception type: {type(e).__name__}")

        return {"message": "Invitation sent", "token": invitation_result.token}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the error for debugging
        import logging
        logging.error(f"Error inviting user to team: {e}")
        # Fix: result is a bool from redis_cache.delete, so cannot access result.token here
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/invitations/{token}/accept")
async def accept_invitation(
    token: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Accept a team invitation"""
    logger = logging.getLogger(__name__)
    logger.info(f"Accepting invitation with token: {token} for user: {current_user.email} (ID: {current_user.id})")

    service = CollaborationService(db)
    try:
        member = service.accept_invitation(token, current_user.id)

        # Invalidate cache for this team's jobs after successful acceptance
        try:
            cache_key = f"{CACHE_KEY_PREFIX}:{member.team_id}:30"  # Default 30 days
            redis_cache.delete(cache_key)
            logger.info(f"Cache invalidated for team {member.team_id} jobs after invitation acceptance")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache after invitation acceptance: {e}")

        logger.info(f"Successfully accepted invitation for user {current_user.email} to join team {member.team_id}")
        return {"message": "Joined team successfully", "team_id": member.team_id}
    except ValueError as e:
        logger.error(f"Failed to accept invitation for user {current_user.email}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error accepting invitation for user {current_user.email}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/teams/{team_id}/jobs")
async def get_team_jobs(
    team_id: int,
    days: int = 30,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all jobs from team members grouped by user with Redis caching"""
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"🔍 API: Fetching jobs for team {team_id}, user {current_user.email}, days={days}")

    membership = db.query(models.TeamMember).filter(
        models.TeamMember.team_id == team_id,
        models.TeamMember.user_id == current_user.id
    ).first()

    if not membership:
        logger.warning(f"❌ API: User {current_user.email} is not a member of team {team_id}")
        raise HTTPException(status_code=403, detail="Not a member of this team")

    # Create cache key for this team's jobs
    cache_key = f"{CACHE_KEY_PREFIX}:{team_id}:{days}"

    # Try to get cached data first
    cached_data = redis_cache.get(cache_key)
    if cached_data:
        jobs_data = json.loads(cached_data)
        logger.info(f"✅ API: Returning cached jobs for team {team_id}")
        return jobs_data

    # Fetch fresh data if not cached
    logger.info(f"🔄 API: Fetching fresh jobs for team {team_id}")
    service = CollaborationService(db)
    jobs_data = service.get_team_jobs(team_id, days)

    logger.info(f"📊 API: Service returned jobs grouped by {len(jobs_data.get('users', {}))} users with {len(jobs_data.get('all_jobs', []))} total jobs for team {team_id}")

    # Validate job data structure
    if not isinstance(jobs_data, dict) or 'users' not in jobs_data or 'all_jobs' not in jobs_data:
        logger.error(f"❌ API: Invalid jobs data structure: {jobs_data}")
        raise HTTPException(status_code=500, detail="Invalid jobs data structure")

    # Validate job objects in the grouped structure
    for user_id, user_data in jobs_data['users'].items():
        for i, job in enumerate(user_data['jobs']):
            job_id = job.get('jobId')
            if not job_id or not isinstance(job_id, str):
                logger.warning(f"⚠️ API: Job {i} for user {user_id} with missing or invalid jobId detected: {job}")
                job['jobId'] = f"unknown-{secrets.token_hex(4)}"

    # Also validate all_jobs array
    for i, job in enumerate(jobs_data['all_jobs']):
        job_id = job.get('jobId')
        if not job_id or not isinstance(job_id, str):
            logger.warning(f"⚠️ API: Job {i} in all_jobs with missing or invalid jobId detected: {job}")
            job['jobId'] = f"unknown-{secrets.token_hex(4)}"

    logger.info(f"✅ API: Validated jobs data for team {team_id}")

    # Cache the validated jobs data
    try:
        redis_cache.set(cache_key, json.dumps(jobs_data, default=str), ex=CACHE_EXPIRY)
        logger.info(f"💾 API: Cached jobs data for team {team_id} (expires in {CACHE_EXPIRY}s)")
    except Exception as e:
        logger.error(f"❌ API: Failed to cache jobs data: {e}")

    return jobs_data

@router.get("/user/teams", response_model=List[schemas.TeamOut])
async def get_user_teams(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all teams the user belongs to"""
    service = CollaborationService(db)
    teams = service.get_user_teams(current_user.id)

    result = []
    for team in teams:
        # Get member count
        member_count = db.query(models.TeamMember).filter(
            models.TeamMember.team_id == team.id
        ).count()
        # Create TeamOut with member_count
        team_dict = {
            "id": team.id,
            "name": team.name,
            "description": team.description,
            "created_at": team.created_at,
            "created_by_id": team.created_by_id,
            "member_count": member_count
        }
        result.append(schemas.TeamOut(**team_dict))

    return result



@router.get("/teams/{team_id}/members", response_model=List[schemas.TeamMemberOut])
async def get_team_members(
    team_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all members of a team"""
    # Check if user is team member
    membership = db.query(models.TeamMember).filter(
        models.TeamMember.team_id == team_id,
        models.TeamMember.user_id == current_user.id
    ).first()

    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this team")

    # Get team members with user details using join
    members = db.query(models.TeamMember).join(models.User).filter(
        models.TeamMember.team_id == team_id
    ).all()

    # Convert to the expected format with user details
    result = []
    for member in members:
        member_dict = {
            "id": member.id,
            "user_id": member.user_id,
            "team_id": member.team_id,
            "role": member.role,
            "joined_at": member.joined_at,
            "user": {
                "id": member.user.id,
                "email": member.user.email,
                "name": member.user.name,
                "created_at": member.user.created_at
            }
        }
        result.append(member_dict)

    return result
