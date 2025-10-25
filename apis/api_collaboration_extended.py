# api_collaboration_extended.py - Extended collaboration API endpoints
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from typing import List, Optional, Dict, Any
from datetime import datetime

from Database.DBConfiguration.database import get_db
from Database.DBmodels import extended_database_models as models
from Database.DBmodels import extended_schemas as schemas
from Authorization.auth import get_current_user

router = APIRouter()

# -------------------------------
# Notification Endpoints
# -------------------------------
@router.get("/notifications", response_model=List[schemas.NotificationOut])
async def get_notifications(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    unread_only: bool = Query(False),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user notifications"""
    query = db.query(models.Notification).filter(models.Notification.user_id == current_user.id)

    if unread_only:
        query = query.filter(models.Notification.is_read == False)

    notifications = query.order_by(desc(models.Notification.created_at)).offset(skip).limit(limit).all()
    return notifications

@router.put("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a notification as read"""
    notification = db.query(models.Notification).filter(
        and_(models.Notification.id == notification_id, models.Notification.user_id == current_user.id)
    ).first()

    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    notification.is_read = True
    notification.read_at = datetime.utcnow()
    db.commit()

    return {"message": "Notification marked as read"}

@router.put("/notifications/mark-all-read")
async def mark_all_notifications_read(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark all user notifications as read"""
    db.query(models.Notification).filter(
        and_(models.Notification.user_id == current_user.id, models.Notification.is_read == False)
    ).update({"is_read": True, "read_at": datetime.utcnow()})

    db.commit()
    return {"message": "All notifications marked as read"}

# -------------------------------
# Activity Feed Endpoints
# -------------------------------
@router.get("/activity", response_model=List[schemas.ActivityLogOut])
async def get_activity_feed(
    team_id: Optional[int] = Query(None),
    user_id: Optional[int] = Query(None),
    resource_type: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get activity feed with filtering"""
    query = db.query(models.ActivityLog)

    if team_id:
        query = query.filter(models.ActivityLog.team_id == team_id)
        # Check team membership
        team_member = db.query(models.TeamMember).filter(
            and_(models.TeamMember.team_id == team_id, models.TeamMember.user_id == current_user.id)
        ).first()
        if not team_member:
            raise HTTPException(status_code=403, detail="Not authorized to view activity for this team")

    if user_id:
        query = query.filter(models.ActivityLog.user_id == user_id)
    if resource_type:
        query = query.filter(models.ActivityLog.resource_type == resource_type)

    activities = query.order_by(desc(models.ActivityLog.created_at)).offset(skip).limit(limit).all()
    return activities

# -------------------------------
# Dashboard Endpoints
# -------------------------------
@router.get("/dashboard/stats", response_model=schemas.DashboardStats)
async def get_dashboard_stats(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics for the current user"""
    # Get user's teams
    user_teams = db.query(models.TeamMember).filter(models.TeamMember.user_id == current_user.id).all()
    team_ids = [tm.team_id for tm in user_teams]

    if not team_ids:
        return schemas.DashboardStats(
            total_teams=0,
            total_projects=0,
            total_tasks=0,
            completed_tasks=0,
            pending_invitations=0,
            unread_notifications=0
        )

    # Calculate stats
    total_teams = len(team_ids)
    total_projects = db.query(models.Project).filter(models.Project.team_id.in_(team_ids)).count()
    total_tasks = 0  # Removed task counting since tasks are removed
    completed_tasks = 0  # Removed task counting since tasks are removed

    # Pending invitations
    pending_invitations = db.query(models.TeamInvitation).filter(
        and_(models.TeamInvitation.email == current_user.email, models.TeamInvitation.status == "pending")
    ).count()

    # Unread notifications
    unread_notifications = db.query(models.Notification).filter(
        and_(models.Notification.user_id == current_user.id, models.Notification.is_read == False)
    ).count()

    # Recent activity
    recent_activity = db.query(models.ActivityLog).filter(
        models.ActivityLog.team_id.in_(team_ids)
    ).order_by(desc(models.ActivityLog.created_at)).limit(10).all()

    return schemas.DashboardStats(
        total_teams=total_teams,
        total_projects=total_projects,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        pending_invitations=pending_invitations,
        unread_notifications=unread_notifications,
        recent_activity=recent_activity
    )

@router.get("/dashboard/data", response_model=schemas.DashboardData)
async def get_dashboard_data(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive dashboard data"""
    stats = await get_dashboard_stats(current_user, db)

    # Get user's teams
    user_teams = db.query(models.TeamMember).filter(models.TeamMember.user_id == current_user.id).all()
    team_ids = [tm.team_id for tm in user_teams]

    my_tasks = []  # Removed task fetching since tasks are removed

    # Recent projects
    recent_projects = db.query(models.Project).filter(
        models.Project.team_id.in_(team_ids)
    ).order_by(desc(models.Project.created_at)).limit(5).all()

    # Team activity
    team_activity = db.query(models.ActivityLog).filter(
        models.ActivityLog.team_id.in_(team_ids)
    ).order_by(desc(models.ActivityLog.created_at)).limit(20).all()

    # Recent notifications
    notifications = db.query(models.Notification).filter(
        models.Notification.user_id == current_user.id
    ).order_by(desc(models.Notification.created_at)).limit(10).all()

    return schemas.DashboardData(
        stats=stats,
        my_tasks=my_tasks,
        recent_projects=recent_projects,
        team_activity=team_activity,
        notifications=notifications
    )

# -------------------------------
# Helper Functions
# -------------------------------
async def check_permission(user_id: int, permission_name: str, db: Session) -> bool:
    """Check if user has a specific permission"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return False

    # Check direct user permissions
    user_perm = db.query(models.User).join(models.User.permissions).filter(
        and_(models.User.id == user_id, models.Permission.name == permission_name)
    ).first()

    if user_perm:
        return True

    # Check role-based permissions
    if user.role_id:
        role_perm = db.query(models.Role).join(models.Role.permissions).filter(
            and_(models.Role.id == user.role_id, models.Permission.name == permission_name)
        ).first()

        if role_perm:
            return True

    return False

async def log_activity(
    user_id: int,
    action: str,
    resource_type: str,
    resource_id: int,
    description: str,
    team_id: Optional[int] = None,
    db: Session = None
):
    """Log user activity"""
    if not db:
        return

    activity = models.ActivityLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        description=description,
        team_id=team_id
    )
    db.add(activity)
    db.commit()

# -------------------------------
# Notification Helper Functions
# -------------------------------
async def create_notification(
    user_id: int,
    notification_type: str,
    title: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    db: Session = None
):
    """Create a notification for a user"""
    if not db:
        return

    notification = models.Notification(
        user_id=user_id,
        type=notification_type,
        title=title,
        message=message,
        data=data
    )
    db.add(notification)
    db.commit()

async def notify_team_members(
    team_id: int,
    notification_type: str,
    title: str,
    message: str,
    exclude_user_id: Optional[int] = None,
    data: Optional[Dict[str, Any]] = None,
    db: Session = None
):
    """Send notification to all team members"""
    if not db:
        return

    team_members = db.query(models.TeamMember).filter(models.TeamMember.team_id == team_id).all()

    for member in team_members:
        if exclude_user_id and member.user_id == exclude_user_id:
            continue

        await create_notification(
            user_id=member.user_id,
            notification_type=notification_type,
            title=title,
            message=message,
            data=data,
            db=db
        )
