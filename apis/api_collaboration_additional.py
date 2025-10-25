from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from Database.DBConfiguration.database import get_db
from Authorization.auth import get_current_user
import Database.DBmodels.schemas as schemas
import Database.DBmodels.database_models as models

# Additional collaboration endpoints
router = APIRouter(prefix="/collaboration", tags=["Collaboration"])

@router.get("/invitations/pending", response_model=List[schemas.TeamInvitationOut])
async def get_pending_invitations(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get pending team invitations for the current user"""
    # Get invitations by email
    invitations = db.query(models.TeamInvitation).filter(
        models.TeamInvitation.email == current_user.email,
        models.TeamInvitation.status == "pending",
        models.TeamInvitation.expires_at > datetime.utcnow()
    ).all()

    result = []
    for invitation in invitations:
        team = db.query(models.Team).filter(models.Team.id == invitation.team_id).first()
        if team:
            invitation_dict = {
                "id": invitation.id,
                "team_id": invitation.team_id,
                "team_name": team.name,
                "email": invitation.email,
                "token": invitation.token,
                "invited_by_id": invitation.invited_by_id,
                "expires_at": invitation.expires_at,
                "status": invitation.status
            }
            result.append(schemas.TeamInvitationOut(**invitation_dict))

    return result

@router.delete("/invitations/{invitation_id}")
async def decline_invitation(
    invitation_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Decline a team invitation"""
    invitation = db.query(models.TeamInvitation).filter(
        models.TeamInvitation.id == invitation_id,
        models.TeamInvitation.email == current_user.email,
        models.TeamInvitation.status == "pending"
    ).first()

    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")

    invitation.status = "declined"
    db.commit()

    return {"message": "Invitation declined"}

@router.get("/notifications")
async def get_notifications(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get notifications for the current user"""
    # Get pending invitations as notifications
    notifications = []

    # Team invitations
    invitations = db.query(models.TeamInvitation).filter(
        models.TeamInvitation.email == current_user.email,
        models.TeamInvitation.status == "pending",
        models.TeamInvitation.expires_at > datetime.utcnow()
    ).all()

    for invitation in invitations:
        team = db.query(models.Team).filter(models.Team.id == invitation.team_id).first()
        if team:
            notifications.append({
                "id": f"invitation_{invitation.id}",
                "type": "invitation",
                "message": f"You have been invited to join team '{team.name}'",
                "timestamp": invitation.created_at.isoformat(),
                "token": invitation.token,
                "team_id": invitation.team_id,
                "team_name": team.name
            })

    # Also get notifications from the notification table
    notification_records = db.query(models.TeamInvitationNotification).filter(
        models.TeamInvitationNotification.user_email == current_user.email,
        models.TeamInvitationNotification.status == "unread"
    ).all()

    for notification in notification_records:
        team = db.query(models.Team).filter(models.Team.id == notification.team_id).first()
        if team:
            # Get the token from the related invitation
            token = notification.invitation.token if notification.invitation else None
            notifications.append({
                "id": f"notification_{notification.id}",
                "type": "invitation",
                "message": f"You have been invited to join team '{team.name}'",
                "timestamp": notification.created_at.isoformat(),
                "token": token,
                "team_id": notification.team_id,
                "team_name": team.name
            })

    return {"notifications": notifications}

@router.put("/notifications/{notification_id}/read")
async def mark_notification_as_read(
    notification_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a notification as read"""
    # Handle both invitation and notification types
    if notification_id.startswith("notification_"):
        # It's a notification record
        notification_record_id = int(notification_id.split("_")[1])
        notification = db.query(models.TeamInvitationNotification).filter(
            models.TeamInvitationNotification.id == notification_record_id,
            models.TeamInvitationNotification.user_email == current_user.email
        ).first()

        if notification:
            notification.status = "read"
            db.commit()
            return {"message": "Notification marked as read"}

    elif notification_id.startswith("invitation_"):
        # It's an invitation - we don't change its status, just acknowledge it
        return {"message": "Notification acknowledged"}

    raise HTTPException(status_code=404, detail="Notification not found")
