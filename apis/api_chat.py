#!/usr/bin/env python3
"""
Chat API endpoints for team messaging
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from typing import List, Optional
from datetime import datetime

from Database.DBConfiguration.database import get_db
from Database.DBmodels import database_models as db_models
from Database.DBmodels import extended_database_models as models
from Database.DBmodels import extended_schemas as schemas
from Database.DBmodels.database_models import TeamMember
from Authorization.auth import get_current_user
from chat.chat_server import chat_server

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/teams/{team_id}/messages", response_model=schemas.ChatMessageOut)
async def send_chat_message(
    team_id: int,
    message: schemas.ChatMessageSend,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message to team chat"""
    # Check if user is team member
    team_member = db.query(TeamMember).filter(
        and_(TeamMember.team_id == team_id, TeamMember.user_id == current_user.id)
    ).first()

    if not team_member:
        raise HTTPException(status_code=403, detail="Not a member of this team")

    # Get or create chat room for the team
    chat_room = db.query(models.ChatRoom).filter(models.ChatRoom.team_id == team_id).first()

    if not chat_room:
        chat_room = models.ChatRoom(
            team_id=team_id,
            name=f"Team {team_id} Chat",
            created_by_id=current_user.id
        )
        db.add(chat_room)
        db.flush()

    # Create the message
    chat_message = models.ChatMessage(
        room_id=chat_room.id,
        user_id=current_user.id,
        content=message.content,
        message_type=message.message_type,
        reply_to_id=message.reply_to_id
    )
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)

    # Broadcast the new message to all connected users in the room
    message_data = {
        "type": "new_message",
        "message": {
            "id": chat_message.id,
            "room_id": chat_message.room_id,
            "user_id": chat_message.user_id,
            "content": chat_message.content,
            "message_type": chat_message.message_type,
            "reply_to_id": chat_message.reply_to_id,
            "created_at": chat_message.created_at.isoformat() if chat_message.created_at else None,
            "user": {
                "id": current_user.id,
                "name": current_user.name,
                "email": current_user.email
            }
        }
    }
    await chat_server.broadcast_message(chat_room.id, message_data)

    return chat_message

@router.get("/teams/{team_id}/messages", response_model=List[schemas.ChatMessageOut])
async def get_chat_messages(
    team_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get chat messages for a team"""
    try:
        # Check if user is team member
        team_member = db.query(TeamMember).filter(
            and_(TeamMember.team_id == team_id, TeamMember.user_id == current_user.id)
        ).first()

        if not team_member:
            raise HTTPException(status_code=403, detail="Not a member of this team")

        # Get chat room for the team
        chat_room = db.query(models.ChatRoom).filter(models.ChatRoom.team_id == team_id).first()

        if not chat_room:
            return []

        # Get messages
        messages = db.query(models.ChatMessage).filter(
            models.ChatMessage.room_id == chat_room.id
        ).order_by(desc(models.ChatMessage.created_at)).offset(skip).limit(limit).all()

        return messages
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error in get_chat_messages: {e}\n{traceback_str}")
        raise HTTPException(status_code=500, detail="Internal Server Error while fetching chat messages")

@router.get("/teams/{team_id}/room", response_model=schemas.ChatRoomOut)
async def get_chat_room(
    team_id: int,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get or create chat room for a team"""
    # Check if user is team member
    team_member = db.query(TeamMember).filter(
        and_(TeamMember.team_id == team_id, TeamMember.user_id == current_user.id)
    ).first()

    if not team_member:
        raise HTTPException(status_code=403, detail="Not a member of this team")

    # Get or create chat room
    chat_room = db.query(models.ChatRoom).filter(models.ChatRoom.team_id == team_id).first()

    if not chat_room:
        chat_room = models.ChatRoom(
            team_id=team_id,
            name=f"Team {team_id} Chat",
            created_by_id=current_user.id
        )
        db.add(chat_room)
        db.commit()
        db.refresh(chat_room)

    return chat_room
