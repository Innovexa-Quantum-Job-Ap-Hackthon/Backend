# extended_database_models.py - Extended database models for collaboration features
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text, Table, Float, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

# Import Base from database.py
from Database.DBConfiguration.database import Base
from Database.DBmodels.database_models import user_permission_association

# Extended User model with additional relationships
class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="role", primaryjoin="Role.id==User.role_id")
    permissions = relationship("RolePermission", back_populates="role")


class Permission(Base):
    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    resource = Column(String)  # e.g., 'team', 'project', 'task'
    action = Column(String)    # e.g., 'create', 'read', 'update', 'delete'

    # Relationships
    users = relationship("User", secondary=user_permission_association, back_populates="permissions")
    roles = relationship("RolePermission", back_populates="permission")


class RolePermission(Base):
    __tablename__ = "role_permissions"

    id = Column(Integer, primary_key=True, index=True)
    role_id = Column(Integer, ForeignKey("roles.id"))
    permission_id = Column(Integer, ForeignKey("permissions.id"))

    # Relationships
    role = relationship("Role", back_populates="permissions")
    permission = relationship("Permission", back_populates="roles")


# Extended Project model with additional fields
class Milestone(Base):
    __tablename__ = "milestones"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text, nullable=True)
    due_date = Column(DateTime)
    status = Column(String, default="pending")  # 'pending', 'completed', 'overdue'

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Foreign keys
    project_id = Column(Integer, ForeignKey("projects.id"))
    created_by_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    project = relationship("Project", back_populates="milestones")
    created_by = relationship("User")


# Notification System
class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String)  # 'invitation', 'task_assigned', 'project_update', etc.
    title = Column(String)
    message = Column(Text)
    data = Column(JSON, nullable=True)  # Additional data for the notification

    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime, nullable=True)
    is_read = Column(Boolean, default=False)

    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    user = relationship("User", back_populates="notifications")


class NotificationPreference(Base):
    __tablename__ = "notification_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    notification_type = Column(String)
    email_enabled = Column(Boolean, default=True)
    in_app_enabled = Column(Boolean, default=True)

    # Relationships
    user = relationship("User")


# Chat System
class ChatRoom(Base):
    __tablename__ = "chat_rooms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    type = Column(String, default="team")  # 'team', 'project', 'direct'
    created_at = Column(DateTime, default=datetime.utcnow)

    # Foreign keys
    team_id = Column(Integer, ForeignKey("teams.id"))
    created_by_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    team = relationship("Team", back_populates="chat_rooms")
    created_by = relationship("User")
    messages = relationship("ChatMessage", back_populates="room")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text)
    message_type = Column(String, default="text")  # 'text', 'file', 'system'
    created_at = Column(DateTime, default=datetime.utcnow)

    # Foreign keys
    room_id = Column(Integer, ForeignKey("chat_rooms.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    reply_to_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=True)

    # Relationships
    room = relationship("ChatRoom", back_populates="messages")
    user = relationship("User", back_populates="chat_messages")
    reply_to = relationship("ChatMessage", remote_side=[id])


# Activity Logging System
class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String)  # 'created', 'updated', 'deleted', 'joined', etc.
    resource_type = Column(String)  # 'team', 'project', 'task', 'file', etc.
    resource_id = Column(Integer)
    description = Column(Text)
    meta_data = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"))
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)

    # Relationships
    user = relationship("User", back_populates="activity_logs")
    team = relationship("Team")


# Default data seeding functions
def create_default_roles_and_permissions(db):
    """Create default roles and permissions"""
    from sqlalchemy.orm import Session

    # Create default permissions (removed task and file permissions)
    permissions_data = [
        # Team permissions
        {"name": "team_create", "resource": "team", "action": "create", "description": "Create teams"},
        {"name": "team_read", "resource": "team", "action": "read", "description": "View teams"},
        {"name": "team_update", "resource": "team", "action": "update", "description": "Update teams"},
        {"name": "team_delete", "resource": "team", "action": "delete", "description": "Delete teams"},
        {"name": "team_invite", "resource": "team", "action": "invite", "description": "Invite users to teams"},

        # Project permissions
        {"name": "project_create", "resource": "project", "action": "create", "description": "Create projects"},
        {"name": "project_read", "resource": "project", "action": "read", "description": "View projects"},
        {"name": "project_update", "resource": "project", "action": "update", "description": "Update projects"},
        {"name": "project_delete", "resource": "project", "action": "delete", "description": "Delete projects"},
    ]

    for perm_data in permissions_data:
        permission = Permission(**perm_data)
        db.add(permission)
    db.commit()

    # Create default roles
    roles_data = [
        {"name": "admin", "description": "Full access to all features"},
        {"name": "member", "description": "Standard team member access"},
        {"name": "viewer", "description": "Read-only access to team resources"},
    ]

    for role_data in roles_data:
        role = Role(**role_data)
        db.add(role)
    db.commit()

    # Assign permissions to roles
    admin_role = db.query(Role).filter(Role.name == "admin").first()
    member_role = db.query(Role).filter(Role.name == "member").first()
    viewer_role = db.query(Role).filter(Role.name == "viewer").first()

    all_permissions = db.query(Permission).all()

    # Admin gets all permissions
    for perm in all_permissions:
        role_perm = RolePermission(role_id=admin_role.id, permission_id=perm.id)
        db.add(role_perm)

    # Member gets most permissions except delete operations
    member_permissions = ["team_create", "team_read", "team_update", "team_invite",
                         "project_create", "project_read", "project_update"]

    for perm in all_permissions:
        if perm.name in member_permissions:
            role_perm = RolePermission(role_id=member_role.id, permission_id=perm.id)
            db.add(role_perm)

    # Viewer gets only read permissions
    viewer_permissions = ["team_read", "project_read"]

    for perm in all_permissions:
        if perm.name in viewer_permissions:
            role_perm = RolePermission(role_id=viewer_role.id, permission_id=perm.id)
            db.add(role_perm)

    db.commit()
