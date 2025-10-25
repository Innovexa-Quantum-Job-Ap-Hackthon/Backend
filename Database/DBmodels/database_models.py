# database_models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text, Table
from sqlalchemy.orm import relationship
from datetime import datetime

# Import Base from database.py
from Database.DBConfiguration.database import Base

# Association table for many-to-many between Users and Teams
user_team_association = Table(
    "user_teams",
    Base.metadata,
    Column("user_id", ForeignKey("users.id")),
    Column("team_id", ForeignKey("teams.id")),
)

# Association table for many-to-many between Users and Permissions
user_permission_association = Table(
    "user_permissions",
    Base.metadata,
    Column("user_id", ForeignKey("users.id")),
    Column("permission_id", ForeignKey("permissions.id")),
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    name = Column(String)
    ibm_api_key_encrypted = Column(String, nullable=True)
    ibm_instance_key_encrypted = Column(String, nullable=True)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=True)

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    ## Relationships for collaboration
    role = relationship("Role", back_populates="users")
    permissions = relationship("Permission", secondary=user_permission_association, back_populates="users")
    teams_created = relationship("Team", back_populates="creator")
    team_memberships = relationship("TeamMember", back_populates="user")
    invitations_sent = relationship("TeamInvitation", back_populates="inviter")
    projects_created = relationship("Project", back_populates="creator")
    job_logs = relationship("JobLogs", back_populates="user")
    jobs = relationship("Job", back_populates="owner")
    notifications = relationship("Notification", back_populates="user")
    chat_messages = relationship("ChatMessage", back_populates="user")
    activity_logs = relationship("ActivityLog", back_populates="user")

class Team(Base):
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    creator = relationship("User", back_populates="teams_created")
    members = relationship("TeamMember", back_populates="team")
    projects = relationship("Project", back_populates="team")
    invitations = relationship("TeamInvitation", back_populates="team")
    jobs = relationship("Job", back_populates="team")
    chat_rooms = relationship("ChatRoom", back_populates="team")

class TeamMember(Base):
    __tablename__ = "team_members"
    
    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String, default="member")  # 'admin', 'member'
    joined_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="members")
    user = relationship("User", back_populates="team_memberships")

class TeamInvitation(Base):
    __tablename__ = "team_invitations"
    
    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    email = Column(String, index=True)
    token = Column(String, unique=True, index=True)
    invited_by_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="pending")  # 'pending', 'accepted', 'expired', 'declined'
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Relationships
    team = relationship("Team", back_populates="invitations")
    inviter = relationship("User", back_populates="invitations_sent")

class TeamInvitationNotification(Base):
    __tablename__ = "team_invitation_notifications"

    id = Column(Integer, primary_key=True, index=True)
    invitation_id = Column(Integer, ForeignKey("team_invitations.id"))
    user_email = Column(String, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="unread")  # 'unread', 'read'

    # Relationships
    invitation = relationship("TeamInvitation")
    team = relationship("Team")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    created_by_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="projects")
    creator = relationship("User", back_populates="projects_created")
    jobs = relationship("JobLogs", back_populates="project")
    milestones = relationship("Milestone", back_populates="project")

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, nullable=True)
    status = Column(String, default="pending")  # e.g. pending, running, completed
    created_at = Column(DateTime, default=datetime.utcnow)

    # Foreign keys
    owner_id = Column(Integer, ForeignKey("users.id"))
    team_id = Column(Integer, ForeignKey("teams.id"))

    # Relationships
    owner = relationship("User", back_populates="jobs")
    team = relationship("Team", back_populates="jobs")

class JobLogs(Base):
    __tablename__ = "JobLogs"

    jobId = Column(String, primary_key=True, index=True)
    JobRasied = Column(DateTime, default=datetime.utcnow)
    Device = Column(String)
    Status = Column(String, default="queued")  # Default status for new jobs
    Shots = Column(Integer)
    JobCompletion = Column(DateTime, nullable=True)  # Only set when job completes

    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Relationships
    user = relationship("User", back_populates="job_logs")
    project = relationship("Project", back_populates="jobs")
