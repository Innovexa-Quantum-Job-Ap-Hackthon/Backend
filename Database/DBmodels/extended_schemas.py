# extended_schemas.py - Extended Pydantic schemas for collaboration features
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

# -------------------------------
# Role and Permission schemas
# -------------------------------
class PermissionBase(BaseModel):
    name: str
    description: Optional[str] = None
    resource: str
    action: str

class PermissionCreate(PermissionBase):
    pass

class PermissionOut(PermissionBase):
    id: int

    model_config = {"from_attributes": True}

class RoleBase(BaseModel):
    name: str
    description: Optional[str] = None

class RoleCreate(RoleBase):
    pass

class RoleOut(RoleBase):
    id: int
    created_at: datetime
    permissions: List[PermissionOut] = []

    model_config = {"from_attributes": True}

class RolePermissionCreate(BaseModel):
    role_id: int
    permission_id: int

# -------------------------------
# Extended User schemas
# -------------------------------
class UserProfile(BaseModel):
    id: int
    email: EmailStr
    name: str
    role_id: Optional[int] = None
    role: Optional[RoleOut] = None
    permissions: List[PermissionOut] = []
    created_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}

# -------------------------------
# Milestone schemas
# -------------------------------
class MilestoneStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    OVERDUE = "overdue"

class MilestoneBase(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: datetime
    status: MilestoneStatus = MilestoneStatus.PENDING

class MilestoneCreate(MilestoneBase):
    project_id: int

class MilestoneUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    status: Optional[MilestoneStatus] = None

class MilestoneOut(MilestoneBase):
    id: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    project_id: int
    created_by_id: int
    created_by: UserProfile

    model_config = {"from_attributes": True}

# -------------------------------
# Notification schemas
# -------------------------------
class NotificationType(str, Enum):
    INVITATION = "invitation"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    PROJECT_UPDATE = "project_update"
    TEAM_UPDATE = "team_update"
    MENTION = "mention"
    SYSTEM = "system"

class NotificationBase(BaseModel):
    type: NotificationType
    title: str
    message: str
    data: Optional[Dict[str, Any]] = None

class NotificationCreate(NotificationBase):
    user_id: int

class NotificationOut(NotificationBase):
    id: int
    created_at: datetime
    read_at: Optional[datetime] = None
    is_read: bool
    user_id: int

    model_config = {"from_attributes": True}

class NotificationPreferenceBase(BaseModel):
    notification_type: str
    email_enabled: bool = True
    in_app_enabled: bool = True

class NotificationPreferenceCreate(NotificationPreferenceBase):
    pass

class NotificationPreferenceOut(NotificationPreferenceBase):
    id: int
    user_id: int

    model_config = {"from_attributes": True}

# -------------------------------
# Chat System schemas
# -------------------------------
class ChatRoomType(str, Enum):
    TEAM = "team"
    PROJECT = "project"
    DIRECT = "direct"

class ChatMessageType(str, Enum):
    TEXT = "text"
    FILE = "file"
    SYSTEM = "system"

class ChatRoomBase(BaseModel):
    name: str
    type: ChatRoomType = ChatRoomType.TEAM

class ChatRoomCreate(ChatRoomBase):
    team_id: int

class ChatRoomOut(ChatRoomBase):
    id: int
    created_at: datetime
    team_id: int
    created_by_id: int
    created_by: UserProfile

    model_config = {"from_attributes": True}

class ChatMessageBase(BaseModel):
    content: str
    message_type: ChatMessageType = ChatMessageType.TEXT
    reply_to_id: Optional[int] = None

class ChatMessageCreate(ChatMessageBase):
    room_id: int

class ChatMessageSend(ChatMessageBase):
    pass  # No room_id, as it's determined by team_id in endpoint

class ChatMessageOut(ChatMessageBase):
    id: int
    created_at: datetime
    room_id: int
    user_id: int
    user: UserProfile
    reply_to_id: Optional[int] = None


# -------------------------------
# Activity Log schemas
# -------------------------------
class ActivityLogBase(BaseModel):
    action: str
    resource_type: str
    resource_id: int
    description: str
    meta_data: Optional[Dict[str, Any]] = None

class ActivityLogCreate(ActivityLogBase):
    user_id: int
    team_id: Optional[int] = None

class ActivityLogOut(ActivityLogBase):
    id: int
    created_at: datetime
    user_id: int
    team_id: Optional[int] = None
    user: UserProfile

    model_config = {"from_attributes": True}

# -------------------------------
# Extended Project schemas
# -------------------------------
class ProjectDetail(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    team_id: int
    created_by_id: int
    created_at: datetime
    status: str = "active"
    progress_percentage: float = 0.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tasks: List = []  # Empty list since tasks are removed
    milestones: List[MilestoneOut] = []
    files: List = []  # Empty list since files are removed

    model_config = {"from_attributes": True}

# -------------------------------
# Team Analytics schemas
# -------------------------------
class TeamAnalytics(BaseModel):
    total_members: int
    total_projects: int
    total_tasks: int = 0  # Set to 0 since tasks are removed
    completed_tasks: int = 0  # Set to 0 since tasks are removed
    active_tasks: int = 0  # Set to 0 since tasks are removed
    overdue_tasks: int = 0  # Set to 0 since tasks are removed
    total_files: int = 0  # Set to 0 since files are removed
    storage_used: int = 0  # Set to 0 since files are removed
    recent_activity: List[ActivityLogOut] = []

# -------------------------------
# Dashboard schemas
# -------------------------------
class DashboardStats(BaseModel):
    total_teams: int
    total_projects: int
    total_tasks: int = 0  # Set to 0 since tasks are removed
    completed_tasks: int = 0  # Set to 0 since tasks are removed
    pending_invitations: int
    unread_notifications: int
    recent_activity: List[ActivityLogOut] = []
    upcoming_deadlines: List[Dict[str, Any]] = []

class DashboardData(BaseModel):
    stats: DashboardStats
    my_tasks: List = []  # Empty list since tasks are removed
    recent_projects: List[ProjectDetail] = []
    team_activity: List[ActivityLogOut] = []
    notifications: List[NotificationOut] = []

# -------------------------------
# API Response schemas for extended features
# -------------------------------
class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int

class SearchFilters(BaseModel):
    query: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    assignee_id: Optional[int] = None
    project_id: Optional[int] = None
    team_id: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

class BulkActionRequest(BaseModel):
    action: str  # 'update_status', 'assign', 'delete', etc.
    item_ids: List[int]
    data: Optional[Dict[str, Any]] = None

# Forward references for circular dependencies
ChatMessageOut.model_rebuild()
