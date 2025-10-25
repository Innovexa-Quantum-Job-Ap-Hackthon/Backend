# schemas.py
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List

# -------------------------------
# User schemas
# -------------------------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: str
    created_at: datetime

    model_config = {
        "from_attributes": True  # Pydantic V2 replacement for orm_mode
    }

# -------------------------------
# Token schemas
# -------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str

class JobLogs(BaseModel):
    jobId: str
    Device: str
    Status: str
    JobRasied: datetime
    Shots: Optional[int] = None
    JobCompletion: Optional[datetime] = None
    user_id: Optional[int] = None

    model_config = {
        "from_attributes": True  # Pydantic V2 replacement for orm_mode
    }



class TokenData(BaseModel):
    email: Optional[str] = None




# -------------------------------
# Team schemas
# -------------------------------
class TeamCreate(BaseModel):
    name: str
    description: Optional[str] = None

class TeamOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    created_by_id: int
    member_count: int

    model_config = {
        "from_attributes": True
    }

class TeamDetail(TeamOut):
    members: List[UserOut] = []

# -------------------------------
# Team Member schemas
# -------------------------------
class TeamMemberBase(BaseModel):
    role: str = "member"

class TeamMemberOut(TeamMemberBase):
    id: int
    user_id: int
    team_id: int
    joined_at: datetime
    user: UserOut

    model_config = {
        "from_attributes": True
    }

# -------------------------------
# Team Invitation schemas
# -------------------------------
class TeamInvitationCreate(BaseModel):
    email: EmailStr

    model_config = {
        "from_attributes": True
    }

class TeamInvitationOut(BaseModel):
    id: int
    team_id: int
    email: str
    invited_by_id: int
    status: str
    created_at: datetime
    expires_at: datetime

    model_config = {
        "from_attributes": True
    }

# -------------------------------
# Project schemas
# -------------------------------
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    team_id: int

class ProjectOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    team_id: int
    created_by_id: int
    created_at: datetime

    model_config = {
        "from_attributes": True
    }

# -------------------------------
# API Response schemas
# -------------------------------
class SuccessResponse(BaseModel):
    success: bool = True
    message: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None

# -------------------------------
# Analytics schemas
# -------------------------------
class AnalyticsRequest(BaseModel):
    days: int = 30

class DevicePerformance(BaseModel):
    device_name: str
    total_jobs: int
    avg_wait_time: Optional[float] = None
    success_rate: Optional[float] = None
    avg_score: Optional[float] = None

class AnalyticsResponse(BaseModel):
    overall_stats: dict
    device_performance: dict

# -------------------------------
# Job Submission schemas
# -------------------------------
class SubmitJobRequest(BaseModel):
    backend_name: str
    shots: int = 1024
    circuit_data: dict

class JobResult(BaseModel):
    job_id: str
    status: str
    results: Optional[dict] = None
    execution_time: Optional[float] = None

# -------------------------------
# Device schemas
# -------------------------------
class DeviceProperties(BaseModel):
    name: str
    status: str
    pending_jobs: int
    qubits: int
    queue_length: int
    avg_runtime: Optional[float] = None
    error_rate: Optional[float] = None
    t1_time: Optional[float] = None
    t2_time: Optional[float] = None
    gate_fidelity: Optional[float] = None
    readout_fidelity: Optional[float] = None
    score: Optional[float] = None
    wait_time: Optional[int] = None
    last_updated: Optional[str] = None
    success_probability: Optional[float] = None
    cost_estimate: Optional[float] = None
    carbon_footprint: Optional[float] = None
    noise_profile: Optional[str] = None
    noise_recommendation: Optional[str] = None

# -------------------------------
# Job Optimization schemas
# -------------------------------
class JobOptimizerRequest(BaseModel):
    qubits_required: int
    gates_required: int = 0
    circuit_depth: int = 0
    priority: str = "balanced"  # "speed", "reliability", "balanced"

class JobOptimizerResponse(BaseModel):
    recommended_device: str
    success_probability: float
    estimated_wait_time: int
    estimated_cost: float
    estimated_carbon: float
    explanation: str
    alternatives: List[dict]

# -------------------------------
# Circuit Compilation schemas
# -------------------------------
class CircuitCompilationRequest(BaseModel):
    circuit_code: str
    circuit_format: str = "qasm"

class CircuitCompilationResult(BaseModel):
    device_name: str
    can_execute: bool
    additional_swaps: int = 0
    circuit_depth: int = 0
    estimated_fidelity: float = 0.0
    compilation_time: float = 0.0
    required_qubits: int = 0
    available_qubits: int = 0
    error_message: Optional[str] = None