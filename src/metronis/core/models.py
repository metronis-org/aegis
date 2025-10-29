"""Core domain models for Metronis platform."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ApplicationType(str, Enum):
    """Supported AI application types."""

    GENERIC = "generic"
    CHATBOT = "chatbot"
    RAG = "rag"
    AGENT = "agent"
    RL_AGENT = "rl_agent"
    CLINICAL_SUPPORT = "clinical_support"
    DIAGNOSTIC = "diagnostic"
    DOCUMENTATION = "documentation"
    TRADING_AGENT = "trading_agent"
    ROBOTICS = "robotics"
    OTHER = "other"


class Severity(str, Enum):
    """Evaluation severity levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvaluationStatus(str, Enum):
    """Trace evaluation status."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class UserContext(BaseModel):
    """User context information for a trace."""
    
    user_id: Optional[str] = None
    session_id: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    """Retrieved context for RAG systems."""

    source: str
    content: str
    relevance_score: float
    citation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RLStep(BaseModel):
    """Single step in an RL episode."""

    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Optional[Dict[str, Any]] = None
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class PolicyInfo(BaseModel):
    """Information about the RL policy."""

    policy_name: str
    policy_version: Optional[str] = None
    action_space: Dict[str, Any] = Field(default_factory=dict)
    state_space: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class AIProcessing(BaseModel):
    """AI processing information for a trace."""

    model: str
    input: str
    output: str
    reasoning_steps: List[str] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None

    # RAG-specific fields
    retrieved_contexts: List[RetrievedContext] = Field(default_factory=list)

    # RL-specific fields
    rl_episode: List[RLStep] = Field(default_factory=list)
    policy_info: Optional[PolicyInfo] = None
    cumulative_reward: Optional[float] = None
    episode_length: Optional[int] = None


class TraceMetadata(BaseModel):
    """Additional metadata for a trace."""

    use_case: Optional[str] = None
    domain: Optional[str] = None  # healthcare, trading, robotics, legal, etc.
    specialty: Optional[str] = None  # cardiology, oncology, etc.
    risk_level: Optional[str] = None  # critical, high, medium, low
    custom_fields: Dict[str, Any] = Field(default_factory=dict)

    # Domain-specific context
    patient_context: Optional[str] = None  # For healthcare
    market_context: Optional[Dict[str, Any]] = None  # For trading
    environment_context: Optional[Dict[str, Any]] = None  # For robotics


class Trace(BaseModel):
    """Core trace model representing an AI interaction."""
    
    trace_id: UUID = Field(default_factory=uuid4)
    organization_id: UUID
    application_id: UUID
    application_type: ApplicationType = ApplicationType.GENERIC
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    user_context: UserContext = Field(default_factory=UserContext)
    ai_processing: AIProcessing
    metadata: TraceMetadata = Field(default_factory=TraceMetadata)
    
    # Internal fields
    evaluation_status: EvaluationStatus = EvaluationStatus.PENDING
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        """Parse timestamp from various formats."""
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class EvaluationIssue(BaseModel):
    """Individual issue found during evaluation."""
    
    type: str
    severity: Severity
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ModuleResult(BaseModel):
    """Result from a single evaluation module."""
    
    module_name: str
    tier_level: int
    passed: bool
    risk_score: Optional[float] = None
    confidence: Optional[float] = None
    issues: List[EvaluationIssue] = Field(default_factory=list)
    execution_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Complete evaluation result for a trace."""
    
    evaluation_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID
    
    # Overall results
    overall_severity: Severity = Severity.LOW
    overall_passed: bool = True
    total_execution_time_ms: float = 0.0
    
    # Module results by tier
    tier1_results: List[ModuleResult] = Field(default_factory=list)
    tier2_results: List[ModuleResult] = Field(default_factory=list)
    tier3_results: List[ModuleResult] = Field(default_factory=list)
    
    # Aggregated information
    all_issues: List[EvaluationIssue] = Field(default_factory=list)
    error_types: List[str] = Field(default_factory=list)
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def add_module_result(self, result: ModuleResult) -> None:
        """Add a module result to the appropriate tier."""
        if result.tier_level == 1:
            self.tier1_results.append(result)
        elif result.tier_level == 2:
            self.tier2_results.append(result)
        elif result.tier_level == 3:
            self.tier3_results.append(result)
        
        # Update aggregated data
        self.all_issues.extend(result.issues)
        self.total_execution_time_ms += result.execution_time_ms or 0.0
        
        # Update overall status
        if not result.passed:
            self.overall_passed = False
        
        # Update severity (take highest)
        for issue in result.issues:
            if issue.severity.value > self.overall_severity.value:
                self.overall_severity = issue.severity
        
        # Update error types
        for issue in result.issues:
            if issue.type not in self.error_types:
                self.error_types.append(issue.type)
    
    def finalize(self) -> None:
        """Finalize the evaluation result."""
        self.completed_at = datetime.utcnow()
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class Organization(BaseModel):
    """Organization model for multi-tenancy."""
    
    organization_id: UUID = Field(default_factory=uuid4)
    name: str
    api_key_hash: str
    
    # Rate limiting
    rate_limit_per_minute: int = 1000
    tier_limits: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }