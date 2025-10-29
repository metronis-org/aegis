"""Database model for traces."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, Integer, Text, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
from sqlalchemy.sql import func

from metronis.infrastructure.database import Base
from metronis.core.models import ApplicationType, EvaluationStatus


class TraceModel(Base):
    """Database model for traces."""
    
    __tablename__ = "traces"
    
    # Primary key
    trace_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Organization and application
    organization_id = Column(UUID(as_uuid=True), nullable=False)
    application_id = Column(UUID(as_uuid=True), nullable=False)
    application_type = Column(
        ENUM(ApplicationType, name="application_type_enum"),
        nullable=False,
        default=ApplicationType.GENERIC
    )
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # User context
    user_id = Column(String(100))
    session_id = Column(UUID(as_uuid=True))
    
    # AI processing info
    model_used = Column(String(100))
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    latency_ms = Column(Integer)
    
    # Trace data (JSONB for flexibility)
    raw_trace = Column(JSONB)
    sanitized_trace = Column(JSONB)
    
    # Evaluation status
    evaluation_status = Column(
        ENUM(EvaluationStatus, name="evaluation_status_enum"),
        nullable=False,
        default=EvaluationStatus.PENDING
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_traces_org_app", "organization_id", "application_id"),
        Index("idx_traces_timestamp", "timestamp"),
        Index("idx_traces_status", "evaluation_status"),
        Index("idx_traces_created", "created_at"),
    )