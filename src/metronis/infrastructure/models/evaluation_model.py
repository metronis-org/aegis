"""Database model for evaluations."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, Integer, Float, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, ENUM
from sqlalchemy.sql import func

from metronis.infrastructure.database import Base
from metronis.core.models import Severity


class EvaluationModel(Base):
    """Database model for evaluation results."""
    
    __tablename__ = "evaluations"
    
    # Primary key
    evaluation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to trace
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.trace_id"), nullable=False)
    
    # Evaluation metadata
    tier_level = Column(Integer, nullable=False)
    evaluation_module = Column(String(100), nullable=False)
    
    # Results
    risk_score = Column(Float)
    error_types = Column(ARRAY(Text))
    severity = Column(ENUM(Severity, name="severity_enum"))
    confidence = Column(Float)
    
    # Detailed output (JSONB for flexibility)
    evaluation_output = Column(JSONB)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_evaluations_trace", "trace_id"),
        Index("idx_evaluations_severity", "severity"),
        Index("idx_evaluations_module", "evaluation_module"),
        Index("idx_evaluations_tier", "tier_level"),
    )


class EvaluationSummaryModel(Base):
    """Database model for evaluation summaries (one per trace)."""
    
    __tablename__ = "evaluation_summaries"
    
    # Primary key (same as trace_id for 1:1 relationship)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.trace_id"), primary_key=True)
    
    # Overall results
    overall_severity = Column(ENUM(Severity, name="severity_enum"), nullable=False)
    overall_passed = Column(String(10), nullable=False)  # "true" or "false"
    total_execution_time_ms = Column(Float, default=0.0)
    
    # Aggregated data
    all_issues = Column(JSONB)
    error_types = Column(ARRAY(Text))
    
    # Tier completion status
    tier1_completed = Column(String(10), default="false")
    tier2_completed = Column(String(10), default="false")
    tier3_completed = Column(String(10), default="false")
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Indexes
    __table_args__ = (
        Index("idx_eval_summary_severity", "overall_severity"),
        Index("idx_eval_summary_completed", "completed_at"),
    )