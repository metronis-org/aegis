"""SQLAlchemy ORM models for database persistence."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Float,
    Integer,
    Boolean,
    Text,
    ForeignKey,
    Index,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from metronis.db.base import Base
from metronis.core.models import Severity, EvaluationStatus


class OrganizationModel(Base):
    """Organization/customer account."""

    __tablename__ = "organizations"

    organization_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    api_key = Column(String(255), unique=True, nullable=False, index=True)
    domain = Column(String(100), nullable=True)
    tier = Column(String(50), default="free")  # free, pro, enterprise
    monthly_trace_limit = Column(Integer, default=10000)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    traces = relationship("TraceModel", back_populates="organization")

    def __repr__(self):
        return f"<Organization {self.name}>"


class TraceModel(Base):
    """Trace of an AI system interaction."""

    __tablename__ = "traces"

    trace_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.organization_id"), nullable=False
    )
    application_id = Column(String(255), nullable=True)
    application_type = Column(String(100), nullable=False)
    session_id = Column(String(255), nullable=True, index=True)

    # AI Processing fields
    model = Column(String(255), nullable=False)
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=False)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    # RL-specific fields (stored as JSON)
    rl_episode = Column(JSONB, nullable=True)
    policy_info = Column(JSONB, nullable=True)
    cumulative_reward = Column(Float, nullable=True)
    episode_length = Column(Integer, nullable=True)

    # RAG-specific fields
    retrieved_contexts = Column(JSONB, nullable=True)

    # Metadata
    domain = Column(String(100), nullable=True, index=True)
    specialty = Column(String(100), nullable=True)
    patient_context = Column(Text, nullable=True)
    market_context = Column(Text, nullable=True)
    environment_context = Column(Text, nullable=True)
    additional_metadata = Column(JSONB, nullable=True)

    # Timestamps
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    organization = relationship("OrganizationModel", back_populates="traces")
    evaluations = relationship("EvaluationResultModel", back_populates="trace")

    # Indexes
    __table_args__ = (
        Index("idx_org_domain_created", "organization_id", "domain", "created_at"),
        Index("idx_model_domain", "model", "domain"),
        Index("idx_session", "session_id"),
    )

    def __repr__(self):
        return f"<Trace {self.trace_id}>"


class EvaluationResultModel(Base):
    """Evaluation result for a trace."""

    __tablename__ = "evaluation_results"

    evaluation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.trace_id"), nullable=False)

    # Overall results
    overall_passed = Column(Boolean, nullable=False)
    overall_severity = Column(SQLEnum(Severity), nullable=True)
    status = Column(SQLEnum(EvaluationStatus), default=EvaluationStatus.PENDING)

    # Execution metrics
    total_execution_time_ms = Column(Float, nullable=False)
    cost = Column(Float, default=0.0)

    # Tier results (stored as JSON for flexibility)
    tier1_results = Column(JSONB, nullable=True)
    tier2_results = Column(JSONB, nullable=True)
    tier3_results = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    trace = relationship("TraceModel", back_populates="evaluations")
    issues = relationship("EvaluationIssueModel", back_populates="evaluation")

    # Indexes
    __table_args__ = (
        Index("idx_trace_created", "trace_id", "created_at"),
        Index("idx_status", "status"),
    )

    def __repr__(self):
        return f"<EvaluationResult {self.evaluation_id}>"


class EvaluationIssueModel(Base):
    """Individual issue found during evaluation."""

    __tablename__ = "evaluation_issues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_results.evaluation_id"),
        nullable=False,
    )

    issue_type = Column(String(255), nullable=False, index=True)
    severity = Column(SQLEnum(Severity), nullable=False, index=True)
    message = Column(Text, nullable=False)
    details = Column(JSONB, nullable=True)

    # Which module detected this issue
    module_name = Column(String(255), nullable=True)
    tier_level = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    evaluation = relationship("EvaluationResultModel", back_populates="issues")

    # Indexes
    __table_args__ = (
        Index("idx_eval_severity", "evaluation_id", "severity"),
        Index("idx_issue_type", "issue_type"),
    )

    def __repr__(self):
        return f"<Issue {self.issue_type} [{self.severity}]>"


class ExpertLabelModel(Base):
    """Expert review labels for active learning."""

    __tablename__ = "expert_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.trace_id"), nullable=False)
    expert_id = Column(String(255), nullable=False)

    # Expert's judgment
    is_safe = Column(Boolean, nullable=False)
    severity = Column(SQLEnum(Severity), nullable=True)
    issues_identified = Column(JSONB, nullable=True)
    feedback = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False)

    # Metadata
    labeled_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    labeling_time_seconds = Column(Float, nullable=True)

    # Indexes
    __table_args__ = (Index("idx_trace_expert", "trace_id", "expert_id"),)

    def __repr__(self):
        return f"<ExpertLabel trace={self.trace_id} safe={self.is_safe}>"


class AlertModel(Base):
    """Alerts sent for critical issues."""

    __tablename__ = "alerts"

    alert_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.trace_id"), nullable=False)
    evaluation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_results.evaluation_id"),
        nullable=True,
    )
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.organization_id"), nullable=False
    )

    # Alert details
    rule_name = Column(String(255), nullable=False)
    severity = Column(String(50), nullable=False)
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSONB, nullable=True)

    # Delivery
    channels = Column(JSONB, nullable=False)  # ["slack", "email"]
    sent_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(255), nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_org_sent", "organization_id", "sent_at"),
        Index("idx_severity_ack", "severity", "acknowledged"),
    )

    def __repr__(self):
        return f"<Alert {self.rule_name} [{self.severity}]>"


class ModelVersionModel(Base):
    """Versioning for trained ML models."""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(100), nullable=False, index=True)
    tier = Column(Integer, nullable=False)
    model_name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)

    # Model metadata
    model_type = Column(String(100), nullable=False)
    model_path = Column(String(500), nullable=False)
    metrics = Column(JSONB, nullable=True)  # accuracy, precision, recall, etc.
    training_samples = Column(Integer, nullable=True)

    # Deployment
    is_production = Column(Boolean, default=False)
    deployed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index("idx_domain_tier_prod", "domain", "tier", "is_production"),
        Index("idx_model_version", "model_name", "version"),
    )

    def __repr__(self):
        return f"<ModelVersion {self.model_name} v{self.version}>"


class UsageMetricModel(Base):
    """Usage metrics for billing."""

    __tablename__ = "usage_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.organization_id"), nullable=False
    )

    # Time period
    date = Column(DateTime, nullable=False, index=True)
    hour = Column(Integer, nullable=True)  # 0-23 for hourly metrics

    # Metrics
    traces_count = Column(Integer, default=0)
    tier1_count = Column(Integer, default=0)
    tier2_count = Column(Integer, default=0)
    tier3_count = Column(Integer, default=0)
    tier4_count = Column(Integer, default=0)

    # Costs
    tier1_cost = Column(Float, default=0.0)
    tier2_cost = Column(Float, default=0.0)
    tier3_cost = Column(Float, default=0.0)
    tier4_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index("idx_org_date", "organization_id", "date"),
        Index("idx_date", "date"),
    )

    def __repr__(self):
        return f"<UsageMetric org={self.organization_id} date={self.date}>"
