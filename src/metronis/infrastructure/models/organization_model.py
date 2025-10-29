"""Database model for organizations."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, Integer, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from metronis.infrastructure.database import Base


class OrganizationModel(Base):
    """Database model for organizations."""
    
    __tablename__ = "organizations"
    
    # Primary key
    organization_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Basic info
    name = Column(String(255), nullable=False)
    api_key_hash = Column(String(255), nullable=False, unique=True)
    
    # Rate limiting and configuration
    rate_limit_per_minute = Column(Integer, default=1000)
    tier_limits = Column(JSONB)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_organizations_api_key", "api_key_hash"),
        Index("idx_organizations_name", "name"),
    )