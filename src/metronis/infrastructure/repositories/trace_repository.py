"""Trace repository for database operations."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Session

from metronis.core.models import Trace
from metronis.db.models import TraceModel


class TraceRepository:
    """Repository for trace database operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, trace: Trace) -> TraceModel:
        """Create a new trace."""
        db_trace = TraceModel(
            trace_id=trace.trace_id,
            organization_id=trace.organization_id,
            application_id=trace.application_id,
            application_type=trace.application_type,
            session_id=trace.session_id,
            model=trace.ai_processing.model,
            input_text=trace.ai_processing.input,
            output_text=trace.ai_processing.output,
            prompt_tokens=trace.ai_processing.prompt_tokens,
            completion_tokens=trace.ai_processing.completion_tokens,
            total_tokens=trace.ai_processing.total_tokens,
            rl_episode=[step.model_dump() for step in trace.ai_processing.rl_episode],
            policy_info=(
                trace.ai_processing.policy_info.model_dump()
                if trace.ai_processing.policy_info
                else None
            ),
            cumulative_reward=trace.ai_processing.cumulative_reward,
            episode_length=trace.ai_processing.episode_length,
            retrieved_contexts=[
                ctx.model_dump() for ctx in trace.ai_processing.retrieved_contexts
            ],
            domain=trace.metadata.domain,
            specialty=trace.metadata.specialty,
            patient_context=trace.metadata.patient_context,
            market_context=trace.metadata.market_context,
            environment_context=trace.metadata.environment_context,
            additional_metadata=trace.metadata.additional_metadata,
            timestamp=trace.timestamp,
        )
        self.db.add(db_trace)
        self.db.commit()
        self.db.refresh(db_trace)
        return db_trace

    def get_by_id(self, trace_id: UUID) -> Optional[TraceModel]:
        """Get trace by ID."""
        return self.db.query(TraceModel).filter(TraceModel.trace_id == trace_id).first()

    def list_by_organization(
        self,
        organization_id: UUID,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TraceModel]:
        """List traces for an organization."""
        query = self.db.query(TraceModel).filter(
            TraceModel.organization_id == organization_id
        )

        if domain:
            query = query.filter(TraceModel.domain == domain)

        return (
            query.order_by(desc(TraceModel.created_at))
            .limit(limit)
            .offset(offset)
            .all()
        )

    def delete(self, trace_id: UUID) -> bool:
        """Delete a trace."""
        trace = self.get_by_id(trace_id)
        if trace:
            self.db.delete(trace)
            self.db.commit()
            return True
        return False
