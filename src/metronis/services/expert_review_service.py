"""
Expert Review Service - Manage active learning labeling tasks
"""

from datetime import datetime
from typing import Dict, List, Optional

import structlog
from sqlalchemy import desc
from sqlalchemy.orm import Session

from metronis.db.models import ExpertLabelModel, TraceModel

logger = structlog.get_logger(__name__)


class ExpertReviewService:
    """Manage expert review tasks for active learning."""

    def __init__(self, db: Session):
        self.db = db

    def get_review_queue(
        self,
        organization_id: str,
        domain: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get traces that need expert review."""

        # Find traces without expert labels, prioritize uncertain ones
        query = (
            self.db.query(TraceModel)
            .filter(TraceModel.organization_id == organization_id)
            .outerjoin(
                ExpertLabelModel, TraceModel.trace_id == ExpertLabelModel.trace_id
            )
            .filter(ExpertLabelModel.label_id.is_(None))  # No label yet
        )

        if domain:
            query = query.filter(TraceModel.domain == domain)

        traces = query.order_by(desc(TraceModel.created_at)).limit(limit).all()

        return [
            {
                "trace_id": str(trace.trace_id),
                "model": trace.model,
                "domain": trace.domain,
                "input_text": trace.input_text[:500],  # Truncate for preview
                "output_text": trace.output_text[:500],
                "created_at": (
                    trace.created_at.isoformat() if trace.created_at else None
                ),
            }
            for trace in traces
        ]

    def submit_label(
        self,
        trace_id: str,
        expert_email: str,
        label: str,  # e.g., 'pass', 'fail', 'needs_review'
        confidence: float,
        notes: Optional[str] = None,
        issue_categories: Optional[List[str]] = None,
    ) -> ExpertLabelModel:
        """Submit expert label for a trace."""

        expert_label = ExpertLabelModel(
            trace_id=trace_id,
            expert_email=expert_email,
            label=label,
            confidence=confidence,
            notes=notes,
            issue_categories=issue_categories or [],
            created_at=datetime.utcnow(),
        )

        self.db.add(expert_label)
        self.db.commit()
        self.db.refresh(expert_label)

        logger.info(
            "Expert label submitted",
            trace_id=trace_id,
            label=label,
            expert_email=expert_email,
        )

        return expert_label

    def get_labeled_traces(
        self,
        organization_id: str,
        expert_email: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get traces that have been labeled."""

        query = (
            self.db.query(TraceModel, ExpertLabelModel)
            .join(ExpertLabelModel, TraceModel.trace_id == ExpertLabelModel.trace_id)
            .filter(TraceModel.organization_id == organization_id)
        )

        if expert_email:
            query = query.filter(ExpertLabelModel.expert_email == expert_email)

        results = query.order_by(desc(ExpertLabelModel.created_at)).limit(limit).all()

        return [
            {
                "trace_id": str(trace.trace_id),
                "model": trace.model,
                "domain": trace.domain,
                "input_text": trace.input_text[:500],
                "output_text": trace.output_text[:500],
                "label": label.label,
                "confidence": label.confidence,
                "expert_email": label.expert_email,
                "notes": label.notes,
                "labeled_at": (
                    label.created_at.isoformat() if label.created_at else None
                ),
            }
            for trace, label in results
        ]

    def get_labeling_stats(self, organization_id: str) -> Dict:
        """Get statistics on labeling progress."""

        total_traces = (
            self.db.query(TraceModel)
            .filter(TraceModel.organization_id == organization_id)
            .count()
        )

        labeled_traces = (
            self.db.query(TraceModel)
            .join(ExpertLabelModel, TraceModel.trace_id == ExpertLabelModel.trace_id)
            .filter(TraceModel.organization_id == organization_id)
            .count()
        )

        unlabeled_traces = total_traces - labeled_traces

        # Labels by expert
        labels_by_expert = {}
        for label in self.db.query(ExpertLabelModel).all():
            email = label.expert_email
            labels_by_expert[email] = labels_by_expert.get(email, 0) + 1

        return {
            "total_traces": total_traces,
            "labeled_traces": labeled_traces,
            "unlabeled_traces": unlabeled_traces,
            "labeling_progress": (
                round(labeled_traces / total_traces * 100, 2) if total_traces > 0 else 0
            ),
            "labels_by_expert": labels_by_expert,
        }
