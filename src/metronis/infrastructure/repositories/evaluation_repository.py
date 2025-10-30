"""Evaluation repository."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from metronis.core.models import EvaluationResult, EvaluationStatus
from metronis.db.models import EvaluationIssueModel, EvaluationResultModel


class EvaluationRepository:
    """Repository for evaluation results."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, result: EvaluationResult) -> EvaluationResultModel:
        """Create evaluation result."""
        db_result = EvaluationResultModel(
            evaluation_id=result.evaluation_id,
            trace_id=result.trace_id,
            overall_passed=result.overall_passed,
            overall_severity=result.overall_severity,
            status=result.status,
            total_execution_time_ms=result.total_execution_time_ms,
            cost=result.cost,
            tier1_results=[r.model_dump() for r in result.tier1_results],
            tier2_results=[r.model_dump() for r in result.tier2_results],
            tier3_results=[r.model_dump() for r in result.tier3_results],
        )
        self.db.add(db_result)

        # Add issues
        for issue in result.all_issues:
            db_issue = EvaluationIssueModel(
                evaluation_id=result.evaluation_id,
                issue_type=issue.type,
                severity=issue.severity,
                message=issue.message,
                details=issue.details,
            )
            self.db.add(db_issue)

        self.db.commit()
        self.db.refresh(db_result)
        return db_result

    def update_status(self, evaluation_id: UUID, status: EvaluationStatus) -> None:
        """Update evaluation status."""
        self.db.query(EvaluationResultModel).filter(
            EvaluationResultModel.evaluation_id == evaluation_id
        ).update({"status": status, "completed_at": datetime.utcnow()})
        self.db.commit()

    def get_by_id(self, evaluation_id: UUID) -> Optional[EvaluationResultModel]:
        """Get evaluation by ID."""
        return (
            self.db.query(EvaluationResultModel)
            .filter(EvaluationResultModel.evaluation_id == evaluation_id)
            .first()
        )

    def get_by_trace_id(self, trace_id: UUID) -> Optional[EvaluationResultModel]:
        """Get latest evaluation for a trace."""
        return (
            self.db.query(EvaluationResultModel)
            .filter(EvaluationResultModel.trace_id == trace_id)
            .order_by(EvaluationResultModel.created_at.desc())
            .first()
        )
