"""Evaluation API routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from metronis.api.dependencies import get_current_user, get_db
from metronis.db.models import OrganizationModel
from metronis.infrastructure.repositories.evaluation_repository import (
    EvaluationRepository,
)
from metronis.infrastructure.repositories.trace_repository import TraceRepository

router = APIRouter()


@router.get("/evaluations/{evaluation_id}")
async def get_evaluation(
    evaluation_id: UUID,
    db: Session = Depends(get_db),
    current_user: OrganizationModel = Depends(get_current_user),
):
    """Get evaluation result by ID."""
    eval_repo = EvaluationRepository(db)
    evaluation = eval_repo.get_by_id(evaluation_id)

    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    # Check access
    trace_repo = TraceRepository(db)
    trace = trace_repo.get_by_id(evaluation.trace_id)

    if trace.organization_id != current_user.organization_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return evaluation


@router.get("/traces/{trace_id}/evaluation")
async def get_trace_evaluation(
    trace_id: UUID,
    db: Session = Depends(get_db),
    current_user: OrganizationModel = Depends(get_current_user),
):
    """Get evaluation result for a trace."""
    trace_repo = TraceRepository(db)
    trace = trace_repo.get_by_id(trace_id)

    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    if trace.organization_id != current_user.organization_id:
        raise HTTPException(status_code=403, detail="Access denied")

    eval_repo = EvaluationRepository(db)
    evaluation = eval_repo.get_by_trace_id(trace_id)

    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return evaluation
