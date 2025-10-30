"""Trace API routes."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from metronis.api.dependencies import get_db, get_current_user
from metronis.core.models import Trace
from metronis.infrastructure.repositories.trace_repository import TraceRepository
from metronis.db.models import OrganizationModel

router = APIRouter()


@router.post("/traces", status_code=201)
async def submit_trace(
    trace: Trace,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: OrganizationModel = Depends(get_current_user),
):
    """Submit a trace for evaluation."""
    # Set organization_id from authenticated user
    trace.organization_id = current_user.organization_id

    # Save trace to database
    trace_repo = TraceRepository(db)
    db_trace = trace_repo.create(trace)

    # Queue for evaluation (background task)
    # background_tasks.add_task(queue_trace_for_evaluation, trace)

    return {
        "trace_id": str(db_trace.trace_id),
        "status": "queued",
        "message": "Trace submitted for evaluation",
    }


@router.get("/traces/{trace_id}")
async def get_trace(
    trace_id: UUID,
    db: Session = Depends(get_db),
    current_user: OrganizationModel = Depends(get_current_user),
):
    """Get trace by ID."""
    trace_repo = TraceRepository(db)
    trace = trace_repo.get_by_id(trace_id)

    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    if trace.organization_id != current_user.organization_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return trace


@router.get("/traces")
async def list_traces(
    domain: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: OrganizationModel = Depends(get_current_user),
):
    """List traces for authenticated organization."""
    trace_repo = TraceRepository(db)
    traces = trace_repo.list_by_organization(
        current_user.organization_id,
        domain=domain,
        limit=limit,
        offset=offset,
    )

    return {
        "traces": traces,
        "total": len(traces),
        "limit": limit,
        "offset": offset,
    }


@router.delete("/traces/{trace_id}")
async def delete_trace(
    trace_id: UUID,
    db: Session = Depends(get_db),
    current_user: OrganizationModel = Depends(get_current_user),
):
    """Delete a trace."""
    trace_repo = TraceRepository(db)
    trace = trace_repo.get_by_id(trace_id)

    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    if trace.organization_id != current_user.organization_id:
        raise HTTPException(status_code=403, detail="Access denied")

    trace_repo.delete(trace_id)

    return {"status": "deleted", "trace_id": str(trace_id)}
