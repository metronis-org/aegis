"""
Trace Explorer API

Advanced search, filtering, and exploration of traces.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import or_, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from metronis.api.dependencies import get_db, get_current_user
from metronis.core.models import Trace, EvaluationResult, Severity
from metronis.db.models import TraceModel, EvaluationResultModel, EvaluationIssueModel

router = APIRouter(prefix="/traces", tags=["traces"])


class TraceSummary(BaseModel):
    """Summary of a trace for list views."""
    trace_id: UUID
    organization_id: UUID
    application_type: str
    domain: Optional[str]
    model: str
    created_at: datetime
    evaluation_status: Optional[str]
    overall_passed: Optional[bool]
    issues_count: int
    execution_time_ms: Optional[float]


class TraceDetail(BaseModel):
    """Full trace details."""
    trace: Trace
    evaluation: Optional[EvaluationResult]
    related_traces: List[TraceSummary]


class TraceSearchFilters(BaseModel):
    """Filters for trace search."""
    domain: Optional[str] = None
    application_type: Optional[str] = None
    model: Optional[str] = None
    passed: Optional[bool] = None
    has_critical_issues: Optional[bool] = None
    severity: Optional[Severity] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    search_text: Optional[str] = None  # Search in input/output


class PaginatedTraces(BaseModel):
    """Paginated trace results."""
    traces: List[TraceSummary]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


@router.get("", response_model=PaginatedTraces)
async def list_traces(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    domain: Optional[str] = Query(None),
    application_type: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    passed: Optional[bool] = Query(None),
    has_critical_issues: Optional[bool] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    search_text: Optional[str] = Query(None),
    sort_by: str = Query("created_at", regex="^(created_at|execution_time)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    List traces with filtering and pagination.

    Supports advanced filtering by domain, model, status, etc.
    """
    # Build query
    query = db.query(TraceModel).filter(
        TraceModel.organization_id == current_user.organization_id
    )

    # Apply filters
    if domain:
        query = query.filter(TraceModel.domain == domain)

    if application_type:
        query = query.filter(TraceModel.application_type == application_type)

    if model:
        query = query.filter(TraceModel.model == model)

    if start_date:
        query = query.filter(TraceModel.created_at >= start_date)

    if end_date:
        query = query.filter(TraceModel.created_at <= end_date)

    if search_text:
        search_pattern = f"%{search_text}%"
        query = query.filter(
            or_(
                TraceModel.input_text.ilike(search_pattern),
                TraceModel.output_text.ilike(search_pattern),
            )
        )

    # Join with evaluations for filtering
    if passed is not None or has_critical_issues is not None:
        query = query.join(EvaluationResultModel)

        if passed is not None:
            query = query.filter(EvaluationResultModel.overall_passed == passed)

        if has_critical_issues:
            query = query.join(EvaluationIssueModel).filter(
                EvaluationIssueModel.severity == Severity.CRITICAL.value
            )

    # Get total count
    total = await db.scalar(query.count())

    # Apply sorting
    if sort_order == "desc":
        query = query.order_by(desc(getattr(TraceModel, sort_by)))
    else:
        query = query.order_by(getattr(TraceModel, sort_by))

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    # Execute query
    results = await db.execute(query)
    traces = results.scalars().all()

    # Convert to summaries
    trace_summaries = []
    for trace in traces:
        # Get evaluation for this trace
        eval_result = await db.scalar(
            db.query(EvaluationResultModel)
            .filter(EvaluationResultModel.trace_id == trace.trace_id)
            .order_by(desc(EvaluationResultModel.created_at))
            .limit(1)
        )

        # Count issues
        issue_count = await db.scalar(
            func.count(EvaluationIssueModel.id)
            .filter(EvaluationIssueModel.evaluation_id == eval_result.evaluation_id)
        ) if eval_result else 0

        trace_summaries.append(
            TraceSummary(
                trace_id=trace.trace_id,
                organization_id=trace.organization_id,
                application_type=trace.application_type,
                domain=trace.domain,
                model=trace.model,
                created_at=trace.created_at,
                evaluation_status=eval_result.status if eval_result else None,
                overall_passed=eval_result.overall_passed if eval_result else None,
                issues_count=issue_count,
                execution_time_ms=eval_result.total_execution_time_ms if eval_result else None,
            )
        )

    return PaginatedTraces(
        traces=trace_summaries,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
        has_prev=page > 1,
    )


@router.get("/{trace_id}", response_model=TraceDetail)
async def get_trace_detail(
    trace_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    Get full details for a specific trace.

    Includes the trace, evaluation results, and related traces.
    """
    # Get trace
    trace = await db.get(TraceModel, trace_id)

    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    if trace.organization_id != current_user.organization_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Get evaluation result
    eval_result = await db.scalar(
        db.query(EvaluationResultModel)
        .filter(EvaluationResultModel.trace_id == trace_id)
        .order_by(desc(EvaluationResultModel.created_at))
        .limit(1)
    )

    # Get related traces (same session/conversation)
    related_traces = []
    if trace.session_id:
        related_query = (
            db.query(TraceModel)
            .filter(
                TraceModel.session_id == trace.session_id,
                TraceModel.trace_id != trace_id,
            )
            .order_by(TraceModel.created_at)
            .limit(10)
        )

        related_results = await db.execute(related_query)
        for related in related_results.scalars():
            related_eval = await db.scalar(
                db.query(EvaluationResultModel)
                .filter(EvaluationResultModel.trace_id == related.trace_id)
                .order_by(desc(EvaluationResultModel.created_at))
                .limit(1)
            )

            issue_count = await db.scalar(
                func.count(EvaluationIssueModel.id)
                .filter(EvaluationIssueModel.evaluation_id == related_eval.evaluation_id)
            ) if related_eval else 0

            related_traces.append(
                TraceSummary(
                    trace_id=related.trace_id,
                    organization_id=related.organization_id,
                    application_type=related.application_type,
                    domain=related.domain,
                    model=related.model,
                    created_at=related.created_at,
                    evaluation_status=related_eval.status if related_eval else None,
                    overall_passed=related_eval.overall_passed if related_eval else None,
                    issues_count=issue_count,
                    execution_time_ms=related_eval.total_execution_time_ms if related_eval else None,
                )
            )

    # Convert to Pydantic models
    trace_model = Trace.from_orm(trace)
    eval_model = EvaluationResult.from_orm(eval_result) if eval_result else None

    return TraceDetail(
        trace=trace_model,
        evaluation=eval_model,
        related_traces=related_traces,
    )


@router.post("/search", response_model=PaginatedTraces)
async def search_traces(
    filters: TraceSearchFilters,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    Advanced trace search with complex filters.

    Allows POST-based search with more complex filter combinations.
    """
    return await list_traces(
        page=page,
        page_size=page_size,
        domain=filters.domain,
        application_type=filters.application_type,
        model=filters.model,
        passed=filters.passed,
        has_critical_issues=filters.has_critical_issues,
        start_date=filters.start_date,
        end_date=filters.end_date,
        search_text=filters.search_text,
        db=db,
        current_user=current_user,
    )


@router.get("/{trace_id}/similar", response_model=List[TraceSummary])
async def find_similar_traces(
    trace_id: UUID,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    Find similar traces based on embeddings or characteristics.

    Uses semantic similarity, same model, same domain, etc.
    """
    # Get source trace
    trace = await db.get(TraceModel, trace_id)

    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    if trace.organization_id != current_user.organization_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Find similar traces
    # In production, this would use vector embeddings for semantic similarity
    # For now, use simpler heuristics

    query = (
        db.query(TraceModel)
        .filter(
            TraceModel.organization_id == current_user.organization_id,
            TraceModel.trace_id != trace_id,
            or_(
                TraceModel.domain == trace.domain,
                TraceModel.model == trace.model,
                TraceModel.application_type == trace.application_type,
            ),
        )
        .limit(limit)
    )

    results = await db.execute(query)
    similar_traces = []

    for similar in results.scalars():
        eval_result = await db.scalar(
            db.query(EvaluationResultModel)
            .filter(EvaluationResultModel.trace_id == similar.trace_id)
            .order_by(desc(EvaluationResultModel.created_at))
            .limit(1)
        )

        issue_count = await db.scalar(
            func.count(EvaluationIssueModel.id)
            .filter(EvaluationIssueModel.evaluation_id == eval_result.evaluation_id)
        ) if eval_result else 0

        similar_traces.append(
            TraceSummary(
                trace_id=similar.trace_id,
                organization_id=similar.organization_id,
                application_type=similar.application_type,
                domain=similar.domain,
                model=similar.model,
                created_at=similar.created_at,
                evaluation_status=eval_result.status if eval_result else None,
                overall_passed=eval_result.overall_passed if eval_result else None,
                issues_count=issue_count,
                execution_time_ms=eval_result.total_execution_time_ms if eval_result else None,
            )
        )

    return similar_traces


@router.delete("/{trace_id}")
async def delete_trace(
    trace_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    Delete a trace and its associated evaluations.
    """
    # Get trace
    trace = await db.get(TraceModel, trace_id)

    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    if trace.organization_id != current_user.organization_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Delete trace (cascade will handle evaluations)
    await db.delete(trace)
    await db.commit()

    return {"status": "deleted", "trace_id": str(trace_id)}


@router.post("/bulk-delete")
async def bulk_delete_traces(
    trace_ids: List[UUID],
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    Delete multiple traces in bulk.
    """
    # Verify all traces belong to user
    traces = await db.execute(
        db.query(TraceModel).filter(
            TraceModel.trace_id.in_(trace_ids),
            TraceModel.organization_id == current_user.organization_id,
        )
    )

    traces_to_delete = traces.scalars().all()

    if len(traces_to_delete) != len(trace_ids):
        raise HTTPException(
            status_code=400,
            detail="Some traces not found or access denied",
        )

    # Delete all traces
    for trace in traces_to_delete:
        await db.delete(trace)

    await db.commit()

    return {
        "status": "deleted",
        "count": len(traces_to_delete),
    }
