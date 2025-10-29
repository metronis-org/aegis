"""Trace ingestion endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
import structlog

from metronis.core.models import Trace, EvaluationResult
from metronis.core.exceptions import ValidationError, TraceNotFoundError
from metronis.services.ingestion.dependencies import (
    get_trace_service,
    get_current_organization,
)


logger = structlog.get_logger(__name__)
router = APIRouter()


class TraceResponse(BaseModel):
    """Response model for trace submission."""
    
    trace_id: UUID
    status: str = "accepted"
    message: str = "Trace queued for evaluation"


class BatchTraceResponse(BaseModel):
    """Response model for batch trace submission."""
    
    accepted: int
    rejected: int
    trace_ids: List[UUID]
    errors: List[str] = []


@router.post("/traces", response_model=TraceResponse)
async def submit_trace(
    trace: Trace,
    trace_service=Depends(get_trace_service),
    organization=Depends(get_current_organization),
):
    """Submit a single trace for evaluation."""
    
    try:
        # Set organization ID
        trace.organization_id = organization.organization_id
        
        # Process the trace
        processed_trace = await trace_service.process_trace(trace)
        
        logger.info(
            "Trace submitted successfully",
            trace_id=str(processed_trace.trace_id),
            organization_id=str(organization.organization_id),
        )
        
        return TraceResponse(
            trace_id=processed_trace.trace_id,
            status="accepted",
        )
        
    except ValidationError as e:
        logger.warning(
            "Trace validation failed",
            error=str(e),
            organization_id=str(organization.organization_id),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {e.message}",
        )
    except Exception as e:
        logger.error(
            "Failed to process trace",
            error=str(e),
            organization_id=str(organization.organization_id),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process trace",
        )


@router.post("/traces/batch", response_model=BatchTraceResponse)
async def submit_batch_traces(
    traces: List[Trace],
    trace_service=Depends(get_trace_service),
    organization=Depends(get_current_organization),
):
    """Submit multiple traces for evaluation."""
    
    accepted = 0
    rejected = 0
    trace_ids = []
    errors = []
    
    for trace in traces:
        try:
            # Set organization ID
            trace.organization_id = organization.organization_id
            
            # Process the trace
            processed_trace = await trace_service.process_trace(trace)
            
            accepted += 1
            trace_ids.append(processed_trace.trace_id)
            
        except ValidationError as e:
            rejected += 1
            errors.append(f"Trace validation failed: {e.message}")
            
        except Exception as e:
            rejected += 1
            errors.append(f"Failed to process trace: {str(e)}")
    
    logger.info(
        "Batch traces processed",
        accepted=accepted,
        rejected=rejected,
        organization_id=str(organization.organization_id),
    )
    
    return BatchTraceResponse(
        accepted=accepted,
        rejected=rejected,
        trace_ids=trace_ids,
        errors=errors,
    )


@router.get("/traces/{trace_id}", response_model=Trace)
async def get_trace(
    trace_id: UUID,
    trace_service=Depends(get_trace_service),
    organization=Depends(get_current_organization),
):
    """Get a trace by ID."""
    
    try:
        trace = await trace_service.get_trace(str(trace_id), organization.organization_id)
        
        if not trace:
            raise TraceNotFoundError(f"Trace {trace_id} not found")
        
        return trace
        
    except TraceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace {trace_id} not found",
        )
    except Exception as e:
        logger.error(
            "Failed to get trace",
            trace_id=str(trace_id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trace",
        )


@router.get("/evaluations/{trace_id}", response_model=EvaluationResult)
async def get_evaluation(
    trace_id: UUID,
    trace_service=Depends(get_trace_service),
    organization=Depends(get_current_organization),
):
    """Get evaluation result for a trace."""
    
    try:
        result = await trace_service.get_evaluation_result(
            str(trace_id), 
            organization.organization_id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation result for trace {trace_id} not found",
            )
        
        return result
        
    except Exception as e:
        logger.error(
            "Failed to get evaluation result",
            trace_id=str(trace_id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evaluation result",
        )