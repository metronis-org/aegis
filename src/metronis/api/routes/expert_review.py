'''
Expert Review API Routes
'''

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional, List

from metronis.db.session import get_db
from metronis.api.dependencies import get_current_user
from metronis.services.expert_review_service import ExpertReviewService
from metronis.db.models import OrganizationModel

router = APIRouter(prefix='/expert-review', tags=['expert-review'])


class SubmitLabelRequest(BaseModel):
    trace_id: str
    expert_email: EmailStr
    label: str  # 'pass', 'fail', 'needs_review'
    confidence: float  # 0.0 - 1.0
    notes: Optional[str] = None
    issue_categories: Optional[List[str]] = None


@router.get('/queue')
async def get_review_queue(
    domain: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Get traces that need expert review.'''
    service = ExpertReviewService(db)
    queue = service.get_review_queue(
        organization_id=str(current_user.organization_id),
        domain=domain,
        limit=limit,
    )
    return {'queue': queue, 'count': len(queue)}


@router.post('/label')
async def submit_label(
    request: SubmitLabelRequest,
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Submit expert label for a trace.'''
    service = ExpertReviewService(db)
    label = service.submit_label(
        trace_id=request.trace_id,
        expert_email=request.expert_email,
        label=request.label,
        confidence=request.confidence,
        notes=request.notes,
        issue_categories=request.issue_categories,
    )
    return {
        'label_id': str(label.label_id),
        'trace_id': str(label.trace_id),
        'message': 'Label submitted successfully',
    }


@router.get('/labeled')
async def get_labeled_traces(
    expert_email: Optional[EmailStr] = Query(None),
    limit: int = Query(100, le=1000),
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Get traces that have been labeled.'''
    service = ExpertReviewService(db)
    labeled = service.get_labeled_traces(
        organization_id=str(current_user.organization_id),
        expert_email=expert_email,
        limit=limit,
    )
    return {'labeled_traces': labeled, 'count': len(labeled)}


@router.get('/stats')
async def get_labeling_stats(
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Get statistics on labeling progress.'''
    service = ExpertReviewService(db)
    stats = service.get_labeling_stats(str(current_user.organization_id))
    return stats
