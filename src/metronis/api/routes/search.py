"""
Elasticsearch API Routes
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from metronis.api.dependencies import get_current_user
from metronis.db.models import OrganizationModel
from metronis.db.session import get_db
from metronis.services.elasticsearch_service import ElasticsearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/traces")
async def search_traces(
    query: Optional[str] = Query(None, description="Full-text search query"),
    domain: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Search traces with full-text and filters."""
    es = ElasticsearchService()
    results = es.search_traces(
        organization_id=str(current_user.organization_id),
        query=query,
        domain=domain,
        model=model,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    return {"results": results, "count": len(results)}


@router.get("/evaluations")
async def search_evaluations(
    severity: Optional[str] = Query(None),
    passed: Optional[bool] = Query(None),
    limit: int = Query(100, le=1000),
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Search evaluations with filters."""
    es = ElasticsearchService()
    results = es.search_evaluations(
        organization_id=str(current_user.organization_id),
        severity=severity,
        passed=passed,
        limit=limit,
    )
    return {"results": results, "count": len(results)}


@router.get("/aggregations")
async def get_aggregations(
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get aggregation statistics."""
    es = ElasticsearchService()
    aggs = es.get_aggregations(str(current_user.organization_id))
    return aggs
