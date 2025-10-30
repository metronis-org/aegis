"""
Analytics API Endpoints

Provides insights, metrics, and trends across evaluations.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import Float, func
from sqlalchemy.ext.asyncio import AsyncSession

from metronis.api.dependencies import get_current_user, get_db
from metronis.core.models import Severity
from metronis.db.models import (
    EvaluationIssueModel,
    EvaluationResultModel,
    TraceModel,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# Response Models
class MetricValue(BaseModel):
    """A single metric value."""

    timestamp: datetime
    value: float


class TimeSeriesMetric(BaseModel):
    """Time series metric data."""

    metric_name: str
    data_points: List[MetricValue]
    aggregation: str  # avg, sum, count, etc.


class DomainStats(BaseModel):
    """Statistics for a domain."""

    domain_name: str
    total_traces: int
    total_evaluations: int
    pass_rate: float
    avg_execution_time_ms: float
    critical_issues_count: int
    high_issues_count: int
    cost_total: float


class IssueBreakdown(BaseModel):
    """Breakdown of issues by type."""

    issue_type: str
    count: int
    severity: str
    percentage: float


class CostBreakdown(BaseModel):
    """Cost breakdown by tier."""

    tier: int
    tier_name: str
    total_cost: float
    invocation_count: int
    avg_cost_per_invocation: float


class ModelPerformance(BaseModel):
    """Performance metrics for AI models."""

    model_name: str
    total_traces: int
    pass_rate: float
    avg_issues_per_trace: float
    critical_issues: int


class AlertStats(BaseModel):
    """Alert statistics."""

    total_alerts: int
    critical_alerts: int
    warning_alerts: int
    info_alerts: int
    alerts_by_domain: Dict[str, int]


class AnalyticsSummary(BaseModel):
    """Overall analytics summary."""

    total_traces: int
    total_evaluations: int
    overall_pass_rate: float
    avg_execution_time_ms: float
    total_cost: float
    domains: List[DomainStats]
    date_range: Dict[str, datetime]


@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    domain: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get overall analytics summary.

    Returns high-level metrics across all evaluations.
    """
    # Default to last 30 days
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Build base query
    query_filters = [
        TraceModel.organization_id == current_user.organization_id,
        TraceModel.created_at >= start_date,
        TraceModel.created_at <= end_date,
    ]

    if domain:
        query_filters.append(TraceModel.domain == domain)

    # Get total traces
    total_traces = (
        await db.scalar(func.count(TraceModel.trace_id).filter(*query_filters)) or 0
    )

    # Get evaluation stats
    eval_stats = await db.execute(
        func.count(EvaluationResultModel.evaluation_id).filter(
            EvaluationResultModel.trace_id.in_(
                db.query(TraceModel.trace_id).filter(*query_filters)
            )
        )
    )
    total_evaluations = eval_stats or 0

    # Calculate pass rate
    passed_evals = (
        await db.scalar(
            func.count(EvaluationResultModel.evaluation_id).filter(
                EvaluationResultModel.trace_id.in_(
                    db.query(TraceModel.trace_id).filter(*query_filters)
                ),
                EvaluationResultModel.overall_passed == True,
            )
        )
        or 0
    )

    pass_rate = (
        (passed_evals / total_evaluations * 100) if total_evaluations > 0 else 0.0
    )

    # Calculate avg execution time
    avg_time = (
        await db.scalar(
            func.avg(EvaluationResultModel.total_execution_time_ms).filter(
                EvaluationResultModel.trace_id.in_(
                    db.query(TraceModel.trace_id).filter(*query_filters)
                )
            )
        )
        or 0.0
    )

    # Calculate total cost
    total_cost = (
        await db.scalar(
            func.sum(EvaluationResultModel.cost).filter(
                EvaluationResultModel.trace_id.in_(
                    db.query(TraceModel.trace_id).filter(*query_filters)
                )
            )
        )
        or 0.0
    )

    # Get domain-specific stats
    domain_stats = await get_domain_statistics(
        db, current_user.organization_id, start_date, end_date
    )

    return AnalyticsSummary(
        total_traces=total_traces,
        total_evaluations=total_evaluations,
        overall_pass_rate=pass_rate,
        avg_execution_time_ms=avg_time,
        total_cost=total_cost,
        domains=domain_stats,
        date_range={"start": start_date, "end": end_date},
    )


@router.get("/timeseries", response_model=List[TimeSeriesMetric])
async def get_timeseries_metrics(
    metric: str = Query(
        ..., description="Metric to retrieve: pass_rate, cost, latency, issue_count"
    ),
    granularity: str = Query("day", description="Granularity: hour, day, week"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    domain: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get time series metrics.

    Provides trend data over time for various metrics.
    """
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Generate time buckets based on granularity
    time_format = {
        "hour": "%Y-%m-%d %H:00:00",
        "day": "%Y-%m-%d",
        "week": "%Y-W%W",
    }.get(granularity, "%Y-%m-%d")

    # Build query based on metric type
    if metric == "pass_rate":
        return await _get_pass_rate_timeseries(
            db, current_user.organization_id, start_date, end_date, domain, time_format
        )
    elif metric == "cost":
        return await _get_cost_timeseries(
            db, current_user.organization_id, start_date, end_date, domain, time_format
        )
    elif metric == "latency":
        return await _get_latency_timeseries(
            db, current_user.organization_id, start_date, end_date, domain, time_format
        )
    elif metric == "issue_count":
        return await _get_issue_count_timeseries(
            db, current_user.organization_id, start_date, end_date, domain, time_format
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown metric: {metric}")


@router.get("/issues/breakdown", response_model=List[IssueBreakdown])
async def get_issues_breakdown(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    domain: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get breakdown of issues by type and severity.
    """
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Query issue breakdown
    query = (
        db.query(
            EvaluationIssueModel.issue_type,
            EvaluationIssueModel.severity,
            func.count(EvaluationIssueModel.id).label("count"),
        )
        .join(EvaluationResultModel)
        .join(TraceModel)
        .filter(
            TraceModel.organization_id == current_user.organization_id,
            TraceModel.created_at >= start_date,
            TraceModel.created_at <= end_date,
        )
    )

    if domain:
        query = query.filter(TraceModel.domain == domain)

    query = query.group_by(
        EvaluationIssueModel.issue_type,
        EvaluationIssueModel.severity,
    ).order_by(func.count(EvaluationIssueModel.id).desc())

    results = await db.execute(query)
    issue_data = results.all()

    # Calculate total for percentages
    total_issues = sum(row.count for row in issue_data)

    return [
        IssueBreakdown(
            issue_type=row.issue_type,
            count=row.count,
            severity=row.severity,
            percentage=(row.count / total_issues * 100) if total_issues > 0 else 0.0,
        )
        for row in issue_data
    ]


@router.get("/cost/breakdown", response_model=List[CostBreakdown])
async def get_cost_breakdown(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    domain: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get cost breakdown by evaluation tier.

    Shows how much each tier is contributing to overall cost.
    """
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # This would query tier-specific cost data from module results
    # Simplified version for now

    tier_names = {
        0: "Tier 0: Pre-processing",
        1: "Tier 1: Heuristics",
        2: "Tier 2: ML/RL",
        3: "Tier 3: LLM/Simulation",
        4: "Tier 4: Expert Review",
    }

    # Mock data - in production, query from module_results table
    return [
        CostBreakdown(
            tier=1,
            tier_name=tier_names[1],
            total_cost=0.0,
            invocation_count=1000,
            avg_cost_per_invocation=0.0,
        ),
        CostBreakdown(
            tier=2,
            tier_name=tier_names[2],
            total_cost=5.0,
            invocation_count=250,
            avg_cost_per_invocation=0.02,
        ),
        CostBreakdown(
            tier=3,
            tier_name=tier_names[3],
            total_cost=50.0,
            invocation_count=40,
            avg_cost_per_invocation=1.25,
        ),
    ]


@router.get("/models/performance", response_model=List[ModelPerformance])
async def get_model_performance(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    domain: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get performance metrics for different AI models.

    Shows which models are performing well vs. poorly.
    """
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Query model performance
    query = (
        db.query(
            TraceModel.model,
            func.count(TraceModel.trace_id).label("total_traces"),
            func.avg(func.cast(EvaluationResultModel.overall_passed, Float)).label(
                "pass_rate"
            ),
        )
        .join(EvaluationResultModel)
        .filter(
            TraceModel.organization_id == current_user.organization_id,
            TraceModel.created_at >= start_date,
            TraceModel.created_at <= end_date,
        )
    )

    if domain:
        query = query.filter(TraceModel.domain == domain)

    query = query.group_by(TraceModel.model).order_by(
        func.count(TraceModel.trace_id).desc()
    )

    results = await db.execute(query)
    model_data = results.all()

    performance_list = []
    for row in model_data:
        # Get issue count for this model
        issue_count = (
            await db.scalar(
                func.count(EvaluationIssueModel.id)
                .join(EvaluationResultModel)
                .join(TraceModel)
                .filter(
                    TraceModel.model == row.model,
                    TraceModel.organization_id == current_user.organization_id,
                    TraceModel.created_at >= start_date,
                    TraceModel.created_at <= end_date,
                )
            )
            or 0
        )

        critical_count = (
            await db.scalar(
                func.count(EvaluationIssueModel.id)
                .join(EvaluationResultModel)
                .join(TraceModel)
                .filter(
                    TraceModel.model == row.model,
                    EvaluationIssueModel.severity == Severity.CRITICAL.value,
                    TraceModel.organization_id == current_user.organization_id,
                    TraceModel.created_at >= start_date,
                    TraceModel.created_at <= end_date,
                )
            )
            or 0
        )

        performance_list.append(
            ModelPerformance(
                model_name=row.model,
                total_traces=row.total_traces,
                pass_rate=row.pass_rate * 100 if row.pass_rate else 0.0,
                avg_issues_per_trace=(
                    issue_count / row.total_traces if row.total_traces > 0 else 0.0
                ),
                critical_issues=critical_count,
            )
        )

    return performance_list


@router.get("/alerts/stats", response_model=AlertStats)
async def get_alert_statistics(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get alert statistics.
    """
    # This would query from alerts table
    # Mock data for now

    return AlertStats(
        total_alerts=45,
        critical_alerts=5,
        warning_alerts=30,
        info_alerts=10,
        alerts_by_domain={
            "healthcare": 25,
            "trading": 15,
            "robotics": 5,
        },
    )


# Helper functions


async def get_domain_statistics(
    db: AsyncSession,
    organization_id: str,
    start_date: datetime,
    end_date: datetime,
) -> List[DomainStats]:
    """Get statistics broken down by domain."""

    # Query domains
    query = (
        db.query(TraceModel.domain)
        .filter(
            TraceModel.organization_id == organization_id,
            TraceModel.created_at >= start_date,
            TraceModel.created_at <= end_date,
        )
        .distinct()
    )

    results = await db.execute(query)
    domains = [row[0] for row in results.all() if row[0]]

    domain_stats = []
    for domain in domains:
        # Get stats for this domain
        total_traces = (
            await db.scalar(
                func.count(TraceModel.trace_id).filter(
                    TraceModel.organization_id == organization_id,
                    TraceModel.domain == domain,
                    TraceModel.created_at >= start_date,
                    TraceModel.created_at <= end_date,
                )
            )
            or 0
        )

        total_evals = (
            await db.scalar(
                func.count(EvaluationResultModel.evaluation_id)
                .join(TraceModel)
                .filter(
                    TraceModel.organization_id == organization_id,
                    TraceModel.domain == domain,
                    TraceModel.created_at >= start_date,
                    TraceModel.created_at <= end_date,
                )
            )
            or 0
        )

        passed_evals = (
            await db.scalar(
                func.count(EvaluationResultModel.evaluation_id)
                .join(TraceModel)
                .filter(
                    TraceModel.organization_id == organization_id,
                    TraceModel.domain == domain,
                    EvaluationResultModel.overall_passed == True,
                    TraceModel.created_at >= start_date,
                    TraceModel.created_at <= end_date,
                )
            )
            or 0
        )

        pass_rate = (passed_evals / total_evals * 100) if total_evals > 0 else 0.0

        avg_time = (
            await db.scalar(
                func.avg(EvaluationResultModel.total_execution_time_ms)
                .join(TraceModel)
                .filter(
                    TraceModel.organization_id == organization_id,
                    TraceModel.domain == domain,
                    TraceModel.created_at >= start_date,
                    TraceModel.created_at <= end_date,
                )
            )
            or 0.0
        )

        critical_issues = (
            await db.scalar(
                func.count(EvaluationIssueModel.id)
                .join(EvaluationResultModel)
                .join(TraceModel)
                .filter(
                    TraceModel.organization_id == organization_id,
                    TraceModel.domain == domain,
                    EvaluationIssueModel.severity == Severity.CRITICAL.value,
                    TraceModel.created_at >= start_date,
                    TraceModel.created_at <= end_date,
                )
            )
            or 0
        )

        high_issues = (
            await db.scalar(
                func.count(EvaluationIssueModel.id)
                .join(EvaluationResultModel)
                .join(TraceModel)
                .filter(
                    TraceModel.organization_id == organization_id,
                    TraceModel.domain == domain,
                    EvaluationIssueModel.severity == Severity.HIGH.value,
                    TraceModel.created_at >= start_date,
                    TraceModel.created_at <= end_date,
                )
            )
            or 0
        )

        total_cost = (
            await db.scalar(
                func.sum(EvaluationResultModel.cost)
                .join(TraceModel)
                .filter(
                    TraceModel.organization_id == organization_id,
                    TraceModel.domain == domain,
                    TraceModel.created_at >= start_date,
                    TraceModel.created_at <= end_date,
                )
            )
            or 0.0
        )

        domain_stats.append(
            DomainStats(
                domain_name=domain,
                total_traces=total_traces,
                total_evaluations=total_evals,
                pass_rate=pass_rate,
                avg_execution_time_ms=avg_time,
                critical_issues_count=critical_issues,
                high_issues_count=high_issues,
                cost_total=total_cost,
            )
        )

    return domain_stats


async def _get_pass_rate_timeseries(
    db: AsyncSession,
    organization_id: str,
    start_date: datetime,
    end_date: datetime,
    domain: Optional[str],
    time_format: str,
) -> List[TimeSeriesMetric]:
    """Get pass rate time series."""
    # Implementation would query database with date bucketing
    # Simplified mock for now
    return [
        TimeSeriesMetric(
            metric_name="pass_rate",
            data_points=[],
            aggregation="avg",
        )
    ]


async def _get_cost_timeseries(
    db: AsyncSession,
    organization_id: str,
    start_date: datetime,
    end_date: datetime,
    domain: Optional[str],
    time_format: str,
) -> List[TimeSeriesMetric]:
    """Get cost time series."""
    return [
        TimeSeriesMetric(
            metric_name="cost",
            data_points=[],
            aggregation="sum",
        )
    ]


async def _get_latency_timeseries(
    db: AsyncSession,
    organization_id: str,
    start_date: datetime,
    end_date: datetime,
    domain: Optional[str],
    time_format: str,
) -> List[TimeSeriesMetric]:
    """Get latency time series."""
    return [
        TimeSeriesMetric(
            metric_name="latency",
            data_points=[],
            aggregation="avg",
        )
    ]


async def _get_issue_count_timeseries(
    db: AsyncSession,
    organization_id: str,
    start_date: datetime,
    end_date: datetime,
    domain: Optional[str],
    time_format: str,
) -> List[TimeSeriesMetric]:
    """Get issue count time series."""
    return [
        TimeSeriesMetric(
            metric_name="issue_count",
            data_points=[],
            aggregation="sum",
        )
    ]
