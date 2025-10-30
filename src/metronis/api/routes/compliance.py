'''
Compliance API Routes
'''

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from pydantic import BaseModel

from metronis.db.session import get_db
from metronis.api.dependencies import get_current_user
from metronis.services.compliance_service import ComplianceService
from metronis.db.models import OrganizationModel

router = APIRouter(prefix='/compliance', tags=['compliance'])


@router.get('/fda-tplc')
async def generate_fda_report(
    start_date: datetime = None,
    end_date: datetime = None,
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Generate FDA Total Product Life Cycle report.'''
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()

    compliance = ComplianceService(db)
    report = compliance.generate_fda_tplc_report(
        str(current_user.organization_id),
        start_date,
        end_date,
    )
    return report


@router.get('/hipaa')
async def generate_hipaa_report(
    start_date: datetime = None,
    end_date: datetime = None,
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Generate HIPAA compliance report.'''
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()

    compliance = ComplianceService(db)
    report = compliance.generate_hipaa_report(
        str(current_user.organization_id),
        start_date,
        end_date,
    )
    return report


@router.get('/soc2')
async def generate_soc2_evidence(
    start_date: datetime = None,
    end_date: datetime = None,
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Generate SOC2 audit evidence.'''
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=90)
    if not end_date:
        end_date = datetime.utcnow()

    compliance = ComplianceService(db)
    report = compliance.generate_soc2_evidence(
        str(current_user.organization_id),
        start_date,
        end_date,
    )
    return report


@router.get('/audit-trail')
async def get_audit_trail(
    start_date: datetime = None,
    end_date: datetime = None,
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Get detailed audit trail.'''
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()

    compliance = ComplianceService(db)
    audit_trail = compliance.generate_audit_trail(
        str(current_user.organization_id),
        start_date,
        end_date,
    )
    return {'audit_trail': audit_trail, 'count': len(audit_trail)}
