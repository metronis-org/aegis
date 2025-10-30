'''
Billing API Routes
'''

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta

from metronis.db.session import get_db
from metronis.api.dependencies import get_current_user
from metronis.services.billing_service import BillingService
from metronis.db.models import OrganizationModel

router = APIRouter(prefix='/billing', tags=['billing'])


class CreateSubscriptionRequest(BaseModel):
    price_id: str = 'price_1234567890'


class UsageRecord(BaseModel):
    metric_type: str
    quantity: int
    metadata: dict = {}


@router.post('/customer')
async def create_customer(
    email: EmailStr,
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Create Stripe customer for organization.'''
    billing = BillingService(db)
    customer_id = billing.create_customer(current_user, email)
    return {'customer_id': customer_id}


@router.post('/subscription')
async def create_subscription(
    request: CreateSubscriptionRequest,
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Create subscription for organization.'''
    billing = BillingService(db)
    subscription = billing.create_subscription(current_user, request.price_id)
    return subscription


@router.post('/usage')
async def record_usage(
    usage: UsageRecord,
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Record usage for billing.'''
    billing = BillingService(db)
    billing.record_usage(
        str(current_user.organization_id),
        usage.metric_type,
        usage.quantity,
        usage.metadata,
    )
    return {'status': 'recorded'}


@router.get('/usage/summary')
async def get_usage_summary(
    start_date: datetime = None,
    end_date: datetime = None,
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Get usage summary for billing period.'''
    if not start_date:
        start_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)
    if not end_date:
        end_date = datetime.utcnow()

    billing = BillingService(db)
    summary = billing.get_usage_summary(
        str(current_user.organization_id),
        start_date,
        end_date,
    )
    return summary


@router.post('/invoice')
async def create_invoice(
    start_date: datetime = None,
    end_date: datetime = None,
    current_user: OrganizationModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    '''Create invoice for billing period.'''
    if not start_date:
        start_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)
    if not end_date:
        end_date = datetime.utcnow()

    billing = BillingService(db)
    invoice = billing.create_invoice(current_user, start_date, end_date)
    return invoice


@router.post('/webhook')
async def stripe_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    '''Handle Stripe webhooks.'''
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')

    billing = BillingService(db)
    result = billing.handle_webhook(payload, sig_header)
    return result
