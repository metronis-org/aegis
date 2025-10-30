'''
Onboarding API Routes - Self-serve signup
'''

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import secrets
import string

from metronis.db.session import get_db
from metronis.db.models import OrganizationModel
from metronis.services.billing_service import BillingService
from datetime import datetime

router = APIRouter(prefix='/onboarding', tags=['onboarding'])


class SignupRequest(BaseModel):
    organization_name: str
    email: EmailStr
    create_stripe_customer: bool = True


def generate_api_key(length: int = 32) -> str:
    '''Generate secure API key.'''
    alphabet = string.ascii_letters + string.digits
    return 'metronis_' + ''.join(secrets.choice(alphabet) for _ in range(length))


@router.post('/signup')
async def signup(
    request: SignupRequest,
    db: Session = Depends(get_db),
):
    '''Self-serve customer signup.'''

    # Check if organization already exists
    existing = db.query(OrganizationModel).filter(
        OrganizationModel.name == request.organization_name
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail='Organization name already exists')

    # Create organization
    api_key = generate_api_key()

    organization = OrganizationModel(
        name=request.organization_name,
        api_key=api_key,
        created_at=datetime.utcnow(),
    )

    db.add(organization)
    db.commit()
    db.refresh(organization)

    # Create Stripe customer
    stripe_customer_id = None
    if request.create_stripe_customer:
        try:
            billing = BillingService(db)
            stripe_customer_id = billing.create_customer(organization, request.email)
        except Exception as e:
            # Non-blocking - log error but continue
            pass

    return {
        'organization_id': str(organization.organization_id),
        'organization_name': organization.name,
        'api_key': api_key,
        'email': request.email,
        'stripe_customer_id': stripe_customer_id,
        'message': 'Welcome to Metronis Aegis! Use your API key to start sending traces.',
        'docs_url': 'https://docs.metronis.ai/quickstart',
    }
