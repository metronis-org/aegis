#!/usr/bin/env python
'''
Customer Onboarding Automation

Automates creation of new customer accounts with:
- Organization creation
- API key generation
- Database setup
- Welcome email
'''

import argparse
import secrets
import string
from datetime import datetime
from sqlalchemy.orm import Session

from metronis.db.session import SessionLocal
from metronis.db.models import OrganizationModel
from metronis.services.billing_service import BillingService


def generate_api_key(length: int = 32) -> str:
    '''Generate secure API key.'''
    alphabet = string.ascii_letters + string.digits
    return 'metronis_' + ''.join(secrets.choice(alphabet) for _ in range(length))


def onboard_customer(
    name: str,
    email: str,
    db: Session,
    create_stripe_customer: bool = True,
) -> dict:
    '''Onboard new customer organization.'''

    print(f'Onboarding customer: {name} ({email})')

    # 1. Create organization
    api_key = generate_api_key()

    organization = OrganizationModel(
        name=name,
        api_key=api_key,
        created_at=datetime.utcnow(),
    )

    db.add(organization)
    db.commit()
    db.refresh(organization)

    print(f'[OK] Organization created: {organization.organization_id}')
    print(f'[OK] API Key: {api_key}')

    # 2. Create Stripe customer (optional)
    stripe_customer_id = None
    if create_stripe_customer:
        try:
            billing = BillingService(db)
            stripe_customer_id = billing.create_customer(organization, email)
            print(f'[OK] Stripe customer created: {stripe_customer_id}')
        except Exception as e:
            print(f'[WARNING] Stripe customer creation failed: {e}')

    # 3. Send welcome email (placeholder)
    print(f'[TODO] Send welcome email to {email}')

    result = {
        'organization_id': str(organization.organization_id),
        'organization_name': name,
        'api_key': api_key,
        'email': email,
        'stripe_customer_id': stripe_customer_id,
        'created_at': organization.created_at.isoformat(),
    }

    print('\n' + '=' * 60)
    print('ONBOARDING COMPLETE')
    print('=' * 60)
    print(f'Organization ID: {result["organization_id"]}')
    print(f'API Key: {result["api_key"]}')
    print(f'Email: {result["email"]}')
    print('=' * 60)

    return result


def main():
    '''Main CLI entrypoint.'''
    parser = argparse.ArgumentParser(description='Onboard new customer')
    parser.add_argument('--name', required=True, help='Organization name')
    parser.add_argument('--email', required=True, help='Customer email')
    parser.add_argument('--no-stripe', action='store_true', help='Skip Stripe customer creation')

    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = onboard_customer(
            name=args.name,
            email=args.email,
            db=db,
            create_stripe_customer=not args.no_stripe,
        )

        # Print integration instructions
        print('\nNext steps:')
        print('1. Share the API key with the customer')
        print('2. Point them to documentation: https://docs.metronis.ai')
        print('3. Schedule onboarding call if needed')

    finally:
        db.close()


if __name__ == '__main__':
    main()
