'''
Unit tests for billing service
'''

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from metronis.services.billing_service import BillingService
from metronis.db.models import OrganizationModel, UsageMetricModel


@pytest.fixture
def db_session():
    '''Mock database session.'''
    return Mock()


@pytest.fixture
def organization():
    '''Mock organization.'''
    org = OrganizationModel(
        organization_id='123e4567-e89b-12d3-a456-426614174000',
        name='Test Org',
        stripe_customer_id='cus_test123',
    )
    return org


@pytest.fixture
def billing_service(db_session):
    '''Billing service instance.'''
    return BillingService(db_session)


class TestBillingService:
    '''Test billing service methods.'''

    @patch('metronis.services.billing_service.stripe.Customer.create')
    def test_create_customer(self, mock_stripe, billing_service, organization, db_session):
        '''Test Stripe customer creation.'''
        mock_stripe.return_value = Mock(id='cus_new123')

        customer_id = billing_service.create_customer(organization, 'test@example.com')

        assert customer_id == 'cus_new123'
        assert organization.stripe_customer_id == 'cus_new123'
        db_session.commit.assert_called_once()

    def test_record_usage(self, billing_service, db_session):
        '''Test usage recording.'''
        billing_service.record_usage(
            organization_id='123e4567-e89b-12d3-a456-426614174000',
            metric_type='trace_evaluation',
            quantity=10,
        )

        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()

    def test_get_usage_summary(self, billing_service, db_session):
        '''Test usage summary calculation.'''
        # Mock query results
        mock_metrics = [
            Mock(
                metric_type='trace_evaluation',
                quantity=100,
            ),
            Mock(
                metric_type='tier3_llm_call',
                quantity=20,
            ),
        ]

        db_session.query.return_value.filter.return_value.all.return_value = mock_metrics

        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()

        summary = billing_service.get_usage_summary(
            '123e4567-e89b-12d3-a456-426614174000',
            start_date,
            end_date,
        )

        assert 'trace_evaluation' in summary['metrics']
        assert summary['metrics']['trace_evaluation']['count'] == 100
        assert summary['total_cost'] > 0
