"""
Billing Service - Stripe Integration

Handles subscription management, usage tracking, and invoicing.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import stripe
import structlog
from sqlalchemy.orm import Session

from metronis.db.models import OrganizationModel, UsageMetricModel

logger = structlog.get_logger(__name__)

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_...")


class BillingService:
    """Manage billing, subscriptions, and usage tracking."""

    def __init__(self, db: Session):
        self.db = db

    def create_customer(self, organization: OrganizationModel, email: str) -> str:
        """Create Stripe customer for organization."""
        try:
            customer = stripe.Customer.create(
                email=email,
                metadata={
                    "organization_id": str(organization.organization_id),
                    "organization_name": organization.name,
                },
            )

            # Store Stripe customer ID
            organization.stripe_customer_id = customer.id
            self.db.commit()

            logger.info(
                "Stripe customer created",
                organization_id=str(organization.organization_id),
                customer_id=customer.id,
            )

            return customer.id

        except stripe.error.StripeError as e:
            logger.error("Stripe customer creation failed", error=str(e))
            raise

    def create_subscription(
        self,
        organization: OrganizationModel,
        price_id: str = "price_1234567890",  # Stripe price ID
    ) -> Dict:
        """Create subscription for organization."""
        try:
            if not organization.stripe_customer_id:
                raise ValueError("Organization has no Stripe customer")

            subscription = stripe.Subscription.create(
                customer=organization.stripe_customer_id,
                items=[{"price": price_id}],
                metadata={
                    "organization_id": str(organization.organization_id),
                },
            )

            logger.info(
                "Subscription created",
                organization_id=str(organization.organization_id),
                subscription_id=subscription.id,
                status=subscription.status,
            )

            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "current_period_end": subscription.current_period_end,
            }

        except stripe.error.StripeError as e:
            logger.error("Subscription creation failed", error=str(e))
            raise

    def record_usage(
        self,
        organization_id: str,
        metric_type: str,
        quantity: int,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record usage for billing."""
        usage_metric = UsageMetricModel(
            organization_id=organization_id,
            metric_type=metric_type,  # e.g., 'trace_evaluation', 'tier3_llm_call'
            quantity=quantity,
            metadata=metadata or {},
            timestamp=datetime.utcnow(),
        )

        self.db.add(usage_metric)
        self.db.commit()

        logger.debug(
            "Usage recorded",
            organization_id=organization_id,
            metric_type=metric_type,
            quantity=quantity,
        )

    def get_usage_summary(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """Get usage summary for billing period."""
        metrics = (
            self.db.query(UsageMetricModel)
            .filter(
                UsageMetricModel.organization_id == organization_id,
                UsageMetricModel.timestamp >= start_date,
                UsageMetricModel.timestamp <= end_date,
            )
            .all()
        )

        summary = {}
        total_cost = 0.0

        for metric in metrics:
            metric_type = metric.metric_type
            if metric_type not in summary:
                summary[metric_type] = {"count": 0, "cost": 0.0}

            # Pricing (example)
            unit_cost = {
                "trace_evaluation": 0.01,  # $0.01 per trace
                "tier3_llm_call": 0.10,  # $0.10 per LLM call
                "expert_label": 0.50,  # $0.50 per expert label
            }.get(metric_type, 0.0)

            summary[metric_type]["count"] += metric.quantity
            cost = metric.quantity * unit_cost
            summary[metric_type]["cost"] += cost
            total_cost += cost

        return {
            "organization_id": organization_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "metrics": summary,
            "total_cost": round(total_cost, 2),
        }

    def create_invoice(
        self,
        organization: OrganizationModel,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """Create invoice for billing period."""
        usage_summary = self.get_usage_summary(
            str(organization.organization_id),
            start_date,
            end_date,
        )

        try:
            # Create invoice items in Stripe
            for metric_type, data in usage_summary["metrics"].items():
                if data["cost"] > 0:
                    stripe.InvoiceItem.create(
                        customer=organization.stripe_customer_id,
                        amount=int(data["cost"] * 100),  # Cents
                        currency="usd",
                        description=f"{metric_type}: {data['count']} units",
                    )

            # Create and finalize invoice
            invoice = stripe.Invoice.create(
                customer=organization.stripe_customer_id,
                auto_advance=True,  # Auto-finalize
            )

            invoice = stripe.Invoice.finalize_invoice(invoice.id)

            logger.info(
                "Invoice created",
                organization_id=str(organization.organization_id),
                invoice_id=invoice.id,
                amount_due=invoice.amount_due / 100,
            )

            return {
                "invoice_id": invoice.id,
                "amount_due": invoice.amount_due / 100,
                "invoice_url": invoice.hosted_invoice_url,
                "status": invoice.status,
            }

        except stripe.error.StripeError as e:
            logger.error("Invoice creation failed", error=str(e))
            raise

    def handle_webhook(self, payload: bytes, sig_header: str) -> Dict:
        """Handle Stripe webhook events."""
        webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_...")

        try:
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)

            logger.info("Webhook received", event_type=event["type"])

            # Handle different event types
            if event["type"] == "invoice.payment_succeeded":
                self._handle_payment_succeeded(event["data"]["object"])

            elif event["type"] == "invoice.payment_failed":
                self._handle_payment_failed(event["data"]["object"])

            elif event["type"] == "customer.subscription.deleted":
                self._handle_subscription_deleted(event["data"]["object"])

            return {"status": "success"}

        except Exception as e:
            logger.error("Webhook handling failed", error=str(e))
            raise

    def _handle_payment_succeeded(self, invoice):
        """Handle successful payment."""
        customer_id = invoice["customer"]
        org = (
            self.db.query(OrganizationModel)
            .filter(OrganizationModel.stripe_customer_id == customer_id)
            .first()
        )

        if org:
            logger.info(
                "Payment succeeded",
                organization_id=str(org.organization_id),
                invoice_id=invoice["id"],
                amount=invoice["amount_paid"] / 100,
            )
            # Update organization status, send receipt email, etc.

    def _handle_payment_failed(self, invoice):
        """Handle failed payment."""
        customer_id = invoice["customer"]
        org = (
            self.db.query(OrganizationModel)
            .filter(OrganizationModel.stripe_customer_id == customer_id)
            .first()
        )

        if org:
            logger.warning(
                "Payment failed",
                organization_id=str(org.organization_id),
                invoice_id=invoice["id"],
            )
            # Send notification, suspend service if needed

    def _handle_subscription_deleted(self, subscription):
        """Handle subscription cancellation."""
        customer_id = subscription["customer"]
        org = (
            self.db.query(OrganizationModel)
            .filter(OrganizationModel.stripe_customer_id == customer_id)
            .first()
        )

        if org:
            logger.info(
                "Subscription cancelled",
                organization_id=str(org.organization_id),
                subscription_id=subscription["id"],
            )
            # Handle subscription end, data retention, etc.
