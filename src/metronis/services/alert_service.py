"""
Alert Service

Sends notifications when critical evaluation issues are detected.
Supports multiple channels: Email, Slack, Webhook, SMS.
"""

import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
import structlog
from pydantic import BaseModel

from metronis.core.interfaces import AlertService as IAlertService
from metronis.core.models import EvaluationResult, Severity, Trace

logger = structlog.get_logger(__name__)


class AlertChannel(str, Enum):
    """Supported alert channels."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertRule(BaseModel):
    """Configuration for an alert rule."""

    name: str
    description: str
    channels: List[AlertChannel]
    severity_threshold: Severity = Severity.HIGH
    domain_filter: Optional[str] = None  # Only alert for specific domain
    enabled: bool = True


class Alert(BaseModel):
    """An alert to be sent."""

    alert_id: str
    rule_name: str
    severity: AlertSeverity
    title: str
    message: str
    trace_id: str
    organization_id: str
    domain: Optional[str] = None
    details: Dict[str, Any]
    timestamp: datetime
    channels: List[AlertChannel]


class AlertService(IAlertService):
    """
    Service for sending alerts when evaluation issues are detected.

    Supports multiple channels and configurable rules.
    """

    def __init__(
        self,
        email_config: Optional[Dict[str, str]] = None,
        slack_webhook_url: Optional[str] = None,
        webhook_url: Optional[str] = None,
        sms_config: Optional[Dict[str, str]] = None,
        pagerduty_key: Optional[str] = None,
    ):
        """
        Initialize alert service.

        Args:
            email_config: Email configuration (SMTP settings)
            slack_webhook_url: Slack incoming webhook URL
            webhook_url: Generic webhook URL
            sms_config: SMS configuration (Twilio, etc.)
            pagerduty_key: PagerDuty integration key
        """
        self.email_config = email_config
        self.slack_webhook_url = slack_webhook_url
        self.webhook_url = webhook_url
        self.sms_config = sms_config
        self.pagerduty_key = pagerduty_key

        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Alert rules
        self.rules: List[AlertRule] = []

        # Alert history (in-memory, should use database in production)
        self.alert_history: List[Alert] = []

        logger.info("Alert service initialized")

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules.append(rule)
        logger.info("Alert rule added", rule=rule.name, channels=rule.channels)

    async def process_evaluation_result(
        self, trace: Trace, result: EvaluationResult
    ) -> None:
        """
        Process an evaluation result and send alerts if needed.

        Args:
            trace: The evaluated trace
            result: Evaluation result
        """
        # Check which rules are triggered
        triggered_rules = self._check_rules(trace, result)

        if not triggered_rules:
            return

        # Create and send alerts
        for rule in triggered_rules:
            await self._send_alert_for_rule(trace, result, rule)

    def _check_rules(self, trace: Trace, result: EvaluationResult) -> List[AlertRule]:
        """Check which alert rules are triggered."""
        triggered = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Check domain filter
            if rule.domain_filter and trace.metadata.domain != rule.domain_filter:
                continue

            # Check severity threshold
            if result.overall_severity:
                severity_order = {
                    Severity.LOW: 0,
                    Severity.MEDIUM: 1,
                    Severity.HIGH: 2,
                    Severity.CRITICAL: 3,
                }

                result_severity_level = severity_order.get(result.overall_severity, 0)
                threshold_level = severity_order.get(rule.severity_threshold, 2)

                if result_severity_level >= threshold_level:
                    triggered.append(rule)

        return triggered

    async def _send_alert_for_rule(
        self, trace: Trace, result: EvaluationResult, rule: AlertRule
    ) -> None:
        """Send alert for a triggered rule."""
        # Create alert
        alert_severity = self._map_severity(result.overall_severity)

        alert = Alert(
            alert_id=f"alert_{result.evaluation_id}",
            rule_name=rule.name,
            severity=alert_severity,
            title=f"{alert_severity.upper()}: {rule.description}",
            message=self._format_alert_message(trace, result),
            trace_id=str(trace.trace_id),
            organization_id=str(trace.organization_id),
            domain=trace.metadata.domain,
            details={
                "evaluation_id": str(result.evaluation_id),
                "overall_passed": result.overall_passed,
                "issues_count": len(result.all_issues),
                "critical_issues": [
                    issue.message
                    for issue in result.all_issues
                    if issue.severity == Severity.CRITICAL
                ],
            },
            timestamp=datetime.utcnow(),
            channels=rule.channels,
        )

        # Store in history
        self.alert_history.append(alert)

        # Send to each channel
        send_tasks = []
        for channel in rule.channels:
            if channel == AlertChannel.EMAIL and self.email_config:
                send_tasks.append(self._send_email(alert))
            elif channel == AlertChannel.SLACK and self.slack_webhook_url:
                send_tasks.append(self._send_slack(alert))
            elif channel == AlertChannel.WEBHOOK and self.webhook_url:
                send_tasks.append(self._send_webhook(alert))
            elif channel == AlertChannel.SMS and self.sms_config:
                send_tasks.append(self._send_sms(alert))
            elif channel == AlertChannel.PAGERDUTY and self.pagerduty_key:
                send_tasks.append(self._send_pagerduty(alert))

        # Send all alerts concurrently
        if send_tasks:
            results = await asyncio.gather(*send_tasks, return_exceptions=True)

            # Log any failures
            for i, result_item in enumerate(results):
                if isinstance(result_item, Exception):
                    logger.error(
                        "Failed to send alert",
                        channel=rule.channels[i],
                        error=str(result_item),
                    )

        logger.info(
            "Alert sent",
            alert_id=alert.alert_id,
            rule=rule.name,
            channels=rule.channels,
        )

    def _format_alert_message(self, trace: Trace, result: EvaluationResult) -> str:
        """Format alert message."""
        msg = f"Evaluation Alert for Trace {trace.trace_id}\n\n"
        msg += f"Domain: {trace.metadata.domain}\n"
        msg += f"Application: {trace.application_type}\n"
        msg += f"Overall Status: {'PASSED' if result.overall_passed else 'FAILED'}\n"
        msg += f"Severity: {result.overall_severity}\n"
        msg += f"Execution Time: {result.total_execution_time_ms:.2f}ms\n\n"

        if result.all_issues:
            msg += f"Issues Detected ({len(result.all_issues)}):\n"
            for issue in result.all_issues[:5]:  # Limit to first 5
                msg += f"  • [{issue.severity}] {issue.message}\n"

            if len(result.all_issues) > 5:
                msg += f"  ... and {len(result.all_issues) - 5} more\n"

        return msg

    def _map_severity(self, severity: Optional[Severity]) -> AlertSeverity:
        """Map evaluation severity to alert severity."""
        if severity == Severity.CRITICAL:
            return AlertSeverity.CRITICAL
        elif severity in [Severity.HIGH, Severity.MEDIUM]:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    async def _send_email(self, alert: Alert) -> None:
        """Send email alert."""
        # This would integrate with SMTP or email service (SendGrid, AWS SES, etc.)
        logger.info("Sending email alert", alert_id=alert.alert_id)

        # Placeholder - implement actual email sending
        # Example with smtplib:
        # import smtplib
        # from email.mime.text import MIMEText
        # ...

        logger.debug(
            "Email alert sent",
            alert_id=alert.alert_id,
            to=self.email_config.get("recipient") if self.email_config else None,
        )

    async def _send_slack(self, alert: Alert) -> None:
        """Send Slack alert."""
        if not self.slack_webhook_url:
            return

        # Format Slack message
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.CRITICAL: "#ff0000",
        }.get(alert.severity, "#808080")

        payload = {
            "text": alert.title,
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Trace ID",
                            "value": alert.trace_id,
                            "short": True,
                        },
                        {
                            "title": "Domain",
                            "value": alert.domain or "N/A",
                            "short": True,
                        },
                        {
                            "title": "Severity",
                            "value": alert.severity.upper(),
                            "short": True,
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True,
                        },
                    ],
                }
            ],
        }

        try:
            response = await self.http_client.post(
                self.slack_webhook_url,
                json=payload,
            )
            response.raise_for_status()
            logger.info("Slack alert sent", alert_id=alert.alert_id)

        except httpx.HTTPError as e:
            logger.error("Failed to send Slack alert", error=str(e))
            raise

    async def _send_webhook(self, alert: Alert) -> None:
        """Send webhook alert."""
        if not self.webhook_url:
            return

        payload = alert.model_dump(mode="json")

        try:
            response = await self.http_client.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            logger.info("Webhook alert sent", alert_id=alert.alert_id)

        except httpx.HTTPError as e:
            logger.error("Failed to send webhook alert", error=str(e))
            raise

    async def _send_sms(self, alert: Alert) -> None:
        """Send SMS alert."""
        # This would integrate with Twilio, AWS SNS, etc.
        logger.info("Sending SMS alert", alert_id=alert.alert_id)

        # Placeholder - implement actual SMS sending
        logger.debug("SMS alert sent", alert_id=alert.alert_id)

    async def _send_pagerduty(self, alert: Alert) -> None:
        """Send PagerDuty alert."""
        if not self.pagerduty_key:
            return

        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "critical",
        }

        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "severity": severity_map.get(alert.severity, "warning"),
                "source": "metronis-evaluation",
                "custom_details": alert.details,
            },
        }

        try:
            response = await self.http_client.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
            )
            response.raise_for_status()
            logger.info("PagerDuty alert sent", alert_id=alert.alert_id)

        except httpx.HTTPError as e:
            logger.error("Failed to send PagerDuty alert", error=str(e))
            raise

    async def get_alert_history(
        self,
        organization_id: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alert history with optional filters."""
        alerts = self.alert_history

        if organization_id:
            alerts = [a for a in alerts if a.organization_id == organization_id]

        if domain:
            alerts = [a for a in alerts if a.domain == domain]

        # Sort by timestamp descending
        alerts = sorted(alerts, key=lambda a: a.timestamp, reverse=True)

        return alerts[:limit]

    async def close(self) -> None:
        """Close HTTP client."""
        await self.http_client.aclose()


# Global singleton
_alert_service: Optional[AlertService] = None


def get_alert_service(
    slack_webhook_url: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> AlertService:
    """Get or create the global alert service instance."""
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService(
            slack_webhook_url=slack_webhook_url,
            webhook_url=webhook_url,
        )

        # Add default rules
        _alert_service.add_rule(
            AlertRule(
                name="critical_healthcare_issues",
                description="Critical issues in healthcare AI",
                channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY],
                severity_threshold=Severity.CRITICAL,
                domain_filter="healthcare",
            )
        )

        _alert_service.add_rule(
            AlertRule(
                name="trading_high_risk",
                description="High risk trading decisions",
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                severity_threshold=Severity.HIGH,
                domain_filter="trading",
            )
        )

    return _alert_service
