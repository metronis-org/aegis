"""Unit tests for core models."""

from datetime import datetime
from uuid import uuid4

import pytest

from metronis.core.models import (
    AIProcessing,
    ApplicationType,
    EvaluationIssue,
    EvaluationResult,
    ModuleResult,
    Severity,
    Trace,
)


class TestTrace:
    """Test cases for Trace model."""

    def test_trace_creation(self):
        """Test basic trace creation."""
        ai_processing = AIProcessing(
            model="gpt-4",
            input="Test input",
            output="Test output",
        )

        trace = Trace(
            organization_id=uuid4(),
            application_id=uuid4(),
            ai_processing=ai_processing,
        )

        assert trace.trace_id is not None
        assert trace.application_type == ApplicationType.GENERIC
        assert trace.timestamp is not None
        assert trace.ai_processing.model == "gpt-4"

    def test_trace_validation(self):
        """Test trace validation."""
        ai_processing = AIProcessing(
            model="gpt-4",
            input="Test input",
            output="Test output",
            tokens_used=100,
            latency_ms=1500,
        )

        trace = Trace(
            organization_id=uuid4(),
            application_id=uuid4(),
            ai_processing=ai_processing,
        )

        # Should not raise any validation errors
        assert trace.ai_processing.tokens_used == 100
        assert trace.ai_processing.latency_ms == 1500

    def test_trace_serialization(self):
        """Test trace JSON serialization."""
        ai_processing = AIProcessing(
            model="gpt-4",
            input="Test input",
            output="Test output",
        )

        trace = Trace(
            organization_id=uuid4(),
            application_id=uuid4(),
            ai_processing=ai_processing,
        )

        # Should serialize to JSON without errors
        json_data = trace.json()
        assert "trace_id" in json_data
        assert "ai_processing" in json_data


class TestEvaluationResult:
    """Test cases for EvaluationResult model."""

    def test_evaluation_result_creation(self):
        """Test basic evaluation result creation."""
        result = EvaluationResult(
            trace_id=uuid4(),
        )

        assert result.evaluation_id is not None
        assert result.overall_passed is True
        assert result.overall_severity == Severity.LOW
        assert len(result.tier1_results) == 0

    def test_add_module_result(self):
        """Test adding module results."""
        result = EvaluationResult(trace_id=uuid4())

        # Add a passing Tier 1 result
        module_result = ModuleResult(
            module_name="TestModule",
            tier_level=1,
            passed=True,
            execution_time_ms=10.0,
        )

        result.add_module_result(module_result)

        assert len(result.tier1_results) == 1
        assert result.overall_passed is True
        assert result.total_execution_time_ms == 10.0

    def test_add_failing_module_result(self):
        """Test adding failing module results."""
        result = EvaluationResult(trace_id=uuid4())

        # Add a failing result with issues
        issue = EvaluationIssue(
            type="test_error",
            severity=Severity.HIGH,
            message="Test error message",
        )

        module_result = ModuleResult(
            module_name="TestModule",
            tier_level=1,
            passed=False,
            issues=[issue],
            execution_time_ms=15.0,
        )

        result.add_module_result(module_result)

        assert len(result.tier1_results) == 1
        assert result.overall_passed is False
        assert result.overall_severity == Severity.HIGH
        assert len(result.all_issues) == 1
        assert "test_error" in result.error_types

    def test_finalize_result(self):
        """Test finalizing evaluation result."""
        result = EvaluationResult(trace_id=uuid4())

        assert result.completed_at is None

        result.finalize()

        assert result.completed_at is not None
        assert isinstance(result.completed_at, datetime)


class TestModuleResult:
    """Test cases for ModuleResult model."""

    def test_module_result_creation(self):
        """Test basic module result creation."""
        result = ModuleResult(
            module_name="TestModule",
            tier_level=1,
            passed=True,
        )

        assert result.module_name == "TestModule"
        assert result.tier_level == 1
        assert result.passed is True
        assert len(result.issues) == 0

    def test_module_result_with_issues(self):
        """Test module result with issues."""
        issue = EvaluationIssue(
            type="format_error",
            severity=Severity.MEDIUM,
            message="Invalid format detected",
        )

        result = ModuleResult(
            module_name="FormatValidator",
            tier_level=1,
            passed=False,
            issues=[issue],
            risk_score=0.6,
            confidence=0.9,
        )

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.issues[0].type == "format_error"
        assert result.risk_score == 0.6
        assert result.confidence == 0.9


class TestEvaluationIssue:
    """Test cases for EvaluationIssue model."""

    def test_issue_creation(self):
        """Test basic issue creation."""
        issue = EvaluationIssue(
            type="safety_violation",
            severity=Severity.CRITICAL,
            message="Harmful content detected",
            details={"pattern": "violence", "confidence": 0.95},
        )

        assert issue.type == "safety_violation"
        assert issue.severity == Severity.CRITICAL
        assert issue.message == "Harmful content detected"
        assert issue.details["pattern"] == "violence"

    def test_issue_serialization(self):
        """Test issue JSON serialization."""
        issue = EvaluationIssue(
            type="test_error",
            severity=Severity.LOW,
            message="Test message",
        )

        json_data = issue.json()
        assert "type" in json_data
        assert "severity" in json_data
        assert "message" in json_data
