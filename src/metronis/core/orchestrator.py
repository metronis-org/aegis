"""
5-Tier Evaluation Orchestrator

Coordinates the evaluation pipeline:
- Tier 0: Pre-processing & routing
- Tier 1: Fast heuristic validators (domain-specific rules)
- Tier 2: RL-specific evaluators + ML classification
- Tier 3: Simulation rollouts + LLM-as-judge
- Tier 4: Expert review (active learning)
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from metronis.core.domain import Domain, DomainRegistry
from metronis.core.interfaces import (
    AlertService,
    DataSanitizer,
    EvaluationModule,
    EvaluationOrchestrator,
)
from metronis.core.models import (
    EvaluationResult,
    EvaluationStatus,
    ModuleResult,
    Severity,
    Trace,
)


class ModuleRegistry:
    """Registry for evaluation modules with domain-aware routing."""

    def __init__(self):
        """Initialize the module registry."""
        self.modules: Dict[str, Dict[int, List[EvaluationModule]]] = {}
        # Structure: {domain_name: {tier_level: [modules]}}

    def register_module(
        self, module: EvaluationModule, domain: str, tier: int
    ) -> None:
        """Register an evaluation module for a specific domain and tier."""
        if domain not in self.modules:
            self.modules[domain] = {}
        if tier not in self.modules[domain]:
            self.modules[domain][tier] = []

        self.modules[domain][tier].append(module)

    def get_modules(self, domain: str, tier: int) -> List[EvaluationModule]:
        """Get all modules for a domain and tier."""
        return self.modules.get(domain, {}).get(tier, [])

    def get_applicable_modules(
        self, trace: Trace, domain: str, tier: int
    ) -> List[EvaluationModule]:
        """Get modules applicable to a specific trace."""
        modules = self.get_modules(domain, tier)
        return [m for m in modules if m.is_applicable(trace)]


class FiveTierOrchestrator(EvaluationOrchestrator):
    """
    Main evaluation orchestrator implementing 5-tier pipeline.

    Cost optimization through early-exit:
    - 70% of traces pass Tier 1 (free) and exit
    - 25% escalate to Tier 2 (RL/ML, ~$0.005/trace)
    - 4% require Tier 3 (simulation + LLM, ~$0.10/trace)
    - 1% need Tier 4 (expert review, ~$10/trace)

    Average cost: ~$0.02/trace (10x cheaper than pure LLM eval)
    """

    def __init__(
        self,
        domain_registry: DomainRegistry,
        module_registry: ModuleRegistry,
        knowledge_base_service: Any,
        alert_service: Optional[AlertService] = None,
        data_sanitizer: Optional[DataSanitizer] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the orchestrator."""
        self.domain_registry = domain_registry
        self.module_registry = module_registry
        self.knowledge_base_service = knowledge_base_service
        self.alert_service = alert_service
        self.data_sanitizer = data_sanitizer
        self.config = config or {}

        # Thresholds
        self.tier2_escalation_threshold = self.config.get(
            "tier2_threshold", 0.7
        )  # Escalate if any Tier 1 issue
        self.tier3_escalation_threshold = self.config.get(
            "tier3_threshold", 0.8
        )  # Escalate if Tier 2 risk > 0.8
        self.tier4_escalation_threshold = self.config.get(
            "tier4_threshold", 0.9
        )  # Escalate if Tier 3 unsafe

    async def evaluate_trace(self, trace: Trace) -> EvaluationResult:
        """
        Orchestrate the complete 5-tier evaluation of a trace.

        Returns:
            EvaluationResult with all tier results
        """
        # Initialize evaluation result
        result = EvaluationResult(trace_id=trace.trace_id)

        try:
            # Tier 0: Pre-processing
            trace = await self._tier0_preprocessing(trace)

            # Get domain for this trace
            domain = self.domain_registry.get_domain_for_trace(
                {
                    "domain": trace.metadata.domain,
                    "application_type": trace.application_type,
                    "specialty": trace.metadata.specialty,
                }
            )

            if not domain:
                # No domain found, use generic evaluation
                domain_name = "generic"
            else:
                domain_name = domain.name

            # Context to pass between tiers
            context = {
                "domain": domain_name,
                "knowledge_base_service": self.knowledge_base_service,
            }

            # Tier 1: Fast heuristic validators
            tier1_results = await self._tier1_evaluate(trace, domain_name, context)
            for tier1_result in tier1_results:
                result.add_module_result(tier1_result)

            # Early exit if all Tier 1 checks passed
            tier1_passed = all(r.passed for r in tier1_results)
            if tier1_passed:
                result.finalize()
                return result

            # Check for critical issues requiring immediate alert
            has_critical = any(
                issue.severity == Severity.CRITICAL
                for tier1_result in tier1_results
                for issue in tier1_result.issues
            )

            if has_critical and self.alert_service:
                await self.alert_service.send_alert(trace, result)

            # Tier 2: RL-specific evaluation + ML classification
            if self._should_run_tier2(tier1_results):
                tier2_results = await self._tier2_evaluate(
                    trace, domain_name, context, tier1_results
                )
                for tier2_result in tier2_results:
                    result.add_module_result(tier2_result)

                # Update context with Tier 2 results
                context["tier2_results"] = tier2_results

                # Early exit if low risk
                tier2_risk = self._aggregate_tier2_risk(tier2_results)
                if tier2_risk < self.tier3_escalation_threshold:
                    result.finalize()
                    return result

            # Tier 3: Simulation rollouts + LLM-as-judge
            if self._should_run_tier3(tier1_results, context.get("tier2_results", [])):
                tier3_results = await self._tier3_evaluate(
                    trace, domain_name, context, tier1_results
                )
                for tier3_result in tier3_results:
                    result.add_module_result(tier3_result)

                # Update context
                context["tier3_results"] = tier3_results

                # Check if expert review needed
                if self._should_run_tier4(tier3_results):
                    # Queue for expert review (Tier 4)
                    await self._queue_for_expert_review(trace, result)

            result.finalize()
            return result

        except Exception as e:
            # Log error and return partial result
            print(f"Error during evaluation: {e}")
            result.overall_passed = False
            result.finalize()
            return result

    async def _tier0_preprocessing(self, trace: Trace) -> Trace:
        """
        Tier 0: Pre-processing

        - Sanitize data (remove PII/PHI)
        - Validate schema
        - Route to correct domain
        """
        # Sanitize if data sanitizer available
        if self.data_sanitizer:
            trace = self.data_sanitizer.sanitize_trace(trace)

        # Validate trace schema (basic checks)
        if not trace.ai_processing or not trace.ai_processing.output:
            raise ValueError("Invalid trace: missing AI output")

        # Infer domain if not specified
        if not trace.metadata.domain:
            # Auto-detect based on application_type
            if trace.application_type in ["clinical_support", "diagnostic"]:
                trace.metadata.domain = "healthcare"
            elif trace.application_type == "trading_agent":
                trace.metadata.domain = "trading"

        return trace

    async def _tier1_evaluate(
        self, trace: Trace, domain: str, context: Dict[str, Any]
    ) -> List[ModuleResult]:
        """
        Tier 1: Fast heuristic validators

        Cost: $0/trace
        Latency: <10ms per module
        Coverage: 70% of traces pass here
        """
        results = []

        # Get applicable Tier 1 modules
        modules = self.module_registry.get_applicable_modules(trace, domain, tier=1)

        # Run all Tier 1 modules in parallel
        tasks = [module.evaluate(trace, context) for module in modules]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, ModuleResult)]

        return valid_results

    async def _tier2_evaluate(
        self,
        trace: Trace,
        domain: str,
        context: Dict[str, Any],
        tier1_results: List[ModuleResult],
    ) -> List[ModuleResult]:
        """
        Tier 2: RL-specific evaluation + ML classification

        Cost: $0.001-0.01/trace
        Latency: 100-500ms
        Coverage: 25% escalate here
        """
        results = []

        # Get Tier 2 modules (RL evaluators + ML classifiers)
        modules = self.module_registry.get_applicable_modules(trace, domain, tier=2)

        # Add Tier 1 summary to context
        context["tier1_summary"] = self._summarize_tier1(tier1_results)

        # Run Tier 2 modules
        for module in modules:
            try:
                result = await module.evaluate(trace, context)
                results.append(result)
            except Exception as e:
                print(f"Error in Tier 2 module {module.name}: {e}")

        return results

    async def _tier3_evaluate(
        self,
        trace: Trace,
        domain: str,
        context: Dict[str, Any],
        tier1_results: List[ModuleResult],
    ) -> List[ModuleResult]:
        """
        Tier 3: Simulation rollouts + LLM-as-judge

        Cost: $0.05-0.50/trace
        Latency: 2-30 seconds
        Coverage: 4% require this tier
        """
        results = []

        # Get Tier 3 modules
        modules = self.module_registry.get_applicable_modules(trace, domain, tier=3)

        # Run sequentially due to high cost
        for module in modules:
            try:
                result = await module.evaluate(trace, context)
                results.append(result)

                # Break early if deemed safe
                if result.passed and result.risk_score and result.risk_score < 0.3:
                    break

            except Exception as e:
                print(f"Error in Tier 3 module {module.name}: {e}")

        return results

    async def _queue_for_expert_review(
        self, trace: Trace, eval_result: EvaluationResult
    ) -> None:
        """
        Tier 4: Queue for expert review

        Cost: $5-20/trace
        Latency: Manual
        Purpose: Ground truth generation, active learning
        """
        # This would queue the trace for expert review
        # Implementation depends on your review platform
        # For now, just log it
        print(f"Trace {trace.trace_id} queued for expert review")

        # Could push to a queue, notify experts, etc.
        # await self.expert_review_service.queue(trace, eval_result)

    def _should_run_tier2(self, tier1_results: List[ModuleResult]) -> bool:
        """Determine if Tier 2 evaluation is needed."""
        # Run Tier 2 if any Tier 1 module found issues
        has_issues = any(len(r.issues) > 0 for r in tier1_results)
        return has_issues

    def _should_run_tier3(
        self, tier1_results: List[ModuleResult], tier2_results: List[ModuleResult]
    ) -> bool:
        """Determine if Tier 3 evaluation is needed."""
        if not tier2_results:
            # No Tier 2 ran, check Tier 1 for high severity
            has_high_severity = any(
                issue.severity in [Severity.HIGH, Severity.CRITICAL]
                for r in tier1_results
                for issue in r.issues
            )
            return has_high_severity

        # Check Tier 2 risk scores
        tier2_risk = self._aggregate_tier2_risk(tier2_results)
        return tier2_risk >= self.tier3_escalation_threshold

    def _should_run_tier4(self, tier3_results: List[ModuleResult]) -> bool:
        """Determine if expert review (Tier 4) is needed."""
        # Queue for expert if Tier 3 marked as unsafe
        has_unsafe = any(
            r.metadata.get("safety_rating") == "UNSAFE" for r in tier3_results
        )

        # Or if high uncertainty
        has_high_uncertainty = any(
            r.confidence and r.confidence < 0.7 for r in tier3_results
        )

        return has_unsafe or has_high_uncertainty

    def _aggregate_tier2_risk(self, tier2_results: List[ModuleResult]) -> float:
        """Aggregate risk scores from Tier 2 modules."""
        if not tier2_results:
            return 0.0

        risk_scores = [
            r.risk_score for r in tier2_results if r.risk_score is not None
        ]
        if not risk_scores:
            return 0.0

        # Take max risk score (conservative)
        return max(risk_scores)

    def _summarize_tier1(self, tier1_results: List[ModuleResult]) -> Dict[str, Any]:
        """Summarize Tier 1 results for passing to subsequent tiers."""
        return {
            "total_modules": len(tier1_results),
            "passed": sum(1 for r in tier1_results if r.passed),
            "failed": sum(1 for r in tier1_results if not r.passed),
            "critical_issues": sum(
                1
                for r in tier1_results
                for issue in r.issues
                if issue.severity == Severity.CRITICAL
            ),
            "high_issues": sum(
                1
                for r in tier1_results
                for issue in r.issues
                if issue.severity == Severity.HIGH
            ),
            "error_types": list(
                set(
                    issue.type
                    for r in tier1_results
                    for issue in r.issues
                )
            ),
        }
