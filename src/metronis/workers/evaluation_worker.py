"""
Evaluation Worker

Consumes traces from the queue and runs them through the 5-tier evaluation pipeline.
"""

import asyncio
from pathlib import Path
from typing import Optional

import structlog

from metronis.core.domain import DomainRegistry
from metronis.core.orchestrator import FiveTierOrchestrator, ModuleRegistry
from metronis.core.models import Trace, EvaluationStatus
from metronis.infrastructure.database import get_database
from metronis.infrastructure.repositories.trace_repository import TraceRepository
from metronis.infrastructure.repositories.evaluation_repository import EvaluationRepository
from metronis.services.knowledge_base_service import KnowledgeBaseService
from metronis.services.phi_detector import get_phi_detector

logger = structlog.get_logger(__name__)


class EvaluationWorker:
    """
    Worker that processes evaluation tasks.

    Reads traces from queue, runs evaluation, stores results.
    """

    def __init__(
        self,
        domain_registry_path: Optional[Path] = None,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize the evaluation worker.

        Args:
            domain_registry_path: Path to domains directory
            redis_url: Redis connection URL for knowledge base caching
        """
        # Initialize domain registry
        if domain_registry_path is None:
            domain_registry_path = Path(__file__).parent.parent.parent.parent / "domains"

        self.domain_registry = DomainRegistry(domain_registry_path)
        logger.info(
            "Loaded domains",
            domains=self.domain_registry.list_domains(),
        )

        # Initialize module registry
        self.module_registry = ModuleRegistry()
        self._register_modules()

        # Initialize services
        self.kb_service = KnowledgeBaseService(redis_url=redis_url)
        self.phi_detector = get_phi_detector()

        # Initialize orchestrator
        self.orchestrator = FiveTierOrchestrator(
            domain_registry=self.domain_registry,
            module_registry=self.module_registry,
            knowledge_base_service=self.kb_service,
            data_sanitizer=None,  # PHI detection done in ingestion
        )

        logger.info("Evaluation worker initialized")

    def _register_modules(self) -> None:
        """Register evaluation modules for each domain."""
        # For now, this is a placeholder
        # In production, would dynamically load modules from domain directories

        # TODO: Auto-load Tier-1 modules from domains/*/tier1_modules/
        # TODO: Auto-load Tier-2 models from domains/*/tier2_models/
        # TODO: Auto-load Tier-3 evaluators from domains/*/tier3_prompts/

        logger.info("Modules registered", registry=self.module_registry.modules)

    async def process_trace(self, trace: Trace) -> None:
        """
        Process a single trace through the evaluation pipeline.

        Args:
            trace: Trace to evaluate
        """
        logger.info(
            "Starting trace evaluation",
            trace_id=str(trace.trace_id),
            domain=trace.metadata.domain,
            application_type=trace.application_type,
        )

        try:
            # Update status to in_progress
            await self._update_trace_status(trace.trace_id, EvaluationStatus.IN_PROGRESS)

            # Run evaluation through orchestrator
            evaluation_result = await self.orchestrator.evaluate_trace(trace)

            # Store evaluation result
            await self._store_evaluation_result(evaluation_result)

            # Update trace status
            final_status = (
                EvaluationStatus.COMPLETED
                if evaluation_result.overall_passed
                else EvaluationStatus.FAILED
            )
            await self._update_trace_status(trace.trace_id, final_status)

            logger.info(
                "Trace evaluation completed",
                trace_id=str(trace.trace_id),
                overall_passed=evaluation_result.overall_passed,
                severity=evaluation_result.overall_severity,
                execution_time_ms=evaluation_result.total_execution_time_ms,
                issues_count=len(evaluation_result.all_issues),
            )

        except Exception as e:
            logger.error(
                "Trace evaluation failed",
                trace_id=str(trace.trace_id),
                error=str(e),
                exc_info=True,
            )

            # Update status to failed
            await self._update_trace_status(trace.trace_id, EvaluationStatus.FAILED)

    async def _update_trace_status(self, trace_id, status: EvaluationStatus) -> None:
        """Update trace status in database."""
        db = get_database()
        trace_repo = TraceRepository(db.session)

        # TODO: Implement update_status method in repository
        # await trace_repo.update_status(str(trace_id), status)

        logger.debug(
            "Trace status updated",
            trace_id=str(trace_id),
            status=status,
        )

    async def _store_evaluation_result(self, result) -> None:
        """Store evaluation result in database."""
        db = get_database()
        eval_repo = EvaluationRepository(db.session)

        await eval_repo.create(result)

        logger.debug(
            "Evaluation result stored",
            evaluation_id=str(result.evaluation_id),
            trace_id=str(result.trace_id),
        )

    async def run(self, queue_name: str = "evaluations") -> None:
        """
        Run the worker in a loop, consuming from the queue.

        Args:
            queue_name: Name of the queue to consume from
        """
        logger.info("Starting evaluation worker", queue=queue_name)

        # In production, this would consume from Kafka or Redis queue
        # For now, this is a placeholder

        try:
            while True:
                # TODO: Implement queue consumption
                # trace = await self.queue.dequeue(queue_name)
                # if trace:
                #     await self.process_trace(trace)
                # else:
                #     await asyncio.sleep(1)  # No work, wait

                await asyncio.sleep(1)  # Placeholder

        except KeyboardInterrupt:
            logger.info("Worker shutting down")
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.kb_service.close()
        logger.info("Worker cleaned up")


async def main():
    """Main entry point for the worker."""
    worker = EvaluationWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
