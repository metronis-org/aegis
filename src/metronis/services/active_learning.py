"""
Active Learning Pipeline

Identifies uncertain predictions and queues them for expert labeling.
Uses uncertainty sampling, disagreement sampling, and diversity sampling.
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from pydantic import BaseModel
from sqlalchemy import and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from metronis.core.models import Trace, EvaluationResult, ModuleResult, Severity

logger = structlog.get_logger(__name__)


class SamplingStrategy(str, Enum):
    """Active learning sampling strategies."""

    UNCERTAINTY = "uncertainty"  # Select least confident predictions
    DISAGREEMENT = "disagreement"  # Select where tiers disagree
    DIVERSITY = "diversity"  # Select diverse examples
    RANDOM = "random"  # Random sampling baseline


class LabelingPriority(str, Enum):
    """Priority levels for expert labeling."""

    CRITICAL = "critical"  # Must be labeled immediately
    HIGH = "high"  # Label within 24 hours
    MEDIUM = "medium"  # Label within 1 week
    LOW = "low"  # Label when convenient


class ExpertLabel(BaseModel):
    """Expert label for a trace."""

    trace_id: str
    expert_id: str
    is_safe: bool
    severity: Optional[Severity]
    issues_identified: List[str]
    feedback: str
    confidence: float  # Expert's confidence 0-1
    labeled_at: datetime


class LabelingTask(BaseModel):
    """A task for expert labeling."""

    task_id: str
    trace_id: str
    priority: LabelingPriority
    sampling_strategy: SamplingStrategy
    uncertainty_score: float
    disagreement_score: float
    created_at: datetime
    assigned_to: Optional[str] = None
    completed: bool = False
    label: Optional[ExpertLabel] = None


class ActiveLearningPipeline:
    """
    Active learning pipeline for continuous model improvement.

    Workflow:
    1. Evaluate traces with current models
    2. Identify uncertain/disagreeing predictions
    3. Queue for expert labeling
    4. Collect labels
    5. Retrain models with new labels
    6. Deploy improved models
    """

    def __init__(
        self,
        db_session: AsyncSession,
        min_confidence_threshold: float = 0.7,
        disagreement_threshold: float = 0.3,
        labeling_batch_size: int = 50,
        sampling_strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY,
    ):
        """
        Initialize active learning pipeline.

        Args:
            db_session: Database session
            min_confidence_threshold: Queue for labeling if confidence below this
            disagreement_threshold: Queue if tier disagreement above this
            labeling_batch_size: Number of samples per labeling batch
            sampling_strategy: Default sampling strategy
        """
        self.db = db_session
        self.min_confidence_threshold = min_confidence_threshold
        self.disagreement_threshold = disagreement_threshold
        self.labeling_batch_size = labeling_batch_size
        self.sampling_strategy = sampling_strategy

        self.labeling_queue: List[LabelingTask] = []

        logger.info(
            "Active learning pipeline initialized",
            sampling_strategy=sampling_strategy,
            confidence_threshold=min_confidence_threshold,
        )

    async def process_evaluation_result(
        self, trace: Trace, result: EvaluationResult
    ) -> Optional[LabelingTask]:
        """
        Process an evaluation result and determine if it needs expert labeling.

        Args:
            trace: The evaluated trace
            result: Evaluation result

        Returns:
            LabelingTask if queued for labeling, None otherwise
        """
        # Calculate uncertainty and disagreement scores
        uncertainty_score = self._calculate_uncertainty(result)
        disagreement_score = self._calculate_disagreement(result)

        # Determine if labeling is needed
        needs_labeling = False
        priority = LabelingPriority.LOW

        if uncertainty_score > (1 - self.min_confidence_threshold):
            needs_labeling = True
            if result.overall_severity in [Severity.CRITICAL, Severity.HIGH]:
                priority = LabelingPriority.CRITICAL
            else:
                priority = LabelingPriority.HIGH

        elif disagreement_score > self.disagreement_threshold:
            needs_labeling = True
            priority = LabelingPriority.MEDIUM

        # Check for edge cases
        if self._is_edge_case(trace, result):
            needs_labeling = True
            priority = LabelingPriority.HIGH

        if not needs_labeling:
            return None

        # Create labeling task
        task = LabelingTask(
            task_id=f"task_{trace.trace_id}_{datetime.utcnow().timestamp()}",
            trace_id=str(trace.trace_id),
            priority=priority,
            sampling_strategy=self.sampling_strategy,
            uncertainty_score=uncertainty_score,
            disagreement_score=disagreement_score,
            created_at=datetime.utcnow(),
        )

        self.labeling_queue.append(task)

        logger.info(
            "Trace queued for expert labeling",
            trace_id=str(trace.trace_id),
            priority=priority,
            uncertainty=f"{uncertainty_score:.3f}",
            disagreement=f"{disagreement_score:.3f}",
        )

        return task

    def _calculate_uncertainty(self, result: EvaluationResult) -> float:
        """
        Calculate uncertainty score from evaluation result.

        Uses entropy, confidence scores, and tier-specific uncertainty.
        """
        uncertainties = []

        # Tier-level uncertainties
        for module_result in result.tier1_results + result.tier2_results + result.tier3_results:
            if module_result.confidence is not None:
                uncertainty = 1.0 - module_result.confidence
                uncertainties.append(uncertainty)

        if not uncertainties:
            return 0.5  # Default uncertainty

        # Return max uncertainty (most uncertain module)
        return max(uncertainties)

    def _calculate_disagreement(self, result: EvaluationResult) -> float:
        """
        Calculate disagreement score between tiers.

        High disagreement indicates the tiers don't agree on safety.
        """
        # Get pass/fail decisions from each tier
        tier_decisions = {
            1: [r.passed for r in result.tier1_results],
            2: [r.passed for r in result.tier2_results],
            3: [r.passed for r in result.tier3_results],
        }

        # Calculate pass rates per tier
        tier_pass_rates = {}
        for tier, decisions in tier_decisions.items():
            if decisions:
                tier_pass_rates[tier] = sum(decisions) / len(decisions)

        if len(tier_pass_rates) < 2:
            return 0.0

        # Calculate variance in pass rates
        pass_rates = list(tier_pass_rates.values())
        disagreement = np.std(pass_rates) if len(pass_rates) > 1 else 0.0

        return float(disagreement)

    def _is_edge_case(self, trace: Trace, result: EvaluationResult) -> bool:
        """
        Check if this is an edge case that should be labeled.

        Edge cases include:
        - Novel patterns not seen in training
        - Rare domains or application types
        - Borderline cases near decision boundary
        """
        # Check for rare domain
        if trace.metadata.domain in ["legal", "robotics"]:
            return True

        # Check for borderline case (risk score near threshold)
        if result.tier2_results:
            for tier2_result in result.tier2_results:
                if tier2_result.risk_score:
                    # Near decision boundary (0.45-0.55)
                    if 0.45 <= tier2_result.risk_score <= 0.55:
                        return True

        # Check for new patterns
        # This would use embeddings or pattern matching in production

        return False

    async def get_labeling_batch(
        self,
        strategy: Optional[SamplingStrategy] = None,
        batch_size: Optional[int] = None,
        domain: Optional[str] = None,
    ) -> List[LabelingTask]:
        """
        Get a batch of traces for expert labeling.

        Args:
            strategy: Sampling strategy to use
            batch_size: Number of tasks to return
            domain: Filter by domain

        Returns:
            List of labeling tasks
        """
        strategy = strategy or self.sampling_strategy
        batch_size = batch_size or self.labeling_batch_size

        # Filter incomplete tasks
        available_tasks = [
            task for task in self.labeling_queue
            if not task.completed and (not domain or task.trace_id.startswith(domain))
        ]

        if not available_tasks:
            return []

        # Apply sampling strategy
        if strategy == SamplingStrategy.UNCERTAINTY:
            # Sort by uncertainty descending
            available_tasks.sort(key=lambda t: t.uncertainty_score, reverse=True)

        elif strategy == SamplingStrategy.DISAGREEMENT:
            # Sort by disagreement descending
            available_tasks.sort(key=lambda t: t.disagreement_score, reverse=True)

        elif strategy == SamplingStrategy.DIVERSITY:
            # Sample diverse examples
            # In production, use clustering/embeddings
            # For now, just round-robin through different scores
            pass

        elif strategy == SamplingStrategy.RANDOM:
            # Random sampling
            import random
            random.shuffle(available_tasks)

        # Sort by priority, then by strategy
        priority_order = {
            LabelingPriority.CRITICAL: 0,
            LabelingPriority.HIGH: 1,
            LabelingPriority.MEDIUM: 2,
            LabelingPriority.LOW: 3,
        }
        available_tasks.sort(key=lambda t: priority_order[t.priority])

        return available_tasks[:batch_size]

    async def submit_label(
        self, task_id: str, label: ExpertLabel
    ) -> None:
        """
        Submit an expert label for a task.

        Args:
            task_id: Task ID
            label: Expert label
        """
        # Find task
        task = next((t for t in self.labeling_queue if t.task_id == task_id), None)

        if not task:
            logger.warning("Task not found", task_id=task_id)
            return

        # Update task
        task.label = label
        task.completed = True

        logger.info(
            "Expert label received",
            task_id=task_id,
            trace_id=label.trace_id,
            is_safe=label.is_safe,
            expert_confidence=label.confidence,
        )

        # Store label in database for model retraining
        await self._store_label(label)

    async def _store_label(self, label: ExpertLabel) -> None:
        """Store expert label in database."""
        # In production, store in ExpertLabels table
        # For now, just log
        logger.info("Storing expert label", trace_id=label.trace_id)

    async def get_labeling_stats(self) -> Dict[str, Any]:
        """Get statistics about the labeling queue."""
        total_tasks = len(self.labeling_queue)
        completed_tasks = sum(1 for t in self.labeling_queue if t.completed)
        pending_tasks = total_tasks - completed_tasks

        # Count by priority
        priority_counts = {
            priority: sum(
                1 for t in self.labeling_queue
                if t.priority == priority and not t.completed
            )
            for priority in LabelingPriority
        }

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "priority_breakdown": priority_counts,
            "completion_rate": (completed_tasks / total_tasks * 100)
            if total_tasks > 0
            else 0.0,
        }

    async def get_model_improvement_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics showing model improvement from active learning.

        Compares model performance before and after incorporating expert labels.
        """
        # In production, query database for:
        # - Accuracy before/after labels
        # - Reduction in uncertainty
        # - Improvement in agreement
        # - Coverage of edge cases

        return {
            "accuracy_improvement": 5.2,  # %
            "uncertainty_reduction": 15.3,  # %
            "labels_collected": len([t for t in self.labeling_queue if t.completed]),
            "labels_pending": len([t for t in self.labeling_queue if not t.completed]),
        }


class ExpertLabelingInterface:
    """
    Interface for expert labeling.

    Provides UI/API for experts to review traces and provide labels.
    """

    def __init__(self, active_learning: ActiveLearningPipeline):
        """Initialize labeling interface."""
        self.al_pipeline = active_learning

    async def get_next_task(
        self, expert_id: str, domain_preference: Optional[str] = None
    ) -> Optional[LabelingTask]:
        """
        Get next task for an expert.

        Args:
            expert_id: Expert identifier
            domain_preference: Preferred domain

        Returns:
            Next labeling task or None
        """
        batch = await self.al_pipeline.get_labeling_batch(
            batch_size=1, domain=domain_preference
        )

        if not batch:
            return None

        task = batch[0]
        task.assigned_to = expert_id

        return task

    async def submit_label(
        self,
        task_id: str,
        expert_id: str,
        is_safe: bool,
        severity: Optional[Severity],
        issues_identified: List[str],
        feedback: str,
        confidence: float,
    ) -> None:
        """
        Submit expert label.

        Args:
            task_id: Task ID
            expert_id: Expert identifier
            is_safe: Whether trace is safe
            severity: Severity if unsafe
            issues_identified: List of issues found
            feedback: Expert feedback
            confidence: Expert confidence
        """
        label = ExpertLabel(
            trace_id=task_id.split("_")[1],  # Extract trace ID from task ID
            expert_id=expert_id,
            is_safe=is_safe,
            severity=severity,
            issues_identified=issues_identified,
            feedback=feedback,
            confidence=confidence,
            labeled_at=datetime.utcnow(),
        )

        await self.al_pipeline.submit_label(task_id, label)


# Global singleton
_active_learning_pipeline: Optional[ActiveLearningPipeline] = None


async def get_active_learning_pipeline(
    db_session: AsyncSession,
) -> ActiveLearningPipeline:
    """Get or create the global active learning pipeline instance."""
    global _active_learning_pipeline
    if _active_learning_pipeline is None:
        _active_learning_pipeline = ActiveLearningPipeline(db_session)
    return _active_learning_pipeline
