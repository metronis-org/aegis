"""
Metronis - Unified AI Evaluation Platform

A comprehensive platform for evaluating AI systems across all domains through
a multi-tiered pipeline combining rule-based checks, ML classification, and
LLM-as-judge evaluation.
"""

__version__ = "0.1.0"
__author__ = "Metronis Team"
__email__ = "team@metronis.ai"

from metronis.core.interfaces import EvaluationModule
from metronis.core.models import EvaluationResult, Trace

__all__ = [
    "Trace",
    "EvaluationResult",
    "EvaluationModule",
]
