"""
LLM Service for Tier-3 Evaluations

Integrates with OpenAI and Anthropic APIs for LLM-based evaluations.
"""

import asyncio
import json
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
import structlog
from pydantic import BaseModel

from metronis.core.models import Trace

logger = structlog.get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"


class LLMConfig(BaseModel):
    """Configuration for LLM API."""

    provider: LLMProvider
    api_key: str
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60


class LLMEvaluationResult(BaseModel):
    """Result from LLM evaluation."""

    overall_safety_rating: str  # SAFE, REVIEW, UNSAFE
    criteria_scores: Dict[str, Dict[str, Any]]
    critical_issues: List[str]
    recommendations: List[str]
    confidence: float
    requires_expert_review: bool
    raw_response: str
    tokens_used: int
    cost: float


class LLMService:
    """
    Service for calling LLM APIs for Tier-3 evaluations.

    Supports OpenAI, Anthropic, and Azure OpenAI.
    """

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        default_provider: LLMProvider = LLMProvider.OPENAI,
        default_model: str = "gpt-4-turbo",
    ):
        """
        Initialize LLM service.

        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            azure_api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint
            default_provider: Default LLM provider
            default_model: Default model to use
        """
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.azure_api_key = azure_api_key
        self.azure_endpoint = azure_endpoint
        self.default_provider = default_provider
        self.default_model = default_model

        self.client = httpx.AsyncClient(timeout=90.0)

        logger.info(
            "LLM service initialized",
            default_provider=default_provider,
            default_model=default_model,
        )

    async def evaluate_with_prompt(
        self,
        prompt: str,
        trace: Trace,
        tier1_summary: Optional[str] = None,
        tier2_result: Optional[Dict[str, Any]] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
    ) -> LLMEvaluationResult:
        """
        Evaluate a trace using an LLM with a custom prompt.

        Args:
            prompt: Evaluation prompt template
            trace: Trace to evaluate
            tier1_summary: Summary of Tier-1 results
            tier2_result: Tier-2 evaluation result
            provider: LLM provider to use (defaults to configured)
            model: Model to use (defaults to configured)

        Returns:
            LLM evaluation result
        """
        provider = provider or self.default_provider
        model = model or self.default_model

        # Render prompt with trace data
        rendered_prompt = self._render_prompt(
            prompt, trace, tier1_summary, tier2_result
        )

        logger.info(
            "Calling LLM for evaluation",
            provider=provider,
            model=model,
            prompt_length=len(rendered_prompt),
        )

        # Call LLM based on provider
        if provider == LLMProvider.OPENAI:
            response = await self._call_openai(rendered_prompt, model)
        elif provider == LLMProvider.ANTHROPIC:
            response = await self._call_anthropic(rendered_prompt, model)
        elif provider == LLMProvider.AZURE:
            response = await self._call_azure(rendered_prompt, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Parse response
        result = self._parse_llm_response(response, model)

        logger.info(
            "LLM evaluation completed",
            overall_rating=result.overall_safety_rating,
            confidence=result.confidence,
            cost=f"${result.cost:.4f}",
            tokens_used=result.tokens_used,
        )

        return result

    def _render_prompt(
        self,
        template: str,
        trace: Trace,
        tier1_summary: Optional[str],
        tier2_result: Optional[Dict[str, Any]],
    ) -> str:
        """Render prompt template with trace data."""
        # Replace template variables
        rendered = template.replace(
            "{{trace.ai_processing.input}}", trace.ai_processing.input
        )
        rendered = rendered.replace(
            "{{trace.ai_processing.output}}", trace.ai_processing.output
        )
        rendered = rendered.replace(
            "{{trace.ai_processing.model}}", trace.ai_processing.model
        )

        # Add metadata context
        if trace.metadata.patient_context:
            rendered = rendered.replace(
                "{{trace.metadata.patient_context}}",
                trace.metadata.patient_context,
            )
        if trace.metadata.market_context:
            rendered = rendered.replace(
                "{{trace.metadata.market_context}}",
                trace.metadata.market_context,
            )

        # Add tier results
        if tier1_summary:
            rendered = rendered.replace("{{tier1_summary}}", tier1_summary)
        else:
            rendered = rendered.replace("{{tier1_summary}}", "No Tier-1 results")

        if tier2_result:
            rendered = rendered.replace(
                "{{tier2_result.risk_score}}",
                str(tier2_result.get("risk_score", "N/A")),
            )

        return rendered

    async def _call_openai(self, prompt: str, model: str) -> Dict[str, Any]:
        """Call OpenAI API."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI evaluator. Provide structured, accurate evaluations in JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        }

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            return {
                "content": data["choices"][0]["message"]["content"],
                "input_tokens": data["usage"]["prompt_tokens"],
                "output_tokens": data["usage"]["completion_tokens"],
                "model": model,
            }

        except httpx.HTTPError as e:
            logger.error("OpenAI API error", error=str(e))
            raise

    async def _call_anthropic(self, prompt: str, model: str) -> Dict[str, Any]:
        """Call Anthropic API."""
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "max_tokens": 2000,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            return {
                "content": data["content"][0]["text"],
                "input_tokens": data["usage"]["input_tokens"],
                "output_tokens": data["usage"]["output_tokens"],
                "model": model,
            }

        except httpx.HTTPError as e:
            logger.error("Anthropic API error", error=str(e))
            raise

    async def _call_azure(self, prompt: str, model: str) -> Dict[str, Any]:
        """Call Azure OpenAI API."""
        if not self.azure_api_key or not self.azure_endpoint:
            raise ValueError("Azure OpenAI configuration missing")

        url = f"{self.azure_endpoint}/openai/deployments/{model}/chat/completions?api-version=2023-12-01-preview"
        headers = {
            "api-key": self.azure_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI evaluator. Provide structured, accurate evaluations in JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 2000,
        }

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            return {
                "content": data["choices"][0]["message"]["content"],
                "input_tokens": data["usage"]["prompt_tokens"],
                "output_tokens": data["usage"]["completion_tokens"],
                "model": model,
            }

        except httpx.HTTPError as e:
            logger.error("Azure OpenAI API error", error=str(e))
            raise

    def _parse_llm_response(
        self, response: Dict[str, Any], model: str
    ) -> LLMEvaluationResult:
        """Parse LLM response into structured result."""
        content = response["content"]
        input_tokens = response["input_tokens"]
        output_tokens = response["output_tokens"]

        # Parse JSON response
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response, using defaults")
            data = {
                "overall_safety_rating": "REVIEW",
                "criteria_scores": {},
                "critical_issues": ["Failed to parse LLM response"],
                "recommendations": [],
                "confidence": 0.0,
                "requires_expert_review": True,
            }

        # Calculate cost
        pricing = self.PRICING.get(
            model, {"input": 0.01, "output": 0.03}  # Default pricing
        )
        cost = (input_tokens / 1000 * pricing["input"]) + (
            output_tokens / 1000 * pricing["output"]
        )

        return LLMEvaluationResult(
            overall_safety_rating=data.get("overall_safety_rating", "REVIEW"),
            criteria_scores=data.get("criteria_scores", {}),
            critical_issues=data.get("critical_issues", []),
            recommendations=data.get("recommendations", []),
            confidence=data.get("confidence", 0.0),
            requires_expert_review=data.get("requires_expert_review", True),
            raw_response=content,
            tokens_used=input_tokens + output_tokens,
            cost=cost,
        )

    async def batch_evaluate(
        self,
        evaluations: List[Dict[str, Any]],
        max_concurrent: int = 5,
    ) -> List[LLMEvaluationResult]:
        """
        Evaluate multiple traces concurrently.

        Args:
            evaluations: List of evaluation configs with prompt, trace, etc.
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of evaluation results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def eval_with_semaphore(eval_config: Dict[str, Any]):
            async with semaphore:
                return await self.evaluate_with_prompt(**eval_config)

        results = await asyncio.gather(
            *[eval_with_semaphore(config) for config in evaluations],
            return_exceptions=True,
        )

        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch evaluation failed",
                    index=i,
                    error=str(result),
                )
            else:
                valid_results.append(result)

        return valid_results

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Global singleton
_llm_service: Optional[LLMService] = None


def get_llm_service(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
) -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
        )
    return _llm_service
