"""Domain specification and registry for plug-and-play architecture."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml
from pydantic import BaseModel, Field

from metronis.core.interfaces import EvaluationModule


class RiskLevel(str, Enum):
    """Risk levels for domains."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class KnowledgeBaseType(str, Enum):
    """Types of knowledge base integrations."""

    REST_API = "rest_api"
    GRAPHQL = "graphql"
    DATABASE = "database"
    VECTOR_DB = "vector_db"
    RULE_ENGINE = "rule_engine"
    FILE_BASED = "file_based"


class KnowledgeBaseConfig(BaseModel):
    """Configuration for a knowledge base integration."""

    name: str
    type: KnowledgeBaseType
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    cache_ttl: int = 86400  # 24 hours
    config: Dict[str, Any] = Field(default_factory=dict)


class SafetyConstraint(BaseModel):
    """A domain-specific safety constraint."""

    name: str
    description: str
    constraint_type: str  # e.g., "range_check", "interaction_check", "format_check"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "high"  # critical, high, medium, low


class Tier2ModelConfig(BaseModel):
    """Configuration for a Tier 2 ML model."""

    name: str
    model_type: str  # e.g., "bert_classifier", "risk_predictor"
    input_features: List[str]
    output: str
    training_data_source: str = "expert_reviews"
    config: Dict[str, Any] = Field(default_factory=dict)


class Tier3EvalConfig(BaseModel):
    """Configuration for a Tier 3 LLM evaluator."""

    name: str
    eval_type: str  # e.g., "clinical_reasoning", "retrieval_quality"
    criteria: List[str]
    prompt_template: str
    config: Dict[str, Any] = Field(default_factory=dict)


class SimulatorConfig(BaseModel):
    """Configuration for domain-specific simulator."""

    name: str
    simulator_type: str  # e.g., "rl_environment", "physics_based", "discrete_event"
    dynamics: str = "stochastic"  # deterministic, stochastic
    state_space: Dict[str, Any] = Field(default_factory=dict)
    action_space: Dict[str, Any] = Field(default_factory=dict)
    reward_function: str = ""
    transitions: str = ""  # How to learn/define state transitions
    config: Dict[str, Any] = Field(default_factory=dict)


class MetricConfig(BaseModel):
    """Configuration for success metrics."""

    primary: str
    secondary: List[str] = Field(default_factory=list)
    rl_specific: List[str] = Field(default_factory=list)


class DomainSpec(BaseModel):
    """Complete specification for a domain."""

    domain_name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    regulatory_frameworks: List[str] = Field(default_factory=list)

    # Entities in this domain
    entities: Dict[str, List[str]] = Field(default_factory=dict)

    # Safety constraints
    safety_constraints: List[SafetyConstraint] = Field(default_factory=list)

    # Knowledge base integrations
    knowledge_bases: List[KnowledgeBaseConfig] = Field(default_factory=list)

    # Success metrics
    success_metrics: MetricConfig = Field(default_factory=MetricConfig)

    # Simulators
    simulators: List[SimulatorConfig] = Field(default_factory=list)

    # Tier 1 modules (auto-generated from constraints)
    tier1_modules: List[str] = Field(default_factory=list)

    # Tier 2 ML models
    tier2_models: List[Tier2ModelConfig] = Field(default_factory=list)

    # Tier 3 evaluators
    tier3_evals: List[Tier3EvalConfig] = Field(default_factory=list)

    # Transfer metrics configuration
    transfer_metrics: List[str] = Field(default_factory=list)

    # Additional configuration
    config: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DomainSpec":
        """Load domain spec from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save domain spec to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


@dataclass
class Domain:
    """Runtime representation of a domain with loaded modules."""

    spec: DomainSpec
    tier1_modules: List[EvaluationModule] = field(default_factory=list)
    tier2_models: Dict[str, Any] = field(default_factory=dict)
    tier3_evaluators: List[Any] = field(default_factory=list)
    knowledge_bases: Dict[str, Any] = field(default_factory=dict)
    simulators: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Get domain name."""
        return self.spec.domain_name

    @property
    def risk_level(self) -> RiskLevel:
        """Get risk level."""
        return self.spec.risk_level

    def get_applicable_tier1_modules(self, trace_type: str) -> List[EvaluationModule]:
        """Get Tier 1 modules applicable to a trace type."""
        # Filter modules based on trace type
        return [m for m in self.tier1_modules if m.is_applicable_for_type(trace_type)]


class DomainRegistry:
    """
    Central registry for all domains.
    Manages loading, caching, and retrieval of domain specifications.
    """

    def __init__(self, domains_path: Path):
        """Initialize the domain registry."""
        self.domains_path = domains_path
        self.domains: Dict[str, Domain] = {}
        self._load_domains()

    def _load_domains(self) -> None:
        """Load all domains from the domains directory."""
        if not self.domains_path.exists():
            raise ValueError(f"Domains path does not exist: {self.domains_path}")

        for domain_dir in self.domains_path.iterdir():
            if not domain_dir.is_dir():
                continue

            domain_spec_path = domain_dir / "domain_spec.yaml"
            if not domain_spec_path.exists():
                continue

            try:
                spec = DomainSpec.from_yaml(domain_spec_path)
                domain = Domain(spec=spec)
                self.domains[spec.domain_name] = domain
                print(f"Loaded domain: {spec.domain_name}")
            except Exception as e:
                print(f"Failed to load domain from {domain_dir}: {e}")

    def get_domain(self, domain_name: str) -> Optional[Domain]:
        """Get a domain by name."""
        return self.domains.get(domain_name)

    def list_domains(self) -> List[str]:
        """List all available domain names."""
        return list(self.domains.keys())

    def register_domain(self, domain: Domain) -> None:
        """Register a new domain."""
        self.domains[domain.name] = domain

    def get_domain_for_trace(self, trace_metadata: Dict[str, Any]) -> Optional[Domain]:
        """Determine which domain to use for a given trace."""
        domain_name = trace_metadata.get("domain")
        if domain_name:
            return self.get_domain(domain_name)

        # Fallback logic: infer domain from application_type, specialty, etc.
        application_type = trace_metadata.get("application_type")
        specialty = trace_metadata.get("specialty")

        if application_type in ["clinical_support", "diagnostic", "documentation"]:
            return self.get_domain("healthcare")
        elif application_type == "trading_agent":
            return self.get_domain("trading")
        elif application_type == "robotics":
            return self.get_domain("robotics")

        return None

    def reload_domain(self, domain_name: str) -> None:
        """Reload a specific domain from disk."""
        domain_dir = self.domains_path / domain_name
        domain_spec_path = domain_dir / "domain_spec.yaml"

        if not domain_spec_path.exists():
            raise ValueError(f"Domain spec not found: {domain_spec_path}")

        spec = DomainSpec.from_yaml(domain_spec_path)
        domain = Domain(spec=spec)
        self.domains[domain_name] = domain

    def create_domain_from_spec(self, spec: DomainSpec, save: bool = True) -> Domain:
        """Create a new domain from a specification."""
        domain = Domain(spec=spec)

        if save:
            domain_dir = self.domains_path / spec.domain_name
            domain_dir.mkdir(parents=True, exist_ok=True)

            spec_path = domain_dir / "domain_spec.yaml"
            spec.to_yaml(spec_path)

            # Create directory structure
            (domain_dir / "tier1_modules").mkdir(exist_ok=True)
            (domain_dir / "tier2_models").mkdir(exist_ok=True)
            (domain_dir / "tier3_prompts").mkdir(exist_ok=True)
            (domain_dir / "simulators").mkdir(exist_ok=True)
            (domain_dir / "knowledge_bases").mkdir(exist_ok=True)
            (domain_dir / "compliance").mkdir(exist_ok=True)

        self.register_domain(domain)
        return domain
