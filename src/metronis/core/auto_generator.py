"""
Auto-generation engine for creating evaluation modules from domain specifications.

This module transforms YAML domain specs into:
1. Tier-1 validators (from safety_constraints)
2. Tier-2 ML model training pipelines
3. Tier-3 LLM evaluation prompts
4. Simulator environments
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from jinja2 import Environment, FileSystemLoader, Template

from metronis.core.domain import DomainSpec, SafetyConstraint
from metronis.core.interfaces import EvaluationModule
from metronis.core.models import EvaluationIssue, ModuleResult, Severity, Trace


class AutoGenerator:
    """Main auto-generation engine."""

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize the auto-generator."""
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "templates"

        self.templates_dir = templates_dir
        self.jinja_env = Environment(loader=FileSystemLoader(str(templates_dir)))

    def generate_tier1_module(
        self, constraint: SafetyConstraint, domain_name: str
    ) -> str:
        """Generate a Tier-1 validator from a safety constraint."""
        template = self.jinja_env.get_template("tier1_validator.py.j2")

        # Convert constraint name to PascalCase class name
        class_name = self._to_pascal_case(constraint.name) + "Validator"

        code = template.render(
            class_name=class_name,
            constraint_name=constraint.name,
            description=constraint.description,
            constraint_type=constraint.constraint_type,
            parameters=constraint.parameters,
            severity=constraint.severity,
            domain_name=domain_name,
        )

        return code

    def generate_tier2_model_scaffold(
        self, model_config: Dict[str, Any], domain_name: str
    ) -> str:
        """Generate a Tier-2 ML model training scaffold."""
        template = self.jinja_env.get_template("tier2_model.py.j2")

        code = template.render(
            model_name=model_config.get("name"),
            model_type=model_config.get("model_type"),
            input_features=model_config.get("input_features", []),
            output=model_config.get("output"),
            training_data_source=model_config.get("training_data_source"),
            config=model_config.get("config", {}),
            domain_name=domain_name,
        )

        return code

    def generate_tier3_prompt(
        self, eval_config: Dict[str, Any], domain_name: str
    ) -> str:
        """Generate a Tier-3 LLM evaluation prompt."""
        template = self.jinja_env.get_template("tier3_prompt.txt.j2")

        prompt = template.render(
            eval_name=eval_config.get("name"),
            eval_type=eval_config.get("eval_type"),
            criteria=eval_config.get("criteria", []),
            domain_name=domain_name,
        )

        return prompt

    def generate_simulator_scaffold(
        self, simulator_config: Dict[str, Any], domain_name: str
    ) -> str:
        """Generate a simulator environment scaffold (Gym-compatible)."""
        template = self.jinja_env.get_template("simulator.py.j2")

        code = template.render(
            simulator_name=simulator_config.get("name"),
            simulator_type=simulator_config.get("simulator_type"),
            state_space=simulator_config.get("state_space", {}),
            action_space=simulator_config.get("action_space", {}),
            reward_function=simulator_config.get("reward_function"),
            dynamics=simulator_config.get("dynamics"),
            config=simulator_config.get("config", {}),
            domain_name=domain_name,
        )

        return code

    def generate_domain_modules(
        self, spec: DomainSpec, output_dir: Path
    ) -> Dict[str, List[Path]]:
        """
        Generate all modules for a domain and save to output directory.

        Returns:
            Dictionary with keys: tier1_modules, tier2_models, tier3_prompts, simulators
            Values are lists of generated file paths.
        """
        domain_dir = output_dir / spec.domain_name
        domain_dir.mkdir(parents=True, exist_ok=True)

        generated = {
            "tier1_modules": [],
            "tier2_models": [],
            "tier3_prompts": [],
            "simulators": [],
        }

        # Generate Tier-1 validators
        tier1_dir = domain_dir / "tier1_modules"
        tier1_dir.mkdir(exist_ok=True)

        for constraint in spec.safety_constraints:
            code = self.generate_tier1_module(constraint, spec.domain_name)
            class_name = self._to_pascal_case(constraint.name) + "Validator"
            file_path = tier1_dir / f"{self._to_snake_case(class_name)}.py"

            with open(file_path, "w") as f:
                f.write(code)

            generated["tier1_modules"].append(file_path)

        # Generate Tier-2 models
        tier2_dir = domain_dir / "tier2_models"
        tier2_dir.mkdir(exist_ok=True)

        for model_config in spec.tier2_models:
            code = self.generate_tier2_model_scaffold(
                (
                    model_config.model_dump()
                    if hasattr(model_config, "model_dump")
                    else model_config
                ),
                spec.domain_name,
            )
            model_name = (
                model_config.name
                if hasattr(model_config, "name")
                else model_config.get("name")
            )
            file_path = tier2_dir / f"{self._to_snake_case(model_name)}.py"

            with open(file_path, "w") as f:
                f.write(code)

            generated["tier2_models"].append(file_path)

        # Generate Tier-3 prompts
        tier3_dir = domain_dir / "tier3_prompts"
        tier3_dir.mkdir(exist_ok=True)

        for eval_config in spec.tier3_evals:
            prompt = self.generate_tier3_prompt(
                (
                    eval_config.model_dump()
                    if hasattr(eval_config, "model_dump")
                    else eval_config
                ),
                spec.domain_name,
            )
            eval_name = (
                eval_config.name
                if hasattr(eval_config, "name")
                else eval_config.get("name")
            )
            file_path = tier3_dir / f"{self._to_snake_case(eval_name)}.txt"

            with open(file_path, "w") as f:
                f.write(prompt)

            generated["tier3_prompts"].append(file_path)

        # Generate simulators
        sim_dir = domain_dir / "simulators"
        sim_dir.mkdir(exist_ok=True)

        for sim_config in spec.simulators:
            code = self.generate_simulator_scaffold(
                (
                    sim_config.model_dump()
                    if hasattr(sim_config, "model_dump")
                    else sim_config
                ),
                spec.domain_name,
            )
            sim_name = (
                sim_config.name
                if hasattr(sim_config, "name")
                else sim_config.get("name")
            )
            file_path = sim_dir / f"{self._to_snake_case(sim_name)}.py"

            with open(file_path, "w") as f:
                f.write(code)

            generated["simulators"].append(file_path)

        # Generate __init__.py files for each directory
        self._generate_init_files(domain_dir, generated)

        return generated

    def _generate_init_files(
        self, domain_dir: Path, generated: Dict[str, List[Path]]
    ) -> None:
        """Generate __init__.py files for module directories."""
        # Main domain __init__.py
        init_content = f'"""Auto-generated modules for {domain_dir.name} domain."""\n\n'

        tier1_dir = domain_dir / "tier1_modules"
        if tier1_dir.exists():
            init_content += "from . import tier1_modules\n"
            # Create tier1_modules/__init__.py
            tier1_init = []
            for file_path in generated["tier1_modules"]:
                module_name = file_path.stem
                class_name = (
                    self._to_pascal_case(module_name.replace("_validator", ""))
                    + "Validator"
                )
                tier1_init.append(f"from .{module_name} import {class_name}")

            with open(tier1_dir / "__init__.py", "w") as f:
                f.write("\n".join(tier1_init) + "\n")

        # Similar for other directories...
        for subdir in ["tier2_models", "tier3_prompts", "simulators"]:
            subdir_path = domain_dir / subdir
            if subdir_path.exists():
                with open(subdir_path / "__init__.py", "w") as f:
                    f.write(f'"""Auto-generated {subdir}."""\n')

        with open(domain_dir / "__init__.py", "w") as f:
            f.write(init_content)

    @staticmethod
    def _to_pascal_case(snake_str: str) -> str:
        """Convert snake_case to PascalCase."""
        return "".join(word.capitalize() for word in snake_str.split("_"))

    @staticmethod
    def _to_snake_case(pascal_str: str) -> str:
        """Convert PascalCase to snake_case."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", pascal_str).lower()


class ConstraintValidator:
    """
    Base class for auto-generated constraint validators.
    Auto-generated validators inherit from this.
    """

    def __init__(self, constraint: SafetyConstraint):
        """Initialize with a constraint."""
        self.constraint = constraint
        self.name = self.constraint.name

    def validate_range(
        self, value: float, min_val: Optional[float], max_val: Optional[float]
    ) -> Optional[str]:
        """Helper: Validate a value is within range."""
        if min_val is not None and value < min_val:
            return f"Value {value} below minimum {min_val}"
        if max_val is not None and value > max_val:
            return f"Value {value} exceeds maximum {max_val}"
        return None

    def validate_existence(
        self, entity: str, knowledge_base: str, kb_service: Any
    ) -> bool:
        """Helper: Check if entity exists in knowledge base."""
        result = kb_service.check_existence(knowledge_base, entity)
        return result.get("exists", False)

    def validate_interaction(
        self, entity1: str, entity2: str, kb_service: Any
    ) -> Optional[Dict[str, Any]]:
        """Helper: Check for interactions between entities."""
        return kb_service.check_interaction(entity1, entity2)


# Example of a fully auto-generated validator
class MedicationOverdoseValidator(EvaluationModule):
    """
    Auto-generated validator for medication overdose checks.
    This is an example of what the auto-generator produces.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the validator."""
        super().__init__(config)
        self.max_dose = self.config.get("max_daily_dose", 500)

    def get_tier_level(self) -> int:
        """Return tier level."""
        return 1

    def evaluate(
        self, trace: Trace, context: Optional[Dict[str, Any]] = None
    ) -> ModuleResult:
        """Evaluate the trace for medication overdose."""
        import time

        start_time = time.time()
        issues: List[EvaluationIssue] = []

        # Extract medications from trace
        medications = self._extract_medications(trace)

        for med in medications:
            dosage = med.get("dosage", 0)
            if dosage > self.max_dose:
                issues.append(
                    EvaluationIssue(
                        type="medication_overdose",
                        severity=Severity.CRITICAL,
                        message=f"Medication dosage {dosage}mg exceeds safe limit of {self.max_dose}mg",
                        details={"medication": med, "max_allowed": self.max_dose},
                    )
                )

        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        return ModuleResult(
            module_name=self.name,
            tier_level=1,
            passed=len(issues) == 0,
            issues=issues,
            execution_time_ms=execution_time,
            metadata={"medications_checked": len(medications)},
        )

    def _extract_medications(self, trace: Trace) -> List[Dict[str, Any]]:
        """Extract medication entities from trace output."""
        # This would use NER or regex to extract medications
        # For now, simplified implementation
        medications = []

        # Check if metadata contains medication context
        if hasattr(trace.metadata, "patient_context"):
            # Parse patient context for medications
            # ... implementation ...
            pass

        return medications

    def is_applicable(self, trace: Trace) -> bool:
        """Check if this module applies to the trace."""
        return trace.application_type in [
            "clinical_support",
            "documentation",
        ] or (trace.metadata.domain == "healthcare")
