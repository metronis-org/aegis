"""
Metronis CLI

Command-line interface for domain generation and management.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
import structlog

from metronis.core.auto_generator import AutoGenerator
from metronis.core.domain import DomainRegistry

logger = structlog.get_logger(__name__)


@click.group()
def cli():
    """Metronis CLI - AI Evaluation Infrastructure"""
    pass


@cli.command()
@click.argument("domain_name")
@click.option(
    "--domains-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to domains directory (defaults to ./domains)",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for generated modules (defaults to ./generated)",
)
@click.option(
    "--tier",
    type=click.Choice(["1", "2", "3", "simulator", "all"]),
    default="all",
    help="Which tier to generate (default: all)",
)
def generate_domain(
    domain_name: str,
    domains_path: Optional[Path],
    output_path: Optional[Path],
    tier: str,
):
    """
    Generate evaluation modules from a domain specification.

    DOMAIN_NAME: Name of the domain to generate (e.g., healthcare, trading)

    Example:
        metronis generate-domain healthcare
        metronis generate-domain trading --tier 1
    """
    # Set default paths
    if domains_path is None:
        domains_path = Path.cwd() / "domains"

    if output_path is None:
        output_path = Path.cwd() / "generated" / domain_name

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'='*80}")
    click.echo(f"Metronis Auto-Generation")
    click.echo(f"{'='*80}\n")

    try:
        # Load domain registry
        click.echo(f"Loading domains from: {domains_path}")
        registry = DomainRegistry(domains_path)

        # Get domain spec
        domain = registry.get_domain(domain_name)
        if domain is None:
            click.secho(
                f"❌ Domain '{domain_name}' not found in registry",
                fg="red",
            )
            click.echo(f"\nAvailable domains: {', '.join(registry.list_domains())}")
            sys.exit(1)

        click.secho(f"✅ Found domain: {domain_name}", fg="green")
        click.echo(f"   Risk Level: {domain.risk_level.value}")
        click.echo(
            f"   Regulatory Frameworks: {', '.join(domain.spec.regulatory_frameworks)}"
        )
        click.echo(f"   Safety Constraints: {len(domain.spec.safety_constraints)}")
        click.echo(f"   Knowledge Bases: {len(domain.spec.knowledge_bases)}")

        # Initialize auto-generator
        generator = AutoGenerator()

        # Generate modules based on tier selection
        generated_files = []

        if tier == "all" or tier == "1":
            click.echo(f"\n{'─'*80}")
            click.echo("Generating Tier-1 Validators...")
            click.echo(f"{'─'*80}")

            tier1_dir = output_path / "tier1_modules"
            tier1_dir.mkdir(exist_ok=True)

            for constraint in domain.spec.safety_constraints:
                try:
                    # Generate module code
                    code = generator.generate_tier1_module(constraint, domain_name)

                    # Write to file
                    filename = f"{constraint.name}_validator.py"
                    filepath = tier1_dir / filename
                    filepath.write_text(code, encoding="utf-8")

                    generated_files.append(filepath)
                    click.secho(f"  ✅ {filename}", fg="green")

                except Exception as e:
                    click.secho(
                        f"  ❌ Failed to generate {constraint.name}: {e}",
                        fg="red",
                    )

            # Create __init__.py
            init_file = tier1_dir / "__init__.py"
            init_file.write_text(
                f'"""Auto-generated Tier-1 validators for {domain_name} domain."""\n',
                encoding="utf-8",
            )
            click.echo(f"  ✅ __init__.py")

        if tier == "all" or tier == "2":
            click.echo(f"\n{'─'*80}")
            click.echo("Generating Tier-2 ML Models...")
            click.echo(f"{'─'*80}")

            tier2_dir = output_path / "tier2_models"
            tier2_dir.mkdir(exist_ok=True)

            for constraint in domain.spec.safety_constraints:
                if constraint.ml_model_type:
                    try:
                        code = generator.generate_tier2_model(constraint, domain_name)
                        filename = f"{constraint.name}_model.py"
                        filepath = tier2_dir / filename
                        filepath.write_text(code, encoding="utf-8")

                        generated_files.append(filepath)
                        click.secho(f"  ✅ {filename}", fg="green")

                    except Exception as e:
                        click.secho(
                            f"  ❌ Failed to generate {constraint.name}: {e}",
                            fg="red",
                        )

            if generated_files:
                init_file = tier2_dir / "__init__.py"
                init_file.write_text(
                    f'"""Auto-generated Tier-2 models for {domain_name} domain."""\n',
                    encoding="utf-8",
                )
                click.echo(f"  ✅ __init__.py")
            else:
                click.echo("  ℹ️  No Tier-2 models to generate (no ml_model_type specified)")

        if tier == "all" or tier == "3":
            click.echo(f"\n{'─'*80}")
            click.echo("Generating Tier-3 LLM Prompts...")
            click.echo(f"{'─'*80}")

            tier3_dir = output_path / "tier3_prompts"
            tier3_dir.mkdir(exist_ok=True)

            for constraint in domain.spec.safety_constraints:
                try:
                    prompt = generator.generate_tier3_prompt(constraint, domain_name)
                    filename = f"{constraint.name}_prompt.txt"
                    filepath = tier3_dir / filename
                    filepath.write_text(prompt, encoding="utf-8")

                    generated_files.append(filepath)
                    click.secho(f"  ✅ {filename}", fg="green")

                except Exception as e:
                    click.secho(
                        f"  ❌ Failed to generate {constraint.name}: {e}",
                        fg="red",
                    )

        if tier == "all" or tier == "simulator":
            click.echo(f"\n{'─'*80}")
            click.echo("Generating Simulator...")
            click.echo(f"{'─'*80}")

            if domain.spec.simulator:
                try:
                    code = generator.generate_simulator(
                        domain.spec.simulator, domain_name
                    )
                    filename = f"{domain_name}_simulator.py"
                    filepath = output_path / filename
                    filepath.write_text(code, encoding="utf-8")

                    generated_files.append(filepath)
                    click.secho(f"  ✅ {filename}", fg="green")

                except Exception as e:
                    click.secho(f"  ❌ Failed to generate simulator: {e}", fg="red")
            else:
                click.echo("  ℹ️  No simulator configuration found")

        # Summary
        click.echo(f"\n{'='*80}")
        click.echo("Generation Complete")
        click.echo(f"{'='*80}\n")
        click.secho(f"✅ Generated {len(generated_files)} files", fg="green")
        click.echo(f"📁 Output directory: {output_path}")

        click.echo("\nNext steps:")
        click.echo(f"  1. Review generated files in {output_path}")
        click.echo(f"  2. Test validators: python -m pytest {output_path}/tier1_modules/")
        click.echo(f"  3. Register modules with ModuleRegistry")

    except Exception as e:
        click.secho(f"\n❌ Error: {e}", fg="red")
        logger.error("Generation failed", error=str(e), exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--domains-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to domains directory (defaults to ./domains)",
)
def list_domains(domains_path: Optional[Path]):
    """List all available domains."""
    if domains_path is None:
        domains_path = Path.cwd() / "domains"

    try:
        registry = DomainRegistry(domains_path)
        domains = registry.list_domains()

        click.echo(f"\n{'='*80}")
        click.echo("Available Domains")
        click.echo(f"{'='*80}\n")

        if not domains:
            click.echo("No domains found.")
            return

        for domain_name in sorted(domains):
            domain = registry.get_domain(domain_name)
            if domain:
                click.secho(f"• {domain_name}", fg="cyan", bold=True)
                click.echo(f"  Risk Level: {domain.risk_level.value}")
                click.echo(
                    f"  Constraints: {len(domain.spec.safety_constraints)}"
                )
                click.echo(
                    f"  Knowledge Bases: {len(domain.spec.knowledge_bases)}"
                )
                if domain.spec.regulatory_frameworks:
                    click.echo(
                        f"  Compliance: {', '.join(domain.spec.regulatory_frameworks[:3])}"
                    )
                click.echo()

    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.argument("domain_name")
@click.option(
    "--domains-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to domains directory (defaults to ./domains)",
)
def inspect_domain(domain_name: str, domains_path: Optional[Path]):
    """Inspect a domain specification in detail."""
    if domains_path is None:
        domains_path = Path.cwd() / "domains"

    try:
        registry = DomainRegistry(domains_path)
        domain = registry.get_domain(domain_name)

        if domain is None:
            click.secho(f"❌ Domain '{domain_name}' not found", fg="red")
            click.echo(f"\nAvailable domains: {', '.join(registry.list_domains())}")
            sys.exit(1)

        click.echo(f"\n{'='*80}")
        click.echo(f"Domain: {domain_name}")
        click.echo(f"{'='*80}\n")

        click.secho("Basic Information:", fg="cyan", bold=True)
        click.echo(f"  Risk Level: {domain.risk_level.value}")
        click.echo(
            f"  Regulatory Frameworks: {', '.join(domain.spec.regulatory_frameworks) or 'None'}"
        )

        if domain.spec.entities:
            click.echo(f"\n{click.style('Entities:', fg='cyan', bold=True)}")
            for entity_type, values in domain.spec.entities.items():
                click.echo(f"  • {entity_type}: {len(values)} values")

        click.echo(
            f"\n{click.style('Safety Constraints:', fg='cyan', bold=True)} ({len(domain.spec.safety_constraints)})"
        )
        for constraint in domain.spec.safety_constraints:
            click.echo(f"  • {constraint.name}")
            click.echo(f"    Type: {constraint.constraint_type}")
            click.echo(f"    Severity: {constraint.severity}")
            if constraint.ml_model_type:
                click.echo(f"    ML Model: {constraint.ml_model_type}")

        click.echo(
            f"\n{click.style('Knowledge Bases:', fg='cyan', bold=True)} ({len(domain.spec.knowledge_bases)})"
        )
        for kb in domain.spec.knowledge_bases:
            click.echo(f"  • {kb.name} ({kb.kb_type})")

        if domain.spec.simulator:
            click.echo(f"\n{click.style('Simulator:', fg='cyan', bold=True)}")
            click.echo(f"  Environment: {domain.spec.simulator.environment_type}")
            click.echo(
                f"  State Space: {domain.spec.simulator.state_space.get('type', 'N/A')}"
            )
            click.echo(
                f"  Action Space: {domain.spec.simulator.action_space.get('type', 'N/A')}"
            )

        click.echo()

    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red")
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
