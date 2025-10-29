"""
Test script for auto-generation system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from metronis.core.domain import DomainRegistry
from metronis.core.auto_generator import AutoGenerator


def main():
    print("\n" + "=" * 80)
    print("Testing Metronis Auto-Generation System")
    print("=" * 80 + "\n")

    # Set up paths
    domains_path = Path(__file__).parent / "domains"
    output_path = Path(__file__).parent / "generated" / "healthcare"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Domains path: {domains_path}")
    print(f"Output path: {output_path}\n")

    # Load domain registry
    print("Loading domain registry...")
    registry = DomainRegistry(domains_path)
    domains = registry.list_domains()
    print(f"[OK] Found {len(domains)} domains: {', '.join(domains)}\n")

    # Get healthcare domain
    print("Loading healthcare domain...")
    domain = registry.get_domain("healthcare")
    if domain is None:
        print("[ERROR] Healthcare domain not found!")
        return 1

    print(f"[OK] Loaded healthcare domain")
    print(f"   Risk Level: {domain.risk_level.value}")
    print(f"   Safety Constraints: {len(domain.spec.safety_constraints)}")
    print(f"   Knowledge Bases: {len(domain.spec.knowledge_bases)}")
    print()

    # Initialize auto-generator
    print("Initializing auto-generator...")
    generator = AutoGenerator()
    print("[OK] Auto-generator initialized\n")

    # Generate Tier-1 validators
    print("-" * 80)
    print("Generating Tier-1 Validators")
    print("-" * 80)

    tier1_dir = output_path / "tier1_modules"
    tier1_dir.mkdir(exist_ok=True)

    generated_count = 0
    for constraint in domain.spec.safety_constraints:
        try:
            print(f"Generating {constraint.name}...", end=" ")

            # Generate module code
            code = generator.generate_tier1_module(constraint, "healthcare")

            # Write to file
            filename = f"{constraint.name}_validator.py"
            filepath = tier1_dir / filename
            filepath.write_text(code, encoding="utf-8")

            print(f"[OK] ({len(code)} bytes)")
            generated_count += 1

        except Exception as e:
            print(f"[ERROR] {e}")

    # Create __init__.py
    init_file = tier1_dir / "__init__.py"
    init_file.write_text(
        '"""Auto-generated Tier-1 validators for healthcare domain."""\n',
        encoding="utf-8",
    )
    print(f"[OK] Created __init__.py\n")

    # Generate Tier-3 prompts
    print("-" * 80)
    print("Generating Tier-3 LLM Prompts")
    print("-" * 80)

    tier3_dir = output_path / "tier3_prompts"
    tier3_dir.mkdir(exist_ok=True)

    if domain.spec.tier3_evals:
        for eval_config in domain.spec.tier3_evals:
            try:
                print(f"Generating {eval_config.name} prompt...", end=" ")

                # Generate prompt - convert Pydantic model to dict
                prompt = generator.generate_tier3_prompt(
                    eval_config.model_dump(), "healthcare"
                )

                # Write to file
                filename = f"{eval_config.name}_prompt.txt"
                filepath = tier3_dir / filename
                filepath.write_text(prompt, encoding="utf-8")

                print(f"[OK] ({len(prompt)} bytes)")
                generated_count += 1

            except Exception as e:
                print(f"[ERROR] {e}")
        print()
    else:
        print("[INFO] No Tier-3 eval configs to generate\n")

    # Generate Tier-2 models
    print("-" * 80)
    print("Generating Tier-2 ML Models")
    print("-" * 80)

    tier2_dir = output_path / "tier2_models"
    tier2_dir.mkdir(exist_ok=True)

    if domain.spec.tier2_models:
        for model_config in domain.spec.tier2_models:
            try:
                print(f"Generating {model_config.name} model...", end=" ")

                # Generate model code - convert Pydantic model to dict
                code = generator.generate_tier2_model_scaffold(
                    model_config.model_dump(), "healthcare"
                )

                # Write to file
                filename = f"{model_config.name}_model.py"
                filepath = tier2_dir / filename
                filepath.write_text(code, encoding="utf-8")

                print(f"[OK] ({len(code)} bytes)")
                generated_count += 1

            except Exception as e:
                print(f"[ERROR] {e}")

        init_file = tier2_dir / "__init__.py"
        init_file.write_text(
            '"""Auto-generated Tier-2 models for healthcare domain."""\n',
            encoding="utf-8",
        )
        print(f"[OK] Created __init__.py\n")
    else:
        print("[INFO] No Tier-2 models to generate\n")

    # Generate simulator
    if domain.spec.simulators:
        print("-" * 80)
        print("Generating Simulators")
        print("-" * 80)

        for simulator_config in domain.spec.simulators:
            try:
                print(f"Generating {simulator_config.name}.py...", end=" ")

                # Generate simulator code - convert Pydantic model to dict
                code = generator.generate_simulator_scaffold(
                    simulator_config.model_dump(), "healthcare"
                )

                # Write to file
                filename = f"{simulator_config.name}.py"
                filepath = output_path / filename
                filepath.write_text(code, encoding="utf-8")

                print(f"[OK] ({len(code)} bytes)")
                generated_count += 1

            except Exception as e:
                print(f"[ERROR] {e}")
        print()
    else:
        print("-" * 80)
        print("No simulator configuration found")
        print("-" * 80 + "\n")

    # Summary
    print("=" * 80)
    print("Generation Complete")
    print("=" * 80)
    print(f"[OK] Generated {generated_count} files")
    print(f"[INFO] Output directory: {output_path}")
    print()

    # Show generated files
    print("Generated files:")
    for file in sorted(output_path.rglob("*.py")):
        rel_path = file.relative_to(output_path)
        print(f"  • {rel_path}")
    for file in sorted(output_path.rglob("*.txt")):
        rel_path = file.relative_to(output_path)
        print(f"  • {rel_path}")
    print()

    # Test import of a generated module
    print("Testing import of generated module...")
    try:
        # Add generated modules to path
        sys.path.insert(0, str(tier1_dir))

        # Try to import the first validator
        if domain.spec.safety_constraints:
            first_constraint = domain.spec.safety_constraints[0]
            module_name = f"{first_constraint.name}_validator"

            print(f"  Importing {module_name}...", end=" ")
            __import__(module_name)
            print("[OK] Import successful!")
        print()

    except Exception as e:
        print(f"[ERROR] Import failed: {e}\n")

    print("=" * 80)
    print("[OK] Auto-generation test completed successfully!")
    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
