"""
Complete demonstration of Metronis Aegis system.

This script demonstrates:
1. Domain registration (Healthcare + Trading)
2. Auto-generation of evaluation modules
3. LLM trace evaluation (clinical decision support)
4. RL agent evaluation (trading)
5. End-to-end 5-tier pipeline
6. SDK usage
"""

import asyncio
from pathlib import Path
from uuid import uuid4

from metronis.core.auto_generator import AutoGenerator
from metronis.core.domain import DomainRegistry, DomainSpec
from metronis.core.models import (
    AIProcessing,
    ApplicationType,
    PolicyInfo,
    RLStep,
    Trace,
    TraceMetadata,
)
from metronis.core.orchestrator import FiveTierOrchestrator, ModuleRegistry
from metronis.services.knowledge_base_service import KnowledgeBaseService
from metronis.sdk.client import MetronisClient


async def demo_domain_registration():
    """Demo 1: Register domains from YAML specs."""
    print("=" * 80)
    print("DEMO 1: Domain Registration")
    print("=" * 80)

    # Initialize domain registry
    domains_path = Path("domains")
    registry = DomainRegistry(domains_path)

    # List loaded domains
    print(f"\nLoaded domains: {registry.list_domains()}")

    # Get healthcare domain
    healthcare = registry.get_domain("healthcare")
    if healthcare:
        print(f"\nHealthcare domain:")
        print(f"  - Risk level: {healthcare.risk_level}")
        print(f"  - Safety constraints: {len(healthcare.spec.safety_constraints)}")
        print(f"  - Knowledge bases: {len(healthcare.spec.knowledge_bases)}")
        print(f"  - Tier-1 modules: {healthcare.spec.tier1_modules}")

    # Get trading domain
    trading = registry.get_domain("trading")
    if trading:
        print(f"\nTrading domain:")
        print(f"  - Risk level: {trading.risk_level}")
        print(f"  - Safety constraints: {len(trading.spec.safety_constraints)}")
        print(f"  - Tier-1 modules: {trading.spec.tier1_modules}")

    return registry


async def demo_auto_generation(registry: DomainRegistry):
    """Demo 2: Auto-generate evaluation modules from domain specs."""
    print("\n" + "=" * 80)
    print("DEMO 2: Auto-Generation of Evaluation Modules")
    print("=" * 80)

    generator = AutoGenerator()

    # Generate modules for healthcare domain
    healthcare = registry.get_domain("healthcare")
    if healthcare:
        print(f"\nGenerating modules for healthcare domain...")

        output_dir = Path("generated")
        generated = generator.generate_domain_modules(healthcare.spec, output_dir)

        print(f"\nGenerated files:")
        print(f"  - Tier-1 validators: {len(generated['tier1_modules'])} files")
        for path in generated["tier1_modules"]:
            print(f"    • {path.name}")

        print(f"  - Tier-2 models: {len(generated['tier2_models'])} files")
        print(f"  - Tier-3 prompts: {len(generated['tier3_prompts'])} files")
        print(f"  - Simulators: {len(generated['simulators'])} files")


async def demo_llm_evaluation():
    """Demo 3: Evaluate a clinical decision support trace."""
    print("\n" + "=" * 80)
    print("DEMO 3: LLM Evaluation (Clinical Decision Support)")
    print("=" * 80)

    # Create a sample trace
    trace = Trace(
        trace_id=uuid4(),
        organization_id=uuid4(),
        application_id=uuid4(),
        application_type=ApplicationType.CLINICAL_SUPPORT,
        ai_processing=AIProcessing(
            model="gpt-4",
            input="Patient is 65yo male with hypertension (BP 150/95). Current meds: aspirin 81mg. What treatment do you recommend?",
            output="I recommend starting lisinopril 10mg once daily. Lisinopril is an ACE inhibitor that is first-line for hypertension. Monitor BP in 2 weeks. Also recommend lifestyle modifications: reduce sodium intake, increase exercise, and weight loss if overweight.",
        ),
        metadata=TraceMetadata(
            domain="healthcare",
            specialty="cardiology",
            risk_level="high",
        ),
    )

    print(f"\nTrace ID: {trace.trace_id}")
    print(f"\nInput: {trace.ai_processing.input}")
    print(f"\nOutput: {trace.ai_processing.output}")

    # Initialize services
    kb_service = KnowledgeBaseService(redis_url="redis://localhost:6379")
    module_registry = ModuleRegistry()

    # Note: In production, modules would be auto-loaded from domain registry
    # For demo, we'll simulate

    print(f"\nEvaluation would run through 5 tiers:")
    print(f"  Tier 0: Pre-processing (PHI detection, routing)")
    print(f"  Tier 1: Medication validator, drug interaction checker")
    print(f"  Tier 2: Safety risk classifier (ML model)")
    print(f"  Tier 3: Clinical reasoning evaluator (LLM-as-judge)")
    print(f"  Tier 4: Expert review (if needed)")

    await kb_service.close()


async def demo_rl_evaluation():
    """Demo 4: Evaluate a trading RL agent."""
    print("\n" + "=" * 80)
    print("DEMO 4: RL Agent Evaluation (Trading)")
    print("=" * 80)

    # Create a sample RL episode
    episode = [
        RLStep(
            state={"cash": 100000, "positions": [], "market_price": 150.0},
            action={"place_order": {"symbol": "AAPL", "quantity": 100, "price": 150.0}},
            reward=0.0,
            next_state={"cash": 85000, "positions": [{"symbol": "AAPL", "quantity": 100}], "market_price": 150.0},
            done=False,
        ),
        RLStep(
            state={"cash": 85000, "positions": [{"symbol": "AAPL", "quantity": 100}], "market_price": 152.0},
            action={"do_nothing": True},
            reward=200.0,
            next_state={"cash": 85000, "positions": [{"symbol": "AAPL", "quantity": 100}], "market_price": 152.0},
            done=False,
        ),
        RLStep(
            state={"cash": 85000, "positions": [{"symbol": "AAPL", "quantity": 100}], "market_price": 155.0},
            action={"close_position": {"symbol": "AAPL", "quantity": 100}},
            reward=500.0,
            next_state={"cash": 100500, "positions": [], "market_price": 155.0},
            done=True,
        ),
    ]

    cumulative_reward = sum(step.reward for step in episode)

    trace = Trace(
        trace_id=uuid4(),
        organization_id=uuid4(),
        application_id=uuid4(),
        application_type=ApplicationType.TRADING_AGENT,
        ai_processing=AIProcessing(
            model="PPO-Trading-Agent-v1",
            input="Trade AAPL with risk limit",
            output=f"Episode completed: +${cumulative_reward} profit",
            rl_episode=episode,
            policy_info=PolicyInfo(
                policy_name="PPO-Trading-Agent",
                policy_version="v1.0",
            ),
            cumulative_reward=cumulative_reward,
            episode_length=len(episode),
        ),
        metadata=TraceMetadata(
            domain="trading",
            risk_level="critical",
        ),
    )

    print(f"\nTrace ID: {trace.trace_id}")
    print(f"\nPolicy: {trace.ai_processing.policy_info.policy_name}")
    print(f"Episode length: {len(episode)} steps")
    print(f"Cumulative reward: ${cumulative_reward}")

    print(f"\nEpisode summary:")
    for i, step in enumerate(episode):
        print(f"  Step {i}: action={list(step.action.keys())[0]}, reward=${step.reward}")

    print(f"\nRL-specific evaluations would check:")
    print(f"  • Reward shaping (detect reward hacking)")
    print(f"  • Exploration efficiency (state coverage)")
    print(f"  • Policy divergence (vs baseline)")
    print(f"  • Safety constraints (position limits, margin)")
    print(f"  • Convergence (training stability)")


async def demo_sdk_usage():
    """Demo 5: SDK client library usage."""
    print("\n" + "=" * 80)
    print("DEMO 5: SDK Client Library")
    print("=" * 80)

    # Note: This requires the API server to be running
    # For demo, we'll show the code without executing

    print("\nExample SDK usage:")
    print("""
    from metronis.sdk import MetronisClient

    # Initialize client
    client = MetronisClient(api_key="your-api-key")

    # Evaluate a clinical decision
    result = await client.evaluate_trace(
        input="Patient with chest pain, what tests?",
        output="Recommend ECG, troponin, and chest X-ray",
        model="gpt-4",
        domain="healthcare",
        application_type="clinical_support"
    )

    print(f"Safe: {result.overall_passed}")
    print(f"Issues: {len(result.all_issues)}")

    # Evaluate an RL episode
    result = await client.evaluate_rl_episode(
        episode=[
            {"state": {...}, "action": {...}, "reward": 1.0},
            {"state": {...}, "action": {...}, "reward": 2.0},
        ],
        policy_name="TradingAgent-v1",
        domain="trading"
    )

    # Get analytics
    analytics = await client.get_analytics(domain="healthcare")
    print(f"Total traces: {analytics['total_traces']}")
    print(f"Pass rate: {analytics['pass_rate']:.1%}")
    """)


async def demo_integration():
    """Demo 6: Integration with LangChain/LlamaIndex."""
    print("\n" + "=" * 80)
    print("DEMO 6: Framework Integrations")
    print("=" * 80)

    print("\nLangChain Integration:")
    print("""
    from langchain.chains import LLMChain
    from metronis.sdk import MetronisClient, LangChainCallback

    client = MetronisClient(api_key="your-key")
    callback = LangChainCallback(client, domain="healthcare")

    chain = LLMChain(llm=llm, callbacks=[callback])
    result = chain.run("What medication for diabetes?")
    # Automatically evaluated by Metronis
    """)

    print("\nLlamaIndex Integration:")
    print("""
    from llama_index import ServiceContext, VectorStoreIndex
    from metronis.sdk import MetronisClient, LlamaIndexCallback

    client = MetronisClient(api_key="your-key")
    callback = LlamaIndexCallback(client, domain="legal")

    service_context = ServiceContext.from_defaults(callback_manager=callback)
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    # All queries automatically evaluated
    """)


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "METRONIS AEGIS - FULL SYSTEM DEMO" + " " * 25 + "║")
    print("║" + " " * 15 + "Universal AI Evaluation Infrastructure" + " " * 24 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    try:
        # Demo 1: Domain registration
        registry = await demo_domain_registration()

        # Demo 2: Auto-generation
        await demo_auto_generation(registry)

        # Demo 3: LLM evaluation
        await demo_llm_evaluation()

        # Demo 4: RL evaluation
        await demo_rl_evaluation()

        # Demo 5: SDK usage
        await demo_sdk_usage()

        # Demo 6: Integrations
        await demo_integration()

        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Start the API server: make dev")
        print("  2. Generate your domain modules: python -m metronis.cli generate-domain <domain>")
        print("  3. Test with your own traces: python examples/test_trace.py")
        print("  4. View dashboard: http://localhost:8000/docs")
        print()

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
