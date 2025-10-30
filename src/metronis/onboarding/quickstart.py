"""
Customer Onboarding & Quickstart

Guides new customers through setup and their first evaluation.
"""

import asyncio
from typing import Optional

import structlog
from pydantic import BaseModel

from metronis.core.models import AIProcessing, Trace
from metronis.sdk.client import MetronisClient

logger = structlog.get_logger(__name__)


class OnboardingStep(BaseModel):
    """A single onboarding step."""

    step_number: int
    title: str
    description: str
    completed: bool = False
    code_example: Optional[str] = None


class OnboardingGuide:
    """
    Interactive onboarding guide for new customers.

    Walks through:
    1. API key setup
    2. First evaluation
    3. Domain selection
    4. SDK integration
    5. Dashboard tour
    """

    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        """
        Initialize onboarding guide.

        Args:
            api_key: Metronis API key
            base_url: API base URL
        """
        self.client = MetronisClient(api_key=api_key, base_url=base_url)
        self.steps = self._create_onboarding_steps()
        self.current_step = 0

    def _create_onboarding_steps(self) -> list[OnboardingStep]:
        """Create onboarding steps."""
        return [
            OnboardingStep(
                step_number=1,
                title="Welcome to Metronis",
                description="Let's get you started with AI evaluation in 5 minutes",
            ),
            OnboardingStep(
                step_number=2,
                title="Install the SDK",
                description="Install the Metronis Python SDK",
                code_example="""
pip install metronis

# Or with poetry:
poetry add metronis
""",
            ),
            OnboardingStep(
                step_number=3,
                title="Setup API Key",
                description="Configure your API key",
                code_example="""
import os
from metronis.sdk import MetronisClient

# Set API key
os.environ["METRONIS_API_KEY"] = "your-api-key"

# Initialize client
client = MetronisClient()
""",
            ),
            OnboardingStep(
                step_number=4,
                title="Your First Evaluation",
                description="Submit your first trace for evaluation",
                code_example="""
# Evaluate a simple LLM output
result = await client.evaluate_trace(
    input="What is the capital of France?",
    output="The capital of France is Paris.",
    model="gpt-4",
    domain="general",
)

print(f"Passed: {result.overall_passed}")
print(f"Issues: {len(result.all_issues)}")
""",
            ),
            OnboardingStep(
                step_number=5,
                title="Choose Your Domain",
                description="Select a domain for specialized evaluation",
                code_example="""
# Healthcare domain
result = await client.evaluate_trace(
    input="Patient has fever and cough. Recommend treatment?",
    output="Recommend rest and fluids. See doctor if worsens.",
    model="gpt-4",
    domain="healthcare",
)

# Trading domain
result = await client.evaluate_trace(
    input="Should I buy AAPL stock?",
    output="Consider diversification and consult financial advisor.",
    model="gpt-4",
    domain="trading",
)
""",
            ),
            OnboardingStep(
                step_number=6,
                title="Integrate with Your App",
                description="Add Metronis to your existing application",
                code_example="""
# LangChain integration
from langchain.callbacks import MetronisCallback

callback = MetronisCallback(
    api_key="your-api-key",
    domain="healthcare",
)

chain.run(input, callbacks=[callback])

# LlamaIndex integration
from llama_index.callbacks import MetronisCallback

callback = MetronisCallback(api_key="your-api-key")
query_engine.query(query, callbacks=[callback])
""",
            ),
            OnboardingStep(
                step_number=7,
                title="View Results in Dashboard",
                description="Check your evaluation results in the web dashboard",
            ),
            OnboardingStep(
                step_number=8,
                title="Set Up Alerts",
                description="Get notified when critical issues are detected",
                code_example="""
# Configure Slack alerts for critical issues
client.configure_alerts(
    channels=["slack"],
    slack_webhook="https://hooks.slack.com/...",
    severity_threshold="high",
)
""",
            ),
        ]

    async def start_onboarding(self) -> None:
        """Start the interactive onboarding process."""
        print("\n" + "=" * 80)
        print("🚀 Welcome to Metronis - AI Evaluation Platform")
        print("=" * 80)
        print("\nLet's get you started in 5 minutes!\n")

        for step in self.steps:
            await self._present_step(step)

        print("\n" + "=" * 80)
        print("✅ Onboarding Complete!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Visit the dashboard: http://localhost:3000")
        print("  2. Check out the docs: https://docs.metronis.ai")
        print("  3. Join our Discord: https://discord.gg/metronis")
        print("\nHappy evaluating! 🎉\n")

    async def _present_step(self, step: OnboardingStep) -> None:
        """Present a single onboarding step."""
        print(f"\n📍 Step {step.step_number}: {step.title}")
        print("-" * 80)
        print(f"{step.description}\n")

        if step.code_example:
            print("Code example:")
            print("```python")
            print(step.code_example.strip())
            print("```\n")

        # Wait for user to continue (in interactive mode)
        input("Press Enter to continue...")

        step.completed = True

    async def run_first_evaluation(self) -> None:
        """
        Run the user's first evaluation as part of onboarding.

        This is a guided, interactive evaluation.
        """
        print("\n" + "=" * 80)
        print("🎯 Let's run your first evaluation!")
        print("=" * 80)

        print("\n1. We'll evaluate a simple LLM output")
        print("2. You'll see how Metronis catches issues")
        print("3. Then we'll check the results\n")

        # Example 1: Safe output
        print("Example 1: Safe Output")
        print("-" * 40)

        result1 = await self.client.evaluate_trace(
            input="What is 2+2?",
            output="2+2 equals 4.",
            model="gpt-4",
            domain="general",
        )

        print(f"✅ Passed: {result1.overall_passed}")
        print(f"   Issues: {len(result1.all_issues)}")
        print(f"   Latency: {result1.total_execution_time_ms:.0f}ms")

        input("\nPress Enter to see an example with issues...")

        # Example 2: Unsafe output (healthcare)
        print("\nExample 2: Unsafe Output (Healthcare)")
        print("-" * 40)

        result2 = await self.client.evaluate_trace(
            input="I have chest pain. What should I do?",
            output="It's probably nothing. Just rest and you'll be fine.",
            model="gpt-4",
            domain="healthcare",
        )

        print(f"❌ Passed: {result2.overall_passed}")
        print(f"   Issues detected: {len(result2.all_issues)}")

        if result2.all_issues:
            print("\n   Critical issues:")
            for issue in result2.all_issues[:3]:
                print(f"     • {issue.message}")

        print(f"\n💡 Metronis caught {len(result2.all_issues)} safety issues!")
        print("   This output should not be shown to users.")

        input("\nPress Enter to continue...")


async def quick_start(
    api_key: str,
    example_domain: str = "general",
) -> None:
    """
    Quick start guide - minimal setup.

    Args:
        api_key: Metronis API key
        example_domain: Domain to use for example
    """
    print("\n" + "=" * 80)
    print("⚡ Metronis Quickstart (30 seconds)")
    print("=" * 80 + "\n")

    # Initialize client
    print("1. Initializing Metronis client...")
    client = MetronisClient(api_key=api_key)

    # Run sample evaluation
    print("2. Running sample evaluation...")

    result = await client.evaluate_trace(
        input="Sample AI input",
        output="Sample AI output",
        model="gpt-4",
        domain=example_domain,
    )

    print(f"\n✅ Evaluation complete!")
    print(f"   Status: {'PASSED' if result.overall_passed else 'FAILED'}")
    print(f"   Issues: {len(result.all_issues)}")
    print(f"   Cost: ${result.cost:.4f}")
    print(f"   Latency: {result.total_execution_time_ms:.0f}ms")

    print("\n3. Check your dashboard: http://localhost:3000")
    print("\n✨ You're all set! Start evaluating your AI.\n")


def print_integration_examples():
    """Print integration examples for popular frameworks."""
    print("\n" + "=" * 80)
    print("📚 Integration Examples")
    print("=" * 80 + "\n")

    examples = {
        "LangChain": """
from langchain import OpenAI
from langchain.callbacks import MetronisCallback

callback = MetronisCallback(
    api_key="your-api-key",
    domain="healthcare",
)

llm = OpenAI()
result = llm("Your prompt", callbacks=[callback])

# Evaluation happens automatically!
""",
        "LlamaIndex": """
from llama_index import GPTSimpleVectorIndex
from llama_index.callbacks import MetronisCallback

callback = MetronisCallback(api_key="your-api-key")

index = GPTSimpleVectorIndex.from_documents(documents)
query_engine = index.as_query_engine(callbacks=[callback])

response = query_engine.query("Your query")
# Evaluation happens automatically!
""",
        "OpenAI SDK": """
import openai
from metronis.sdk import MetronisClient

client = MetronisClient(api_key="your-api-key")

# Make OpenAI call
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
)

# Evaluate with Metronis
result = await client.evaluate_trace(
    input="Hello",
    output=response.choices[0].message.content,
    model="gpt-4",
    domain="general",
)
""",
        "Anthropic SDK": """
import anthropic
from metronis.sdk import MetronisClient

client = MetronisClient(api_key="your-api-key")

# Make Anthropic call
anthropic_client = anthropic.Anthropic()
message = anthropic_client.messages.create(
    model="claude-3-opus",
    messages=[{"role": "user", "content": "Hello"}],
)

# Evaluate with Metronis
result = await client.evaluate_trace(
    input="Hello",
    output=message.content[0].text,
    model="claude-3-opus",
    domain="general",
)
""",
    }

    for framework, code in examples.items():
        print(f"{framework}:")
        print("```python")
        print(code.strip())
        print("```\n")


if __name__ == "__main__":
    # Example usage
    guide = OnboardingGuide(api_key="demo-key")
    asyncio.run(guide.start_onboarding())
