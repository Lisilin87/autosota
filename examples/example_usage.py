"""Example usage of AutoSOTA."""

import os
from pathlib import Path

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

from src.main import cli
from src.core import PaperTask, TaskStatus
from src.services import SchedulerService


def example_basic_usage():
    """Basic usage example."""

    # Create a sample paper file
    paper_path = Path("examples/sample_paper.txt")
    paper_path.parent.mkdir(parents=True, exist_ok=True)

    paper_content = """
    Sample Paper: Improving Image Classification
    
    Abstract:
    We propose a novel approach for image classification that achieves
    state-of-the-art results on ImageNet.
    
    GitHub: https://github.com/example/image-classification
    
    Results:
    - Top-1 Accuracy: 85.5%
    - Top-5 Accuracy: 97.2%
    
    The model is trained on ImageNet dataset with 1.2M images.
    """

    paper_path.write_text(paper_content)

    # Create task
    task = PaperTask(
        paper_id="sample-001",
        title="Sample Paper: Improving Image Classification",
        paper_path=str(paper_path),
        repo_url="https://github.com/example/image-classification",
        conference="CVPR2024",
        domain="computer-vision",
        status=TaskStatus.NEW,
    )

    # Configuration
    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout_sec": 120,
            "max_retries": 3,
        },
        "workspace_root": "./workspaces",
        "max_iterations": 3,
        "docker": {
            "enabled": False,  # Disable Docker for local testing
        },
        "monitor": {
            "stuck_timeout_sec": 1200,
            "repeated_error_threshold": 3,
        },
        "supervisor": {
            "strict_mode": True,
            "audit_all_ideas": True,
        },
    }

    # Run scheduler
    try:
        scheduler = SchedulerService(
            task=task,
            config=config,
            workspace_root="./workspaces",
            max_iterations=3,
        )

        result = scheduler.run()

        print("Optimization completed!")
        print(f"Status: {result['status']}")
        print(f"Total iterations: {result['total_iterations']}")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure to set OPENAI_API_KEY environment variable")


def example_cli_usage():
    """CLI usage example."""
    import click

    # Using CLI
    ctx = click.Context(cli)

    # Run command
    cli.run(
        ctx,
        run,
        paper_path="examples/sample_paper.txt",
        repo_url="https://github.com/example/image/image-classification",
        paper_id="sample-001",
        max_iterations=3,
    )


if __name__ == "__main__":
    print("AutoSOTA Example Usage")
    print("=" * 50)
    print()
    print("Note: This example requires:")
    print("1. OPENAI_API_KEY environment variable set")
    print("2. A valid GitHub repository URL")
    print("3. Internet connection for LLM API calls")
    print()
    print("To run this example:")
    print("1. Set your API key: export OPENAI_API_KEY='your-key'")
    print("2. Update the repo_url to a real repository")
    print("3. Run: python examples/example_usage.py")
    print()

    # Uncomment to run (after setting up API key)
    # example_basic_usage()
