"""Main entry point for AutoSOTA CLI."""

import sys
import click
from pathlib import Path
from typing import Optional

from src.core import (
    get_logger,
    ensure_dir,
    read_json,
    write_json,
    PaperTask,
    TaskStatus,
)
from src.services.scheduler_service import SchedulerService


logger = get_logger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """AutoSOTA: End-to-End Automated Research System."""
    pass


@cli.command()
@click.option("--paper-path", required=True, help="Path to paper PDF/HTML")
@click.option("--repo-url", help="GitHub repository URL")
@click.option("--paper-id", help="Custom paper ID")
@click.option("--conference", help="Conference name")
@click.option("--domain", help="Research domain")
@click.option("--workspace-root", default="./workspaces", help="Workspace root directory")
@click.option("--config", default="configs/default.yaml", help="Configuration file path")
@click.option("--max-iterations", default=8, help="Maximum optimization iterations")
def run(
    paper_path: str,
    repo_url: Optional[str],
    paper_id: Optional[str],
    conference: Optional[str],
    domain: Optional[str],
    workspace_root: str,
    config: str,
    max_iterations: int,
):
    """Run AutoSOTA on a paper."""
    try:
        logger.info(f"Starting AutoSOTA for paper: {paper_path}")

        # Load configuration
        config_data = read_json(config) if Path(config).exists() else {}

        # Create paper task
        if not paper_id:
            paper_id = Path(paper_path).stem

        task = PaperTask(
            paper_id=paper_id,
            title=Path(paper_path).stem,
            paper_path=paper_path,
            repo_url=repo_url,
            conference=conference,
            domain=domain,
            status=TaskStatus.NEW,
        )

        # Initialize scheduler
        scheduler = SchedulerService(
            task=task,
            config=config_data,
            workspace_root=workspace_root,
            max_iterations=max_iterations,
        )

        # Run the optimization pipeline
        result = scheduler.run()

        logger.info(f"AutoSOTA completed successfully for paper: {paper_id}")
        logger.info(f"Result: {result}")

    except Exception as e:
        logger.error(f"AutoSOTA failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option("--paper-id", required=True, help="Paper ID to resume")
@click.option("--workspace-root", default="./workspaces", help="Workspace root directory")
@click.option("--config", default="configs/default.yaml", help="Configuration file path")
def resume(
    paper_id: str,
    workspace_root: str,
    config: str,
):
    """Resume a stopped AutoSOTA run."""
    try:
        logger.info(f"Resuming AutoSOTA for paper: {paper_id}")

        # Load configuration
        config_data = read_json(config) if Path(config).exists() else {}

        # Initialize scheduler with existing task
        scheduler = SchedulerService.resume(
            paper_id=paper_id,
            workspace_root=workspace_root,
            config=config_data,
        )

        # Run the optimization pipeline
        result = scheduler.run()

        logger.info(f"AutoSOTA resumed and completed for paper: {paper_id}")
        logger.info(f"Result: {result}")

    except Exception as e:
        logger.error(f"Resume failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option("--paper-id", required=True, help="Paper ID to check")
@click.option("--workspace-root", default="./workspaces", help="Workspace root directory")
def status(paper_id: str, workspace_root: str):
    """Check the status of an AutoSOTA run."""
    try:
        workspace_path = Path(workspace_root) / paper_id
        state_file = workspace_path / "state.json"

        if not state_file.exists():
            click.echo(f"No state found for paper: {paper_id}")
            return

        state = read_json(str(state_file))

        click.echo(f"Paper ID: {state['paper_id']}")
        click.echo(f"Phase: {state['phase']}")
        click.echo(f"Iteration: {state['iteration']}")
        click.echo(f"Status: {state['status']}")
        click.echo(f"Last Updated: {state.get('updated_at', 'N/A')}")

    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option("--paper-id", required=True, help="Paper ID to export")
@click.option("--output-dir", help="Output directory for exported artifacts")
@click.option("--workspace-root", default="./workspaces", help="Workspace root directory")
def export(paper_id: str, output_dir: Optional[str], workspace_root: str):
    """Export results and artifacts from an AutoSOTA run."""
    try:
        from src.services.scheduler_service import SchedulerService

        workspace_path = Path(workspace_root) / paper_id

        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = workspace_path / "export"

        ensure_dir(str(output_path))

        # Export key artifacts
        artifacts = [
            "resource_manifest.json",
            "rubric.json",
            "target_metric.json",
            "init_report.json",
            "code_analysis.md",
            "idea_library.md",
            "research_report.md",
            "scores.jsonl",
            "state.json",
            "best_patch.diff",
            "final_report.md",
        ]

        exported = []
        for artifact in artifacts:
            src_file = workspace_path / artifact
            if src_file.exists():
                dst_file = output_path / artifact
                import shutil

                shutil.copy2(src_file, dst_file)
                exported.append(artifact)

        click.echo(f"Exported {len(exported)} artifacts to: {output_path}")
        for artifact in exported:
            click.echo(f"  - {artifact}")

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
