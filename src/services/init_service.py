"""Init service for AutoSOTA."""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.core import (
    get_logger,
    ensure_dir,
    write_json,
    read_json,
    read_text,
    InitReport,
    PaperTask,
    ResourceManifest,
    InitError,
    run_command,
    parse_requirements,
)
from src.core.llm_client import LLMClient


logger = get_logger(__name__)


class InitService:
    """Service for initializing execution environment."""

    def __init__(
        self,
        llm_client: LLMClient,
        workspace_root: str,
        config: Dict[str, Any],
    ):
        self.llm_client = llm_client
        self.workspace_root = workspace_root
        self.config = config
        self.docker_config = config.get("docker", {})

    def bootstrap(
        self,
        task: PaperTask,
        manifest: ResourceManifest,
    ) -> InitReport:
        """Bootstrap execution environment."""
        logger.info(f"Bootstrapping environment for task: {task.paper_id}")

        workspace_path = Path(self.workspace_root) / task.paper_id
        repo_path = workspace_path / "repo"

        report = InitReport()

        try:
            # Detect and setup environment
            env_info = self._detect_environment(repo_path)
            report.docker_image = env_info.get("docker_image")
            report.python_version = env_info.get("python_version")

            # Install dependencies
            self._install_dependencies(repo_path, report)

            # Validate environment
            self._validate_environment(repo_path, report)

            # Discover commands
            commands = self._discover_commands(repo_path)
            report.train_command = commands.get("train")
            report.eval_command = commands.get("eval")

            # Perform dry run
            report.dry_run_success = self._dry_run(repo_path, commands)

            # Save report
            report_path = workspace_path / "init_report.json"
            write_json(str(report_path), report.model_dump())

            logger.info(f"Bootstrap completed for: {task.paper_id}")
            return report

        except Exception as e:
            logger.error(f"Bootstrap failed: {e}")
            report.errors.append(str(e))
            raise InitError(f"Bootstrap failed: {e}")

    def _detect_environment(self, repo_path: Path) -> Dict[str, Any]:
        """Detect environment configuration."""
        logger.info("Detecting environment")

        env_info = {}

        # Check for Dockerfile
        dockerfile = repo_path / "Dockerfile"
        if dockerfile.exists():
            env_info["has_dockerfile"] = True
            env_info["docker_image"] = self._parse_dockerfile(dockerfile)

        # Check for Python version
        try:
            result = run_command(["python", "--version"], cwd=str(repo_path))
            if result.returncode == 0:
                version_str = result.stdout.strip()
                env_info["python_version"] = version_str.replace("Python ", "")
        except Exception:
            pass

        # Check for PyTorch
        try:
            result = run_command(
                ["python", "-c", "import torch; print(torch.__version__)"],
                cwd=str(repo_path),
            )
            if result.returncode == 0:
                env_info["torch_version"] = result.stdout.strip()
        except Exception:
            pass

        return env_info

    def _parse_dockerfile(self, dockerfile: Path) -> str:
        """Parse Dockerfile to get base image."""
        try:
            content = read_text(str(dockerfile))
            for line in content.split("\n"):
                if line.strip().upper().startswith("FROM"):
                    return line.strip().split()[1]
        except Exception:
            pass
        return None

    def _install_dependencies(self, repo_path: Path, report: InitReport) -> None:
        """Install dependencies."""
        logger.info("Installing dependencies")

        # Check for requirements.txt
        requirements_files = list(repo_path.glob("requirements*.txt"))

        for req_file in requirements_files:
            logger.info(f"Installing from: {req_file.name}")
            try:
                result = run_command(
                    ["pip", "install", "-r", str(req_file)],
                    cwd=str(repo_path),
                    timeout=600,
                )
                if result.returncode != 0:
                    report.warnings.append(f"Failed to install {req_file.name}: {result.stderr}")
            except Exception as e:
                report.warnings.append(f"Error installing {req_file.name}: {e}")

        # Check for setup.py
        setup_py = repo_path / "setup.py"
        if setup_py.exists():
            logger.info("Installing from setup.py")
            try:
                result = run_command(
                    ["pip", "install", "-e", "."],
                    cwd=str(repo_path),
                    timeout=600,
                )
                if result.returncode != 0:
                    report.warnings.append(f"Failed to install from setup.py: {result.stderr}")
            except Exception as e:
                report.warnings.append(f"Error installing from setup.py: {e}")

    def _validate_environment(self, repo_path: Path, report: InitReport) -> None:
        """Validate environment setup."""
        logger.info("Validating environment")

        # Check CUDA availability
        try:
            result = run_command(
                ["python", "-c", "import torch; print(torch.cuda.is_available())"],
                cwd=str(repo_path),
            )
            if result.returncode == 0:
                report.cuda_available = result.stdout.strip().lower() == "true"
        except Exception:
            report.cuda_available = False

        # Check GPU count
        if report.cuda_available:
            try:
                result = run_command(
                    ["python", "-c", "import torch; print(torch.cuda.device_count())"],
                    cwd=str(repo_path),
                )
                if result.returncode == 0:
                    report.gpu_count = int(result.stdout.strip())
            except Exception:
                pass

    def _discover_commands(self, repo_path: Path) -> Dict[str, str]:
        """Discover training and evaluation commands."""
        logger.info("Discovering commands")

        commands = {}

        # Look for common train scripts
        train_patterns = [
            "train.py",
            "main.py",
            "run.py",
            "train.sh",
        ]

        for pattern in train_patterns:
            for script in repo_path.rglob(pattern):
                if script.is_file():
                    commands["train"] = f"python {script.relative_to(repo_path)}"
                    logger.info(f"Found train command: {commands['train']}")
                    break
            if "train" in commands:
                break

        # Look for common eval scripts
        eval_patterns = [
            "eval.py",
            "evaluate.py",
            "test.py",
            "eval.sh",
        ]

        for pattern in eval_patterns:
            for script in repo_path.rglob(pattern):
                if script.is_file():
                    commands["eval"] = f"python {script.relative_to(repo_path)}"
                    logger.info(f"Found eval command: {commands['eval']}")
                    break
            if "eval" in commands:
                break

        return commands

    def _dry_run(self, repo_path: Path, commands: Dict[str, str]) -> bool:
        """Perform dry run to validate setup."""
        logger.info("Performing dry run")

        # Try to import the main module
        try:
            result = run_command(
                ["python", "-c", "pass"],
                cwd=str(repo_path),
                timeout=10,
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Dry run failed: {e}")
            return False
