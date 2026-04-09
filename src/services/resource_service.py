"""Resource service for AutoSOTA."""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

import yaml

from src.core import (
    get_logger,
    ensure_dir,
    write_json,
    read_text,
    write_text,
    ResourceManifest,
    PaperTask,
    ResourceError,
    run_command,
    get_file_tree,
)
from src.core.llm_client import LLMClient


logger = get_logger(__name__)


class ResourceService:
    """Service for preparing resources (repo, data, models, checkpoints)."""

    def __init__(
        self,
        llm_client: LLMClient,
        workspace_root: str,
        config: Dict[str, Any],
    ):
        self.llm_client = llm_client
        self.workspace_root = workspace_root
        self.config = config
        self.resource_config = config.get("resource", {})

    def prepare(self, task: PaperTask) -> ResourceManifest:
        """Prepare resources for the task."""
        logger.info(f"Preparing resources for task: {task.paper_id}")

        # Create workspace
        workspace_path = Path(self.workspace_root) / task.paper_id
        ensure_dir(str(workspace_path))

        # Load paper content
        paper_content = self._load_paper_content(task.paper_path)

        # Extract repo candidates
        repo_candidates = self._extract_repo_candidates(paper_content, task.repo_url)

        # Select and clone repo
        selected_repo = self._select_and_clone_repo(repo_candidates, workspace_path)

        # Analyze repo
        repo_info = self._analyze_repo(workspace_path / "repo")

        # Extract resource information
        resource_info = self._extract_resources(paper_content, repo_info, task)

        # Create manifest
        manifest = ResourceManifest(
            repo_candidates=repo_candidates,
            selected_repo=selected_repo,
            dataset_items=resource_info.get("datasets", []),
            model_items=resource_info.get("models", []),
            checkpoint_items=resource_info.get("checkpoints", []),
            readiness_signals=repo_info.get("readiness_signals", {}),
            local_paths={
                "repo": str(workspace_path / "repo"),
                "resources": str(workspace_path / "resources"),
            },
        )

        # Save manifest
        manifest_path = workspace_path / "resource_manifest.json"
        write_json(str(manifest_path), manifest.model_dump())

        logger.info(f"Resource preparation completed for: {task.paper_id}")
        return manifest

    def _load_paper_content(self, paper_path: str) -> str:
        """Load paper content from file."""
        logger.info(f"Loading paper content from: {paper_path}")

        if not os.path.exists(paper_path):
            raise ResourceError(f"Paper file not found: {paper_path}")

        # For PDF files, we'd need to extract text
        # For now, assume it's already text/markdown
        try:
            return read_text(paper_path)
        except Exception as e:
            raise ResourceError(f"Failed to load paper content: {e}")

    def _extract_repo_candidates(
        self,
        paper_content: str,
        provided_repo_url: Optional[str],
    ) -> List[str]:
        """Extract repository candidates from paper content."""
        logger.info("Extracting repository candidates")

        candidates = []

        # Add provided repo URL if available
        if provided_repo_url:
            candidates.append(provided_repo_url)

        # Look for GitHub URLs in paper content
        github_patterns = [
            r"https?://github\.com/[\w-]+/[\w-]+",
            r"github\.com/[\w-]+/[\w-]+",
        ]

        for pattern in github_patterns:
            matches = re.findall(pattern, paper_content, re.IGNORECASE)
            for match in matches:
                if not match.startswith("http"):
                    match = "https://" + match
                if match not in candidates:
                    candidates.append(match)

        # Look for "code", "implementation", "project page" keywords
        keywords = ["code", "implementation", "project page", "repository", "source code"]
        for keyword in keywords:
            pattern = rf"{keyword}[^:]*:\s*(https?://[^\s]+)"
            matches = re.findall(pattern, paper_content, re.IGNORECASE)
            for match in matches:
                if match not in candidates:
                    candidates.append(match)

        logger.info(f"Found {len(candidates)} repository candidates")
        return candidates

    def _select_and_clone_repo(
        self,
        candidates: List[str],
        workspace_path: Path,
    ) -> Optional[str]:
        """Select and clone the best repository."""
        if not candidates:
            logger.warning("No repository candidates found")
            return None

        # For now, just use the first candidate
        # In a full implementation, we'd use LLM to rank and select
        selected_url = candidates[0]

        logger.info(f"Selected repository: {selected_url}")

        # Clone repository
        repo_path = workspace_path / "repo"

        try:
            # Remove existing repo if present
            if repo_path.exists():
                import shutil

                shutil.rmtree(repo_path)

            # Clone with depth 1 for shallow clone
            shallow = self.resource_config.get("shallow_clone", True)
            depth_arg = ["--depth", "1"] if shallow else []

            result = run_command(
                ["git", "clone"] + depth_arg + [selected_url, str(repo_path)],
                timeout=300,
            )

            if result.returncode != 0:
                raise ResourceError(f"Failed to clone repository: {result.stderr}")

            logger.info(f"Repository cloned to: {repo_path}")
            return selected_url

        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise ResourceError(f"Failed to clone repository: {e}")

    def _analyze_repo(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze repository structure and readiness."""
        logger.info(f"Analyzing repository: {repo_path}")

        if not repo_path.exists():
            return {"readiness_signals": {}}

        readiness_signals = {}

        # Check for README
        readme_files = list(repo_path.glob("README*"))
        readiness_signals["has_readme"] = len(readme_files) > 0

        # Check for requirements.txt
        requirements_files = list(repo_path.glob("requirements*.txt"))
        readiness_signals["has_requirements"] = len(requirements_files) > 0

        # Check for setup.py or pyproject.toml
        setup_files = list(repo_path.glob("setup.py")) + list(repo_path.glob("pyproject.toml"))
        readiness_signals["has_setup"] = len(setup_files) > 0

        # Check for Dockerfile
        dockerfile = repo_path / "Dockerfile"
        readiness_signals["has_dockerfile"] = dockerfile.exists()

        # Check for common train/eval scripts
        train_scripts = []
        eval_scripts = []

        for pattern in ["*train*.py", "*eval*.py", "*test*.py"]:
            for script in repo_path.glob(pattern):
                if "train" in script.name.lower():
                    train_scripts.append(script.name)
                elif "eval" in script.name.lower() or "test" in script.name.lower():
                    eval_scripts.append(script.name)

        readiness_signals["has_train_script"] = len(train_scripts) > 0
        readiness_signals["has_eval_script"] = len(eval_scripts) > 0

        # Get file tree
        file_tree = get_file_tree(str(repo_path), max_depth=3)

        return {
            "readiness_signals": readiness_signals,
            "file_tree": file_tree,
            "train_scripts": train_scripts,
            "eval_scripts": eval_scripts,
        }

    def _extract_resources(
        self,
        paper_content: str,
        repo_info: Dict[str, Any],
        task: PaperTask,
    ) -> Dict[str, Any]:
        """Extract dataset, model, and checkpoint information."""
        logger.info("Extracting resource information")

        # For now, use simple pattern matching
        # In a full implementation, we'd use LLM for better extraction

        datasets = []
        models = []
        checkpoints = []

        # Look for dataset mentions
        dataset_patterns = [
            r"dataset[:\s]+([^\n,]+)",
            r"using\s+([A-Z][a-zA-Z]+\s+dataset)",
            r"evaluated\s+on\s+([^\n,]+)",
        ]

        for pattern in dataset_patterns:
            matches = re.findall(pattern, paper_content, re.IGNORECASE)
            for match in matches:
                datasets.append(
                    {
                        "name": match.strip(),
                        "source": "paper",
                        "description": f"Dataset mentioned in paper",
                    }
                )

        # Look for model mentions
        model_patterns = [
            r"(BERT|GPT|T5|ViT|ResNet|Transformer)\s*(\d*)",
            r"pre-trained\s+on\s+([^\n,]+)",
        ]

        for pattern in model_patterns:
            matches = re.findall(pattern, paper_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    model_name = match[0] + match[1] if match[1] else match[0]
                else:
                    model_name = match
                models.append(
                    {
                        "name": model_name,
                        "source": "paper",
                        "description": f"Model mentioned in paper",
                    }
                )

        # Look for checkpoint links
        checkpoint_patterns = [
            r"(https?://[^\s]*checkpoint[^\s]*)",
            r"(https?://[^\s]*weights[^\s]*)",
            r"(https?://[^\s]*model[^\s]*\.(pth|pt|bin|ckpt))",
        ]

        for pattern in checkpoint_patterns:
            matches = re.findall(pattern, paper_content, re.IGNORECASE)
            for match in matches:
                checkpoints.append(
                    {
                        "name": Path(urlparse(match).path).name,
                        "url": match,
                        "description": f"Checkpoint link in paper",
                    }
                )

        return {
            "datasets": datasets[:5],  # Limit to top 5
            "models": models[:5],
            "checkpoints": checkpoints[:5],
        }
