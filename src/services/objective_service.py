"""Objective service for AutoSOTA."""

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.core import (
    get_logger,
    write_json,
    read_text,
    ObjectiveRubricNode,
    PaperTask,
    ResourceManifest,
    ObjectiveError,
)
from src.core.llm_client import LLMClient


logger = get_logger(__name__)


class ObjectiveService:
    """Service for building objective rubrics."""

    def __init__(
        self,
        llm_client: LLMClient,
        workspace_root: str,
        config: Dict[str, Any],
    ):
        self.llm_client = llm_client
        self.workspace_root = workspace_root
        self.config = config
        self.objective_config = config.get("objective", {})

    def build_rubric(
        self,
        task: PaperTask,
        manifest: ResourceManifest,
    ) -> ObjectiveRubricNode:
        """Build objective rubric for the task."""
        logger.info(f"Building rubric for task: {task.paper_id}")

        # Load paper content
        paper_content = self._load_paper_content(task.paper_path)

        # Load repo info if available
        repo_info = self._load_repo_info(task.paper_id)

        # Build rubric using LLM
        rubric = self._build_rubric_with_llm(
            task=task,
            paper_content=paper_content,
            repo_info=repo_info,
            manifest=manifest,
        )

        # Save rubric
        workspace_path = Path(self.workspace_root) / task.paper_id
        rubric_path = workspace_path / "rubric.json"
        write_json(str(rubric_path), rubric.model_dump())

        logger.info(f"Rubric built and saved to: {rubric_path}")
        return rubric

    def detect_primary_metric(self, paper_text: str) -> Dict[str, Any]:
        """Detect primary metric from paper text."""
        logger.info("Detecting primary metric")

        # For now, use simple pattern matching
        # In a full implementation, we'd use LLM for better detection

        metric_patterns = [
            r"(accuracy|acc|precision|recall|f1|f1-score|auc|roc|mAP|AP)\s*[:=]\s*([\d.]+)",
            r"achieved\s+([\d.]+)%?\s+(accuracy|acc|precision|recall|f1)",
            r"(\w+)\s+of\s+([\d.]+)",
        ]

        detected_metrics = []

        for pattern in metric_patterns:
            matches = re.findall(pattern, paper_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    metric_name, value = match[0], match[1]
                else:
                    metric_name, value = match[0], match[1]
                detected_metrics.append(
                    {
                        "name": metric_name.lower(),
                        "value": float(value),
                    }
                )

        if detected_metrics:
            # Return the first detected metric
            return detected_metrics[0]

        # Default fallback
        return {
            "name": "accuracy",
            "direction": "max",
            "baseline_value": None,
            "target_value": None,
        }

    def _load_paper_content(self, paper_path: str) -> str:
        """Load paper content."""
        try:
            return read_text(paper_path)
        except Exception as e:
            raise ObjectiveError(f"Failed to load paper content: {e}")

    def _load_repo_info(self, paper_id: str) -> Dict[str, Any]:
        """Load repository information."""
        workspace_path = Path(self.workspace_root) / paper_id
        manifest_path = workspace_path / "resource_manifest.json"

        if manifest_path.exists():
            return read_json(str(manifest_path))

        return {}

    def _build_rubric_with_llm(
        self,
        task: PaperTask,
        paper_content: str,
        repo_info: Dict[str, Any],
        manifest: ResourceManifest,
    ) -> ObjectiveRubricNode:
        """Build rubric using LLM."""
        # Load prompt template
        template_path = Path("configs/prompts/objective.yaml")
        template = self.llm_client.load_prompt_template(str(template_path))

        # Format prompt
        variables = {
            "title": task.title,
            "paper_content": paper_content[:5000],  # Truncate for context
            "repo_info": json.dumps(repo_info, indent=2),
            "readiness_signals": json.dumps(manifest.readiness_signals, indent=2),
        }

        system_prompt, user_prompt = self.llm_client.format_prompt(template, variables)

        # Call LLM
        response = self.llm_client.chat_json(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
        )

        # Parse rubric from response
        rubric_data = response.get("rubric", {})
        primary_metric = response.get("primary_metric", {})

        # Save primary metric
        workspace_path = Path(self.workspace_root) / task.paper_id
        metric_path = workspace_path / "target_metric.json"
        write_json(str(metric_path), primary_metric)

        # Convert to ObjectiveRubricNode
        return self._dict_to_rubric_node(rubric_data)

    def _dict_to_rubric_node(self, data: Dict[str, Any]) -> ObjectiveRubricNode:
        """Convert dict to ObjectiveRubricNode."""
        children = []
        if "children" in data:
            for child_data in data["children"]:
                children.append(self._dict_to_rubric_node(child_data))

        return ObjectiveRubricNode(
            node_id=data.get("node_id", "root"),
            name=data.get("name", "Root"),
            description=data.get("description", ""),
            depth=data.get("depth", 0),
            weight=data.get("weight", 1.0),
            pass_fail=data.get("pass_fail"),
            evidence=data.get("evidence", []),
            children=children,
        )

    def _build_default_rubric(self) -> ObjectiveRubricNode:
        """Build default rubric structure."""
        return ObjectiveRubricNode(
            node_id="root",
            name="Reproduce Paper Results",
            description="Overall goal of reproducing the paper's reported results",
            depth=0,
            weight=1.0,
            children=[
                ObjectiveRubricNode(
                    node_id="env_ready",
                    name="Environment Ready",
                    description="Environment is set up and dependencies installed",
                    depth=1,
                    weight=0.2,
                    children=[
                        ObjectiveRubricNode(
                            node_id="deps_installed",
                            name="Dependencies Installed",
                            description="All required dependencies are installed",
                            depth=2,
                            weight=0.5,
                        ),
                        ObjectiveRubricNode(
                            node_id="resources_ready",
                            name="Resources Ready",
                            description="Datasets and models are downloaded",
                            depth=2,
                            weight=0.5,
                        ),
                    ],
                ),
                ObjectiveRubricNode(
                    node_id="baseline_runs",
                    name="Baseline Runs",
                    description="Baseline experiments run successfully",
                    depth=1,
                    weight=0.4,
                    children=[
                        ObjectiveRubricNode(
                            node_id="train_runs",
                            name="Training Runs",
                            description="Training completes without errors",
                            depth=2,
                            weight=0.5,
                        ),
                        ObjectiveRubricNode(
                            node_id="eval_runs",
                            name="Evaluation Runs",
                            description="Evaluation completes and produces metrics",
                            depth=2,
                            weight=0.5,
                        ),
                    ],
                ),
                ObjectiveRubricNode(
                    node_id="metrics_match",
                    name="Metrics Match Paper",
                    description="Reported metrics match paper values",
                    depth=1,
                    weight=0.4,
                    children=[
                        ObjectiveRubricNode(
                            node_id="primary_metric",
                            name="Primary Metric",
                            description="Primary metric is close to paper value",
                            depth=2,
                            weight=0.6,
                        ),
                        ObjectiveRubricNode(
                            node_id="secondary_metrics",
                            name="Secondary Metrics",
                            description="Secondary metrics are reasonable",
                            depth=2,
                            weight=0.4,
                        ),
                    ],
                ),
            ],
        )
