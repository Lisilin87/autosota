"""Idea service for AutoSOTA."""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.core import (
    get_logger,
    write_json,
    read_json,
    IdeaItem,
    IdeaType,
    IdeaGranularity,
    IdeaStatus,
    RiskLevel,
    IdeatorError,
)
from src.core.llm_client import LLMClient


logger = get_logger(__name__)


class IdeaService:
    """Service for generating and managing optimization ideas."""

    def __init__(
        self,
        llm_client: LLMClient,
        workspace_root: str,
        config: Dict[str, Any],
    ):
        self.llm_client = llm_client
        self.workspace_root = workspace_root
        self.config = config

    def build_initial_library(
        self,
        paper_id: str,
        code_analysis: str,
        research_report: str,
        baseline_results: Dict[str, Any],
        red_line_constraints: Dict[str, Any],
    ) -> List[IdeaItem]:
        """Build initial idea library."""
        logger.info(f"Building initial idea library for: {paper_id}")

        # Load prompt template
        template_path = Path("configs/prompts/ideator.yaml")
        template = self.llm_client.load_prompt_template(str(template_path))

        # Format prompt
        variables = {
            "code_analysis": code_analysis[:3000],
            "research_report": research_report[:2000],
            "baseline_results": json.dumps(baseline_results, indent=2),
            "execution_history": "[]",
            "red_line_constraints": json.dumps(red_line_constraints, indent=2),
        }

        system_prompt, user_prompt = self.llm_client.format_prompt(template, variables)

        # Call LLM
        response = self.llm_client.chat_json(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
        )

        # Parse ideas from response
        ideas_data = response.get("ideas", [])
        ideas = []

        for idea_data in ideas_data:
            idea = IdeaItem(
                idea_id=idea_data.get("idea_id", "") or str(uuid.uuid4())[:8],
                title=idea_data.get("title", "Untitled Idea"),
                idea_type=IdeaType(idea_data.get("idea_type", "PARAM")),
                granularity=IdeaGranularity(idea_data.get("granularity", "micro")),
                priority=idea_data.get("priority", 5),
                risk=RiskLevel(idea_data.get("risk", "medium")),
                rationale=idea_data.get("rationale", ""),
                assumptions=idea_data.get("assumptions", []),
                status=IdeaStatus.PENDING,
                redline_audit=idea_data.get("redline_audit", {}),
            )
            ideas.append(idea)

        # Save idea library
        self._save_idea_library(paper_id, ideas)

        logger.info(f"Generated {len(ideas)} ideas")
        return ideas

    def update_library(
        self,
        paper_id: str,
        iteration_result: Dict[str, Any],
    ) -> List[IdeaItem]:
        """Update idea library based on iteration results."""
        logger.info(f"Updating idea library for: {paper_id}")

        # Load existing ideas
        ideas = self._load_idea_library(paper_id)

        # Update idea status based on result
        idea_id = iteration_result.get("idea_id")
        success = iteration_result.get("success", False)
        metrics = iteration_result.get("metrics", {})

        for idea in ideas:
            if idea.idea_id == idea_id:
                if success:
                    idea.status = IdeaStatus.DONE
                else:
                    idea.status = IdeaStatus.REJECTED

                # Add to history
                idea.history.append(
                    {
                        "iteration": iteration_result.get("iteration"),
                        "success": success,
                        "metrics": metrics,
                        "timestamp": iteration_result.get("timestamp"),
                    }
                )
                break

        # Save updated library
        self._save_idea_library(paper_id, ideas)

        return ideas

    def select_next_idea(
        self,
        paper_id: str,
        iteration: int,
        recent_ideas: List[str],
    ) -> Optional[IdeaItem]:
        """Select next idea to try."""
        logger.info(f"Selecting next idea for iteration {iteration}")

        # Load ideas
        ideas = self._load_idea_library(paper_id)

        # Filter available ideas
        available = [
            idea
            for idea in ideas
            if idea.status == IdeaStatus.PENDING and idea.redline_audit.get("final") == "CLEARED"
        ]

        if not available:
            logger.info("No more available ideas")
            return None

        # Check for leap path condition
        # If last 3 iterations were all PARAM ideas, force a CODE/ALGO idea
        if len(recent_ideas) >= 3:
            recent_types = [self._get_idea_type(paper_id, idea_id) for idea_id in recent_ideas[-3:]]
            if all(t == IdeaType.PARAM for t in recent_types):
                # Force non-PARAM idea
                non_param = [idea for idea in available if idea.idea_type != IdeaType.PARAM]
                if non_param:
                    logger.info("Forcing leap path: selecting non-PARAM idea")
                    return sorted(non_param, key=lambda x: -x.priority)[0]

        # Select highest priority idea
        selected = sorted(available, key=lambda x: (-x.priority, x.risk))[0]

        logger.info(f"Selected idea: {selected.idea_id} - {selected.title}")
        return selected

    def get_idea_library(self, paper_id: str) -> List[IdeaItem]:
        """Get current idea library."""
        return self._load_idea_library(paper_id)

    def _load_idea_library(self, paper_id: str) -> List[IdeaItem]:
        """Load idea library from file."""
        workspace_path = Path(self.workspace_root) / paper_id
        library_path = workspace_path / "memory" / "idea_library.md"

        if not library_path.exists():
            return []

        try:
            # For now, load from JSON if it exists
            json_path = workspace_path / "idea_library.json"
            if json_path.exists():
                data = read_json(str(json_path))
                return [IdeaItem(**item) for item in data.get("ideas", [])]
        except Exception as e:
            logger.warning(f"Failed to load idea library: {e}")

        return []

    def _save_idea_library(self, paper_id: str, ideas: List[IdeaItem]) -> None:
        """Save idea library to file."""
        workspace_path = Path(self.workspace_root) / paper_id
        memory_dir = workspace_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = workspace_path / "idea_library.json"
        write_json(str(json_path), {"ideas": [idea.model_dump() for idea in ideas]})

        # Save as markdown
        md_path = memory_dir / "idea_library.md"
        self._write_idea_library_md(md_path, ideas)

    def _write_idea_library_md(self, path: Path, ideas: List[IdeaItem]) -> None:
        """Write idea library as markdown."""
        lines = ["# Idea Library\n"]
        lines.append("## Ideas\n")
        lines.append("| ID | Title | Type | Granularity | Priority | Risk | Status |")
        lines.append("|---|---|---|---|---|---|---|")

        for idea in ideas:
            lines.append(
                f"| {idea.idea_id} | {idea.title} | {idea.idea_type} | "
                f"{idea.granularity} | {idea.priority} | {idea.risk} | {idea.status} |"
            )

        lines.append("\n## Red Line Audit\n")
        lines.append("| ID | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Final |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for idea in ideas:
            audit = idea.redline_audit
            lines.append(
                f"| {idea.idea_id} | {audit.get('R1', '-')} | {audit.get('R2', '-')} | "
                f"{audit.get('R3', '-')} | {audit.get('R4', '-')} | {audit.get('R5', '-')} | "
                f"{audit.get('R6', '-')} | {audit.get('R7', '-')} | {audit.get('final', '-')} |"
            )

        lines.append("\n## Execution History\n")
        lines.append("| Iteration | Idea ID | Change | Result | Conclusion |")
        lines.append("|---|---|---|---|---|")

        for idea in ideas:
            for history in idea.history:
                lines.append(
                    f"| {history.get('iteration', '-')} | {idea.idea_id} | "
                    f"{history.get('change', '-')} | {history.get('result', '-')} | "
                    f"{history.get('conclusion', '-')} |"
                )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _get_idea_type(self, paper_id: str, idea_id: str) -> Optional[IdeaType]:
        """Get idea type by ID."""
        ideas = self._load_idea_library(paper_id)
        for idea in ideas:
            if idea.idea_id == idea_id:
                return idea.idea_type
        return None
