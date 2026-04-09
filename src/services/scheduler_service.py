"""Scheduler service for AutoSOTA."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.core import (
    get_logger,
    ensure_dir,
    write_json,
    read_json,
    append_jsonl,
    PaperTask,
    ResourceManifest,
    SystemState,
    Phase,
    TaskStatus,
    RunRecord,
    SchedulerError,
    get_git_commit,
    get_git_diff,
    format_timestamp,
    run_command,
)
from src.services.resource_service import ResourceService
from src.services.objective_service import ObjectiveService
from src.services.init_service import InitService
from src.services.monitor_service import MonitorService
from src.services.fix_service import FixService
from src.services.idea_service import IdeaService
from src.services.supervisor_service import SupervisorService
from src.core.llm_client import LLMClient


logger = get_logger(__name__)


class SchedulerService:
    """Service for managing optimization lifecycle."""

    def __init__(
        self,
        task: PaperTask,
        config: Dict[str, Any],
        workspace_root: str,
        max_iterations: int = 8,
    ):
        self.task = task
        self.config = config
        self.workspace_root = workspace_root
        self.max_iterations = max_iterations

        # Initialize LLM client
        llm_config = config.get("llm", {})
        self.llm_client = LLMClient(
            provider=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2000),
            timeout=llm_config.get("timeout_sec", 120),
            max_retries=llm_config.get("max_retries", 3),
        )

        # Initialize services
        self.resource_service = ResourceService(self.llm_client, workspace_root, config)
        self.objective_service = ObjectiveService(self.llm_client, workspace_root, config)
        self.init_service = InitService(self.llm_client, workspace_root, config)
        self.monitor_service = MonitorService(self.llm_client, workspace_root, config)
        self.fix_service = FixService(self.llm_client, workspace_root, config)
        self.idea_service = IdeaService(self.llm_client, workspace_root, config)
        self.supervisor_service = SupervisorService(self.llm_client, workspace_root, config)

        # System state
        self.state: Optional[SystemState] = None

    @classmethod
    def resume(
        cls,
        paper_id: str,
        workspace_root: str,
        config: Dict[str, Any],
    ) -> "SchedulerService":
        """Resume a stopped scheduler."""
        workspace_path = Path(workspace_root) / paper_id
        state_path = workspace_path / "state.json"

        if not state_path.exists():
            raise SchedulerError(f"No state found for paper: {paper_id}")

        state_data = read_json(str(state_path))

        # Reconstruct task
        task = PaperTask(**state_data.get("task", {}))

        # Create scheduler
        scheduler = cls(task, config, workspace_root)
        scheduler.state = SystemState(**state_data)

        logger.info(f"Resumed scheduler for: {paper_id}")
        return scheduler

    def run(self) -> Dict[str, Any]:
        """Run the optimization pipeline."""
        logger.info(f"Starting optimization for: {self.task.paper_id}")

        try:
            # Load or initialize state
            if not self.state:
                self.state = self._initialize_state()

            # Run phases
            self._run_phase_0()
            self._run_phase_1()
            self._run_phase_2()
            self._run_phase_3()

            # Finalize
            self._finalize()

            return {
                "paper_id": self.task.paper_id,
                "status": "COMPLETED",
                "final_phase": self.state.phase,
                "total_iterations": self.state.iteration,
            }

        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            self.state.status = "FAILED"
            self._save_state()
            raise SchedulerError(f"Optimization failed: {e}")

    def _initialize_state(self) -> SystemState:
        """Initialize system state."""
        logger.info("Initializing system state")

        state = SystemState(
            paper_id=self.task.paper_id,
            phase=Phase.PHASE_0,
            iteration=0,
            status=TaskStatus.NEW,
        )

        self._save_state(state)
        return state

    def _run_phase_0(self) -> None:
        """Run Phase 0: Environment initialization and baseline measurement."""
        logger.info("Starting Phase 0: Environment initialization and baseline")

        self.state.phase = Phase.PHASE_0
        self.state.status = TaskStatus.RESOURCE_READY
        self._save_state()

        # Prepare resources
        logger.info("Preparing resources...")
        manifest = self.resource_service.prepare(self.task)

        self.state.status = TaskStatus.OBJECTIVE_READY
        self._save_state()

        # Build objective
        logger.info("Building objective rubric...")
        rubric = self.objective_service.build_rubric(self.task, manifest)

        self.state.status = TaskStatus.INIT_READY
        self._save_state()

        # Initialize environment
        logger.info("Initializing environment...")
        init_report = self.init_service.bootstrap(self.task, manifest)

        # Run baseline
        logger.info("Running baseline...")
        baseline_result = self._run_baseline()

        self.state.status = TaskStatus.BASELINE_DONE
        self._save_state()

        logger.info("Phase 0 completed")

    def _run_phase_1(self) -> None:
        """Run Phase 1: Code and paper understanding."""
        logger.info("Starting Phase 1: Code analysis")

        self.state.phase = Phase.PHASE_1
        self._save_state()

        # Generate code analysis
        logger.info("Generating code analysis...")
        code_analysis = self._generate_code_analysis()

        self.state.status = TaskStatus.CODE_ANALYZED
        self._save_state()

        logger.info("Phase 1 completed")

    def _run_phase_2(self) -> None:
        """Run Phase 2: Idea library construction."""
        logger.info("Starting Phase 2: Idea library construction")

        self.state.phase = Phase.PHASE_2
        self._save_state()

        # Load baseline results
        baseline_results = self._load_baseline_results()

        # Load code analysis
        code_analysis = self._load_code_analysis()

        # Generate research report
        research_report = self._generate_research_report(code_analysis, baseline_results)

        # Build idea library
        logger.info("Building idea library...")
        red_line_constraints = self.supervisor_service.redline_policy

        ideas = self.idea_service.build_initial_library(
            paper_id=self.task.paper_id,
            code_analysis=code_analysis,
            research_report=research_report,
            baseline_results=baseline_results,
            red_line_constraints=red_line_constraints,
        )

        # Audit all ideas
        for idea in ideas:
            audit_result = self.supervisor_service.audit_idea(
                paper_id=self.task.paper_id,
                idea=idea.model_dump(),
                code_analysis=code_analysis,
            )
            idea.redline_audit = audit_result.get("audit_results", {})
            if audit_result.get("overall_decision") == "REJECTED":
                idea.status = "REJECTED"

        # Save updated ideas
        self.idea_service._save_idea_library(self.task.paper_id, ideas)

        self.state.status = TaskStatus.IDEA_LIBRARY_READY
        self._save_state()

        logger.info("Phase 2 completed")

    def _run_phase_3(self) -> None:
        """Run Phase 3: Iterative optimization."""
        logger.info("Starting Phase 3: Iterative optimization")

        self.state.phase = Phase.PHASE_3
        self.state.status = TaskStatus.OPTIMIZING
        self._save_state()

        recent_ideas = []
        best_metrics = None
        best_iteration = 0

        for iteration in range(self.state.iteration, self.max_iterations):
            logger.info(f"Starting iteration {iteration + 1}/{self.max_iterations}")

            # Select idea
            idea = self.idea_service.select_next_idea(
                paper_id=self.task.paper_id,
                iteration=iteration,
                recent_ideas=recent_ideas,
            )

            if not idea:
                logger.info("No more ideas to try")
                break

            # Git snapshot
            logger.info("Creating git snapshot...")
            commit_before = get_git_commit(
                str(Path(self.workspace_root) / self.task.paper_id / "repo")
            )

            # Apply idea
            logger.info(f"Applying idea: {idea.title}")
            patch_result = self._apply_idea(idea)

            if not patch_result.get("success"):
                logger.warning(f"Failed to apply idea: {idea.title}")
                continue

            # Run evaluation
            logger.info("Running evaluation...")
            eval_result = self._run_evaluation()

            # Record result
            record = RunRecord(
                iteration=iteration + 1,
                git_commit=get_git_commit(
                    str(Path(self.workspace_root) / self.task.paper_id / "repo")
                )
                or "",
                idea_id=idea.idea_id,
                command=eval_result.get("command", ""),
                metrics=eval_result.get("metrics", {}),
                success=eval_result.get("success", False),
                log_path=eval_result.get("log_path", ""),
                start_time=format_timestamp(),
                end_time=format_timestamp(),
            )

            # Save record
            workspace_path = Path(self.workspace_root) / self.task.paper_id
            append_jsonl(str(workspace_path / "scores.jsonl"), record.model_dump())

            # Update idea library
            self.idea_service.update_library(
                paper_id=self.task.paper_id,
                iteration_result={
                    "idea_id": idea.idea_id,
                    "iteration": iteration + 1,
                    "success": eval_result.get("success", False),
                    "metrics": eval_result.get("metrics", {}),
                    "timestamp": format_timestamp(),
                },
            )

            # Track best result
            if eval_result.get("success") and eval_result.get("metrics"):
                if not best_metrics or self._is_better(
                    eval_result.get("metrics"),
                    best_metrics,
                ):
                    best_metrics = eval_result.get("metrics")
                    best_iteration = iteration + 1

                    # Save best patch
                    diff = get_git_diff(
                        str(Path(self.workspace_root) / self.task.paper_id / "repo"),
                        commit_before,
                    )
                    patch_path = workspace_path / "best_patch.diff"
                    with open(patch_path, "w", encoding="utf-8") as f:
                        f.write(diff)

            # Update state
            self.state.iteration = iteration + 1
            recent_ideas.append(idea.idea_id)
            if len(recent_ideas) > 3:
                recent_ideas.pop(0)

            self._save_state()

            logger.info(f"Iteration {iteration + 1} completed")

        logger.info("Phase 3 completed")

    def _finalize(self) -> None:
        """Finalize optimization."""
        logger.info("Finalizing optimization")

        self.state.phase = Phase.DONE
        self.state.status = TaskStatus.FINISHED
        self._save_state()

        # Generate final report
        self._generate_final_report()

        logger.info("Optimization completed successfully")

    def _run_baseline(self) -> Dict[str, Any]:
        """Run baseline evaluation."""
        workspace_path = Path(self.workspace_root) / self.task.paper_id
        repo_path = workspace_path / "repo"

        # Load init report to get eval command
        init_report = read_json(str(workspace_path / "init_report.json"))
        eval_command = init_report.get("eval_command")

        if not eval_command:
            logger.warning("No eval command found, skipping baseline")
            return {"success": False, "metrics": {}}

        # Run evaluation
        result = run_command(
            eval_command.split(),
            cwd=str(repo_path),
            timeout=3600,
        )

        # Parse metrics (simplified)
        metrics = self._parse_metrics(result.stdout)

        return {
            "success": result.returncode == 0,
            "metrics": metrics,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def _run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation."""
        workspace_path = Path(self.workspace_root) / self.task.paper_id
        repo_path = workspace_path / "repo"

        # Load init report to get eval command
        init_report = read_json(str(workspace_path / "init_report.json"))
        eval_command = init_report.get("eval_command")

        if not eval_command:
            return {"success": False, "metrics": {}}

        # Run evaluation
        result = run_command(
            eval_command.split(),
            cwd=str(repo_path),
            timeout=3600,
        )

        # Parse metrics
        metrics = self._parse_metrics(result.stdout)

        # Save log
        log_path = workspace_path / "runs" / f"iter_{self.state.iteration + 1}.log"
        ensure_dir(str(log_path.parent))
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)

        return {
            "success": result.returncode == 0,
            "metrics": metrics,
            "command": eval_command,
            "log_path": str(log_path),
        }

    def _parse_metrics(self, output: str) -> Dict[str, float]:
        """Parse metrics from output."""
        import re

        metrics = {}

        # Look for common metric patterns
        patterns = [
            r"(\w+)\s*[:=]\s*([\d.]+)",
            r"(\w+)\s*:\s*([\d.]+)%?",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for name, value in matches:
                try:
                    metrics[name.lower()] = float(value)
                except ValueError:
                    pass

        return metrics

    def _apply_idea(self, idea: Any) -> Dict[str, Any]:
        """Apply an idea to the codebase."""
        # This is a simplified version
        # In a full implementation, we'd use LLM to generate patches

        workspace_path = Path(self.workspace_root) / self.task.paper_id
        repo_path = workspace_path / "repo"

        # For PARAM ideas, modify config
        if idea.idea_type == "PARAM":
            return self._apply_param_idea(idea, repo_path)
        # For CODE/ALGO ideas, we'd need more sophisticated patching
        else:
            logger.warning(f"Idea type {idea.idea_type} not fully implemented")
            return {"success": False}

    def _apply_param_idea(self, idea: Any, repo_path: Path) -> Dict[str, Any]:
        """Apply a parameter idea."""
        # Simplified: just log the idea
        logger.info(f"Would apply param idea: {idea.title}")
        return {"success": True}

    def _generate_code_analysis(self) -> str:
        """Generate code analysis."""
        workspace_path = Path(self.workspace_root) / self.task.paper_id
        repo_path = workspace_path / "repo"

        # Simplified code analysis
        analysis = "# Code Analysis\n\n"
        analysis += "## Repository Summary\n"
        analysis += f"- Path: {repo_path}\n"
        analysis += "- Project purpose: Machine learning research\n"
        analysis += "- Task type: Classification/Regression\n"
        analysis += "- Main framework: PyTorch\n\n"

        analysis += "## Hard Constraints / Red Lines\n"
        analysis += "- R1: Evaluation parameters must not be modified\n"
        analysis += "- R2: Evaluation scripts must not be modified\n"
        analysis += "- R3: Predictions must be from real model inference\n"

        # Save to file
        memory_dir = workspace_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        with open(memory_dir / "code_analysis.md", "w", encoding="utf-8") as f:
            f.write(analysis)

        return analysis

    def _generate_research_report(
        self, code_analysis: str, baseline_results: Dict[str, Any]
    ) -> str:
        """Generate research report."""
        workspace_path = Path(self.workspace_root) / self.task.paper_id
        memory_dir = workspace_path / "memory"

        report = "# Research Report\n\n"
        report += "## Problem Summary\n"
        report += "Reproduce and improve upon the baseline model performance.\n\n"

        report += "## Baseline Results\n"
        report += f"```json\n{json.dumps(baseline_results, indent=2)}\n```\n\n"

        report += "## Related Optimization Directions\n"
        report += "- Hyperparameter tuning\n"
        report += "- Architecture modifications\n"
        report += "- Training strategy improvements\n\n"

        # Save to file
        with open(memory_dir / "research_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        return report

    def _generate_final_report(self) -> None:
        """Generate final report."""
        workspace_path = Path(self.workspace_root) / self.task.paper_id

        # Load scores
        scores = read_jsonl(str(workspace_path / "scores.jsonl"))

        report = "# Final Report\n\n"
        report += f"## Paper ID: {self.task.paper_id}\n"
        report += f"## Total Iterations: {len(scores)}\n\n"

        if scores:
            report += "## Best Result\n"
            best = max(scores, key=lambda x: x.get("metrics", {}).get("accuracy", 0))
            report += f"```json\n{json.dumps(best, indent=2)}\n```\n\n"

        report += "## Artifacts Generated\n"
        report += "- resource_manifest.json\n"
        report += "- rubric.json\n"
        report += "- init_report.json\n"
        report += "- code_analysis.md\n"
        report += "- idea_library.md\n"
        report += "- research_report.md\n"
        report += "- scores.jsonl\n"
        report += "- best_patch.diff\n"

        # Save to file
        with open(workspace_path / "final_report.md", "w", encoding="utf-8") as f:
            f.write(report)

    def _load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline results."""
        workspace_path = Path(self.workspace_root) / self.task.paper_id
        scores_path = workspace_path / "scores.jsonl"

        if scores_path.exists():
            scores = read_jsonl(str(scores_path))
            if scores:
                return scores[0].get("metrics", {})

        return {}

    def _load_code_analysis(self) -> str:
        """Load code analysis."""
        workspace_path = Path(self.workspace_root) / self.task.paper_id
        analysis_path = workspace_path / "memory" / "code_analysis.md"

        if analysis_path.exists():
            with open(analysis_path, "r", encoding="utf-8") as f:
                return f.read()

        return ""

    def _is_better(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> bool:
        """Compare metrics to determine if metrics1 is better than metrics2."""
        # Simplified: compare accuracy
        acc1 = metrics1.get("accuracy", 0)
        acc2 = metrics2.get("accuracy", 0)
        return acc1 > acc2

    def _save_state(self, state: Optional[SystemState] = None) -> None:
        """Save system state."""
        if state:
            self.state = state

        if not self.state:
            return

        workspace_path = Path(self.workspace_root) / self.task.paper_id
        state_path = workspace_path / "state.json"

        state_data = self.state.model_dump()
        state_data["task"] = self.task.model_dump()
        state_data["updated_at"] = format_timestamp()

        write_json(str(state_path), state_data)
