"""Supervisor service for AutoSOTA."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.core import (
    get_logger,
    read_json,
    write_json,
    SupervisorError,
    RedLineViolation,
)
from src.core.llm_client import LLMClient


logger = get_logger(__name__)


class SupervisorService:
    """Service for enforcing red line constraints and auditing compliance."""

    RED_LINES = {
        "R1": "Evaluation Parameters Fixed",
        "R2": "Evaluation Scripts Immutable",
        "R3": "Real Model Predictions",
        "R4": "Metric Balance",
        "R5": "No Test Set Leakage",
        "R6": "Dataset Distribution Unchanged",
        "R7": "Paper-Specific Constraints",
    }

    def __init__(
        self,
        llm_client: LLMClient,
        workspace_root: str,
        config: Dict[str, Any],
    ):
        self.llm_client = llm_client
        self.workspace_root = workspace_root
        self.config = config
        self.supervisor_config = config.get("supervisor", {})

        self.strict_mode = self.supervisor_config.get("strict_mode", True)
        self.audit_all_ideas = self.supervisor_config.get("audit_all_ideas", True)

        # Load red line policy
        self.redline_policy = self._load_redline_policy()

    def audit_idea(
        self,
        paper_id: str,
        idea: Dict[str, Any],
        code_analysis: str,
    ) -> Dict[str, Any]:
        """Audit an idea for red line compliance."""
        logger.info(f"Auditing idea: {idea.get('idea_id')}")

        audit_results = {}
        violations = []

        # Perform red line checks
        for red_line, description in self.RED_LINES.items():
            result = self._check_red_line(
                red_line,
                idea,
                code_analysis,
            )
            audit_results[red_line] = result

            if result["status"] == "FAIL":
                violations.append(
                    {
                        "red_line": red_line,
                        "description": description,
                        "details": result["details"],
                        "severity": result.get("severity", "medium"),
                    }
                )

        # Determine overall decision
        overall_decision = "CLEARED"
        if violations:
            if self.strict_mode:
                overall_decision = "REJECTED"
            else:
                # Check for critical violations
                critical = [v for v in violations if v.get("severity") == "critical"]
                if critical:
                    overall_decision = "REJECTED"

        return {
            "audit_results": audit_results,
            "overall_decision": overall_decision,
            "violations": violations,
            "recommendations": self._generate_recommendations(violations),
        }

    def audit_patch(
        self,
        paper_id: str,
        diff_text: str,
    ) -> Dict[str, Any]:
        """Audit a code patch for red line violations."""
        logger.info("Auditing code patch")

        violations = []

        # Check R2: Evaluation scripts immutable
        eval_patterns = [
            "*eval*.py",
            "*metric*.py",
            "*test*.py",
        ]

        for pattern in eval_patterns:
            if pattern in diff_text.lower() or "eval" in diff_text.lower():
                violations.append(
                    {
                        "red_line": "R2",
                        "description": "Evaluation Scripts Immutable",
                        "details": f"Patch may modify evaluation-related files matching {pattern}",
                        "severity": "critical",
                    }
                )

        # Check R3: Real model predictions
        forbidden_patterns = [
            "hardcoded",
            "fake.*prediction",
            "mock.*result",
            "return.*constant",
        ]

        for pattern in forbidden_patterns:
            import re

            if re.search(pattern, diff_text, re.IGNORECASE):
                violations.append(
                    {
                        "red_line": "R3",
                        "description": "Real Model Predictions",
                        "details": f"Patch contains forbidden pattern: {pattern}",
                        "severity": "critical",
                    }
                )

        # Check R1: Evaluation parameters fixed
        param_patterns = [
            "recall@",
            "history_window",
            "context_window",
            "max_length",
        ]

        for pattern in param_patterns:
            if pattern in diff_text.lower():
                violations.append(
                    {
                        "red_line": "R1",
                        "description": "Evaluation Parameters Fixed",
                        "details": f"Patch may modify evaluation parameter: {pattern}",
                        "severity": "critical",
                    }
                )

        return {
            "violations": violations,
            "approved": len(violations) == 0 or not self.strict_mode,
        }

    def audit_result(
        self,
        paper_id: str,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
    ) -> Dict[str, Any]:
        """Audit results for metric integrity."""
        logger.info("Auditing results")

        violations = []

        # Check R4: Metric balance
        # Primary metric improvement should not come at cost of other metrics
        if metrics_before and metrics_after:
            for metric, value_before in metrics_before.items():
                if metric in metrics_after:
                    value_after = metrics_after[metric]
                    change = (
                        (value_after - value_before) / abs(value_before) if value_before != 0 else 0
                    )

                    # If metric degraded significantly (>5%)
                    if change < -0.05:
                        violations.append(
                            {
                                "red_line": "R4",
                                "description": "Metric Balance",
                                "details": f"Metric {metric} degraded by {abs(change) * 100:.1f}%",
                                "severity": "high",
                                "metric": metric,
                                "before": value_before,
                                "after": value_after,
                                "change": change,
                            }
                        )

        return {
            "violations": violations,
            "approved": len(violations) == 0,
        }

    def _load_redline_policy(self) -> Dict[str, Any]:
        """Load red line policy from file."""
        policy_path = self.supervisor_config.get("redline_policy", "configs/policies/redlines.yaml")

        if Path(policy_path).exists():
            try:
                with open(policy_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load redline policy: {e}")

        return {}

    def _check_red_line(
        self,
        red_line: str,
        idea: Dict[str, Any],
        code_analysis: str,
    ) -> Dict[str, Any]:
        """Check a specific red line."""

        if red_line == "R1":
            return self._check_r1(idea)
        elif red_line == "R2":
            return self._check_r2(idea)
        elif red_line == "R3":
            return self._check_r3(idea)
        elif red_line == "R4":
            return self._check_r4(idea)
        elif red_line == "R5":
            return self._check_r5(idea)
        elif red_line == "R6":
            return self._check_r6(idea)
        elif red_line == "R7":
            return self._check_r7(idea, code_analysis)
        else:
            return {"status": "PASS", "details": "Unknown red line"}

    def _check_r1(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Check R1: Evaluation parameters fixed."""
        title = idea.get("title", "").lower()
        rationale = idea.get("rationale", "").lower()

        param_keywords = ["recall@", "k=", "window", "max_length"]

        for keyword in param_keywords:
            if keyword in title or keyword in rationale:
                return {
                    "status": "FAIL",
                    "details": f"Idea may modify evaluation parameter: {keyword}",
                    "severity": "critical",
                }

        return {"status": "PASS", "details": "No evaluation parameter modification detected"}

    def _check_r2(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Check R2: Evaluation scripts immutable."""
        title = idea.get("title", "").lower()
        rationale = idea.get("rationale", "").lower()

        eval_keywords = ["eval script", "metric function", "test script", "modify eval"]

        for keyword in eval_keywords:
            if keyword in title or keyword in rationale:
                return {
                    "status": "FAIL",
                    "details": f"Idea may modify evaluation scripts: {keyword}",
                    "severity": "critical",
                }

        return {"status": "PASS", "details": "No evaluation script modification detected"}

    def _check_r3(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Check R3: Real model predictions."""
        title = idea.get("title", "").lower()
        rationale = idea.get("rationale", "").lower()

        fake_keywords = ["hardcode", "fake", "mock", "constant output"]

        for keyword in fake_keywords:
            if keyword in title or keyword in rationale:
                return {
                    "status": "FAIL",
                    "details": f"Idea may use fake predictions: {keyword}",
                    "severity": "critical",
                }

        return {"status": "PASS", "details": "No fake prediction detected"}

    def _check_r4(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Check R4: Metric balance."""
        # This is more of a result check, but we can flag risky ideas
        rationale = idea.get("rationale", "").lower()

        if "ignore" in rationale or "sacrifice" in rationale:
            return {
                "status": "FAIL",
                "details": "Idea may sacrifice other metrics",
                "severity": "high",
            }

        return {"status": "PASS", "details": "No metric sacrifice detected"}

    def _check_r5(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Check R5: No test set leakage."""
        title = idea.get("title", "").lower()
        rationale = idea.get("rationale", "").lower()

        leakage_keywords = ["test set", "validation set", "leak", "peek"]

        for keyword in leakage_keywords:
            if keyword in title or keyword in rationale:
                return {
                    "status": "FAIL",
                    "details": f"Idea may cause test set leakage: {keyword}",
                    "severity": "critical",
                }

        return {"status": "PASS", "details": "No test set leakage detected"}

    def _check_r6(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Check R6: Dataset distribution unchanged."""
        title = idea.get("title", "").lower()
        rationale = idea.get("rationale", "").lower()

        modify_keywords = ["filter", "resample", "subset", "remove data", "relabel"]

        for keyword in modify_keywords:
            if keyword in title or keyword in rationale:
                return {
                    "status": "FAIL",
                    "details": f"Idea may modify dataset distribution: {keyword}",
                    "severity": "high",
                }

        return {"status": "PASS", "details": "No dataset modification detected"}

    def _check_r7(self, idea: Dict[str, Any], code_analysis: str) -> Dict[str, Any]:
        """Check R7: Paper-specific constraints."""
        # Check if code_analysis has hard constraints section
        if "hard constraints" in code_analysis.lower() or "red lines" in code_analysis.lower():
            return {"status": "PASS", "details": "Paper constraints documented"}

        return {
            "status": "WARN",
            "details": "Paper-specific constraints not explicitly documented",
            "severity": "low",
        }

    def _generate_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for fixing violations."""
        recommendations = []

        for violation in violations:
            red_line = violation.get("red_line")
            details = violation.get("details", "")

            if red_line == "R1":
                recommendations.append("Avoid modifying evaluation parameters")
            elif red_line == "R2":
                recommendations.append("Do not modify evaluation or metric scripts")
            elif red_line == "R3":
                recommendations.append("Ensure predictions come from real model inference")
            elif red_line == "R4":
                recommendations.append("Monitor all metrics, not just primary metric")
            elif red_line == "R5":
                recommendations.append("Ensure no test set information leaks into training")
            elif red_line == "R6":
                recommendations.append("Do not modify dataset distribution")
            elif red_line == "R7":
                recommendations.append("Document paper-specific constraints in code_analysis.md")

        return recommendations
