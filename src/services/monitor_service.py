"""Monitor service for AutoSOTA."""

import re
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.core import (
    get_logger,
    read_jsonl,
    MonitorError,
    MonitorAction,
    AgentResponse,
)
from src.core.llm_client import LLMClient


logger = get_logger(__name__)


class MonitorService:
    """Service for monitoring execution and detecting issues."""

    def __init__(
        self,
        llm_client: LLMClient,
        workspace_root: str,
        config: Dict[str, Any],
    ):
        self.llm_client = llm_client
        self.workspace_root = workspace_root
        self.config = config
        self.monitor_config = config.get("monitor", {})

        self.stuck_timeout = self.monitor_config.get("stuck_timeout_sec", 1200)
        self.repeated_error_threshold = self.monitor_config.get("repeated_error_threshold", 3)
        self.log_check_interval = self.monitor_config.get("log_check_interval_sec", 10)

    def inspect(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect current execution state."""
        logger.info("Inspecting execution state")

        paper_id = run_context.get("paper_id")
        workspace_path = Path(self.workspace_root) / paper_id

        # Get log content
        log_content = self._get_log_content(workspace_path, run_context)

        # Get process status
        process_status = self._get_process_status(run_context)

        # Get latest results
        latest_results = self._get_latest_results(workspace_path)

        # Get previous state
        previous_state = self._get_previous_state(workspace_path)

        # Detect issues
        detected_issues = self._detect_issues(
            log_content,
            process_status,
            latest_results,
            previous_state,
        )

        # Determine action
        action = self._determine_action(detected_issues, process_status)

        return {
            "current_phase": self._detect_current_phase(log_content, process_status),
            "risk_level": self._assess_risk_level(detected_issues),
            "detected_issues": detected_issues,
            "action": action,
            "guidance": self._generate_guidance(action, detected_issues),
            "confidence": self._calculate_confidence(detected_issues),
        }

    def _get_log_content(self, workspace_path: Path, run_context: Dict[str, Any]) -> str:
        """Get recent log content."""
        log_path = workspace_path / "runs" / "latest.log"

        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # Return last 100 lines
                    return "".join(lines[-100:])
            except Exception as e:
                logger.warning(f"Failed to read log: {e}")

        return ""

    def _get_process_status(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get process status."""
        pid = run_context.get("pid")

        if pid:
            try:
                import psutil

                process = psutil.Process(pid)
                return {
                    "running": process.is_running(),
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "create_time": process.create_time(),
                }
            except Exception:
                pass

        return {"running": False}

    def _get_latest_results(self, workspace_path: Path) -> List[Dict[str, Any]]:
        """Get latest run results."""
        scores_path = workspace_path / "scores.jsonl"

        if scores_path.exists():
            try:
                records = read_jsonl(str(scores_path))
                return records[-5:] if records else []
            except Exception as e:
                logger.warning(f"Failed to read scores: {e}")

        return []

    def _get_previous_state(self, workspace_path: Path) -> Dict[str, Any]:
        """Get previous monitor state."""
        state_path = workspace_path / "monitor_state.json"

        if state_path.exists():
            try:
                return read_json(str(state_path))
            except Exception:
                pass

        return {}

    def _detect_issues(
        self,
        log_content: str,
        process_status: Dict[str, Any],
        latest_results: List[Dict[str, Any]],
        previous_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect issues in execution."""
        issues = []

        # Check for stuck process
        if process_status.get("running"):
            create_time = process_status.get("create_time")
            if create_time:
                elapsed = time.time() - create_time
                if elapsed > self.stuck_timeout:
                    issues.append(
                        {
                            "type": "stuck_process",
                            "description": f"Process stuck for {elapsed:.0f} seconds",
                            "severity": "high",
                            "evidence": [f"PID: {process_status.get('pid')}"],
                        }
                    )

        # Check for repeated errors
        error_patterns = [
            r"Error:",
            r"Exception:",
            r"Traceback",
            r"Failed",
        ]

        error_count = 0
        for pattern in error_patterns:
            error_count += len(re.findall(pattern, log_content, re.IGNORECASE))

        if error_count >= self.repeated_error_threshold:
            issues.append(
                {
                    "type": "repeated_errors",
                    "description": f"Detected {error_count} errors in recent logs",
                    "severity": "medium",
                    "evidence": [f"Error count: {error_count}"],
                }
            )

        # Check for missing metrics
        if not latest_results:
            issues.append(
                {
                    "type": "missing_metrics",
                    "description": "No metrics produced in recent runs",
                    "severity": "medium",
                    "evidence": [],
                }
            )

        # Check for OOM errors
        if "out of memory" in log_content.lower() or "cuda out of memory" in log_content.lower():
            issues.append(
                {
                    "type": "oom_error",
                    "description": "Out of memory error detected",
                    "severity": "high",
                    "evidence": ["CUDA out of memory"],
                }
            )

        # Check for timeout
        if "timeout" in log_content.lower():
            issues.append(
                {
                    "type": "timeout",
                    "description": "Timeout detected",
                    "severity": "medium",
                    "evidence": ["Timeout in logs"],
                }
            )

        return issues

    def _detect_current_phase(self, log_content: str, process_status: Dict[str, Any]) -> str:
        """Detect current execution phase."""
        if not process_status.get("running"):
            return "DONE"

        phase_keywords = {
            "SETUP": ["setup", "initializing", "bootstrapping"],
            "INSTALL": ["install", "dependency", "requirements"],
            "PREPARE": ["prepare", "loading", "dataset"],
            "RUNNING": ["training", "epoch", "batch"],
            "EVALUATING": ["evaluating", "testing", "validation"],
        }

        log_lower = log_content.lower()

        for phase, keywords in phase_keywords.items():
            for keyword in keywords:
                if keyword in log_lower:
                    return phase

        return "RUNNING"

    def _assess_risk_level(self, issues: List[Dict[str, Any]]) -> str:
        """Assess overall risk level."""
        if not issues:
            return "low"

        severities = [issue.get("severity", "low") for issue in issues]

        if "critical" in severities:
            return "critical"
        elif "high" in severities:
            return "high"
        elif "medium" in severities:
            return "medium"
        else:
            return "low"

    def _determine_action(
        self, issues: List[Dict[str, Any]], process_status: Dict[str, Any]
    ) -> str:
        """Determine appropriate action based on issues."""
        if not issues:
            return MonitorAction.CONTINUE

        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        high_issues = [i for i in issues if i.get("severity") == "high"]

        if critical_issues:
            return MonitorAction.TERMINATE
        elif high_issues:
            if any(i.get("type") == "stuck_process" for i in high_issues):
                return MonitorAction.TERMINATE
            elif any(i.get("type") == "oom_error" for i in high_issues):
                return MonitorAction.FALLBACK
            else:
                return MonitorAction.RESUME_WITH_GUIDANCE
        else:
            return MonitorAction.CONTINUE

    def _generate_guidance(self, action: str, issues: List[Dict[str, Any]]) -> str:
        """Generate guidance for the action."""
        if action == MonitorAction.CONTINUE:
            return "Continue execution normally"
        elif action == MonitorAction.RESUME_WITH_GUIDANCE:
            issue_types = [i.get("type") for i in issues]
            return f"Address issues: {', '.join(issue_types)} and resume"
        elif action == MonitorAction.FALLBACK:
            return "Use fallback strategy (e.g., reduce batch size, use CPU)"
        elif action == MonitorAction.TERMINATE:
            return "Terminate execution due to critical issues"
        elif action == MonitorAction.ROLLBACK:
            return "Rollback to last known good state"
        else:
            return "No specific guidance"

    def _calculate_confidence(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the assessment."""
        if not issues:
            return 1.0

        # More issues = lower confidence
        return max(0.5, 1.0 - len(issues) * 0.1)
