"""Fix service for AutoSOTA."""

import re
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.core import (
    get_logger,
    ErrorSignature,
    ErrorCategory,
    FixError,
)
from src.core.llm_client import LLMClient


logger = get_logger(__name__)


class FixService:
    """Service for fixing errors and issues."""

    def __init__(
        self,
        llm_client: LLMClient,
        workspace_root: str,
        config: Dict[str, Any],
    ):
        self.llm_client = llm_client
        self.workspace_root = workspace_root
        self.config = config

        self.repair_history: List[Dict[str, Any]] = []

    def analyze_error(
        self,
        exception: str,
        log_tail: str,
        env_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze error and suggest repair strategies."""
        logger.info("Analyzing error")

        # Normalize error signature
        error_signature = self._normalize_error(exception, log_tail)

        # Check if we've tried this before
        if self._is_repeated_attempt(error_signature):
            logger.warning(f"Repeated error detected: {error_signature.fingerprint}")
            return {
                "error_signature": error_signature.model_dump(),
                "repair_strategies": [],
                "needs_rollback": True,
                "needs_human_review": True,
                "explanation": "This error has been seen before without successful repair",
            }

        # Generate repair strategies
        repair_strategies = self._generate_repair_strategies(
            error_signature,
            log_tail,
            env_info,
        )

        # Record attempt
        self._record_attempt(error_signature, repair_strategies)

        return {
            "error_signature": error_signature.model_dump(),
            "repair_strategies": [s.model_dump() for s in repair_strategies],
            "needs_rollback": self._needs_rollback(error_signature),
            "needs_human_review": self._needs_human_review(error_signature),
            "explanation": self._generate_explanation(error_signature, repair_strategies),
        }

    def _normalize_error(self, exception: str, log_tail: str) -> ErrorSignature:
        """Normalize error to signature."""
        # Extract error type
        error_type = self._extract_error_type(exception)

        # Generate fingerprint
        fingerprint = self._generate_fingerprint(exception, log_tail)

        # Determine category
        category = self._determine_category(exception, log_tail)

        # Generate root cause hint
        root_cause_hint = self._generate_root_cause_hint(category, exception, log_tail)

        return ErrorSignature(
            category=category,
            fingerprint=fingerprint,
            root_cause_hint=root_cause_hint,
        )

    def _extract_error_type(self, exception: str) -> str:
        """Extract error type from exception string."""
        # Look for common error patterns
        patterns = [
            r"(\w+Error)",
            r"(\w+Exception)",
            r"ModuleNotFoundError:\s*(\w+)",
            r"FileNotFoundError:\s*(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, exception)
            if match:
                return match.group(1)

        return "UnknownError"

    def _generate_fingerprint(self, exception: str, log_tail: str) -> str:
        """Generate unique fingerprint for error."""
        # Combine exception and last few lines of log
        content = exception + "\n" + log_tail[-500:]

        # Remove variable parts (numbers, paths, timestamps)
        content = re.sub(r"\d+", "N", content)
        content = re.sub(r"/[\w/.-]+", "/PATH", content)
        content = re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", content)
        content = re.sub(r"\d{2}:\d{2}:\d{2}", "TIME", content)

        # Generate hash
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _determine_category(self, exception: str, log_tail: str) -> ErrorCategory:
        """Determine error category."""
        exception_lower = exception.lower()
        log_lower = log_tail.lower()

        # Check for dependency errors
        if any(
            keyword in exception_lower
            for keyword in [
                "modulenotfounderror",
                "importerror",
                "no module named",
            ]
        ):
            return ErrorCategory.MISSING_DEPENDENCY

        # Check for version conflicts
        if any(
            keyword in exception_lower
            for keyword in [
                "version conflict",
                "incompatible",
                "requires a different",
            ]
        ):
            return ErrorCategory.VERSION_CONFLICT

        # Check for missing files
        if any(
            keyword in exception_lower
            for keyword in [
                "filenotfounderror",
                "no such file",
                "file not found",
            ]
        ):
            return ErrorCategory.MISSING_FILE

        # Check for missing checkpoints
        if any(
            keyword in exception_lower
            for keyword in [
                "checkpoint",
                "checkpoint not found",
                "load checkpoint",
            ]
        ):
            return ErrorCategory.MISSING_CHECKPOINT

        # Check for path errors
        if any(
            keyword in exception_lower
            for keyword in [
                "path error",
                "invalid path",
                "path does not exist",
            ]
        ):
            return ErrorCategory.PATH_ERROR

        # Check for OOM
        if any(
            keyword in log_lower
            for keyword in [
                "out of memory",
                "cuda out of memory",
                "memory error",
            ]
        ):
            return ErrorCategory.OOM

        # Check for CUDA errors
        if any(
            keyword in exception_lower
            for keyword in [
                "cuda error",
                "cuda not available",
                "gpu not found",
            ]
        ):
            return ErrorCategory.CUDA_UNAVAILABLE

        # Check for metric parse errors
        if any(
            keyword in exception_lower
            for keyword in [
                "metric parse",
                "failed to parse metric",
                "invalid metric",
            ]
        ):
            return ErrorCategory.METRIC_PARSE_FAIL

        # Check for script not found
        if any(
            keyword in exception_lower
            for keyword in [
                "script not found",
                "command not found",
                "no such file or directory",
            ]
        ):
            return ErrorCategory.SCRIPT_NOT_FOUND

        # Check for permission errors
        if any(
            keyword in exception_lower
            for keyword in [
                "permission denied",
                "access denied",
                "permission error",
            ]
        ):
            return ErrorCategory.PERMISSION_ERROR

        # Check for timeout
        if any(
            keyword in exception_lower
            for keyword in [
                "timeout",
                "timed out",
            ]
        ):
            return ErrorCategory.TIMEOUT

        return ErrorCategory.UNKNOWN

    def _generate_root_cause_hint(
        self,
        category: ErrorCategory,
        exception: str,
        log_tail: str,
    ) -> str:
        """Generate hint about root cause."""
        hints = {
            ErrorCategory.MISSING_DEPENDENCY: "Required Python package not installed",
            ErrorCategory.VERSION_CONFLICT: "Package version incompatibility",
            ErrorCategory.MISSING_FILE: "Referenced file does not exist",
            ErrorCategory.MISSING_CHECKPOINT: "Model checkpoint not found or not downloaded",
            ErrorCategory.PATH_ERROR: "Invalid file or directory path",
            ErrorCategory.OOM: "Insufficient GPU memory",
            ErrorCategory.CUDA_UNAVAILABLE: "CUDA or GPU not available",
            ErrorCategory.METRIC_PARSE_FAIL: "Failed to parse evaluation metrics",
            ErrorCategory.SCRIPT_NOT_FOUND: "Execution script not found",
            ErrorCategory.PERMISSION_ERROR: "Insufficient file system permissions",
            ErrorCategory.TIMEOUT: "Operation exceeded time limit",
            ErrorCategory.UNKNOWN: "Unknown error cause",
        }

        return hints.get(category, "Unknown error cause")

    def _is_repeated_attempt(self, error_signature: ErrorSignature) -> bool:
        """Check if this error has been attempted before."""
        fingerprint = error_signature.fingerprint

        for attempt in self.repair_history:
            if attempt.get("fingerprint") == fingerprint:
                # Check if we've tried multiple strategies
                if attempt.get("attempt_count", 0) >= 3:
                    return True

        return False

    def _generate_repair_strategies(
        self,
        error_signature: ErrorSignature,
        log_tail: str,
        env_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate repair strategies based on error category."""
        strategies = []

        category = error_signature.category

        if category == ErrorCategory.MISSING_DEPENDENCY:
            strategies.append(
                {
                    "strategy": "Install missing dependency",
                    "commands": self._extract_install_command(error_signature.root_cause_hint),
                    "patches": [],
                    "confidence": 0.8,
                    "estimated_success_rate": 0.9,
                }
            )

        elif category == ErrorCategory.VERSION_CONFLICT:
            strategies.append(
                {
                    "strategy": "Pin package versions",
                    "commands": ["pip install --upgrade pip"],
                    "patches": [],
                    "confidence": 0.6,
                    "estimated_success_rate": 0.7,
                }
            )

        elif category == ErrorCategory.MISSING_FILE:
            strategies.append(
                {
                    "strategy": "Create missing file or fix path",
                    "commands": [],
                    "patches": self._generate_path_patches(log_tail),
                    "confidence": 0.7,
                    "estimated_success_rate": 0.6,
                }
            )

        elif category == ErrorCategory.OOM:
            strategies.append(
                {
                    "strategy": "Reduce batch size",
                    "commands": [],
                    "patches": self._generate_batch_size_patches(),
                    "confidence": 0.9,
                    "estimated_success_rate": 0.8,
                }
            )

        elif category == ErrorCategory.CUDA_UNAVAILABLE:
            strategies.append(
                {
                    "strategy": "Use CPU fallback",
                    "commands": [],
                    "patches": self._generate_cpu_fallback_patches(),
                    "confidence": 0.7,
                    "estimated_success_rate": 0.5,
                }
            )

        elif category == ErrorCategory.TIMEOUT:
            strategies.append(
                {
                    "strategy": "Increase timeout or reduce workload",
                    "commands": [],
                    "patches": self._generate_timeout_patches(),
                    "confidence": 0.6,
                    "estimated_success_rate": 0.5,
                }
            )

        return strategies

    def _extract_install_command(self, hint: str) -> List[str]:
        """Extract install command from hint."""
        # Try to extract package name
        match = re.search(r"no module named ['\"]?(\w+)['\"]?", hint, re.IGNORECASE)
        if match:
            package = match.group(1)
            return [f"pip install {package}"]
        return ["pip install -r requirements.txt"]

    def _generate_path_patches(self, log_tail: str) -> List[Dict[str, Any]]:
        """Generate path patches from log."""
        patches = []

        # Look for file paths in error
        matches = re.findall(r"File ['\"](.+?)['\"]", log_tail)
        for match in matches:
            patches.append(
                {
                    "file": "config.py",
                    "changes": f"# Fix path: {match}",
                }
            )

        return patches

    def _generate_batch_size_patches(self) -> List[Dict[str, Any]]:
        """Generate batch size reduction patches."""
        return [
            {
                "file": "config.py",
                "changes": "# Reduce batch size for OOM\nbatch_size = 16  # Reduced from 32",
            }
        ]

    def _generate_cpu_fallback_patches(self) -> List[Dict[str, Any]]:
        """Generate CPU fallback patches."""
        return [
            {
                "file": "config.py",
                "changes": "# Use CPU fallback\ndevice = 'cpu'  # Changed from 'cuda'",
            }
        ]

    def _generate_timeout_patches(self) -> List[Dict[str, Any]]:
        """Generate timeout patches."""
        return [
            {
                "file": "config.py",
                "changes": "# Increase timeout\ntimeout = 3600  # Increased from 1800",
            }
        ]

    def _needs_rollback(self, error_signature: ErrorSignature) -> bool:
        """Determine if rollback is needed."""
        critical_categories = [
            ErrorCategory.OOM,
            ErrorCategory.CUDA_UNAVAILABLE,
            ErrorCategory.TIMEOUT,
        ]
        return error_signature.category in critical_categories

    def _needs_human_review(self, error_signature: ErrorCategory) -> bool:
        """Determine if human review is needed."""
        return error_signature.category == ErrorCategory.UNKNOWN

    def _generate_explanation(
        self,
        error_signature: ErrorSignature,
        repair_strategies: List[Dict[str, Any]],
    ) -> str:
        """Generate explanation of the analysis."""
        explanation = f"Error category: {error_signature.category}\n"
        explanation += f"Root cause: {error_signature.root_cause_hint}\n"
        explanation += f"Found {len(repair_strategies)} repair strategies\n"
        return explanation

    def _record_attempt(
        self,
        error_signature: ErrorSignature,
        repair_strategies: List[Dict[str, Any]],
    ) -> None:
        """Record repair attempt."""
        existing = None
        for attempt in self.repair_history:
            if attempt.get("fingerprint") == error_signature.fingerprint:
                existing = attempt
                break

        if existing:
            existing["attempt_count"] = existing.get("attempt_count", 0) + 1
            existing["strategies_tried"].extend([s.get("strategy") for s in repair_strategies])
        else:
            self.repair_history.append(
                {
                    "fingerprint": error_signature.fingerprint,
                    "category": error_signature.category,
                    "attempt_count": 1,
                    "strategies_tried": [s.get("strategy") for s in repair_strategies],
                }
            )
