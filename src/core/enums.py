"""Enumerations for AutoSOTA."""

from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration."""

    NEW = "NEW"
    RESOURCE_READY = "RESOURCE_READY"
    OBJECTIVE_READY = "OBJECTIVE_READY"
    INIT_READY = "INIT_READY"
    BASELINE_DONE = "BASELINE_DONE"
    CODE_ANALYZED = "CODE_ANALYZED"
    IDEA_LIBRARY_READY = "IDEA_LIBRARY_READY"
    OPTIMIZING = "OPTIMIZING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    NEED_REVIEW = "NEED_REVIEW"


class Phase(str, Enum):
    """System phase enumeration."""

    PHASE_0 = "PHASE_0"
    PHASE_1 = "PHASE_1"
    PHASE_2 = "PHASE_2"
    PHASE_3 = "PHASE_3"
    DONE = "DONE"


class IdeaType(str, Enum):
    """Idea type enumeration."""

    PARAM = "PARAM"
    CODE = "CODE"
    ALGO = "ALGO"


class IdeaGranularity(str, Enum):
    """Idea granularity enumeration."""

    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"


class IdeaStatus(str, Enum):
    """Idea status enumeration."""

    PENDING = "PENDING"
    CLEARED = "CLEARED"
    REJECTED = "REJECTED"
    DONE = "DONE"


class RiskLevel(str, Enum):
    """Risk level enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ErrorCategory(str, Enum):
    """Error category enumeration."""

    MISSING_DEPENDENCY = "missing_dependency"
    VERSION_CONFLICT = "version_conflict"
    MISSING_FILE = "missing_file"
    MISSING_CHECKPOINT = "missing_checkpoint"
    PATH_ERROR = "path_error"
    OOM = "oom"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    METRIC_PARSE_FAIL = "metric_parse_fail"
    SCRIPT_NOT_FOUND = "script_not_found"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class MonitorAction(str, Enum):
    """Monitor action enumeration."""

    CONTINUE = "CONTINUE"
    RESUME_WITH_GUIDANCE = "RESUME_WITH_GUIDANCE"
    FALLBACK = "FALLBACK"
    TERMINATE = "TERMINATE"
    ROLLBACK = "ROLLBACK"


class OptimizationDirection(str, Enum):
    """Optimization direction enumeration."""

    MAX = "max"
    MIN = "min"
