"""Custom exceptions for AutoSOTA."""


class AutoSOTAError(Exception):
    """Base exception for AutoSOTA."""

    pass


class ResourceError(AutoSOTAError):
    """Exception raised when resource preparation fails."""

    pass


class ObjectiveError(AutoSOTAError):
    """Exception raised when objective building fails."""

    pass


class InitError(AutoSOTAError):
    """Exception raised when initialization fails."""

    pass


class MonitorError(AutoSOTAError):
    """Exception raised when monitoring fails."""

    pass


class FixError(AutoSOTAError):
    """Exception raised when fixing fails."""

    pass


class IdeatorError(AutoSOTAError):
    """Exception raised when ideation fails."""

    pass


class SchedulerError(AutoSOTAError):
    """Exception raised when scheduling fails."""

    pass


class SupervisorError(AutoSOTAError):
    """Exception raised when supervision fails."""

    pass


class DockerError(AutoSOTAError):
    """Exception raised when Docker operations fail."""

    pass


class ParseError(AutoSOTAError):
    """Exception raised when parsing fails."""

    pass


class StorageError(AutoSOTAError):
    """Exception raised when storage operations fail."""

    pass


class RedLineViolation(AutoSOTAError):
    """Exception raised when red line constraints are violated."""

    pass


class TimeoutError(AutoSOTAError):
    """Exception raised when operation times out."""

    pass


class DependencyError(AutoSOTAError):
    """Exception raised when dependency resolution fails."""

    pass
