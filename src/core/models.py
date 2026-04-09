"""Core data models for AutoSOTA."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime


class PaperTask(BaseModel):
    """Represents a paper replication task."""

    paper_id: str = Field(..., description="Unique identifier for the paper")
    title: str = Field(..., description="Paper title")
    paper_path: str = Field(..., description="Local path to paper PDF/HTML")
    repo_url: Optional[str] = Field(None, description="GitHub repository URL")
    conference: Optional[str] = Field(None, description="Conference name")
    domain: Optional[str] = Field(None, description="Research domain")
    target_metric: Optional[str] = Field(None, description="Primary metric to optimize")
    target_direction: Optional[Literal["max", "min"]] = Field(
        None, description="Optimization direction"
    )
    baseline_metric: Optional[float] = Field(
        None, description="Baseline metric value from paper"
    )
    status: str = Field(default="NEW", description="Task status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ResourceManifest(BaseModel):
    """Manifest of resources prepared for the task."""

    repo_candidates: List[str] = Field(
        default_factory=list, description="Candidate repository URLs"
    )
    selected_repo: Optional[str] = Field(None, description="Selected repository URL")
    dataset_items: List[Dict[str, Any]] = Field(
        default_factory=list, description="Dataset information"
    )
    model_items: List[Dict[str, Any]] = Field(
        default_factory=list, description="Base model information"
    )
    checkpoint_items: List[Dict[str, Any]] = Field(
        default_factory=list, description="Checkpoint information"
    )
    readiness_signals: Dict[str, Any] = Field(
        default_factory=dict, description="Repository readiness signals"
    )
    local_paths: Dict[str, str] = Field(
        default_factory=dict, description="Local resource paths"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ObjectiveRubricNode(BaseModel):
    """Node in the objective rubric tree."""

    node_id: str = Field(..., description="Unique node identifier")
    name: str = Field(..., description="Node name")
    description: str = Field(..., description="Node description")
    depth: int = Field(..., description="Depth in the rubric tree")
    weight: float = Field(..., description="Node weight")
    pass_fail: Optional[bool] = Field(None, description="Pass/fail status")
    evidence: List[str] = Field(default_factory=list, description="Evidence sources")
    children: List["ObjectiveRubricNode"] = Field(
        default_factory=list, description="Child nodes"
    )


ObjectiveRubricNode.model_rebuild()


class IdeaItem(BaseModel):
    """Represents an optimization idea."""

    idea_id: str = Field(..., description="Unique idea identifier")
    title: str = Field(..., description="Idea title")
    idea_type: Literal["PARAM", "CODE", "ALGO"] = Field(..., description="Idea type")
    granularity: Literal["micro", "meso", "macro"] = Field(
        ..., description="Idea granularity"
    )
    priority: int = Field(..., description="Priority (higher is more important)")
    risk: Literal["low", "medium", "high"] = Field(..., description="Risk level")
    rationale: str = Field(..., description="Rationale for the idea")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions")
    status: Literal["PENDING", "CLEARED", "REJECTED", "DONE"] = Field(
        default="PENDING", description="Idea status"
    )
    redline_audit: Dict[str, Any] = Field(
        default_factory=dict, description="Red line audit results"
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Execution history"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RunRecord(BaseModel):
    """Record of a single optimization iteration."""

    iteration: int = Field(..., description="Iteration number")
    git_commit: str = Field(..., description="Git commit hash")
    idea_id: Optional[str] = Field(None, description="Applied idea ID")
    command: str = Field(..., description="Command executed")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Metrics obtained"
    )
    success: bool = Field(..., description="Whether the run succeeded")
    log_path: str = Field(..., description="Path to log file")
    start_time: str = Field(..., description="Start time (ISO format)")
    end_time: str = Field(..., description="End time (ISO format)")
    duration_sec: Optional[float] = Field(None, description="Duration in seconds")


class ErrorSignature(BaseModel):
    """Normalized error signature for fix service."""

    category: str = Field(..., description="Error category")
    fingerprint: str = Field(..., description="Error fingerprint")
    root_cause_hint: Optional[str] = Field(None, description="Root cause hint")


class SystemState(BaseModel):
    """Global system state for persistence."""

    paper_id: str = Field(..., description="Paper ID")
    phase: Literal["PHASE_0", "PHASE_1", "PHASE_2", "PHASE_3", "DONE"] = Field(
        default="PHASE_0", description="Current phase"
    )
    iteration: int = Field(default=0, description="Current iteration")
    status: str = Field(default="NEW", description="System status")
    container_id: Optional[str] = Field(None, description="Docker container ID")
    pid: Optional[int] = Field(None, description="Process ID")
    assigned_gpu: Optional[str] = Field(None, description="Assigned GPU")
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat time")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class InitReport(BaseModel):
    """Report from initialization service."""

    docker_image: Optional[str] = Field(None, description="Docker image used")
    python_version: Optional[str] = Field(None, description="Python version")
    torch_version: Optional[str] = Field(None, description="PyTorch version")
    cuda_available: Optional[bool] = Field(None, description="CUDA availability")
    gpu_count: Optional[int] = Field(None, description="GPU count")
    train_command: Optional[str] = Field(None, description="Discovered train command")
    eval_command: Optional[str] = Field(None, description="Discovered eval command")
    dry_run_success: bool = Field(default=False, description="Dry run success")
    warnings: List[str] = Field(
        default_factory=list, description="Initialization warnings"
    )
    errors: List[str] = Field(default_factory=list, description="Initialization errors")


class AgentResponse(BaseModel):
    """Standardized response from all agents."""

    thought_summary: str = Field(..., description="Summary of agent's reasoning")
    decision: str = Field(..., description="Agent's decision")
    artifacts: List[str] = Field(
        default_factory=list, description="Generated artifacts"
    )
    actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Actions taken"
    )
    risks: List[str] = Field(default_factory=list, description="Identified risks")
    next_step: Optional[str] = Field(None, description="Suggested next step")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
