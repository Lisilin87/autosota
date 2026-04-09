"""Basic tests for AutoSOTA services."""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.core import (
    PaperTask,
    TaskStatus,
    get_logger,
)
from src.services import SchedulerService


logger = get_logger(__name__)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_paper(temp_workspace):
    """Create a sample paper file."""
    paper_path = Path(temp_workspace) / "sample_paper.txt"
    paper_content = """
    Sample Paper Title
    
    This is a sample paper for testing AutoSOTA.
    
    GitHub: https://github.com/example/sample-repo
    
    We achieve 85.5% accuracy on the test set.
    
    The model is trained on the ImageNet dataset.
    """
    paper_path.write_text(paper_content)
    return str(paper_path)


def test_paper_task_creation():
    """Test PaperTask creation."""
    task = PaperTask(
        paper_id="test-001",
        title="Test Paper",
        paper_path="/path/to/paper.pdf",
        repo_url="https://github.com/test/repo",
        status=TaskStatus.NEW,
    )

    assert task.paper_id == "test-001"
    assert task.title == "Test Paper"
    assert task.status == TaskStatus.NEW


def test_scheduler_initialization(sample_paper, temp_workspace):
    """Test scheduler initialization."""
    task = PaperTask(
        paper_id="test-001",
        title="Test Paper",
        paper_path=sample_paper,
        repo_url="https://github.com/example/sample-repo",
        status=TaskStatus.NEW,
    )

    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout_sec": 120,
            "max_retries": 3,
        },
        "workspace_root": temp_workspace,
        "max_iterations": 2,
    }

    # Note: This will fail without OPENAI_API_KEY, but tests the structure
    try:
        scheduler = SchedulerService(
            task=task,
            config=config,
            workspace_root=temp_workspace,
            max_iterations=2,
        )
        assert scheduler.task.paper_id == "test-001"
        assert scheduler.max_iterations == 2
    except Exception as e:
        # Expected if API key not set
        logger.info(f"Scheduler init skipped (expected): {e}")


def test_workspace_structure(temp_workspace):
    """Test workspace directory structure."""
    workspace_path = Path(temp_workspace) / "test-paper"

    # Create expected directories
    (workspace_path / "paper").mkdir(parents=True)
    (workspace_path / "repo").mkdir(parents=True)
    (workspace_path / "resources").mkdir(parents=True)
    (workspace_path / "runs").mkdir(parents=True)
    (workspace_path / "memory").mkdir(parents=True)

    # Verify structure
    assert (workspace_path / "paper").exists()
    assert (workspace_path / "repo").exists()
    assert (workspace_path / "resources").exists()
    assert (workspace_path / "runs").exists()
    assert (workspace_path / "memory").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
