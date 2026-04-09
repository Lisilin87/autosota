"""Utility functions for AutoSOTA."""

import os
import json
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str) -> Dict[str, Any]:
    """Read JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any, indent: int = 2) -> None:
    """Write JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_text(path: str) -> str:
    """Read text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    """Write text file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def append_jsonl(path: str, data: Dict[str, Any]) -> None:
    """Append to JSONL file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file."""
    if not os.path.exists(path):
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def run_command(
    cmd: List[str],
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run shell command."""
    kwargs = {"cwd": cwd, "timeout": timeout}
    if capture:
        kwargs.update({"capture_output": True, "text": True})

    return subprocess.run(cmd, **kwargs)


def get_git_commit(path: str) -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = run_command(["git", "rev-parse", "HEAD"], cwd=path, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_diff(path: str, commit: Optional[str] = None) -> str:
    """Get git diff."""
    try:
        if commit:
            result = run_command(["git", "diff", commit, "HEAD"], cwd=path, timeout=10)
        else:
            result = run_command(["git", "diff"], cwd=path, timeout=10)
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return ""


def get_file_tree(path: str, max_depth: int = 3) -> List[str]:
    """Get file tree structure."""
    tree = []
    root = Path(path)

    for item in root.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(root)
            parts = rel_path.parts
            if len(parts) <= max_depth:
                tree.append(str(rel_path))

    return sorted(tree)


def parse_requirements(path: str) -> List[str]:
    """Parse requirements.txt file."""
    if not os.path.exists(path):
        return []

    requirements = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime to ISO string."""
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat()


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp string."""
    return datetime.fromisoformat(ts)


def sanitize_filename(name: str) -> str:
    """Sanitize string for use as filename."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_list(nested: List[List[Any]]) -> List[Any]:
    """Flatten nested list."""
    return [item for sublist in nested for item in sublist]
