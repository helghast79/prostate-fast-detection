"""Repo path helpers for prostate detect tasks."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TASK_PROSTATE_DETECT = "prostate_detect"
KNOWN_TASKS = (TASK_PROSTATE_DETECT,)


def task_scripts(task: str) -> Path:
    return REPO_ROOT / "scripts" / task


def task_outputs(task: str) -> Path:
    return REPO_ROOT / "outputs" / task


def task_logs(task: str) -> Path:
    return REPO_ROOT / "logs" / task


def ensure_task_dirs(task: str) -> None:
    if task not in KNOWN_TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {KNOWN_TASKS}")
    task_outputs(task).mkdir(parents=True, exist_ok=True)
    task_logs(task).mkdir(parents=True, exist_ok=True)
