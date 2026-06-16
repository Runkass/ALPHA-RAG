"""Canonical paths after repo layout split (project vs agent vs archive)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Agent workflow (plans, state, Cursor rules)
AGENT_DIR = ROOT / "agent"
AGENT_PLAN_DIR = AGENT_DIR / "plan"
AGENT_SPECS_DIR = AGENT_DIR / "specs"
STATE_PATH = AGENT_DIR / "STATE.md"
JOURNAL_PATH = AGENT_PLAN_DIR / "journal.md"

# Competition inputs / docs (not agent plans)
MATERIALS_DIR = ROOT / "materials"
COMPETITION_TEXT = MATERIALS_DIR / "competition_rag.txt"

# Submissions
ACTIVE_SUBMISSION = ROOT / "submission.csv"
ARCHIVE_SUBMISSIONS = ROOT / "archive" / "submissions"
ARCHIVE_LOGS = ROOT / "archive" / "logs"
BACKUP_FULL = ARCHIVE_SUBMISSIONS / "submission.csv.bak.FULL"


def submission_backup(name: str) -> Path:
    """e.g. submission_backup('phase7-step5') -> archive/.../submission.csv.bak.phase7-step5"""
    if name.startswith("submission.csv.bak."):
        return ARCHIVE_SUBMISSIONS / name
    return ARCHIVE_SUBMISSIONS / f"submission.csv.bak.{name}"
