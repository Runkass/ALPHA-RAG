from __future__ import annotations

import ctypes
import os
import sys
from contextlib import contextmanager

from .config import PROJECT_ROOT

LOCK = PROJECT_ROOT / "data" / "pipeline.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    access = 0x00100000 | 0x1000  # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
    handle = ctypes.windll.kernel32.OpenProcess(access, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


@contextmanager
def exclusive_pipeline_lock():
    """Fail if another generate_submission holds the lock."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            old_pid = int(LOCK.read_text(encoding="utf-8").strip())
        except ValueError:
            old_pid = -1
        if _pid_alive(old_pid) and old_pid != os.getpid():
            print(
                f"Another pipeline is running (pid={old_pid}). "
                "Stop it before starting a second generate_submission.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        LOCK.unlink(missing_ok=True)
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)


@contextmanager
def pipeline_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)
