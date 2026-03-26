"""Instance lock — ensures only one hybro-hub daemon runs per machine.

Also owns LOCK_FILE and LOG_FILE path constants, keeping daemon lifecycle
concerns separate from configuration loading.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import IO, Any

from .config import HYBRO_DIR

logger = logging.getLogger(__name__)

LOCK_FILE = HYBRO_DIR / "hub.lock"
LOG_FILE = HYBRO_DIR / "hub.log"


def acquire_instance_lock() -> IO[Any]:
    """Acquire an exclusive lock on ~/.hybro/hub.lock.

    Returns the open file object — it must stay open for the lock to be held.
    The file handle is inherited across fork() so the daemon child keeps the lock
    after the parent exits.  Call write_lock_pid() in the child once its final PID
    is known.

    Raises SystemExit with a clear message if another instance is already running.
    """
    HYBRO_DIR.mkdir(parents=True, exist_ok=True)
    lock_fh: IO[Any] = open(LOCK_FILE, "w", encoding="utf-8")  # noqa: SIM115

    try:
        import fcntl  # Unix only
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        # Windows: use msvcrt
        import msvcrt
        try:
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            lock_fh.close()
            import sys
            logger.error(
                "Another hybro-hub instance is already running on this machine. "
                "Stop it before starting a new one."
            )
            sys.exit(1)
    except OSError:
        lock_fh.close()
        import sys
        logger.error(
            "Another hybro-hub instance is already running on this machine. "
            "Stop it before starting a new one."
        )
        sys.exit(1)

    return lock_fh


def write_lock_pid(lock_fh: IO[Any]) -> None:
    """Write (or overwrite) the current process's PID into the lock file."""
    lock_fh.seek(0)
    lock_fh.write(str(os.getpid()))
    lock_fh.flush()


def read_lock_pid() -> int | None:
    """Read the daemon PID from the lock file. Returns None if not found."""
    try:
        text = LOCK_FILE.read_text(encoding="utf-8").strip()
        return int(text) if text else None
    except (FileNotFoundError, ValueError, OSError):
        return None
