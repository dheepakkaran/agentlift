"""Blind prediction lock: the simulation is fixed before it is compared to reality."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


class LockError(RuntimeError):
    """Raised when a lock is missing or does not match the simulation file."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def lock_path(out_dir: Path, mode: str) -> Path:
    return Path(out_dir) / f"prediction_lock_{mode}.json"


def write(sim_path: Path, out_dir: Path, mode: str, config: dict) -> Path:
    """Record the hash, time, config and row count of a simulation before evaluation."""
    sim_path = Path(sim_path)
    if not sim_path.exists():
        raise LockError(f"cannot lock a simulation that does not exist: {sim_path}")
    rows = sum(1 for line in sim_path.read_text().splitlines() if line.strip())
    payload = {
        "sim_file": str(sim_path),
        "sha256": file_sha256(sim_path),
        "locked_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "n_rows": rows,
        "config": config,
    }
    path = lock_path(out_dir, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def verify(sim_path: Path, out_dir: Path, mode: str) -> dict:
    """Return the lock, or raise if it is missing or its hash does not match."""
    path = lock_path(out_dir, mode)
    if not path.exists():
        raise LockError(
            f"no prediction lock at {path}. Run `lock` before `evaluate` so the simulation "
            "is fixed before it meets the real results."
        )
    payload = json.loads(path.read_text())
    actual = file_sha256(sim_path)
    if payload["sha256"] != actual:
        raise LockError(
            f"simulation file has changed since it was locked "
            f"(locked {payload['sha256'][:12]}, found {actual[:12]})"
        )
    return payload
