"""Process-lifetime ownership for a training output directory."""

from __future__ import annotations

import atexit
import datetime as dt
import fcntl
import hashlib
import json
import os
import socket
import uuid
from pathlib import Path


class ActiveRunError(RuntimeError):
    """Raised when another live process already owns an output directory."""


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ActiveRunLock:
    """An advisory lock that is released by the kernel if the process dies.

    The JSON owner record is diagnostic only. Ownership is determined by
    ``flock``, so a SIGKILL or node loss cannot leave an unrecoverable lock.
    """

    def __init__(
        self,
        out_dir: str,
        *,
        run_id: str,
        config_path: str | None = None,
        config_identity: str | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.run_id = run_id
        self.config_path = config_path
        self.config_identity = config_identity
        self.lock_path = self.out_dir / ".active-run.lock"
        self.owner_path = self.out_dir / ".active-run-owner.json"
        self._fd: int | None = None
        self._nonce: str | None = None

    def acquire(self) -> "ActiveRunLock":
        if self._fd is not None:
            raise RuntimeError("active-run lock is already acquired")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            try:
                owner = self.owner_path.read_text(encoding="utf-8").strip()
            except OSError:
                owner = "owner metadata unavailable"
            raise ActiveRunError(
                f"output directory is owned by another live process: "
                f"{self.out_dir}; {owner}"
            ) from exc

        self._fd = fd
        self._nonce = uuid.uuid4().hex
        owner = {
            "schema_version": 1,
            "acquired_utc": _utc_now(),
            "config_file_sha256": _file_sha256(self.config_path),
            "config_identity": self.config_identity,
            "config_path": self.config_path,
            "hostname": socket.gethostname(),
            "lock_nonce": self._nonce,
            "pid": os.getpid(),
            "run_id": self.run_id,
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        temporary = self.owner_path.with_name(
            f"{self.owner_path.name}.tmp.{os.getpid()}.{self._nonce}"
        )
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(owner, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.owner_path)
        atexit.register(self.release)
        return self

    def release(self) -> None:
        fd = self._fd
        nonce = self._nonce
        if fd is None:
            return
        try:
            try:
                current = json.loads(self.owner_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
            if nonce and current.get("lock_nonce") == nonce:
                self.owner_path.unlink(missing_ok=True)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            self._fd = None
            self._nonce = None
            try:
                atexit.unregister(self.release)
            except Exception:
                pass

    def __enter__(self) -> "ActiveRunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
