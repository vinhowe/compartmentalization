import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict

from .config.job_config import JobConfig


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s.strip())[:80].strip("-_.")


# Keep the old name for internal callers
_slug = slug


def _canonical_cfg_dict(cfg: JobConfig) -> dict:
    d = asdict(cfg)
    return {
        **d,
        "lr": d["lr"],
        "seed": d["training"]["seed"],
    }


def cfg_hash(cfg: JobConfig) -> str:
    s = json.dumps(_canonical_cfg_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2s(s.encode(), digest_size=4).hexdigest()


def _shortsha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def run_dirs(
    root: str, project: str, group: str | None, name: str, cfg: JobConfig, runid: str
) -> tuple[str, str, str]:
    project_dir = os.path.join(root, _slug(project))
    group_dir = os.path.join(project_dir, _slug(group or "default"))
    run_dir = os.path.join(
        group_dir,
        f"{_timestamp()}__{_slug(name)}__{cfg_hash(cfg)}__s{cfg.training.seed}__{_shortsha()}__{runid}",
    )
    return (
        project_dir,
        group_dir,
        run_dir,
    )


def write_meta(out_dir: str, cfg: JobConfig) -> None:
    import json

    meta = os.path.join(out_dir, "meta")
    os.makedirs(meta, exist_ok=True)

    with open(os.path.join(meta, "config.json"), "w") as f:
        json.dump(_canonical_cfg_dict(cfg), f, indent=2)

    # Freeze the exact interpreter environment. Prefer the running Python;
    # ``uv pip freeze`` without --python can silently inspect another venv.
    try:
        pf = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        pf = subprocess.check_output(
            ["uv", "pip", "freeze", "--python", sys.executable],
            stderr=subprocess.DEVNULL,
        ).decode()
    with open(os.path.join(meta, "pip_freeze.txt"), "w") as f:
        f.write(pf)

    # A content-addressed deployment intentionally has no .git directory.
    # Retain its manifest directly and treat Git metadata as optional context.
    deployment_manifest = os.environ.get("DEPLOYMENT_MANIFEST")
    if deployment_manifest and os.path.isfile(deployment_manifest):
        copied_manifest = os.path.join(meta, "deployment-manifest.json")
        shutil.copyfile(deployment_manifest, copied_manifest)
        with open(deployment_manifest, "rb") as handle:
            deployment_sha256 = hashlib.sha256(handle.read()).hexdigest()
        with open(os.path.join(meta, "deployment_manifest_sha256.txt"), "w") as f:
            f.write(deployment_sha256 + "\n")

    try:
        full = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        full = "unavailable:content-addressed-deployment"
    with open(os.path.join(meta, "git_commit.txt"), "w") as f:
        f.write(full + "\n")

    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode()
    except (OSError, subprocess.CalledProcessError):
        diff = ""
    with open(os.path.join(meta, "git_diff.patch"), "w") as f:
        f.write(diff)


def append_to_experiment_log(
    project_dir: str, group_dir: str, run_dir: str, cfg: JobConfig
) -> None:
    # Ensure parent directories exist so appending does not fail
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(group_dir, exist_ok=True)
    run_id = os.path.basename(run_dir)
    output = f"[{_timestamp()}] {cfg.logging.wandb_run_name} / {run_id} ({cfg_hash(cfg)})\n"
    output += f"# {cfg.logging.wandb_notes}\n"
    output += json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")) + "\n"
    with open(os.path.join(project_dir, "experiment_log.txt"), "a") as f:
        f.write(output)
    with open(os.path.join(group_dir, "experiment_log.txt"), "a") as f:
        f.write(output)
