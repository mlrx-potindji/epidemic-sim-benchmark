"""Environment metadata shared by experiment reports."""
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def cpu_model() -> str | None:
    cpu_info = Path("/proc/cpuinfo")
    if not cpu_info.exists():
        return platform.processor() or None
    for line in cpu_info.read_text().splitlines():
        if line.lower().startswith(("model name", "hardware")):
            return line.split(":", 1)[-1].strip()
    return platform.processor() or None


def collect_environment_metadata() -> dict:
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cpus = os.cpu_count()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "available_cpu_count": available_cpus,
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "packages": {
            name: package_version(name)
            for name in ("mesa", "esper", "numpy", "scipy", "networkx", "matplotlib")
        },
    }