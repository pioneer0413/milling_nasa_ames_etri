from __future__ import annotations

import os
import platform
import subprocess
import sys


def collect_environment(root: str = ".") -> dict:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    for package in ["numpy", "pandas", "sklearn", "torch"]:
        try:
            module = __import__(package)
            env[package] = getattr(module, "__version__", "unknown")
        except Exception:
            env[package] = None
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = torch.version.cuda
        env["gpu_count"] = int(torch.cuda.device_count())
        env["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    env["git"] = git_state(root)
    return env


def git_state(root: str = ".") -> dict:
    def run(args):
        return subprocess.run(args, cwd=root, capture_output=True, text=True, check=False).stdout.strip()

    commit = run(["git", "rev-parse", "HEAD"])
    dirty = bool(run(["git", "status", "--porcelain"]))
    return {"commit_hash": commit or None, "dirty": dirty}
