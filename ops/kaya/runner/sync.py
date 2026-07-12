"""Source sync between the local repo and the Kaya mirror (push / pull / mkdir)."""

from __future__ import annotations

import subprocess

from .config import LOCAL_ROOT, KayaConfig, quote
from .remote import run_local, ssh_script


def ensure_remote_dirs(config: KayaConfig) -> None:
    """Create the remote mirror and artifact directories."""

    dirs = [
        config.remote_root,
        config.remote_path("cache"),
        config.remote_path("data"),
        config.remote_path("env"),
        config.remote_path("results"),
        config.remote_path("logs"),
    ]
    ssh_script(config, "mkdir -p " + " ".join(quote(path) for path in dirs))


def push(config: KayaConfig) -> None:
    """Sync source code to Kaya, excluding machine-local artifacts."""

    ensure_remote_dirs(config)
    command = ["rsync", "-az", "--delete"]
    for pattern in config.rsync_excludes:
        command.extend(["--exclude", pattern])
    command.extend([f"{LOCAL_ROOT}/", f"{config.ssh_alias}:{config.remote_root}/"])
    run_local(command)
    ensure_remote_dirs(config)


def pull(config: KayaConfig) -> None:
    """Pull Kaya results and logs back into the local repository."""

    for key in ("results", "logs"):
        local_dir = LOCAL_ROOT / config.raw["paths"][key]
        local_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{config.ssh_alias}:{config.remote_path(key)}/"
        try:
            run_local(["rsync", "-az", remote, f"{local_dir}/"])
        except subprocess.CalledProcessError:
            pass
