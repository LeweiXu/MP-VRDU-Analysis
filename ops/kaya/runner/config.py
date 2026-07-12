"""Kaya config dataclasses, shared constants, and small quoting helpers."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SSH_KEEPALIVE_OPTS = ["-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]

LOCAL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.json"
HEADER_PREFIX = "# kaya:"
BOOL_TRUE = {"1", "true", "yes", "y", "on"}
BOOL_FALSE = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class KayaConfig:
    """Resolved Kaya configuration from `ops/kaya/config.json`."""

    raw: dict[str, Any]
    path: Path

    @property
    def ssh_alias(self) -> str:
        return str(self.raw["ssh_alias"])

    @property
    def remote_root(self) -> str:
        return str(self.raw["remote_root"]).rstrip("/")

    def remote_path(self, key: str) -> str:
        return f"{self.remote_root}/{self.raw['paths'][key].strip('/')}"

    @property
    def slurm(self) -> dict[str, Any]:
        return dict(self.raw["slurm"])

    @property
    def rsync_excludes(self) -> list[str]:
        return list(self.raw["rsync_excludes"])


@dataclass(frozen=True)
class RunSettings:
    """Resolved execution hints for a runnable Python file."""

    target: str
    activate_env: bool
    offline: bool
    job_name: str | None = None


@dataclass(frozen=True)
class SqueueJobStatus:
    """One parsed `squeue` row for a submitted SLURM job."""

    job_id: str
    partition: str
    name: str
    user: str
    state: str
    elapsed: str
    time_limit: str
    nodes: str
    reason: str


def load_config(path: Path = DEFAULT_CONFIG) -> KayaConfig:
    """Load the static Kaya JSON configuration."""

    return KayaConfig(json.loads(path.read_text()), path)


def quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def strip_separator(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def shell_join(command: list[str]) -> str:
    """Quote a command list for remote shell execution."""

    return " ".join(quote(part) for part in command)


def parse_bool(value: str, *, field: str) -> bool:
    """Parse a Kaya header boolean."""

    normalized = value.strip().lower()
    if normalized in BOOL_TRUE:
        return True
    if normalized in BOOL_FALSE:
        return False
    raise ValueError(f"invalid boolean for {field}: {value!r}")
