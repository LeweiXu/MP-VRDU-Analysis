"""Sync the repo to Kaya and run login-node or SLURM jobs.

Purpose:
    Owns common Kaya mechanics only: source sync, remote directory setup,
    module/env activation, secret forwarding, login-node execution, generated
    sbatch submission, job watching, and pulling logs/results. Task-specific work
    remains in separate runnable files under `kaya/`.

Pipeline role:
    Provides the local control plane for all Kaya operations. The core pipeline
    never hard-codes cluster paths; this runner reads `kaya/config.json` and
    exports root-relative artifact paths on the remote side.

CLI:
    `python -m kaya.kaya [--config PATH] COMMAND [command-options]`

Arguments:
    --config PATH: alternate Kaya JSON config (default: `kaya/config.json`).
    COMMAND: one of `show-config`, `push`, `pull`, `run`, `submit`, `watch`,
        `cancel`.

    Runner options (`--gres`, `--time`, `--no-wait`, …) may appear before or
    after PROGRAM; everything after the first `--` is forwarded to PROGRAM.

    `run` options: --target {auto,login,gpu}, --env/--no-env,
    --offline/--online, SLURM overrides for generated GPU Python jobs,
    --no-push, --no-wait, --no-pull, --tail-lines, PROGRAM, and forwarded
    PROGRAM arguments after `--`.

    `submit` options: --env/--no-env, --offline/--online, explicit SLURM
    overrides, --no-push, --no-wait, --no-pull, --tail-lines, PROGRAM
    (`.py` or `.sbatch`), and forwarded PROGRAM arguments after `--`.

    `watch` options: optional JOB_ID, --job-name, --no-pull, --tail-lines.

    `cancel` options: JOB_ID... (explicit ids), --all (all your jobs),
    --job-name NAME, --state STATE (restrict --all/--job-name).
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SSH_KEEPALIVE_OPTS = ["-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]


LOCAL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
HEADER_PREFIX = "# kaya:"
BOOL_TRUE = {"1", "true", "yes", "y", "on"}
BOOL_FALSE = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class KayaConfig:
    """Resolved Kaya configuration from `kaya/config.json`."""

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


def run_local(
    command: list[str],
    *,
    input_text: str | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a local command, raising on failure."""

    return subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=capture,
        check=True,
    )


def guard_process_group(script: str) -> str:
    """Prepend a trap that kills the whole remote process group on interrupt.

    A bare `bash --login -s` child is not a session leader, so OpenSSH has
    nothing to hang up when the client disconnects and orphaned grandchildren
    (e.g. a Python download) keep running on the login node indefinitely.
    This trap is a backstop: if the shell itself is interrupted or hung up
    while a foreground child is still running, it kills its own process
    group (which any synchronous foreground child shares). It deliberately
    does NOT trap EXIT: doing so would fire this same kill on a normal,
    successful exit too, sending the shell itself a self-inflicted SIGTERM
    right as it's finishing cleanly, which turns a successful run into an
    apparent SSH failure (exit 255) instead of a clean 0.
    """

    return "\n".join(["trap 'kill -- -$$ 2>/dev/null || true' HUP TERM INT", script])


def ssh_script(
    config: KayaConfig,
    script: str,
    *,
    capture: bool = False,
    interruptible: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a login-shell script on Kaya.

    By default the script is piped over stdin to a plain (non-pty) shell,
    which is fine for short, deterministic commands whose stdout may need
    clean parsing (e.g. an sbatch job id). Set `interruptible=True` for
    longer-running foreground work (see `handle_run`): the script is staged
    as a remote file and executed under a real pty (`ssh -tt`) plus SSH
    keepalives, so a local Ctrl-C (or a dead connection) reliably tears down
    the whole remote process tree instead of leaving it orphaned. A pty is
    required for this because OpenSSH only sends a hangup to the process
    group of a pty-attached session, not a plain piped exec; running the
    script from a file (rather than piping it over stdin) avoids the pty
    echoing the script text back into captured output.
    """

    wrapped = guard_process_group(script)
    if not interruptible:
        return run_local(["ssh", config.ssh_alias, "bash", "--login", "-s"], input_text=wrapped, capture=capture)

    remote_path = f"{config.remote_path('logs')}/run_{int(time.time())}_{id(script) % 100000}.sh"
    write_remote_file(config, remote_path, wrapped)
    remote_command = f"bash --login {quote(remote_path)}; rc=$?; rm -f {quote(remote_path)}; exit $rc"
    command = ["ssh", "-tt", *SSH_KEEPALIVE_OPTS, config.ssh_alias, remote_command]
    return run_local(command, capture=capture)


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from a local .env file."""

    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def secret_exports(config: KayaConfig, *, include: bool) -> list[str]:
    """Return token exports from the configured local .env file."""

    if not include:
        return []
    secrets = dict(config.raw.get("secrets", {}))
    env_file = LOCAL_ROOT / secrets.get("env_file", ".env")
    names = list(secrets.get("forward", ["HF_TOKEN"]))
    dotenv = load_dotenv(env_file)
    exports: list[str] = []
    for name in names:
        value = dotenv.get(name)
        if value:
            exports.append(f"export {name}={quote(value)}")
    return exports


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


def module_commands(config: KayaConfig) -> list[str]:
    """Return shell commands to load configured Kaya modules."""

    commands: list[str] = []
    for module in config.raw.get("modules", []):
        name = quote(module["name"])
        if module.get("optional"):
            commands.append(f"module load {name} 2>/dev/null || true")
        else:
            commands.append(f"module load {name}")
    commands.append("module list 2>&1 | sed 's/^/[modules] /'")
    return commands


def artifact_exports(config: KayaConfig, *, offline: bool) -> list[str]:
    """Return root-relative artifact environment exports generated from config."""

    cache = config.remote_path("cache")
    values = {
        "PYTHONPATH": config.remote_root,
        # HF weights/datasets
        "HF_HOME": cache,
        "HF_HUB_CACHE": cache,
        "HF_DATASETS_CACHE": f"{cache}/datasets",
        "HF_XET_CACHE": f"{cache}/xet",
        "TORCH_HOME": f"{cache}/torch",
        # package caches: keep conda pkgs and pip wheels in-project, not in $HOME
        "CONDA_PKGS_DIRS": f"{cache}/conda-pkgs",
        "PIP_CACHE_DIR": f"{cache}/pip",
        "MODEL_CACHE_DIR": f"{cache}/datalab/models",
        # PaddleOCR-VL: pull from HF (contained) rather than the Paddle model hub
        "PADDLE_HOME": f"{cache}/paddle",
        "PADDLEOCR_HOME": f"{cache}/paddleocr",
        "PADDLE_PDX_CACHE_HOME": f"{cache}/paddlex",
        "PADDLE_PDX_MODEL_SOURCE": "huggingface",
        # MinerU aux models: pull from HF and cache ModelScope in-project
        "MINERU_MODEL_SOURCE": "huggingface",
        "MODELSCOPE_CACHE": f"{cache}/modelscope",
        # torch compile / triton kernel caches (else they write to $HOME)
        "TRITON_CACHE_DIR": f"{cache}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache}/inductor",
        "MPLCONFIGDIR": f"{cache}/matplotlib",
        # catch-all: redirect any XDG-cache-respecting tool into the project
        "XDG_CACHE_HOME": f"{cache}/xdg",
    }
    if offline:
        values.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    exports = [f"export {name}={quote(value)}" for name, value in values.items()]
    return ["unset TRANSFORMERS_CACHE", *exports]


def activate_env_commands(config: KayaConfig) -> list[str]:
    """Return shell commands to activate the configured conda environment."""

    return [
        'source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true',
        "conda deactivate 2>/dev/null || true",
        f"conda activate {quote(config.remote_path('env'))}",
    ]


def remote_prelude(
    config: KayaConfig,
    *,
    activate: bool,
    offline: bool,
    include_secrets: bool = False,
) -> str:
    """Build the standard remote shell prelude for login or compute commands."""

    lines = [
        "set -euo pipefail",
        f"cd {quote(config.remote_root)}",
        *module_commands(config),
        *artifact_exports(config, offline=offline),
        *secret_exports(config, include=include_secrets and not offline),
    ]
    if activate:
        lines.extend(activate_env_commands(config))
    return "\n".join(lines)


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


def parse_kaya_header(path: Path, *, max_lines: int = 40) -> dict[str, str]:
    """Parse `# kaya: key=value` hints from the top of a Python script."""

    hints: dict[str, str] = {}
    if path.suffix != ".py" or not path.is_file():
        return hints
    for index, line in enumerate(path.read_text(errors="replace").splitlines()):
        if index >= max_lines:
            break
        stripped = line.strip()
        if not stripped.startswith(HEADER_PREFIX):
            continue
        payload = stripped[len(HEADER_PREFIX) :].strip()
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)
        hints[key.strip().lower().replace("_", "-")] = value.strip()
    return hints


def resolve_run_settings(
    path: Path | None,
    *,
    target_override: str = "auto",
    activate_override: bool | None = None,
    offline_override: bool | None = None,
    default_target: str = "login",
) -> RunSettings:
    """Resolve execution settings from Python headers and CLI overrides."""

    header = parse_kaya_header(path) if path else {}
    target = header.get("target", default_target)
    if target_override != "auto":
        target = target_override
    if target not in {"login", "gpu"}:
        raise ValueError(f"invalid target {target!r}; expected login or gpu")

    activate = True
    if "env" in header:
        activate = parse_bool(header["env"], field="env")
    if activate_override is not None:
        activate = activate_override

    offline = target == "gpu"
    if "offline" in header:
        offline = parse_bool(header["offline"], field="offline")
    if offline_override is not None:
        offline = offline_override

    return RunSettings(target=target, activate_env=activate, offline=offline, job_name=header.get("job-name"))


def local_source_path(value: str) -> Path | None:
    """Return a repo-local source path if `value` names one."""

    path = Path(value)
    if not path.is_absolute():
        path = LOCAL_ROOT / path
    try:
        resolved = path.resolve()
        resolved.relative_to(LOCAL_ROOT)
    except (FileNotFoundError, ValueError):
        return None
    return resolved if resolved.exists() else None


def remote_source_path(config: KayaConfig, local_path: Path) -> str:
    """Map a local repo source file to its path inside the Kaya mirror."""

    rel = local_path.resolve().relative_to(LOCAL_ROOT).as_posix()
    return f"{config.remote_root}/{rel}"


def python_module_name(local_path: Path) -> str | None:
    """Return a repo-local Python module name when the path is importable."""

    if local_path.suffix != ".py":
        return None
    rel = local_path.resolve().relative_to(LOCAL_ROOT).with_suffix("")
    parts = rel.parts
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def python_command(config: KayaConfig, local_path: Path, args: list[str]) -> list[str]:
    """Build a remote Python command for a repo-local file."""

    module_name = python_module_name(local_path)
    if module_name:
        return ["python", "-m", module_name, *args]
    return ["python", remote_source_path(config, local_path), *args]


def write_remote_file(config: KayaConfig, remote_path: str, content: str) -> None:
    """Write a small text file to Kaya via SSH stdin."""

    command = f"mkdir -p {quote(str(Path(remote_path).parent))} && cat > {quote(remote_path)}"
    run_local(["ssh", config.ssh_alias, command], input_text=content)


def slurm_options(config: KayaConfig, args: argparse.Namespace, *, job_name: str) -> list[str]:
    """Return sbatch CLI options from config plus explicit overrides."""

    slurm = config.slurm
    options = [
        f"--job-name={job_name}",
        f"--partition={args.partition or slurm['partition']}",
        f"--gres={args.gres or slurm['gres']}",
        f"--nodes={slurm['nodes']}",
        f"--ntasks={slurm['ntasks']}",
        f"--cpus-per-task={args.cpus_per_task or slurm['cpus_per_task']}",
        f"--mem={args.mem or slurm['mem']}",
        f"--time={args.time or slurm['time']}",
    ]
    account = args.account if args.account is not None else slurm.get("account")
    qos = args.qos if args.qos is not None else slurm.get("qos")
    if account:
        options.append(f"--account={account}")
    if qos:
        options.append(f"--qos={qos}")
    return options


def explicit_sbatch_options(args: argparse.Namespace) -> list[str]:
    """Return only SLURM options explicitly supplied for an existing sbatch file."""

    mapping = [
        ("job_name", "job-name"),
        ("partition", "partition"),
        ("gres", "gres"),
        ("account", "account"),
        ("qos", "qos"),
        ("cpus_per_task", "cpus-per-task"),
        ("mem", "mem"),
        ("time", "time"),
    ]
    options: list[str] = []
    for attr, option in mapping:
        value = getattr(args, attr)
        if value is not None:
            options.append(f"--{option}={value}")
    return options


def generated_python_sbatch(
    config: KayaConfig,
    args: argparse.Namespace,
    local_path: Path,
    script_args: list[str],
    settings: RunSettings,
) -> tuple[str, str]:
    """Build a generated sbatch wrapper for a Python source file."""

    job_name = args.job_name or settings.job_name or local_path.stem.replace("_", "-")
    logs_dir = config.remote_path("logs")
    command = python_command(config, local_path, script_args)
    directives = slurm_options(config, args, job_name=job_name)
    content = "\n".join(
        [
            "#!/bin/bash --login",
            *[f"#SBATCH {option}" for option in directives],
            f"#SBATCH --output={logs_dir}/%x_%j.out",
            f"#SBATCH --error={logs_dir}/%x_%j.err",
            "",
            remote_prelude(config, activate=settings.activate_env, offline=settings.offline),
            "echo \"[job] host=$(hostname) pwd=$(pwd)\"",
            "echo \"[job] command: " + shell_join(command).replace('"', '\\"') + "\"",
            shell_join(command),
            "",
        ]
    )
    return job_name, content


def submit_remote_sbatch(
    config: KayaConfig,
    remote_sbatch: str,
    sbatch_args: list[str],
    program_args: list[str],
) -> str:
    """Submit an existing remote sbatch file and return the job id."""

    command = ["sbatch", "--parsable", *sbatch_args, remote_sbatch, *program_args]
    result = ssh_script(config, f"cd {quote(config.remote_root)}\n{shell_join(command)}", capture=True)
    job_id = result.stdout.strip()
    (LOCAL_ROOT / ".kaya_last_job").write_text(job_id + "\n")
    print(f"Submitted job {job_id}")
    return job_id


def parse_squeue_row(line: str) -> SqueueJobStatus | None:
    """Parse one pipe-delimited `squeue` status row."""

    parts = line.rstrip("\n").split("|", 8)
    if len(parts) != 9:
        return None
    return SqueueJobStatus(*parts)


def squeue_statuses(config: KayaConfig, job_id: str) -> list[SqueueJobStatus]:
    """Return current queue rows for a job id."""

    simple_job_id = job_id.split(";", 1)[0]
    fmt = "%i|%P|%j|%u|%t|%M|%l|%D|%R"
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=10",
            config.ssh_alias,
            f"squeue -h -j {quote(simple_job_id)} -o {quote(fmt)}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"squeue exited {result.returncode}"
        raise RuntimeError(message)
    statuses: list[SqueueJobStatus] = []
    for line in result.stdout.splitlines():
        status = parse_squeue_row(line)
        if status is not None:
            statuses.append(status)
    return statuses


def format_wait_status(status: SqueueJobStatus) -> str:
    """Return one human-readable queue status line."""

    state_names = {
        "PD": "pending",
        "R": "running",
        "CG": "completing",
        "CF": "configuring",
        "S": "suspended",
    }
    state = state_names.get(status.state, status.state)
    if status.state == "PD":
        return (
            f"Job {status.job_id} pending for {status.elapsed}/{status.time_limit} "
            f"on partition {status.partition}: {status.reason}"
        )
    if status.state in {"R", "CG"}:
        return (
            f"Job {status.job_id} {state} for {status.elapsed}/{status.time_limit} "
            f"on {status.reason} ({status.nodes} node(s))"
        )
    return (
        f"Job {status.job_id} {state} elapsed={status.elapsed}/{status.time_limit} "
        f"partition={status.partition} reason={status.reason}"
    )


def wait_for_job(config: KayaConfig, job_id: str) -> None:
    """Block until a SLURM job leaves the queue and print accounting output."""

    poll = int(config.slurm.get("poll_seconds", 10))
    simple_job_id = job_id.split(";", 1)[0]
    start = time.monotonic()
    print(f"Waiting for job {simple_job_id} to leave the queue (poll every {poll}s)...")
    while True:
        try:
            statuses = squeue_statuses(config, simple_job_id)
        except RuntimeError as exc:
            waited_s = int(time.monotonic() - start)
            print(f"[wait {waited_s}s] squeue failed: {exc}; retrying in {poll}s")
            time.sleep(poll)
            continue
        waited_s = int(time.monotonic() - start)
        if not statuses:
            print(f"Job {simple_job_id} left the queue after {waited_s}s.")
            break
        primary = statuses[0]
        print(f"[wait {waited_s}s] {format_wait_status(primary)}", flush=True)
        time.sleep(poll)
    subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=10",
            config.ssh_alias,
            f"sacct -j {quote(simple_job_id)} --format=JobID,JobName,State,Elapsed,ExitCode --noheader",
        ],
        text=True,
        check=False,
    )


def last_job_id() -> str:
    """Return the locally remembered Kaya job id."""

    path = LOCAL_ROOT / ".kaya_last_job"
    if not path.is_file():
        raise SystemExit("no job id supplied and .kaya_last_job does not exist")
    return path.read_text().strip()


def print_log_tails(config: KayaConfig, job_id: str, lines: int, *, job_name: str | None = None) -> None:
    """Print local stdout/stderr tails for logs matching a SLURM job id."""

    simple_job_id = job_id.split(";", 1)[0]
    logs_dir = LOCAL_ROOT / config.raw["paths"]["logs"]
    candidates: list[Path] = []
    if job_name:
        candidates.extend([logs_dir / f"{job_name}_{simple_job_id}.out", logs_dir / f"{job_name}_{simple_job_id}.err"])
    candidates.extend(sorted(logs_dir.glob(f"*_{simple_job_id}.out")))
    candidates.extend(sorted(logs_dir.glob(f"*_{simple_job_id}.err")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        print(f"\n===== {path} =====")
        content = path.read_text(errors="replace").splitlines()
        for line in content[-lines:]:
            print(line)


def handle_watch(config: KayaConfig, args: argparse.Namespace) -> None:
    """Wait for a SLURM job, pull artifacts, and print log tails."""

    job_id = args.job_id or last_job_id()
    wait_for_job(config, job_id)
    if not args.no_pull:
        pull(config)
    print_log_tails(config, job_id, args.tail_lines, job_name=args.job_name)


def user_jobs(config: KayaConfig) -> list[SqueueJobStatus]:
    """Return the current user's queued/running jobs (empty if none)."""

    fmt = "%i|%P|%j|%u|%t|%M|%l|%D|%R"
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=10",
            config.ssh_alias,
            f'squeue -h -u "$(whoami)" -o {quote(fmt)}',
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"squeue exited {result.returncode}"
        raise RuntimeError(message)
    statuses: list[SqueueJobStatus] = []
    for line in result.stdout.splitlines():
        status = parse_squeue_row(line)
        if status is not None:
            statuses.append(status)
    return statuses


def print_user_jobs(config: KayaConfig, label: str) -> int:
    """Print the user's current jobs and return the count."""

    try:
        jobs = user_jobs(config)
    except RuntimeError as exc:
        print(f"[cancel] could not list jobs ({label}): {exc}")
        return -1
    print(f"[cancel] {len(jobs)} job(s) {label}:")
    for status in jobs:
        print("  " + format_wait_status(status))
    return len(jobs)


def handle_kill(config: KayaConfig, args: argparse.Namespace) -> None:
    """Cancel a single SLURM job by id (defaults to the last submitted job)."""

    job_id = args.job_id or last_job_id()
    simple_job_id = job_id.split(";", 1)[0]
    print(f"[kill] scancel {simple_job_id} on {config.ssh_alias}")
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", config.ssh_alias, f"scancel {quote(simple_job_id)}"],
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"scancel exited {result.returncode}")
    print(f"[kill] cancelled {simple_job_id}")


def handle_cancel(config: KayaConfig, args: argparse.Namespace) -> None:
    """Cancel SLURM jobs: explicit id(s), by --job-name, or --all (this user)."""

    if not (args.all or args.job_name or args.job_id):
        raise SystemExit("cancel needs one or more job ids, --job-name NAME, or --all")

    print_user_jobs(config, "before cancel")

    scope = f" --state={quote(args.state)}" if args.state else ""
    commands: list[str] = []
    if args.job_id:
        commands.append("scancel " + " ".join(quote(job_id) for job_id in args.job_id))
    if args.all:
        commands.append(f'scancel -u "$(whoami)"{scope}')
    if args.job_name:
        commands.append(f'scancel -u "$(whoami)" --name={quote(args.job_name)}{scope}')

    remote = " && ".join(commands)
    print(f"[cancel] running on {config.ssh_alias}: {remote}")
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", config.ssh_alias, remote],
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"scancel exited {result.returncode}")
    print_user_jobs(config, "remaining")


def _clear_cache_targets(config: KayaConfig, args: argparse.Namespace) -> tuple[list[str], str | None]:
    """Return (paths to rm -rf, logs dir to empty or None), all relative to root.

    Scopes the removal to the generation caches and built tables so a stray
    argument can never delete the whole remote root. `renders`/`marker` (the
    expensive reproducible parse caches) are kept unless --renders/--all, so a
    fresh run does not re-parse every page.
    """

    results_rel = str(config.raw["paths"]["results"]).strip("/")
    logs_rel = str(config.raw["paths"]["logs"]).strip("/")
    cache_rel = f"{results_rel}/cache"
    tables_rel = f"{results_rel}/tables"

    rm_targets: list[str] = []
    clear_logs: str | None = None
    if args.all:
        rm_targets = [cache_rel, tables_rel]
        clear_logs = logs_rel
    else:
        modes = [args.mode] if args.mode else ["full", "smoke"]
        for mode in modes:
            if args.experiment:
                rm_targets.append(f"{cache_rel}/{mode}/{args.experiment}")
            else:
                rm_targets.append(f"{cache_rel}/{mode}")
            rm_targets.append(f"{tables_rel}/{mode}")
        if args.renders:
            rm_targets.append(f"{cache_rel}/renders")
            rm_targets.append(f"{cache_rel}/marker")
        if args.logs:
            clear_logs = logs_rel

    # Safety: every target must stay inside results/ or logs/, no traversal.
    allowed = (results_rel, logs_rel)
    for rel in [*rm_targets, *( [clear_logs] if clear_logs else [] )]:
        if ".." in rel.split("/") or not rel.startswith(allowed):
            raise SystemExit(f"refusing to clear unexpected path: {rel!r}")
    return rm_targets, clear_logs


def handle_clear_cache(config: KayaConfig, args: argparse.Namespace) -> None:
    """Remove cached generation results (and optionally logs) on Kaya.

    Clears the per-experiment prediction/result caches and built tables so the
    next run starts fresh. Remote by default; --local mirrors the same removals
    in the local repo. Destructive, so it prints the targets and needs --yes
    (or a 'y' confirmation) unless --dry-run.
    """

    rm_targets, clear_logs = _clear_cache_targets(config, args)
    scope = "remote+local" if args.local else "remote"

    print(f"[clear-cache] {scope} under {config.remote_root} :")
    for rel in rm_targets:
        print(f"  rm -rf   {rel}")
    if clear_logs:
        print(f"  empty    {clear_logs}/ (keep the dir; sbatch needs it)")
    if args.local:
        print(f"  (and the same paths under {LOCAL_ROOT})")

    if args.dry_run:
        print("[clear-cache] dry run, nothing removed")
        return
    if not args.yes:
        reply = input("[clear-cache] proceed? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("[clear-cache] aborted")
            return

    lines = ["set -u"]
    for rel in rm_targets:
        lines.append(f"rm -rf {quote(config.remote_root + '/' + rel)}")
    if clear_logs:
        logs_abs = quote(config.remote_root + "/" + clear_logs)
        lines.append(f"mkdir -p {logs_abs}")
        lines.append(f"find {logs_abs} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} +")
    lines.append("echo '[clear-cache] remote done'")
    ssh_script(config, "\n".join(lines))

    if args.local:
        for rel in rm_targets:
            shutil.rmtree(LOCAL_ROOT / rel, ignore_errors=True)
        if clear_logs:
            logs_dir = LOCAL_ROOT / clear_logs
            logs_dir.mkdir(parents=True, exist_ok=True)
            for child in logs_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        print("[clear-cache] local done")


def handle_run(config: KayaConfig, args: argparse.Namespace) -> None:
    """Run a Python file or command on login, or dispatch a Python file to GPU."""

    target = local_source_path(args.program)
    forwarded = strip_separator(args.program_args)
    settings = resolve_run_settings(
        target,
        target_override=args.target,
        activate_override=args.activate_env,
        offline_override=args.offline,
        default_target="login",
    )
    if not args.no_push:
        push(config)
    if settings.target == "gpu":
        if target is None or target.suffix != ".py":
            raise SystemExit("run --target gpu requires a repo-local .py file")
        submit_python(config, args, target, forwarded, settings)
        return
    command = python_command(config, target, forwarded) if target and target.suffix == ".py" else [args.program, *forwarded]
    script = "\n".join(
        [
            remote_prelude(
                config,
                activate=settings.activate_env,
                offline=settings.offline,
                include_secrets=True,
            ),
            shell_join(command),
        ]
    )
    ssh_script(config, script, interruptible=True)


def submit_python(
    config: KayaConfig,
    args: argparse.Namespace,
    local_path: Path,
    program_args: list[str],
    settings: RunSettings,
) -> str:
    """Generate and submit an sbatch wrapper for a Python source file."""

    job_name, content = generated_python_sbatch(config, args, local_path, program_args, settings)
    script_remote = f"{config.remote_path('logs')}/generated_{job_name}_{int(time.time())}.sbatch"
    write_remote_file(config, script_remote, content)
    job_id = submit_remote_sbatch(config, script_remote, [], [])
    if not args.no_wait:
        wait_for_job(config, job_id)
        if not args.no_pull:
            pull(config)
        print_log_tails(config, job_id, args.tail_lines, job_name=job_name)
    return job_id


def handle_submit(config: KayaConfig, args: argparse.Namespace) -> None:
    """Submit a Python file through a generated wrapper, or an existing sbatch file."""

    local_path = local_source_path(args.program)
    if local_path is None:
        raise SystemExit(f"submit requires a repo-local .py or .sbatch file: {args.program}")
    if local_path.suffix not in {".py", ".sbatch"}:
        raise SystemExit("submit supports only .py and .sbatch files")
    forwarded = strip_separator(args.program_args)
    if not args.no_push:
        push(config)
    if local_path.suffix == ".sbatch":
        job_name = args.job_name or local_path.stem
        job_id = submit_remote_sbatch(
            config,
            remote_source_path(config, local_path),
            explicit_sbatch_options(args),
            forwarded,
        )
        if not args.no_wait:
            wait_for_job(config, job_id)
            if not args.no_pull:
                pull(config)
            print_log_tails(config, job_id, args.tail_lines, job_name=job_name)
        return
    settings = resolve_run_settings(
        local_path,
        target_override="gpu",
        activate_override=args.activate_env,
        offline_override=args.offline,
        default_target="gpu",
    )
    submit_python(config, args, local_path, forwarded, settings)


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    """Add options shared by `run` and Python `submit`."""

    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument("--env", dest="activate_env", action="store_true", default=None, help="activate the configured Kaya conda environment")
    env_group.add_argument("--no-env", dest="activate_env", action="store_false", help="do not activate the configured Kaya conda environment")
    online_group = parser.add_mutually_exclusive_group()
    online_group.add_argument("--offline", dest="offline", action="store_true", default=None, help="force Hugging Face offline mode")
    online_group.add_argument("--online", dest="offline", action="store_false", help="allow online Hugging Face access and forward configured secrets on login jobs")


def add_slurm_args(parser: argparse.ArgumentParser) -> None:
    """Add SLURM options for generated Python jobs and sbatch overrides."""

    parser.add_argument("--job-name", help="SLURM job name for generated Python jobs; explicit override for .sbatch")
    parser.add_argument("--partition", help="SLURM partition for generated Python jobs; explicit override for .sbatch")
    parser.add_argument("--gres", help="SLURM generic resource request for generated Python jobs; explicit override for .sbatch")
    parser.add_argument("--account", help="SLURM account/allocation for generated Python jobs; explicit override for .sbatch")
    parser.add_argument("--qos", help="SLURM QOS for generated Python jobs; explicit override for .sbatch")
    parser.add_argument("--cpus-per-task", type=int, help="CPUs per task for generated Python jobs; explicit override for .sbatch")
    parser.add_argument("--mem", help="memory request, e.g. 64G, for generated Python jobs; explicit override for .sbatch")
    parser.add_argument("--time", help="wall time, e.g. 00:30:00, for generated Python jobs; explicit override for .sbatch")


def add_job_lifecycle_args(parser: argparse.ArgumentParser) -> None:
    """Add common sync/wait/log options."""

    parser.add_argument("--no-push", action="store_true", help="do not rsync the repo before running/submitting")
    parser.add_argument("--no-wait", action="store_true", help="submit and return immediately")
    parser.add_argument("--no-pull", action="store_true", help="do not pull logs/results after the job exits")
    parser.add_argument("--tail-lines", type=int, default=120, help="number of log lines to print after a waited job")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync this repo to Kaya and run login-node or SLURM jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Runner options come before the program path. Everything after the
            program path, usually separated with --, is forwarded to the script.

            Examples:
              python -m kaya.kaya show-config
              python -m kaya.kaya push
              python -m kaya.kaya run scripts/setup_env.py
              python -m kaya.kaya run scripts/prestage.py -- --skip-models
              python -m kaya.kaya submit --time 00:05:00 scripts/gpu_test.py
              python -m kaya.kaya submit kaya/example.sbatch -- --script-arg value
              python -m kaya.kaya watch

            Python files may declare defaults in header comments:
              # kaya: target=login|gpu
              # kaya: env=true|false
              # kaya: offline=true|false
              # kaya: job-name=optional_name
            """
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to Kaya JSON config")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("show-config", help="print the resolved static Kaya config")
    sub.add_parser("push", help="rsync source to Kaya remote_root using configured excludes")
    sub.add_parser("pull", help="pull remote logs/ and results/ back to the local repo")

    run = sub.add_parser(
        "run",
        help="run a Python file or command on the login node, or a header-selected GPU Python file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Everything after -- is forwarded to the Python file or command.",
    )
    run.add_argument("--target", choices=["auto", "login", "gpu"], default="auto", help="execution target; auto reads a Python header and otherwise uses login")
    add_common_run_args(run)
    add_slurm_args(run)
    add_job_lifecycle_args(run)
    run.add_argument("program", help="repo-local .py file to run, or a command name for login-node execution")

    submit = sub.add_parser(
        "submit",
        help="submit a repo-local .py file via a generated sbatch wrapper, or submit an existing .sbatch file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Everything after -- is forwarded to the Python script or sbatch file.\n"
            "For .py files, omitted SLURM options come from kaya/config.json.\n"
            "For .sbatch files, only explicitly supplied SLURM options are passed as overrides."
        ),
    )
    add_common_run_args(submit)
    add_slurm_args(submit)
    add_job_lifecycle_args(submit)
    submit.add_argument("program", help="repo-local .py or .sbatch file")

    watch = sub.add_parser("watch", help="wait for a job id, pull logs/results, and print matching log tails")
    watch.add_argument("job_id", nargs="?", help="SLURM job id; defaults to .kaya_last_job")
    watch.add_argument("--job-name", help="job name used in logs/<job>_<id>.out matching")
    watch.add_argument("--no-pull", action="store_true", help="do not pull logs/results before printing tails")
    watch.add_argument("--tail-lines", type=int, default=120, help="number of log lines to print")

    kill = sub.add_parser("kill", help="cancel a single SLURM job by id (defaults to the last submitted job)")
    kill.add_argument("job_id", nargs="?", help="SLURM job id to cancel; defaults to .kaya_last_job")

    cancel = sub.add_parser("cancel", help="cancel your SLURM jobs: explicit ids, --job-name, or --all")
    cancel.add_argument("job_id", nargs="*", help="specific job id(s) to cancel")
    cancel.add_argument("--all", action="store_true", help="cancel all of your jobs")
    cancel.add_argument("--job-name", help="cancel your jobs with this SLURM job name")
    cancel.add_argument("--state", help="restrict --all/--job-name to a SLURM state, e.g. PENDING or RUNNING")

    clear = sub.add_parser(
        "clear-cache",
        help="remove cached generation results (and optionally logs) on Kaya to start fresh",
    )
    clear.add_argument("--mode", choices=("full", "smoke"), help="restrict to one mode (default: both)")
    clear.add_argument("--experiment", help="restrict to one experiment dir under the mode(s)")
    clear.add_argument("--renders", action="store_true", help="also drop the render/marker parse caches (else kept)")
    clear.add_argument("--logs", action="store_true", help="also empty the logs/ directory (keeps the dir)")
    clear.add_argument("--all", action="store_true", help="drop the whole results/cache + results/tables + logs")
    clear.add_argument("--local", action="store_true", help="mirror the same removals in the local repo")
    clear.add_argument("--dry-run", action="store_true", help="print targets without removing anything")
    clear.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return parser


def split_forwarded_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv on the first standalone `--` into (runner args, forwarded).

    Everything after the first `--` is forwarded verbatim to the program. This
    is done before argparse so runner options (`--gres`, `--time`, `--no-wait`)
    work whether they come before or after the program path, instead of being
    swallowed by a REMAINDER positional when placed after it.
    """

    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :]
    return list(argv), []


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    runner_args, forwarded = split_forwarded_args(raw)
    args = build_parser().parse_args(runner_args)
    args.program_args = forwarded
    config = load_config(args.config)

    if args.command == "show-config":
        print(json.dumps(config.raw, indent=2, sort_keys=True))
    elif args.command == "push":
        push(config)
    elif args.command == "pull":
        pull(config)
    elif args.command == "run":
        handle_run(config, args)
    elif args.command == "submit":
        handle_submit(config, args)
    elif args.command == "watch":
        handle_watch(config, args)
    elif args.command == "kill":
        handle_kill(config, args)
    elif args.command == "cancel":
        handle_cancel(config, args)
    elif args.command == "clear-cache":
        handle_clear_cache(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
