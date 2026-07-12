"""Remote shell execution on Kaya: run commands over SSH and build the standard
login/compute prelude (modules, artifact-cache exports, secrets, env activation)."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .config import LOCAL_ROOT, SSH_KEEPALIVE_OPTS, KayaConfig, quote


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


def write_remote_file(config: KayaConfig, remote_path: str, content: str) -> None:
    """Write a small text file to Kaya via SSH stdin."""

    command = f"mkdir -p {quote(str(Path(remote_path).parent))} && cat > {quote(remote_path)}"
    run_local(["ssh", config.ssh_alias, command], input_text=content)


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
