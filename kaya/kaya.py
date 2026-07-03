"""Python CLI for syncing and running MP-VRDU jobs on Kaya.

All site-specific names live in `kaya/config.json`. The CLI uses SSH, rsync,
and SLURM but avoids shell-side project configuration files.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOCAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


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


def load_config(path: Path = DEFAULT_CONFIG) -> KayaConfig:
    """Load the static Kaya JSON configuration."""

    return KayaConfig(json.loads(path.read_text()), path)


def quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def strip_separator(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def run_local(command: list[str], *, input_text: str | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a local command, raising on failure."""

    return subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=capture,
        check=True,
    )


def ssh_script(config: KayaConfig, script: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a login-shell script on Kaya."""

    remote_command = "bash --login -lc " + quote(script)
    return run_local(["ssh", config.ssh_alias, remote_command], capture=capture)


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


def exports(config: KayaConfig, *, offline: bool) -> list[str]:
    """Return root-relative artifact environment exports generated from config."""

    cache = config.remote_path("cache")
    values = {
        "HF_HOME": cache,
        "HF_DATASETS_CACHE": f"{cache}/datasets",
        "TORCH_HOME": f"{cache}/torch",
        "PIP_CACHE_DIR": f"{cache}/pip",
    }
    if offline:
        values.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    return [f"export {name}={quote(value)}" for name, value in values.items()]


def activate_env_commands(config: KayaConfig) -> list[str]:
    """Return shell commands to activate the configured conda environment."""

    return [
        'source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true',
        "conda deactivate 2>/dev/null || true",
        f"conda activate {quote(config.remote_path('env'))}",
    ]


def remote_prelude(config: KayaConfig, *, activate: bool, offline: bool) -> str:
    """Build the standard remote shell prelude for login or compute commands."""

    lines = [
        "set -euo pipefail",
        f"cd {quote(config.remote_root)}",
        *module_commands(config),
        *exports(config, offline=offline),
    ]
    if activate:
        lines.extend(activate_env_commands(config))
    return "\n".join(lines)


def shell_join(command: list[str]) -> str:
    """Quote a command list for remote shell execution."""

    return " ".join(quote(part) for part in command)


def run_login(config: KayaConfig, command: list[str], *, activate: bool = True, offline: bool = False) -> None:
    """Run code directly on Kaya's login node."""

    if not command:
        raise SystemExit("run-login requires a command after --")
    script = "\n".join([remote_prelude(config, activate=activate, offline=offline), shell_join(command)])
    ssh_script(config, script)


def setup_env(config: KayaConfig, *, do_push: bool = True) -> None:
    """Build or update the conda environment on Kaya's login node."""

    if do_push:
        push(config)
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {quote(config.remote_root)}",
            *module_commands(config),
            *exports(config, offline=False),
            f"mkdir -p {quote(Path(config.remote_path('env')).parent)} {quote(config.remote_path('cache'))}",
            f"if [[ ! -d {quote(config.remote_path('env') + '/conda-meta')} ]]; then conda create -p {quote(config.remote_path('env'))} python={quote(config.raw['python_version'])} -y; fi",
            *activate_env_commands(config),
            "python -m pip install --upgrade pip wheel setuptools",
            f"python -m pip install --extra-index-url {quote(config.raw['pip']['torch_index_url'])} -r {quote(config.remote_root + '/requirements.txt')}",
            "python -m pip check",
            "python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)\"",
        ]
    )
    ssh_script(config, script)


def prestage(config: KayaConfig, *, model_ids: list[str], stage_dataset: bool, do_push: bool = True) -> None:
    """Download model snapshots and stage MMLongBench on Kaya's login node."""

    if do_push:
        push(config)
    hf = dict(config.raw.get("hf", {}))
    hf_args = ["--max-workers", str(hf.get("max_workers", 1))]
    if not hf.get("disable_xet", True):
        hf_args.append("--enable-xet")
    commands: list[str] = []
    for model_id in model_ids:
        commands.append(
            shell_join(
                [
                    "python",
                    "kaya/download_hf.py",
                    "--model",
                    model_id,
                    "--cache-dir",
                    config.remote_path("cache"),
                    *hf_args,
                ]
            )
        )
    if stage_dataset:
        commands.append(
            shell_join(
                [
                    "python",
                    "kaya/download_hf.py",
                    "--dataset",
                    config.raw["datasets"]["mmlongbench"],
                    "--cache-dir",
                    config.remote_path("cache"),
                    "--data-dir",
                    config.remote_path("data"),
                    "--stage-mmlongbench",
                    *hf_args,
                ]
            )
        )
    script = "\n".join(
        [
            remote_prelude(config, activate=True, offline=False),
            f"mkdir -p {quote(config.remote_path('cache'))} {quote(config.remote_path('data'))}",
            *commands,
            f"du -sh {quote(config.remote_path('cache'))} {quote(config.remote_path('data'))} 2>/dev/null || true",
        ]
    )
    ssh_script(config, script)


def write_remote_file(config: KayaConfig, remote_path: str, content: str) -> None:
    """Write a small text file to Kaya via SSH stdin."""

    command = f"mkdir -p {quote(str(Path(remote_path).parent))} && cat > {quote(remote_path)}"
    run_local(["ssh", config.ssh_alias, command], input_text=content)


def slurm_script(config: KayaConfig, args: argparse.Namespace, command: list[str]) -> str:
    """Build an sbatch script for a Python-managed compute job."""

    slurm = config.slurm
    partition = args.partition or slurm["partition"]
    gres = args.gres or slurm["gres"]
    cpus = args.cpus_per_task or slurm["cpus_per_task"]
    mem = args.mem or slurm["mem"]
    time_limit = args.time or slurm["time"]
    job_name = args.job_name
    logs_dir = config.remote_path("logs")
    account = args.account if getattr(args, "account", None) is not None else slurm.get("account")
    qos = args.qos if getattr(args, "qos", None) is not None else slurm.get("qos")
    optional_directives = []
    if account:
        optional_directives.append(f"#SBATCH --account={account}")
    if qos:
        optional_directives.append(f"#SBATCH --qos={qos}")
    return "\n".join(
        [
            "#!/bin/bash --login",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --gres={gres}",
            *optional_directives,
            f"#SBATCH --nodes={slurm['nodes']}",
            f"#SBATCH --ntasks={slurm['ntasks']}",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={logs_dir}/%x_%j.out",
            f"#SBATCH --error={logs_dir}/%x_%j.err",
            "",
            remote_prelude(config, activate=True, offline=not args.no_offline),
            "echo \"[job] host=$(hostname) pwd=$(pwd)\"",
            "echo \"[job] command: " + shell_join(command).replace('"', '\\"') + "\"",
            shell_join(command),
            "",
        ]
    )


def wait_for_job(config: KayaConfig, job_id: str) -> None:
    """Block until a SLURM job leaves the queue and print accounting output."""

    poll = int(config.slurm.get("poll_seconds", 10))
    print(f"Waiting for job {job_id} to leave the queue...")
    while True:
        result = subprocess.run(
            ["ssh", config.ssh_alias, f"squeue -h -j {quote(job_id)}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if not result.stdout.strip():
            break
        time.sleep(poll)
    subprocess.run(
        [
            "ssh",
            config.ssh_alias,
            f"sacct -j {quote(job_id)} --format=JobID,JobName,State,Elapsed,ExitCode --noheader",
        ],
        text=True,
        check=False,
    )


def print_log_tails(config: KayaConfig, job_name: str, job_id: str, lines: int) -> None:
    """Print local copies of a job's stdout/stderr tails."""

    simple_job_id = job_id.split(";", 1)[0]
    logs_dir = LOCAL_ROOT / config.raw["paths"]["logs"]
    for suffix in ("out", "err"):
        path = logs_dir / f"{job_name}_{simple_job_id}.{suffix}"
        if not path.exists():
            continue
        print(f"\n===== {path} =====")
        content = path.read_text(errors="replace").splitlines()
        for line in content[-lines:]:
            print(line)


def run_gpu(config: KayaConfig, args: argparse.Namespace, command: list[str]) -> str:
    """Submit a compute-node job, optionally wait, pull logs, and print tails."""

    if not command:
        raise SystemExit("run-gpu requires a command after --")
    if not args.no_push:
        push(config)
    timestamp = int(time.time())
    script_remote = f"{config.remote_path('logs')}/generated_{args.job_name}_{timestamp}.sbatch"
    write_remote_file(config, script_remote, slurm_script(config, args, command))
    submit = ssh_script(
        config,
        f"cd {quote(config.remote_root)} && sbatch --parsable {quote(script_remote)}",
        capture=True,
    )
    job_id = submit.stdout.strip()
    print(f"Submitted job {job_id}")
    (LOCAL_ROOT / ".kaya_last_job").write_text(job_id + "\n")
    if not args.no_wait:
        wait_for_job(config, job_id)
        if not args.no_pull:
            pull(config)
            print_log_tails(config, args.job_name, job_id, args.tail_lines)
    return job_id


def run_probe(config: KayaConfig, args: argparse.Namespace) -> None:
    """Run `cli.run_probe` on Kaya login or compute."""

    probe_args = ["python", "-m", "cli.run_probe", args.probe]
    if args.heavy:
        probe_args.append("--run-heavy")
    if args.json:
        probe_args.append("--json")
    for model_id in args.model_id or []:
        probe_args.extend(["--model-id", model_id])
    probe_args.extend(strip_separator(args.extra))

    if args.target == "login":
        if not args.no_push:
            push(config)
        run_login(config, probe_args, activate=True, offline=args.offline)
        return

    gpu_args = argparse.Namespace(
        job_name=args.job_name,
        partition=args.partition,
        gres=args.gres,
        account=args.account,
        qos=args.qos,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time=args.time,
        no_offline=args.no_offline,
        no_push=args.no_push,
        no_wait=args.no_wait,
        no_pull=args.no_pull,
        tail_lines=args.tail_lines,
    )
    run_gpu(config, gpu_args, probe_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("show-config")
    sub.add_parser("push")
    sub.add_parser("pull")

    setup = sub.add_parser("setup-env")
    setup.add_argument("--no-push", action="store_true")

    prestage_parser = sub.add_parser("prestage")
    prestage_parser.add_argument("--no-push", action="store_true")
    prestage_parser.add_argument("--skip-dataset", action="store_true")
    prestage_parser.add_argument("--skip-models", action="store_true")
    prestage_parser.add_argument("--model-id", action="append")

    login = sub.add_parser("run-login")
    login.add_argument("--no-push", action="store_true")
    login.add_argument("--no-activate", action="store_true")
    login.add_argument("--offline", action="store_true")
    login.add_argument("remote_command", nargs=argparse.REMAINDER)

    gpu = sub.add_parser("run-gpu")
    add_gpu_args(gpu, default_job_name="mpvrdu_job")
    gpu.add_argument("remote_command", nargs=argparse.REMAINDER)

    probe = sub.add_parser("run-probe")
    probe.add_argument("probe", choices=["list", "local", "all", "loader", "scanned", "boxes", "model-family", "retrieval", "unanswerable", "doc-type"])
    probe.add_argument("--target", choices=["login", "gpu"], default="gpu")
    probe.add_argument("--heavy", action="store_true")
    probe.add_argument("--json", action="store_true")
    probe.add_argument("--model-id", action="append")
    probe.add_argument("--offline", action="store_true", help="use HF offline mode on login-node probe")
    add_gpu_args(probe, default_job_name="mpvrdu_probe")
    probe.add_argument("extra", nargs=argparse.REMAINDER)

    gpu_test = sub.add_parser("gpu-test")
    add_gpu_args(gpu_test, default_job_name="mpvrdu_gpu_test")

    return parser


def add_gpu_args(parser: argparse.ArgumentParser, *, default_job_name: str) -> None:
    """Add common compute-node options to a subparser."""

    parser.add_argument("--job-name", default=default_job_name)
    parser.add_argument("--partition")
    parser.add_argument("--gres")
    parser.add_argument("--account")
    parser.add_argument("--qos")
    parser.add_argument("--cpus-per-task", type=int)
    parser.add_argument("--mem")
    parser.add_argument("--time")
    parser.add_argument("--no-offline", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--tail-lines", type=int, default=120)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "show-config":
        print(json.dumps(config.raw, indent=2, sort_keys=True))
    elif args.command == "push":
        push(config)
    elif args.command == "pull":
        pull(config)
    elif args.command == "setup-env":
        setup_env(config, do_push=not args.no_push)
    elif args.command == "prestage":
        model_ids = [] if args.skip_models else (args.model_id or list(config.raw["models"]))
        prestage(config, model_ids=model_ids, stage_dataset=not args.skip_dataset, do_push=not args.no_push)
    elif args.command == "run-login":
        if not args.no_push:
            push(config)
        run_login(
            config,
            strip_separator(args.remote_command),
            activate=not args.no_activate,
            offline=args.offline,
        )
    elif args.command == "run-gpu":
        run_gpu(config, args, strip_separator(args.remote_command))
    elif args.command == "run-probe":
        run_probe(config, args)
    elif args.command == "gpu-test":
        run_gpu(
            config,
            args,
            [
                "python",
                "-c",
                "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')",
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
