"""Build sbatch directives and submit jobs: config+override SLURM options, a
generated wrapper for a repo-local Python file, and submitting an existing sbatch."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import LOCAL_ROOT, KayaConfig, RunSettings, quote, shell_join
from .remote import remote_prelude, ssh_script
from .sources import python_command, spec_job_name


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

    job_name = (args.job_name or settings.job_name or spec_job_name(script_args)
                or local_path.stem.replace("_", "-"))
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
