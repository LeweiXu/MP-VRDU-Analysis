"""Sync the repo to Kaya and run login-node or SLURM jobs.

Purpose:
    Owns common Kaya mechanics only: source sync, remote directory setup,
    module/env activation, secret forwarding, login-node execution, generated
    sbatch submission, job watching, cluster status, and pulling logs/results.
    Task-specific work remains in separate runnable files under `ops/`.

Pipeline role:
    Provides the local control plane for all Kaya operations. The core pipeline
    never hard-codes cluster paths; this runner reads `ops/kaya/config.json` and
    exports root-relative artifact paths on the remote side.

CLI:
    `python -m ops.kaya.kaya [--config PATH] COMMAND [command-options]`

    COMMAND is one of `show-config`, `push`, `pull`, `run`, `submit`, `watch`,
    `kill`, `cancel`, `clear-cache`, `status`. Command internals live in the
    `ops/kaya/runner/` subpackage; this file only parses args and dispatches.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from .runner.commands import (
    handle_cancel,
    handle_clear_cache,
    handle_kill,
    handle_run,
    handle_submit,
    handle_watch,
)
from .runner.config import DEFAULT_CONFIG, KayaConfig, load_config
from .runner.sources import spec_arg
from .runner.status import add_status_args, run_status
from .runner.sync import pull, push

# Re-exported for callers that import these from `ops.kaya.kaya`
# (e.g. ops/scripts/prestage.py, ops/scripts/setup_env.py, tests/test_preflight.py).
__all__ = ["KayaConfig", "load_config", "spec_arg", "push", "pull", "main"]


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
              python -m ops.kaya.kaya show-config
              python -m ops.kaya.kaya push
              python -m ops.kaya.kaya run ops/scripts/setup_env.py
              python -m ops.kaya.kaya run ops/scripts/prestage.py -- --skip-models
              python -m ops.kaya.kaya submit --time 00:05:00 ops/scripts/gpu_test.py
              python -m ops.kaya.kaya submit ops/kaya/example.sbatch -- --script-arg value
              python -m ops.kaya.kaya watch
              python -m ops.kaya.kaya status

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
            "For .py files, omitted SLURM options come from ops/kaya/config.json.\n"
            "For .sbatch files, only explicitly supplied SLURM options are passed as overrides."
        ),
    )
    add_common_run_args(submit)
    add_slurm_args(submit)
    add_job_lifecycle_args(submit)
    submit.add_argument("--no-preflight", action="store_true",
                        help="skip the login-node preflight check for a spec-driven generation submit")
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

    status = sub.add_parser(
        "status",
        help="print GPU partition node/queue/start-estimate status (read-only, over SSH)",
    )
    add_status_args(status)

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
    elif args.command == "status":
        return run_status(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
