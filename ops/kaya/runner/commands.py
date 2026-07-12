"""Command handlers: run, submit (with preflight), watch, kill, cancel, clear-cache.

`kaya.py` parses args and dispatches here; each handler wires the lower-level
modules (sync, remote, slurm, jobs) into one user-facing command."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

from .config import LOCAL_ROOT, KayaConfig, RunSettings, quote, shell_join, strip_separator
from .jobs import last_job_id, print_log_tails, print_user_jobs, wait_for_job
from .remote import remote_prelude, ssh_script, write_remote_file
from .slurm import (
    explicit_sbatch_options,
    generated_python_sbatch,
    submit_remote_sbatch,
)
from .sources import (
    local_source_path,
    python_command,
    remote_source_path,
    resolve_run_settings,
    spec_arg,
)
from .sync import pull, push


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


def run_preflight(config: KayaConfig, spec: str, gres: str | None) -> None:
    """Run the login-node preflight for a spec; abort submission if it fails.

    Streams `ops/scripts/preflight.py` on the login node (core env, offline) so a
    parse error, empty corpus, or unstaged weight is caught here, before the job
    clears the queue. A nonzero preflight raises `SystemExit`.
    """

    preflight = LOCAL_ROOT / "ops" / "scripts" / "preflight.py"
    extra = ["--gres", gres] if gres else []
    command = python_command(config, preflight, ["--spec", spec, *extra])
    script = "\n".join([remote_prelude(config, activate=True, offline=True), shell_join(command)])
    print(f"[preflight] {config.ssh_alias}: {shell_join(command)}")
    try:
        ssh_script(config, script)
    except subprocess.CalledProcessError:
        raise SystemExit(
            "[preflight] FAILED - not submitting. Fix the issues above, or pass "
            "--no-preflight to override."
        )
    print("[preflight] passed")


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
    # Preflight a spec-driven generation submit on the login node before the queue.
    spec = spec_arg(forwarded)
    if spec and not args.no_preflight:
        run_preflight(config, spec, args.gres)
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


def handle_watch(config: KayaConfig, args: argparse.Namespace) -> None:
    """Wait for a SLURM job, pull artifacts, and print log tails."""

    job_id = args.job_id or last_job_id()
    wait_for_job(config, job_id)
    if not args.no_pull:
        pull(config)
    print_log_tails(config, job_id, args.tail_lines, job_name=args.job_name)


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
