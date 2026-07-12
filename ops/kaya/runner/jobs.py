"""Monitor submitted SLURM jobs: parse squeue, wait for a job to leave the queue,
list the user's jobs, and print local log tails."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .config import LOCAL_ROOT, KayaConfig, SqueueJobStatus, quote

_QUEUE_FMT = "%i|%P|%j|%u|%t|%M|%l|%D|%R"


def parse_squeue_row(line: str) -> SqueueJobStatus | None:
    """Parse one pipe-delimited `squeue` status row."""

    parts = line.rstrip("\n").split("|", 8)
    if len(parts) != 9:
        return None
    return SqueueJobStatus(*parts)


def _squeue(config: KayaConfig, selector: str) -> list[SqueueJobStatus]:
    """Run `squeue -h -o <fmt>` with a job/user selector and parse the rows."""

    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", config.ssh_alias, f"squeue -h {selector} -o {quote(_QUEUE_FMT)}"],
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


def squeue_statuses(config: KayaConfig, job_id: str) -> list[SqueueJobStatus]:
    """Return current queue rows for a job id."""

    simple_job_id = job_id.split(";", 1)[0]
    return _squeue(config, f"-j {quote(simple_job_id)}")


def user_jobs(config: KayaConfig) -> list[SqueueJobStatus]:
    """Return the current user's queued/running jobs (empty if none)."""

    return _squeue(config, '-u "$(whoami)"')


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
