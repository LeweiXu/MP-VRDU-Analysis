#!/usr/bin/env python3
"""Print Kaya GPU partition node, memory, queue, and start-estimate status.

This is a read-only operational helper that gets SLURM state over SSH.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "kaya" / "config.json"


@dataclass(frozen=True)
class NodeStatus:
    """One SLURM node's GPU, CPU, and memory allocation status."""

    name: str
    state: str
    gres: str
    gpu_alloc: int
    gpu_total: int
    cpu_alloc: int
    cpu_total: int
    mem_alloc_gib: float
    mem_total_gib: float
    mem_free_gib: float


@dataclass(frozen=True)
class QueueRow:
    """One squeue row."""

    job_id: str
    name: str
    user: str
    state: str
    elapsed: str
    time_limit: str
    nodes: str
    gres: str
    reason_or_nodelist: str
    submitted: str
    started: str
    queue_wait: str


@dataclass(frozen=True)
class StartEstimate:
    """One `squeue --start` estimate row."""

    job_id: str
    name: str
    user: str
    state: str
    start_time_raw: str
    start_time_scheduler: str
    wait: str
    wait_seconds: int | None
    reason_or_nodelist: str


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ssh-alias", help="SSH host override; defaults to kaya/config.json")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--user", help="user for the 'Your Jobs' section; defaults to remote whoami")
    parser.add_argument("--limit", type=int, default=25, help="maximum rows in the Your Jobs section")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--no-start", action="store_true", help="skip squeue --start estimates")
    return parser


def load_ssh_alias(config_path: Path, override: str | None) -> str:
    """Return the SSH alias from CLI or config."""

    if override:
        return override
    try:
        return str(json.loads(config_path.read_text())["ssh_alias"])
    except Exception as exc:
        raise SystemExit(f"could not read ssh_alias from {config_path}: {exc}") from exc


def remote(ssh_alias: str, command: str) -> str:
    """Run a read-only command on Kaya and return stdout."""

    result = subprocess.run(
        ["ssh", ssh_alias, command],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"ssh exited {result.returncode}"
        raise SystemExit(message)
    return result.stdout


def remote_user(ssh_alias: str, override: str | None) -> str | None:
    """Return the remote username for user-specific queue filtering."""

    if override:
        return override
    try:
        value = remote(ssh_alias, "whoami").strip()
    except SystemExit:
        return None
    return value or None


def parse_utc_offset(value: str) -> timezone:
    """Parse a numeric UTC offset such as `+0800`."""

    match = re.fullmatch(r"([+-])(\d{2})(\d{2})", value.strip())
    if not match:
        return datetime.now().astimezone().tzinfo or timezone.utc  # type: ignore[return-value]
    sign = 1 if match.group(1) == "+" else -1
    delta = timedelta(hours=int(match.group(2)), minutes=int(match.group(3)))
    return timezone(sign * delta)


def remote_timezone(ssh_alias: str) -> tuple[tzinfo, str]:
    """Return the scheduler/login-node timezone info and display name."""

    output = remote(ssh_alias, "date '+%z|%Z'").strip()
    offset, _, name = output.partition("|")
    return parse_utc_offset(offset), name or offset


def format_dt(value: datetime, tz_name: str | None = None) -> str:
    """Return a timestamp with an explicit timezone label."""

    label = tz_name or value.tzname() or value.strftime("%z")
    return value.strftime("%Y-%m-%d %H:%M:%S ") + label


def format_wait(delta: timedelta) -> str:
    """Return a compact non-negative wait duration."""

    seconds = max(0, int(delta.total_seconds()))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_tres_count(tres: str, key: str) -> int:
    """Extract an integer TRES value such as `gres/gpu=2`."""

    match = re.search(rf"(?:^|,){re.escape(key)}=(\d+)", tres or "")
    return int(match.group(1)) if match else 0


def parse_mem_mb(value: str | None) -> int:
    """Parse SLURM memory strings into MiB-ish integer units."""

    if not value:
        return 0
    text = value.strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTP]?)", text)
    if not match:
        return 0
    number = float(match.group(1))
    unit = match.group(2)
    scale = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "P": 1024 * 1024 * 1024}
    return int(number * scale[unit])


def mib_to_gib(value: int) -> float:
    """Convert MiB to GiB rounded for display."""

    return round(value / 1024, 1)


def key_values(block: str) -> dict[str, str]:
    """Parse `scontrol show nodes` key=value tokens from one node block."""

    values: dict[str, str] = {}
    for key, value in re.findall(r"(\S+?)=(\S+)", block):
        values[key] = value
    return values


def node_names(ssh_alias: str, partition: str) -> list[str]:
    """Return node names in a partition."""

    output = remote(ssh_alias, f"sinfo -h -p {shlex.quote(partition)} -N -o '%N'")
    return [line.strip() for line in output.splitlines() if line.strip()]


def load_nodes(ssh_alias: str, partition: str) -> list[NodeStatus]:
    """Load per-node status through scontrol."""

    names = node_names(ssh_alias, partition)
    if not names:
        return []
    command = "scontrol show nodes " + shlex.quote(",".join(names))
    blocks = [block for block in remote(ssh_alias, command).split("\n\n") if block.strip()]
    nodes: list[NodeStatus] = []
    for block in blocks:
        values = key_values(block)
        cfg_tres = values.get("CfgTRES", "")
        alloc_tres = values.get("AllocTRES", "")
        gpu_total = parse_tres_count(cfg_tres, "gres/gpu") or parse_tres_count(values.get("Gres", ""), "gpu")
        mem_total = parse_mem_mb(values.get("RealMemory"))
        mem_alloc = parse_tres_mem(alloc_tres)
        nodes.append(
            NodeStatus(
                name=values.get("NodeName", "?"),
                state=values.get("State", "?"),
                gres=values.get("Gres", ""),
                gpu_alloc=parse_tres_count(alloc_tres, "gres/gpu"),
                gpu_total=gpu_total,
                cpu_alloc=int(values.get("CPUAlloc", "0")),
                cpu_total=int(values.get("CPUTot", "0")),
                mem_alloc_gib=mib_to_gib(mem_alloc),
                mem_total_gib=mib_to_gib(mem_total),
                mem_free_gib=mib_to_gib(parse_mem_mb(values.get("FreeMem"))),
            )
        )
    return sorted(nodes, key=lambda node: node.name)


def parse_tres_mem(tres: str) -> int:
    """Extract memory from an AllocTRES string."""

    match = re.search(r"(?:^|,)mem=([^,]+)", tres or "")
    return parse_mem_mb(match.group(1)) if match else 0


def parse_slurm_time(value: str, scheduler_tz: tzinfo) -> datetime | None:
    """Parse a SLURM ISO timestamp, returning None for unavailable values."""

    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=scheduler_tz)
    except ValueError:
        return None


def queue_wait(
    *, submitted: str, started: str, state: str, scheduler_tz: tzinfo, now: datetime
) -> str:
    """Return time spent queued for a running or pending job."""

    submitted_at = parse_slurm_time(submitted, scheduler_tz)
    if submitted_at is None:
        return "unknown"
    started_at = parse_slurm_time(started, scheduler_tz) if state == "R" else None
    return format_wait((started_at or now) - submitted_at)


def parse_queue(output: str, *, scheduler_tz: tzinfo, now: datetime | None = None) -> list[QueueRow]:
    """Parse pipe-delimited squeue output."""

    rows: list[QueueRow] = []
    current_time = now or datetime.now(scheduler_tz)
    for line in output.splitlines():
        if not line.strip() or line.startswith("JOBID|"):
            continue
        parts = line.split("|", 10)
        if len(parts) == 11:
            rows.append(
                QueueRow(
                    *parts,
                    queue_wait=queue_wait(
                        submitted=parts[9],
                        started=parts[10],
                        state=parts[3],
                        scheduler_tz=scheduler_tz,
                        now=current_time,
                    ),
                )
            )
    return rows


def load_queue(
    ssh_alias: str, partition: str, *, scheduler_tz: tzinfo, user: str | None = None
) -> list[QueueRow]:
    """Load running/pending queue rows."""

    user_arg = f"-u {shlex.quote(user)} " if user else ""
    fmt = "%i|%j|%u|%t|%M|%l|%D|%b|%R|%V|%S"
    command = (
        f"squeue {user_arg}-p {shlex.quote(partition)} -t RUNNING,PENDING "
        f"-h -o {shlex.quote(fmt)}"
    )
    return parse_queue(remote(ssh_alias, command), scheduler_tz=scheduler_tz)


def parse_start_estimates(
    output: str,
    *,
    scheduler_tz: tzinfo,
    scheduler_tz_name: str,
    now: datetime | None = None,
) -> list[StartEstimate]:
    """Parse pipe-delimited `squeue --start` output."""

    rows: list[StartEstimate] = []
    current_time = now or datetime.now(scheduler_tz)
    for line in output.splitlines():
        if not line.strip() or line.startswith("JOBID|"):
            continue
        parts = line.split("|", 5)
        if len(parts) == 6:
            raw_start = parts[4]
            if raw_start == "N/A":
                rows.append(StartEstimate(*parts[:4], raw_start, "N/A", "N/A", None, parts[5]))
                continue
            scheduler_start = parse_slurm_time(raw_start, scheduler_tz)
            if scheduler_start is None:
                rows.append(StartEstimate(*parts[:4], raw_start, raw_start, "unknown", None, parts[5]))
                continue
            wait_seconds = max(0, int((scheduler_start - current_time).total_seconds()))
            rows.append(
                StartEstimate(
                    *parts[:4],
                    raw_start,
                    format_dt(scheduler_start, scheduler_tz_name),
                    format_wait(timedelta(seconds=wait_seconds)),
                    wait_seconds,
                    parts[5],
                )
            )
    return sorted(rows, key=lambda row: (row.wait_seconds is None, row.wait_seconds or 0))


def load_start_estimates(
    ssh_alias: str,
    partition: str,
    *,
    scheduler_tz: tzinfo,
    scheduler_tz_name: str,
) -> list[StartEstimate]:
    """Load scheduler start estimates for pending jobs."""

    fmt = "%i|%j|%u|%t|%S|%R"
    command = (
        f"squeue -p {shlex.quote(partition)} -t PENDING "
        f"--start -h -o {shlex.quote(fmt)}"
    )
    return parse_start_estimates(
        remote(ssh_alias, command),
        scheduler_tz=scheduler_tz,
        scheduler_tz_name=scheduler_tz_name,
    )


def summarize(nodes: list[NodeStatus]) -> dict[str, Any]:
    """Return aggregate node/GPU/memory counts."""

    return {
        "nodes": len(nodes),
        "gpus_alloc": sum(node.gpu_alloc for node in nodes),
        "gpus_total": sum(node.gpu_total for node in nodes),
        "cpus_alloc": sum(node.cpu_alloc for node in nodes),
        "cpus_total": sum(node.cpu_total for node in nodes),
        "mem_alloc_gib": round(sum(node.mem_alloc_gib for node in nodes), 1),
        "mem_total_gib": round(sum(node.mem_total_gib for node in nodes), 1),
        "mem_free_gib": round(sum(node.mem_free_gib for node in nodes), 1),
    }


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    """Print a compact aligned table."""

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def print_text(
    *,
    ssh_alias: str,
    partition: str,
    nodes: list[NodeStatus],
    queue: list[QueueRow],
    mine: list[QueueRow],
    starts: list[StartEstimate],
    limit: int,
    scheduler_tz_name: str,
) -> None:
    """Print human-readable status."""

    summary = summarize(nodes)
    now = datetime.now().astimezone()
    print(f"Kaya status via {ssh_alias!r}, partition {partition!r} at {format_dt(now)}")
    print(f"Scheduler/login-node timezone: {scheduler_tz_name}; local timezone: {now.tzname() or now.strftime('%z')}")
    print(
        "Summary: "
        f"GPUs {summary['gpus_alloc']}/{summary['gpus_total']} allocated, "
        f"CPUs {summary['cpus_alloc']}/{summary['cpus_total']} allocated, "
        f"memory {summary['mem_alloc_gib']}/{summary['mem_total_gib']} GiB allocated "
        f"({summary['mem_free_gib']} GiB reported free)"
    )
    print()
    print("Nodes")
    print_table(
        ["node", "state", "gpu", "cpu", "mem_alloc/total", "free_mem", "gres"],
        [
            [
                node.name,
                node.state,
                f"{node.gpu_alloc}/{node.gpu_total}",
                f"{node.cpu_alloc}/{node.cpu_total}",
                f"{node.mem_alloc_gib}/{node.mem_total_gib}G",
                f"{node.mem_free_gib}G",
                node.gres,
            ]
            for node in nodes
        ],
    )
    print()
    print("Running/Pending Jobs")
    print_queue(queue)
    print()
    print("Your Jobs")
    print_queue(mine[:limit], show_queue_time=True)
    if starts:
        print()
        print("Scheduler Start Estimates")
        print_table(
            ["job_id", "state", "scheduler_time", "wait", "name", "user", "reason/nodelist"],
            [
                [
                    row.job_id,
                    row.state,
                    row.start_time_scheduler,
                    row.wait,
                    row.name,
                    row.user,
                    row.reason_or_nodelist,
                ]
                for row in starts
            ],
        )
        print()
        print(
            "Start estimates come from `squeue --start`. Treat them as best-effort "
            "backfill predictions, not reservations; priority, reservations, and "
            "other jobs can change them."
        )


def print_queue(rows: list[QueueRow], *, show_queue_time: bool = False) -> None:
    """Print queue rows or an empty message."""

    if not rows:
        print("(none)")
        return
    headers = ["job_id", "state", "elapsed", "limit", "nodes", "gres", "name", "user"]
    if show_queue_time:
        headers.extend(["submitted", "queue_wait"])
    headers.append("reason/nodelist")
    table_rows: list[list[object]] = []
    for row in rows:
        values: list[object] = [
            row.job_id,
            row.state,
            row.elapsed,
            row.time_limit,
            row.nodes,
            row.gres,
            row.name,
            row.user,
        ]
        if show_queue_time:
            values.extend([row.submitted, row.queue_wait])
        values.append(row.reason_or_nodelist)
        table_rows.append(values)
    print_table(headers, table_rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    ssh_alias = load_ssh_alias(args.config, args.ssh_alias)
    nodes = load_nodes(ssh_alias, args.partition)
    scheduler_tz, scheduler_tz_name = remote_timezone(ssh_alias)
    queue = load_queue(ssh_alias, args.partition, scheduler_tz=scheduler_tz)
    user = remote_user(ssh_alias, args.user)
    mine = (
        load_queue(ssh_alias, args.partition, scheduler_tz=scheduler_tz, user=user)
        if user
        else []
    )
    local_tz = datetime.now().astimezone().tzinfo or timezone.utc
    starts = (
        []
        if args.no_start
        else load_start_estimates(
            ssh_alias,
            args.partition,
            scheduler_tz=scheduler_tz,
            scheduler_tz_name=scheduler_tz_name,
        )
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ssh_alias": ssh_alias,
                    "partition": args.partition,
                    "user": user,
                    "scheduler_timezone": scheduler_tz_name,
                    "local_timezone": datetime.now(local_tz).tzname(),
                    "summary": summarize(nodes),
                    "nodes": [asdict(node) for node in nodes],
                    "queue": [asdict(row) for row in queue],
                    "my_jobs": [asdict(row) for row in mine],
                    "start_estimates": [asdict(row) for row in starts],
                    "start_estimate_note": (
                        "squeue --start estimates are best-effort scheduler predictions, not guarantees"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print_text(
        ssh_alias=ssh_alias,
        partition=args.partition,
        nodes=nodes,
        queue=queue,
        mine=mine,
        starts=starts,
        limit=max(1, args.limit),
        scheduler_tz_name=scheduler_tz_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
