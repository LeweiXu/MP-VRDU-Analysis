"""Tests for the Kaya status table data and ordering."""

from datetime import datetime, timezone

from ops.kaya.runner.status import (
    CompletedJob,
    QueueRow,
    parse_completed_jobs,
    parse_queue,
    print_text,
)


def test_queue_wait_uses_start_for_running_and_now_for_pending() -> None:
    """Queue waits stop at start for running jobs and keep growing for pending jobs."""

    output = "\n".join(
        [
            "1|running|me|R|01:00|02:00|1|gpu:1|node1|2026-07-11T10:00:00|2026-07-11T10:30:00",
            "2|pending|me|PD|0:00|02:00|1|gpu:1|Priority|2026-07-11T11:00:00|N/A",
        ]
    )

    rows = parse_queue(
        output,
        scheduler_tz=timezone.utc,
        now=datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc),
    )

    assert rows[0].submitted == "2026-07-11T10:00:00"
    assert rows[0].queue_wait == "00:30:00"
    assert rows[1].queue_wait == "01:00:00"


def test_completed_jobs_are_newest_first_and_limited() -> None:
    """Completed jobs are ordered by end time and capped at the requested count."""

    output = "\n".join(
        [
            "3|older|me|COMPLETED|01:00:00|02:00:00|2026-07-11T12:00:00|2026-07-11T13:00:00|gpu|cpu=4,gres/gpu=2",
            "4|failed|me|FAILED|00:10:00|02:00:00|2026-07-11T13:00:00|2026-07-11T13:10:00|gpu|cpu=4,gres/gpu=2",
            "2|newer|me|COMPLETED|00:30:00|02:00:00|2026-07-11T14:00:00|2026-07-11T14:30:00|gpu|cpu=4,gres/gpu=2",
        ]
    )

    rows = parse_completed_jobs(output, partition="gpu", limit=1)

    assert [row.job_id for row in rows] == ["2"]


def test_shared_queue_and_completed_tables_ignore_your_jobs_limit(capsys) -> None:
    """The row limit only truncates Your Jobs."""

    def queue_row(job_id: str) -> QueueRow:
        return QueueRow(
            job_id,
            "job",
            "me",
            "PD",
            "0:00",
            "1:00",
            "1",
            "gpu:1",
            "Priority",
            "2026-07-11T12:00:00",
            "N/A",
            "00:10:00",
        )

    def completed_row(job_id: str) -> CompletedJob:
        return CompletedJob(
            job_id,
            "job",
            "me",
            "COMPLETED",
            "01:00:00",
            "02:00:00",
            "2026-07-11T12:00:00",
            "2026-07-11T13:00:00",
            "gpu",
            "cpu=4,gres/gpu=2",
        )

    print_text(
        ssh_alias="kaya",
        partition="gpu",
        nodes=[],
        queue=[queue_row("shared-1"), queue_row("shared-2")],
        mine=[queue_row("mine-1"), queue_row("mine-2")],
        completed=[completed_row("done-1"), completed_row("done-2")],
        limit=1,
        scheduler_tz_name="UTC",
    )

    output = capsys.readouterr().out
    assert "shared-1" in output and "shared-2" in output
    assert "mine-1" in output and "mine-2" not in output
    assert "done-1" in output and "done-2" in output
    assert "Scheduler Start Estimates" not in output
