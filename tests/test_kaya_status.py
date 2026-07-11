"""Tests for the Kaya status table data and ordering."""

from datetime import datetime, timezone

from ops.scripts.kaya_status import (
    QueueRow,
    StartEstimate,
    parse_queue,
    parse_start_estimates,
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


def test_start_estimates_are_sorted_by_wait_with_unavailable_last() -> None:
    """Start estimates use ascending wait order and put N/A at the end."""

    output = "\n".join(
        [
            "3|later|other|PD|2026-07-11T14:00:00|Priority",
            "4|unknown|other|PD|N/A|Resources",
            "2|sooner|other|PD|2026-07-11T12:30:00|Priority",
        ]
    )

    rows = parse_start_estimates(
        output,
        scheduler_tz=timezone.utc,
        scheduler_tz_name="UTC",
        now=datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc),
    )

    assert [row.job_id for row in rows] == ["2", "3", "4"]
    assert [row.wait for row in rows] == ["00:30:00", "02:00:00", "N/A"]


def test_shared_queue_and_start_tables_ignore_your_jobs_limit(capsys) -> None:
    """Only Your Jobs is truncated by the row limit."""

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

    def start_row(job_id: str, wait_seconds: int) -> StartEstimate:
        return StartEstimate(
            job_id,
            "job",
            "me",
            "PD",
            "2026-07-11T13:00:00",
            "2026-07-11 13:00:00 UTC",
            "01:00:00",
            wait_seconds,
            "Priority",
        )

    print_text(
        ssh_alias="kaya",
        partition="gpu",
        nodes=[],
        queue=[queue_row("shared-1"), queue_row("shared-2")],
        mine=[queue_row("mine-1"), queue_row("mine-2")],
        starts=[start_row("start-1", 1), start_row("start-2", 2)],
        limit=1,
        scheduler_tz_name="UTC",
    )

    output = capsys.readouterr().out
    assert "shared-1" in output and "shared-2" in output
    assert "mine-1" in output and "mine-2" not in output
    assert "start-1" in output and "start-2" in output
    assert "local_time" not in output
