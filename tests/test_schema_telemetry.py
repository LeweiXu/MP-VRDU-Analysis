"""The per-cell result contract carries the full uniform telemetry schema, and
the truncation fields are a zero-canary now that the input-token cap is gone."""

from __future__ import annotations

import dataclasses

import pytest

from conftest import require

# Fields fixed by the telemetry schema, collected on every cell.
REQUIRED_TELEMETRY = {
    # identity / provenance
    "status", "skipped_reason", "bin_label", "scan_label", "machine",
    # tokens (cap removed: fed must equal total)
    "total_text_tokens", "total_visual_tokens", "text_tokens_fed", "output_tokens",
    # latency split
    "latency_s", "prefill_latency_s", "decode_latency_s",
    # memory
    "peak_vram_bytes",
    # skips / canary
    "oom_occurred", "tokens_dropped",
}


def _row_fields() -> set[str]:
    row = require("schema", "ResultRow")
    assert dataclasses.is_dataclass(row), "ResultRow must be a dataclass contract"
    return {f.name for f in dataclasses.fields(row)}


def test_result_row_has_all_telemetry_fields() -> None:
    missing = REQUIRED_TELEMETRY - _row_fields()
    assert not missing, f"ResultRow missing telemetry fields: {sorted(missing)}"


def test_status_values_are_ok_oom_error() -> None:
    # status is one of ok / oom / error; a helper validates it.
    valid = require("schema", "STATUS_VALUES")
    assert set(valid) == {"ok", "oom", "error"}


def test_truncation_is_a_zero_canary() -> None:
    # With no cap, text_tokens_fed == total_text_tokens, so tokens_dropped == 0
    # and truncation_occurred is False. A nonzero value is a bug signal.
    dropped = require("schema", "tokens_dropped")
    occurred = require("schema", "truncation_occurred")
    assert dropped(total_text_tokens=1000, text_tokens_fed=1000) == 0
    assert occurred(total_text_tokens=1000, text_tokens_fed=1000) is False
