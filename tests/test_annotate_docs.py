"""Focused tests for the interactive document annotation prompts."""

from __future__ import annotations

import argparse

from ops.scripts.annotate_docs import (
    BIN_LABELS,
    SCAN_LABELS,
    _annotate_row,
    _prompt_multi,
    build_parser,
    cmd_sheet,
    read_sheet,
    write_sheet,
)


def _row() -> dict[str, str]:
    return {
        "doc_id": "example.pdf",
        "pdf_path": "example.pdf",
        "doc_type": "Report",
        "auto_scan": "scanned",
        "page_count": "1",
        "bin_label": "mixed-modality",
        "scan_label": "digital",
        "dominant_visual": "",
        "notes": "keep me",
    }


def test_annotate_row_defaults_to_saved_single_choice_labels(monkeypatch) -> None:
    answers = iter(["", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    row = _row()

    _annotate_row(
        row,
        open_cmd=None,
        do_open=False,
        fields=[
            ("bin_label", BIN_LABELS, None),
            ("scan_label", SCAN_LABELS, "auto_scan"),
        ],
        prompt_notes=False,
    )

    assert row["bin_label"] == "mixed-modality"
    assert row["scan_label"] == "digital"


def test_dominant_visual_accepts_space_separated_choices(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "1 3")

    assert _prompt_multi("dominant_visual", ("tables", "charts", "figures"), None) == "tables;figures"


def test_no_notes_flag_is_available_and_preserves_notes(monkeypatch) -> None:
    args = build_parser().parse_args(["annotate", "--no-notes"])
    row = _row()

    _annotate_row(row, open_cmd=None, do_open=False, fields=[], prompt_notes=not args.no_notes)

    assert args.no_notes
    assert row["notes"] == "keep me"


def test_sheet_fills_only_empty_scan_labels(tmp_path) -> None:
    sheet = tmp_path / "labels.csv"
    rows = [
        {**_row(), "doc_id": "empty.pdf", "scan_label": "", "auto_scan": "scanned"},
        {**_row(), "doc_id": "labelled.pdf", "scan_label": "digital", "auto_scan": "scanned"},
    ]
    write_sheet(rows, sheet)

    result = cmd_sheet(
        argparse.Namespace(output=sheet, force=False, fill_empty_scan_labels=True)
    )
    updated = read_sheet(sheet)

    assert result == 0
    assert [row["scan_label"] for row in updated] == ["scanned", "digital"]


def test_sheet_fill_scan_labels_requires_existing_sheet(tmp_path) -> None:
    result = cmd_sheet(
        argparse.Namespace(
            output=tmp_path / "missing.csv",
            force=False,
            fill_empty_scan_labels=True,
        )
    )

    assert result == 1
