"""Hand-annotate the 135 MMLongBench documents, and score the result.

Purpose:
    The paper bins the seven native `doc_type` labels into three domains
    (text_heavy / in_between / visual_heavy, see `data/binning.py`). That mapping
    is an assumption. This builds a per-document sheet so a human can look at each
    doc and record what it actually is, then scores the hand labels against the
    doc_type-derived bin to see whether the assumption holds. Two axes are the
    point (text/visual bin, scanned vs digital-born); two more are captured while
    we are in there (dominant visual element, multi-column layout). The sheet is
    seeded with auto-guesses so a human edits rather than starts blank.

Pipeline role:
    A standalone research utility that reads only the dataset (parquet + PDFs) and
    writes a committable CSV under `annotations/`. Not part of the run pipeline.

Arguments:
    Subcommands `annotate` (interactive, resumable), `sheet` (build the blank
    auto-seeded CSV), and `score` (agreement report). See `--help` for each.

Workflow:
    # interactive: opens each PDF in turn and prompts a menu per field; writes
    # after every document, so it's resumable (re-run to pick up where you left off)
    python -m scripts.annotate_docs annotate
    # score the hand labels against the doc_type-derived bin
    python -m scripts.annotate_docs score
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT  # noqa: E402
from data.binning import doc_type_bin  # noqa: E402
from data.loader import load_raw_mmlongbench, resolve_pdf  # noqa: E402
from data.render import classify_scanned  # noqa: E402

DEFAULT_SHEET = ROOT / "annotations" / "doc_labels.csv"

# Column order. auto_* are machine-seeded priors; the *_label / dominant_visual /
# multi_column / notes columns are for the human.
COLUMNS = [
    "doc_id",
    "pdf_path",
    "doc_type",
    "auto_bin",
    "auto_scan",
    "avg_chars_per_page",
    "page_count",
    "bin_label",
    "scan_label",
    "dominant_visual",
    "multi_column",
    "notes",
]

# The menu fields, in the order the interactive annotator prompts them. Each is
# (column, options, auto-seed column). The auto column supplies the default when
# the row hasn't been annotated yet.
CHOICE_FIELDS: list[tuple[str, list[str], str | None]] = [
    ("bin_label", ["text_heavy", "in_between", "visual_heavy"], "auto_bin"),
    ("scan_label", ["scanned", "digital"], "auto_scan"),
    ("dominant_visual", ["tables", "charts", "figures", "photos", "none"], None),
    ("multi_column", ["single", "multi"], None),
]

# Allowed human values (for a light validation pass in `score`).
VALID = {field: set(options) for field, options, _ in CHOICE_FIELDS}


def unique_docs() -> dict[str, str]:
    """Return `{doc_id: doc_type}` for the 135 documents (one type per doc)."""

    doc_types: dict[str, str] = {}
    for row in load_raw_mmlongbench():
        doc_id = str(row.get("doc_id") or "")
        if not doc_id:
            continue
        doc_types.setdefault(doc_id, str(row.get("doc_type") or ""))
    return doc_types


def build_rows() -> list[dict[str, str]]:
    """Build the seeded annotation rows, one per document."""

    rows: list[dict[str, str]] = []
    for doc_id, doc_type in sorted(unique_docs().items()):
        try:
            pdf = resolve_pdf(doc_id)
        except FileNotFoundError:
            print(f"  ! no PDF for {doc_id}, skipping")
            continue
        scan = classify_scanned(pdf)
        rows.append(
            {
                "doc_id": doc_id,
                "pdf_path": str(pdf.relative_to(ROOT)) if pdf.is_relative_to(ROOT) else str(pdf),
                "doc_type": doc_type,
                "auto_bin": doc_type_bin(doc_type),
                "auto_scan": scan.label,
                "avg_chars_per_page": str(scan.avg_chars_per_page),
                "page_count": str(scan.page_count),
                "bin_label": "",
                "scan_label": "",
                "dominant_visual": "",
                "multi_column": "",
                "notes": "",
            }
        )
    return rows


def write_sheet(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cmd_sheet(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.force:
        print(f"{args.output} already exists; refusing to overwrite hand labels. Use --force to rebuild.")
        return 1
    rows = build_rows()
    write_sheet(rows, args.output)
    print(f"wrote {len(rows)} document rows: {args.output}")
    print("fill bin_label / scan_label / dominant_visual / multi_column, then run: annotate_docs score")
    return 0


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


def invalid_values(rows: list[dict[str, str]]) -> list[str]:
    """Return human-readable complaints for out-of-vocabulary filled values."""

    problems: list[str] = []
    for row in rows:
        for column, allowed in VALID.items():
            value = (row.get(column) or "").strip()
            if value and value not in allowed:
                problems.append(f"{row['doc_id']}: {column}={value!r} not in {sorted(allowed)}")
    return problems


def score_sheet(rows: list[dict[str, str]]) -> dict:
    """Aggregate a filled annotation sheet into the numbers `score` prints.

    Axis 1 (the point): agreement between the human `bin_label` and the
    doc_type-derived `auto_bin`, overall + per doc_type, with the disagreements.
    Axis 2: scanned fraction, hand vs the auto heuristic. Plus distributions of
    the two extra axes.
    """

    total = len(rows)
    labelled = [r for r in rows if (r.get("bin_label") or "").strip()]
    bin_agree = sum(1 for r in labelled if r["bin_label"].strip() == r["auto_bin"].strip())
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # doc_type -> [agree, total]
    mismatches: list[dict[str, str]] = []
    for r in labelled:
        match = r["bin_label"].strip() == r["auto_bin"].strip()
        per_type[r["doc_type"]][0] += int(match)
        per_type[r["doc_type"]][1] += 1
        if not match:
            mismatches.append(
                {"doc_id": r["doc_id"], "doc_type": r["doc_type"], "auto": r["auto_bin"], "human": r["bin_label"].strip()}
            )

    scan_labelled = [r for r in rows if (r.get("scan_label") or "").strip()]
    return {
        "total": total,
        "bin_labelled": len(labelled),
        "bin_agree": bin_agree,
        "per_type": {k: tuple(v) for k, v in per_type.items()},
        "mismatches": mismatches,
        "auto_scanned": sum(1 for r in rows if r.get("auto_scan") == "scanned"),
        "scan_labelled": len(scan_labelled),
        "human_scanned": sum(1 for r in scan_labelled if r["scan_label"].strip() == "scanned"),
        "scan_agree": sum(1 for r in scan_labelled if r["scan_label"].strip() == r["auto_scan"].strip()),
        "dominant_visual": dict(Counter((r.get("dominant_visual") or "").strip() for r in rows if (r.get("dominant_visual") or "").strip())),
        "multi_column": dict(Counter((r.get("multi_column") or "").strip() for r in rows if (r.get("multi_column") or "").strip())),
    }


def cmd_score(args: argparse.Namespace) -> int:
    rows = read_sheet(args.sheet)
    for problem in invalid_values(rows):
        print(f"  ! {problem}")
    s = score_sheet(rows)

    print(f"\nbin_label filled for {s['bin_labelled']}/{s['total']} docs")
    if s["bin_labelled"]:
        print(f"human bin vs doc_type-derived bin: {s['bin_agree']}/{s['bin_labelled']} agree ({_pct(s['bin_agree'], s['bin_labelled'])})")
        print("  by doc_type:")
        for doc_type in sorted(s["per_type"]):
            agree_t, total_t = s["per_type"][doc_type]
            print(f"    {doc_type:<32} {agree_t}/{total_t} agree ({_pct(agree_t, total_t)})")
        if s["mismatches"]:
            print("  disagreements (the assumption misses these):")
            for m in s["mismatches"]:
                print(f"    {m['doc_id']} ({m['doc_type']}): auto={m['auto']} human={m['human']}")

    print(f"\nauto scanned fraction: {s['auto_scanned']}/{s['total']} ({_pct(s['auto_scanned'], s['total'])})")
    if s["scan_labelled"]:
        print(f"human scanned fraction: {s['human_scanned']}/{s['scan_labelled']} ({_pct(s['human_scanned'], s['scan_labelled'])})")
        print(f"human vs auto scan agreement: {s['scan_agree']}/{s['scan_labelled']} ({_pct(s['scan_agree'], s['scan_labelled'])})")

    for column in ("dominant_visual", "multi_column"):
        if s[column]:
            print(f"\n{column}: " + ", ".join(f"{k}={v}" for k, v in sorted(s[column].items())))
    return 0


def row_is_annotated(row: dict[str, str]) -> bool:
    """True once every menu field has a value (notes stays optional)."""

    return all((row.get(field) or "").strip() for field, _, _ in CHOICE_FIELDS)


class _Quit(Exception):
    """Raised from a prompt when the user asks to save and exit."""


def _prompt_choice(field: str, options: list[str], default: str | None) -> str | None:
    """Prompt one menu field. Return the chosen value, or None to leave it as-is."""

    print(f"\n{field}:")
    for i, option in enumerate(options, 1):
        marker = "  <- default" if option == default else ""
        print(f"  {i}) {option}{marker}")
    while True:
        raw = input("  select [number / Enter=default / s=skip / q=save+quit]: ").strip().lower()
        if raw == "q":
            raise _Quit
        if raw == "s":
            return None
        if raw == "" and default is not None:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  invalid; enter one of the listed numbers")


def _prompt_notes(default: str) -> str:
    shown = f" [{default}]" if default else ""
    raw = input(f"\nnotes{shown} (Enter to keep): ").strip()
    return raw or default


def open_document(pdf_path: Path, open_cmd: str | None) -> None:
    """Best-effort open the PDF in a viewer; never fatal if it can't."""

    if open_cmd:
        argv = [*open_cmd.split(), str(pdf_path)]
    else:
        opener = next((exe for exe in ("xdg-open", "wslview", "open") if shutil.which(exe)), None)
        if opener is None:
            print("  (no opener found; open the path above manually)")
            return
        argv = [opener, str(pdf_path)]
    try:
        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        print(f"  (could not open document: {exc}; open the path above manually)")


def _annotate_row(row: dict[str, str], *, open_cmd: str | None, do_open: bool) -> None:
    """Prompt every menu field for one document, updating `row` in place."""

    print("\n" + "=" * 72)
    print(
        f"{row['doc_id']}  |  doc_type={row['doc_type']}  "
        f"auto_bin={row['auto_bin']}  auto_scan={row['auto_scan']}  pages={row['page_count']}"
    )
    pdf_path = Path(row["pdf_path"])
    if not pdf_path.is_absolute():
        pdf_path = ROOT / pdf_path
    print(f"  {pdf_path}")
    if do_open:
        open_document(pdf_path, open_cmd)
    for field, options, auto_col in CHOICE_FIELDS:
        current = (row.get(field) or "").strip()
        default = current or (row.get(auto_col, "") if auto_col else "") or None
        choice = _prompt_choice(field, options, default)
        if choice is not None:
            row[field] = choice
    row["notes"] = _prompt_notes((row.get("notes") or "").strip())


def cmd_annotate(args: argparse.Namespace) -> int:
    path = args.sheet
    if not path.exists():
        write_sheet(build_rows(), path)
        print(f"seeded a new sheet: {path}")
    rows = read_sheet(path)

    pending = rows if args.redo_all else [row for row in rows if not row_is_annotated(row)]
    if not pending:
        print("every document is already annotated. Use --redo-all to revisit them.")
        return 0

    print(f"{len(pending)} of {len(rows)} documents to annotate. Answers save after each doc; 'q' saves and exits.")
    try:
        for row in pending:
            _annotate_row(row, open_cmd=args.open_cmd, do_open=args.open)
            write_sheet(rows, path)
    except (_Quit, KeyboardInterrupt):
        write_sheet(rows, path)
        done = sum(1 for row in rows if row_is_annotated(row))
        print(f"\nsaved {path}  ({done}/{len(rows)} done). Re-run to continue.")
        return 0
    write_sheet(rows, path)
    print(f"\ndone; wrote {path}. Now: python -m scripts.annotate_docs score")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    annotate = sub.add_parser("annotate", help="interactively annotate each document (resumable)")
    annotate.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    annotate.add_argument("--redo-all", action="store_true", help="revisit already-annotated docs too")
    annotate.add_argument("--no-open", dest="open", action="store_false", help="do not auto-open the PDF viewer")
    annotate.add_argument("--open-cmd", help="viewer command to open each PDF (default: xdg-open/wslview/open)")

    sheet = sub.add_parser("sheet", help="build the blank (auto-seeded) annotation sheet, no prompting")
    sheet.add_argument("--output", type=Path, default=DEFAULT_SHEET)
    sheet.add_argument("--force", action="store_true", help="overwrite an existing sheet (loses hand labels)")

    score = sub.add_parser("score", help="score filled hand labels vs the doc_type-derived bin")
    score.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "annotate":
        return cmd_annotate(args)
    if args.command == "sheet":
        return cmd_sheet(args)
    if args.command == "score":
        return cmd_score(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
