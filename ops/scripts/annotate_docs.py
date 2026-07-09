"""Hand-annotate the 135 MMLongBench documents and check annotator agreement.

Purpose:
    The bin axis (text-dominant / mixed-modality / visual-dominant) is the whole
    thesis, so it is labelled by hand rather than derived from the native
    `doc_type`. This builds a per-document sheet a human fills in: the dominant
    modality bin, scan status (digital / scanned), and, optionally, the dominant
    visual element. The sheet is seeded with an auto scan guess so a human edits
    rather than starts blank. The vocab is the single source of truth in
    `data/annotations.py`; this script just collects it.

    Reliability: to report inter-annotator agreement (Cohen's kappa, same 0.75 bar
    as the judge gate), draw a blind subset with `kappa-sheet`, have a second
    annotator fill it, then run `kappa`. The subset sheet is stripped of the first
    annotator's labels so the second pass is genuinely blind.

Pipeline role:
    A standalone research utility. It reads the parquet for the doc_id/doc_type
    list and opens each PDF from the per-doc_type split tree built by
    `ops/scripts/split_docs_by_type.py` (`.data/mmlongbench_docs_split/`), then
    writes a committable CSV under `annotations/` that `data/annotations.py` reads.
    Run `split_docs_by_type` first so the split tree exists.

Workflow:
    # interactive: opens each PDF in turn and prompts a menu per field; writes
    # after every document, so it's resumable (re-run to pick up where you left off)
    python -m ops.scripts.annotate_docs annotate
    # skip the exploratory dominant_visual question entirely:
    python -m ops.scripts.annotate_docs annotate --no-dominant-visual
    # inter-annotator agreement on a blind 25-doc subset:
    python -m ops.scripts.annotate_docs kappa-sheet --n 25 --seed 0
    python -m ops.scripts.annotate_docs annotate --sheet annotations/kappa_subset.csv
    python -m ops.scripts.annotate_docs kappa
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DEFAULT_PATHS, ROOT  # noqa: E402
from data.annotations import BIN_LABELS, SCAN_LABELS, VISUAL_KINDS  # noqa: E402
from data.loader import load_raw_mmlongbench  # noqa: E402
from data.render import classify_scanned  # noqa: E402
from ops.scripts.split_docs_by_type import SPLIT_DIRNAME, safe_dirname  # noqa: E402

DEFAULT_SHEET = ROOT / "annotations" / "doc_labels.csv"
DEFAULT_KAPPA_SHEET = ROOT / "annotations" / "kappa_subset.csv"

# Column order. auto_scan is a machine-seeded prior; the *_label / dominant_visual
# / notes columns are for the human. doc_type is kept as context only (binning no
# longer derives from it).
COLUMNS = [
    "doc_id",
    "pdf_path",
    "doc_type",
    "auto_scan",
    "avg_chars_per_page",
    "page_count",
    "bin_label",
    "scan_label",
    "dominant_visual",
    "notes",
]

# The menu fields, in prompt order. Each is (column, options, auto-seed column).
# The auto column supplies the default when the row hasn't been annotated yet;
# None means there is no machine prior (the human picks from scratch).
CHOICE_FIELDS: list[tuple[str, tuple[str, ...], str | None]] = [
    ("bin_label", BIN_LABELS, None),
    ("scan_label", SCAN_LABELS, "auto_scan"),
    ("dominant_visual", VISUAL_KINDS, None),
]

# Fields that gate "is this row done". dominant_visual is exploratory (per the
# pivot) so it never gates completion, and can be turned off entirely.
REQUIRED_FIELDS = ("bin_label", "scan_label")

# Fields that accept more than one value; stored as a ";"-joined string.
MULTI_FIELDS = {"dominant_visual"}
MULTI_SEP = ";"

# Allowed human values (for a light validation pass in `score`).
VALID = {field: set(options) for field, options, _ in CHOICE_FIELDS}


def split_multi(value: str) -> list[str]:
    """Split a stored multi-value cell into its tokens."""

    return [token.strip() for token in (value or "").split(MULTI_SEP) if token.strip()]


def unique_docs() -> dict[str, str]:
    """Return `{doc_id: doc_type}` for the 135 documents (one type per doc)."""

    doc_types: dict[str, str] = {}
    for row in load_raw_mmlongbench():
        doc_id = str(row.get("doc_id") or "")
        if not doc_id:
            continue
        doc_types.setdefault(doc_id, str(row.get("doc_type") or ""))
    return doc_types


def resolve_split_pdf(doc_id: str, doc_type: str) -> Path:
    """Find a document's PDF inside the per-doc_type split tree.

    `ops/scripts/split_docs_by_type.py` mirrors the corpus into
    `.data/mmlongbench_docs_split/<doc_type>/<file>.pdf`, keeping the original
    filename. That tree is the one handed to annotators, so the sheet points at
    it rather than at the flat `documents/` dir.
    """

    folder = DEFAULT_PATHS.data_dir / SPLIT_DIRNAME / safe_dirname(doc_type)
    names = [doc_id]
    if not doc_id.lower().endswith(".pdf"):
        names.append(f"{doc_id}.pdf")
    for name in names:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no PDF for {doc_id!r} under {folder}; run ops/scripts/split_docs_by_type.py first"
    )


def build_rows() -> list[dict[str, str]]:
    """Build the seeded annotation rows, one per document."""

    rows: list[dict[str, str]] = []
    for doc_id, doc_type in sorted(unique_docs().items()):
        try:
            pdf = resolve_split_pdf(doc_id, doc_type)
        except FileNotFoundError:
            print(f"  ! no split PDF for {doc_id}, skipping")
            continue
        scan = classify_scanned(pdf)
        rows.append(
            {
                "doc_id": doc_id,
                "pdf_path": str(pdf.relative_to(ROOT)) if pdf.is_relative_to(ROOT) else str(pdf),
                "doc_type": doc_type,
                "auto_scan": scan.label,
                "avg_chars_per_page": str(scan.avg_chars_per_page),
                "page_count": str(scan.page_count),
                "bin_label": "",
                "scan_label": "",
                "dominant_visual": "",
                "notes": "",
            }
        )
    return rows


def write_sheet(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        # extrasaction="ignore" so a sheet still carrying a retired column rewrites
        # cleanly instead of raising; the stale column is dropped on the next write.
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
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
    print("fill bin_label / scan_label (+ optional dominant_visual), then run: annotate_docs score")
    return 0


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


def invalid_values(rows: list[dict[str, str]]) -> list[str]:
    """Return human-readable complaints for out-of-vocabulary filled values."""

    problems: list[str] = []
    for row in rows:
        for column, allowed in VALID.items():
            value = (row.get(column) or "").strip()
            if not value:
                continue
            tokens = split_multi(value) if column in MULTI_FIELDS else [value]
            for token in tokens:
                if token not in allowed:
                    problems.append(f"{row['doc_id']}: {column}={token!r} not in {sorted(allowed)}")
    return problems


def score_sheet(rows: list[dict[str, str]]) -> dict:
    """Aggregate a filled annotation sheet into the numbers `score` prints.

    Reports the label distributions plus, for scan status, how often the human
    label matches the auto heuristic (the one machine prior we seed). There is no
    doc_type-derived bin to compare against: bins are hand-labelled by design.
    """

    total = len(rows)
    bin_labelled = [r for r in rows if (r.get("bin_label") or "").strip()]
    scan_labelled = [r for r in rows if (r.get("scan_label") or "").strip()]
    return {
        "total": total,
        "bin_labelled": len(bin_labelled),
        "bin_dist": dict(Counter(r["bin_label"].strip() for r in bin_labelled)),
        "scan_labelled": len(scan_labelled),
        "auto_scanned": sum(1 for r in rows if r.get("auto_scan") == "scanned"),
        "human_scanned": sum(1 for r in scan_labelled if r["scan_label"].strip() == "scanned"),
        "scan_agree": sum(1 for r in scan_labelled if r["scan_label"].strip() == (r.get("auto_scan") or "").strip()),
        "dominant_visual": dict(Counter(token for r in rows for token in split_multi(r.get("dominant_visual") or ""))),
    }


def cmd_score(args: argparse.Namespace) -> int:
    rows = read_sheet(args.sheet)
    for problem in invalid_values(rows):
        print(f"  ! {problem}")
    s = score_sheet(rows)

    print(f"\nbin_label filled for {s['bin_labelled']}/{s['total']} docs")
    if s["bin_dist"]:
        print("  bin distribution:")
        for label in BIN_LABELS:
            print(f"    {label:<16} {s['bin_dist'].get(label, 0)}")

    print(f"\nscan_label filled for {s['scan_labelled']}/{s['total']} docs")
    print(f"auto scanned fraction: {s['auto_scanned']}/{s['total']} ({_pct(s['auto_scanned'], s['total'])})")
    if s["scan_labelled"]:
        print(f"human scanned fraction: {s['human_scanned']}/{s['scan_labelled']} ({_pct(s['human_scanned'], s['scan_labelled'])})")
        print(f"human vs auto scan agreement: {s['scan_agree']}/{s['scan_labelled']} ({_pct(s['scan_agree'], s['scan_labelled'])})")

    if s["dominant_visual"]:
        print("\ndominant_visual: " + ", ".join(f"{k}={v}" for k, v in sorted(s["dominant_visual"].items())))
    return 0


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa for two raters over single-label categorical judgements.

    `pairs` is a list of (rater_a, rater_b) labels, one per shared item. Returns
    the chance-corrected agreement. When chance agreement is 1 (both raters used a
    single category throughout) kappa is 1.0 if they always agreed, else 0.0.
    """

    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(1 for a, b in pairs if a == b)
    p_o = observed / n
    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    categories = set(count_a) | set(count_b)
    p_e = sum((count_a.get(c, 0) / n) * (count_b.get(c, 0) / n) for c in categories)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def cmd_kappa_sheet(args: argparse.Namespace) -> int:
    """Build a blind subset sheet for a second annotator.

    Samples N documents that the primary sheet has already labelled, and writes a
    fresh sheet with the label columns blanked (only the auto_scan prior is kept),
    so the second pass never sees the first annotator's answers.
    """

    if not args.primary.exists():
        print(f"no primary sheet at {args.primary}; run `annotate` first.")
        return 1
    if args.output.exists() and not args.force:
        print(f"{args.output} already exists; use --force to overwrite the blind subset.")
        return 1

    rows = read_sheet(args.primary)
    labelled = [r for r in rows if all((r.get(f) or "").strip() for f in REQUIRED_FIELDS)]
    if len(labelled) < args.n:
        print(f"only {len(labelled)} fully-labelled docs in {args.primary}; need {args.n}. Annotate more first.")
        return 1

    chosen = random.Random(args.seed).sample(labelled, args.n)
    blind: list[dict[str, str]] = []
    for row in sorted(chosen, key=lambda r: r["doc_id"]):
        blank = {col: row.get(col, "") for col in COLUMNS}
        for field in ("bin_label", "scan_label", "dominant_visual", "notes"):
            blank[field] = ""
        blind.append(blank)
    write_sheet(blind, args.output)
    print(f"wrote a blind {args.n}-doc subset (seed={args.seed}): {args.output}")
    print(f"have a second annotator run: annotate_docs annotate --sheet {args.output}")
    print("then: annotate_docs kappa")
    return 0


def cmd_kappa(args: argparse.Namespace) -> int:
    """Report Cohen's kappa between the primary sheet and a filled subset sheet."""

    if not args.primary.exists():
        print(f"no primary sheet at {args.primary}")
        return 1
    if not args.subset.exists():
        print(f"no subset sheet at {args.subset}; build one with `kappa-sheet` and annotate it.")
        return 1

    primary = {r["doc_id"]: r for r in read_sheet(args.primary)}
    subset = {r["doc_id"]: r for r in read_sheet(args.subset)}
    shared = sorted(set(primary) & set(subset))
    if not shared:
        print("the two sheets share no doc_ids.")
        return 1

    print(f"comparing {len(shared)} shared documents\n")
    exit_code = 0
    for field in REQUIRED_FIELDS:
        labelled = [
            (d, (primary[d].get(field) or "").strip(), (subset[d].get(field) or "").strip())
            for d in shared
        ]
        labelled = [(d, a, b) for d, a, b in labelled if a and b]
        if not labelled:
            print(f"{field}: not enough filled values in both sheets")
            continue
        pairs = [(a, b) for _, a, b in labelled]
        agree = sum(1 for a, b in pairs if a == b)
        kappa = cohen_kappa(pairs)
        gate = "PASS" if kappa >= args.gate else "below gate"
        print(f"{field}: kappa = {kappa:.3f}  ({gate}, gate>={args.gate})")
        print(f"  raw agreement: {agree}/{len(pairs)} ({_pct(agree, len(pairs))})")
        for doc_id, a, b in labelled:
            if a != b:
                print(f"    {doc_id}: primary={a} subset={b}")
        if kappa < args.gate:
            exit_code = 1
    return exit_code


def row_is_annotated(row: dict[str, str]) -> bool:
    """True once every required field has a value (dominant_visual/notes optional)."""

    return all((row.get(field) or "").strip() for field in REQUIRED_FIELDS)


class _Quit(Exception):
    """Raised from a prompt when the user asks to save and exit."""


def _prompt_choice(field: str, options: tuple[str, ...], default: str | None) -> str | None:
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


def _prompt_multi(field: str, options: tuple[str, ...], default: str | None) -> str | None:
    """Prompt a multi-select field. Return a ";"-joined value, or None to skip."""

    current = split_multi(default or "")
    print(f"\n{field} (multiple allowed):")
    for i, option in enumerate(options, 1):
        marker = "  <- current" if option in current else ""
        print(f"  {i}) {option}{marker}")
    while True:
        raw = input("  select [comma-separated numbers / Enter=default / s=skip / q=save+quit]: ").strip().lower()
        if raw == "q":
            raise _Quit
        if raw == "s":
            return None
        if raw == "" and current:
            return MULTI_SEP.join(current)
        tokens = [t for t in re.split(r"[,\s]+", raw) if t]
        if tokens and all(t.isdigit() and 1 <= int(t) <= len(options) for t in tokens):
            chosen: list[str] = []
            for t in tokens:
                option = options[int(t) - 1]
                if option not in chosen:
                    chosen.append(option)
            return MULTI_SEP.join(chosen)
        print("  invalid; enter one or more listed numbers separated by commas")


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


def _annotate_row(row: dict[str, str], *, open_cmd: str | None, do_open: bool, fields: list[tuple[str, tuple[str, ...], str | None]]) -> None:
    """Prompt the given menu fields for one document, updating `row` in place."""

    print("\n" + "=" * 72)
    print(
        f"{row['doc_id']}  |  doc_type={row['doc_type']}  "
        f"auto_scan={row['auto_scan']}  pages={row['page_count']}"
    )
    pdf_path = Path(row["pdf_path"])
    if not pdf_path.is_absolute():
        pdf_path = ROOT / pdf_path
    print(f"  {pdf_path}")
    if do_open:
        open_document(pdf_path, open_cmd)
    for field, options, auto_col in fields:
        current = (row.get(field) or "").strip()
        default = current or (row.get(auto_col, "") if auto_col else "") or None
        prompt = _prompt_multi if field in MULTI_FIELDS else _prompt_choice
        choice = prompt(field, options, default)
        if choice is not None:
            row[field] = choice
    row["notes"] = _prompt_notes((row.get("notes") or "").strip())


def cmd_annotate(args: argparse.Namespace) -> int:
    path = args.sheet
    if not path.exists():
        write_sheet(build_rows(), path)
        print(f"seeded a new sheet: {path}")
    rows = read_sheet(path)

    fields = [f for f in CHOICE_FIELDS if not (args.no_dominant_visual and f[0] == "dominant_visual")]
    pending = rows if args.redo_all else [row for row in rows if not row_is_annotated(row)]
    if not pending:
        print("every document is already annotated. Use --redo-all to revisit them.")
        return 0

    if args.no_dominant_visual:
        print("(dominant_visual prompt disabled)")
    print(f"{len(pending)} of {len(rows)} documents to annotate. Answers save after each doc; 'q' saves and exits.")
    try:
        for row in pending:
            _annotate_row(row, open_cmd=args.open_cmd, do_open=args.open, fields=fields)
            write_sheet(rows, path)
    except (_Quit, KeyboardInterrupt):
        write_sheet(rows, path)
        done = sum(1 for row in rows if row_is_annotated(row))
        print(f"\nsaved {path}  ({done}/{len(rows)} done). Re-run to continue.")
        return 0
    write_sheet(rows, path)
    print(f"\ndone; wrote {path}. Now: python -m ops.scripts.annotate_docs score")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    annotate = sub.add_parser("annotate", help="interactively annotate each document (resumable)")
    annotate.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    annotate.add_argument("--redo-all", action="store_true", help="revisit already-annotated docs too")
    annotate.add_argument("--no-dominant-visual", action="store_true", help="skip the exploratory dominant_visual question")
    annotate.add_argument("--no-open", dest="open", action="store_false", help="do not auto-open the PDF viewer")
    annotate.add_argument("--open-cmd", help="viewer command to open each PDF (default: xdg-open/wslview/open)")

    sheet = sub.add_parser("sheet", help="build the blank (auto-seeded) annotation sheet, no prompting")
    sheet.add_argument("--output", type=Path, default=DEFAULT_SHEET)
    sheet.add_argument("--force", action="store_true", help="overwrite an existing sheet (loses hand labels)")

    score = sub.add_parser("score", help="report label distributions and scan-vs-auto agreement")
    score.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)

    kappa_sheet = sub.add_parser("kappa-sheet", help="build a blind subset sheet for a second annotator")
    kappa_sheet.add_argument("--primary", type=Path, default=DEFAULT_SHEET, help="the already-annotated sheet to sample from")
    kappa_sheet.add_argument("--output", type=Path, default=DEFAULT_KAPPA_SHEET)
    kappa_sheet.add_argument("--n", type=int, default=25, help="number of documents in the subset")
    kappa_sheet.add_argument("--seed", type=int, default=0, help="sampling seed (record it for reproducibility)")
    kappa_sheet.add_argument("--force", action="store_true", help="overwrite an existing subset sheet")

    kappa = sub.add_parser("kappa", help="report Cohen's kappa between the primary and a filled subset sheet")
    kappa.add_argument("--primary", type=Path, default=DEFAULT_SHEET)
    kappa.add_argument("--subset", type=Path, default=DEFAULT_KAPPA_SHEET)
    kappa.add_argument("--gate", type=float, default=0.75, help="agreement bar to flag against (default 0.75)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "annotate":
        return cmd_annotate(args)
    if args.command == "sheet":
        return cmd_sheet(args)
    if args.command == "score":
        return cmd_score(args)
    if args.command == "kappa-sheet":
        return cmd_kappa_sheet(args)
    if args.command == "kappa":
        return cmd_kappa(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
