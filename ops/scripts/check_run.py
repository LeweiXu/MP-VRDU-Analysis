"""Post-generation health check: scan a spec's run_tags for failed or missing cells.

Every cell writes exactly one row with a `status` (ok / oom / error), so after a
generate run this reads the predictions.jsonl each run wrote and reports, per run_tag
and task, how many cells are ok vs oom vs error, plus (when the dataset is on hand)
how many are missing versus expected. It exits nonzero if any task looks broken, so
it can gate a run. Read-only: it never touches the caches.

    python -m ops.scripts.check_run --spec ops/specs/h100.yaml
    python -m ops.scripts.check_run --all   # every ops/specs/*.yaml + a summary table
"""

# kaya: target=login
# kaya: env=true
# kaya: offline=true

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SPEC = ROOT / "ops" / "specs" / "h100.yaml"
# Spec files skipped by the --all sweep: templates and smoke-test specs are not real
# runs, so they have no cells to check.
ALL_SKIP_RE = re.compile(r"(template|smoke)", re.IGNORECASE)
# Fraction of oom+error cells at or above which a task is called broken. On an
# H100 a healthy run is ~0; a nonzero cluster usually means a broken parser env or
# a bad model load, not sporadic noise.
DEFAULT_FAIL_RATE = 0.02


def summarize_status(path: Path) -> tuple[Counter, Counter]:
    """Return (status counts, failure-reason histogram) for one predictions.jsonl."""

    counts: Counter = Counter()
    reasons: Counter = Counter()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        status = row.get("status") or "ok"
        counts[status] += 1
        if status != "ok":
            first_line = (row.get("skipped_reason") or "").splitlines()[:1]
            reasons[(first_line[0] if first_line else "")[:100]] += 1
    return counts, reasons


def verdict(counts: Counter, expected: int | None, fail_rate: float) -> tuple[str, str]:
    """Classify a task's rows as OK / WARN / FAIL with a one-line reason."""

    total = sum(counts.values())
    bad = counts.get("oom", 0) + counts.get("error", 0)
    rate = bad / total if total else 0.0
    missing = (expected - total) if (expected is not None and expected > total) else 0
    if total == 0:
        return "FAIL", "no rows written"
    if rate >= fail_rate:
        return "FAIL", f"{bad}/{total} cells failed ({rate:.1%})"
    if missing:
        return "WARN", f"{missing} cells missing (have {total}/{expected})"
    if bad:
        return "WARN", f"{bad}/{total} cells failed ({rate:.1%})"
    return "OK", f"{total} cells ok"


def _expected_rows(config, task, questions, limit) -> int:
    """Cells one task would emit: generation_cells x model_specs x visual_resolutions.

    The driver re-runs the base cells once per resolution (each is a distinct cell key),
    so a multi-resolution spec (e.g. the resolution ladder) emits len(cells) x specs x
    resolutions rows, not just len(cells) x specs.
    """

    from experiments.engine.driver import build_retrievers

    task_questions = list(task.resolve_questions(config, questions))
    if limit is not None:
        task_questions = task_questions[:limit]
    specs = task.model_specs(config)
    if not specs:
        return 0
    resolutions = config.visual_resolutions or (config.visual_resolution,)
    cells = task.generation_cells(config, task_questions, retrievers=build_retrievers(config))
    return len(cells) * len(specs) * len(resolutions)


DETAIL_HEADER = f"{'verdict':7} {'run_tag':24} {'task':20} {'ok':>6} {'oom':>4} {'err':>4} {'miss':>6}  note"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="spec file to check (default: h100.yaml)")
    parser.add_argument("--all", dest="check_all", action="store_true",
                        help="check every ops/specs/*.yaml (skipping *template*/*smoke* specs) "
                             "and print a cross-run summary table")
    parser.add_argument("--fail-rate", type=float, default=DEFAULT_FAIL_RATE,
                        help="oom+error fraction at/above which a task fails (default 0.02)")
    parser.add_argument("--no-expected", action="store_true",
                        help="skip the expected/missing check (report status counts only)")
    parser.add_argument("--show-reasons", type=int, default=5, help="top-N failure reasons to print per task")
    return parser


def report_spec_file(path: Path, questions, args, summary: list) -> bool:
    """Print the detailed per-run table for one spec file and collect summary rows.

    Appends one `(label, run_tag, task, ok, oom, err, done, expected)` tuple per run to
    `summary`. Returns True if any run in the file is a FAIL.
    """

    from experiments.corpus.yaml_spec import config_from_spec, corpus_limit, load_yaml_specs
    from experiments.engine.paths import experiment_paths
    from experiments.registry import resolve

    try:
        specs = load_yaml_specs(path)
    except Exception as exc:  # noqa: BLE001 - a bad spec is reported, not fatal to the sweep
        print(f"[check] {path.name}: could not parse ({type(exc).__name__}: {exc})\n")
        return False

    print(f"[check] {path.name}  ({len(specs)} run(s))")
    print(DETAIL_HEADER)
    print("-" * len(DETAIL_HEADER))

    any_fail = False
    for spec in specs:
        config = config_from_spec(spec)
        limit = corpus_limit(spec)
        for task in resolve(spec.task_name):
            paths = experiment_paths(config, task.name)
            model_specs = task.model_specs(config)
            side_name = ("retrieval.jsonl" if config.text_retrievers
                         else "classifier.jsonl" if config.classifier_spec else None)

            expected = None
            if questions is not None and model_specs:
                try:
                    expected = _expected_rows(config, task, questions, limit)
                except Exception:  # noqa: BLE001 - fall back to status-only for this task
                    expected = None

            if model_specs:
                if not paths.predictions.exists():
                    counts, reasons = Counter(), Counter()
                    label, note = "FAIL", "predictions.jsonl missing (task did not run)"
                else:
                    counts, reasons = summarize_status(paths.predictions)
                    label, note = verdict(counts, expected, args.fail_rate)
            else:
                counts, reasons = Counter(), Counter()
                label, note = "OK", "side-only"

            if side_name:
                side_path = Path(paths.side_dir) / side_name
                if not side_path.exists() or side_path.stat().st_size == 0:
                    label = "FAIL" if not model_specs else ("WARN" if label == "OK" else label)
                    note += f"; side artifact {side_name} missing/empty"

            if label == "FAIL":
                any_fail = True
            total = sum(counts.values())
            miss = (expected - total) if (expected is not None and expected > total) else 0
            print(f"{label:7} {(config.run_tag or '-'):24} {task.name:20} "
                  f"{counts.get('ok', 0):6} {counts.get('oom', 0):4} {counts.get('error', 0):4} {miss:6}  {note}")
            for reason, count in reasons.most_common(args.show_reasons):
                print(f"{'':7} {'':24} {'':20} {'':6} {'':4} {'':4} {'':6}    {count}x  {reason}")
            summary.append((label, config.run_tag or "-", task.name,
                            counts.get("ok", 0), counts.get("oom", 0), counts.get("error", 0), total, expected))
    print()
    return any_fail


def print_summary(summary: list) -> None:
    """Print a compact one-line-per-run summary table across every checked spec."""

    if not summary:
        return
    header = (f"{'verdict':7} {'run_tag':24} {'ok':>7} {'oom':>6} {'err':>5} "
              f"{'done':>7} {'expected':>9} {'done%':>6} {'oom%':>6}")
    print("=" * len(header))
    print("SUMMARY")
    print(header)
    print("-" * len(header))
    for label, run_tag, _task, ok, oom, err, done, expected in summary:
        donep = f"{100 * done / expected:.0f}%" if expected else "-"
        oomp = f"{100 * oom / done:.0f}%" if done else "-"
        exp = str(expected) if expected is not None else "-"
        print(f"{label:7} {run_tag:24} {ok:7} {oom:6} {err:5} {done:7} {exp:>9} {donep:>6} {oomp:>6}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from experiments.corpus.yaml_spec import config_from_spec, load_yaml_specs

    if args.check_all:
        spec_files = [p for p in sorted((ROOT / "ops" / "specs").glob("*.yaml"))
                      if not ALL_SKIP_RE.search(p.name)]
    else:
        spec_files = [args.spec]

    questions = None
    if not args.no_expected:
        try:
            from data.binning import stamp_bins
            from data.loader import load_mmlongbench

            # Any spec gives the data dir; they all read the same MMLongBench corpus.
            base_config = config_from_spec(load_yaml_specs(spec_files[0])[0])
            questions = stamp_bins(load_mmlongbench(base_config.paths.data_dir), require_complete=False)
        except Exception as exc:  # noqa: BLE001 - expected counts are a best-effort extra
            print(f"[check] dataset unavailable for expected counts ({type(exc).__name__}: {exc}); status only\n")

    summary: list = []
    any_fail = False
    for path in spec_files:
        any_fail = report_spec_file(path, questions, args, summary) or any_fail

    if args.check_all or len(spec_files) > 1:
        print_summary(summary)
    print("[check] RESULT:", "FAIL - some tasks look broken (see above)" if any_fail else "OK - no broken tasks")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
