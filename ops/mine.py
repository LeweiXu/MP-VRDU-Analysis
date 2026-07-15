"""Mining entry point: builds the mined_* deployment tables from judged caches.

Reads explicit run_tags (defaults to the canonical set), runs each mined builder,
and writes CSV + a combined markdown under results/tables/mined/, plus a candidates
summary at docs/generated/mined_tables.md. Kept out of the task->table build routing
so a mined table never misfires on the wrong run_tag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import ROOT, ExperimentConfig
from experiments.engine.paths import configure_logging, experiment_paths, log
from reporting.tables import _common as common
from reporting.tables import _markdown as md
from reporting.tables import (
    mined_abstention_by_doctype,
    mined_oom_frontier,
    mined_prefill_cost,
    mined_quant_sensitivity,
    mined_scan_vs_digital,
    mined_vram_headroom,
)

G1 = "G1_oracle_ladder"
G3 = "G3_hallucination"

# The canonical run_tags each mined table reads. Override on the CLI if the tags move.
DEFAULT_TAGS = {
    "representation": "g1-representation-full",
    "resolution": "g1-resolution-full",
    "quantization": "g1-quantization-full",
    "reasoner": "g1-reasoner-full",
    "hallucination": "g3-hallucination-full",
}

# question / caveat metadata for the candidates summary (the user decides inclusion).
SUMMARY = {
    "mined_scan_vs_digital": (
        "Does the cheap T rung collapse on scanned docs, shifting the frontier to TLV/V?",
        "on-thesis deployment finding; scanned T is empty by design.",
    ),
    "mined_prefill_cost": (
        "What does each representation cost to ingest, on the clean (decode-free) axis?",
        "prefill + input tokens only; uncontaminated by the verbose-answer inflation.",
    ),
    "mined_vram_headroom": (
        "How close does each spec/rung/resolution run to the 16 GB V100 ceiling?",
        "peak VRAM is a clean signal; negative headroom means it OOMs a V100.",
    ),
    "mined_quant_sensitivity": (
        "What accuracy and VRAM does 4/8/16-bit cost, per doc_type?",
        "delta vs the 16-bit baseline of the same model; blank when no baseline cached.",
    ),
    "mined_oom_frontier": (
        "Where do cells OOM by rung, resolution, and pages-fed?",
        "from status rows; the empirical 'what fits on a V100' map.",
    ),
    "mined_abstention_by_doctype": (
        "How does abstention on unanswerable questions vary by prompt mode and doc_type?",
        "extends the planned hallucination table with the doc_type axis.",
    ),
}

# Tables that are defined now but cannot be built until a pending generation pass lands.
BLOCKED = [
    ("mined_evidence_survival",
     "Given the gold page was / was not retrieved in the top-k, what is the downstream accuracy?",
     "BLOCKED on the H100 G2 inference remainder; separates retrieval failure from reasoning failure."),
    ("mined_truncation_incidence",
     "Where does the qwen3-embedding retrieval cap still truncate pages, and does it hurt retrieval accuracy?",
     "OPTIONAL/low-value: reads the qwen3-embedding memo rows (seq_len_cap/truncated_pages in "
     "retrievers/text.py), not the eval rows. Ship only if truncation is actually nonzero (it should read zero)."),
]


def _results(run_tag: str, task: str) -> list:
    paths = experiment_paths(ExperimentConfig(run_tag=run_tag), task)
    return common.load_ok_rows(paths.results) if paths.results.exists() else []


def _predictions(run_tag: str, task: str) -> list:
    paths = experiment_paths(ExperimentConfig(run_tag=run_tag), task)
    return common.read_jsonl(paths.predictions)


def build_mined(tags: dict[str, str]) -> tuple[list, dict[str, int]]:
    """Run every mined builder over its source run_tags; return tables + per-table n."""

    rep = _results(tags["representation"], G1)
    quant = _results(tags["quantization"], G1)
    resolution = _results(tags["resolution"], G1)
    g3 = _results(tags["hallucination"], G3)
    # OOM lives in the raw predictions (status rows), pooled across the G1 runs.
    oom_pred: list = []
    for tag in {tags["representation"], tags["resolution"], tags["quantization"], tags["reasoner"]}:
        oom_pred += _predictions(tag, G1)

    specs = [
        (mined_scan_vs_digital.build, rep, len(rep)),
        (mined_prefill_cost.build, rep, len(rep)),
        (mined_vram_headroom.build, rep + resolution + quant, len(rep) + len(resolution) + len(quant)),
        (mined_quant_sensitivity.build, quant + rep, len(quant) + len(rep)),
        (mined_oom_frontier.build, oom_pred, len(oom_pred)),
        (mined_abstention_by_doctype.build, g3, len(g3)),
    ]
    tables = []
    counts: dict[str, int] = {}
    for build_fn, rows, n in specs:
        if not rows:
            log.warning("mine: %s skipped (no source rows yet — judge its run first)",
                        getattr(build_fn, "__module__", build_fn))
            continue
        try:
            table = build_fn(rows)
        except Exception as exc:  # noqa: BLE001 - one bad table must not sink the rest
            log.warning("mine: builder failed (%s): %s", getattr(build_fn, "__module__", build_fn), exc)
            continue
        tables.append(table)
        counts[table.key] = n
    return tables, counts


def write_summary(tables: list, counts: dict[str, int], path: Path) -> None:
    """Write the candidates summary: one line per table (question / n / caveat)."""

    lines = [
        "# Mined tables — candidates",
        "",
        "Generated by `ops/mine.py`. Each surviving candidate answers a deployment "
        "question a reader of an MP-VRDU paper would ask; the user selects which "
        "graduate to the paper. Contaminated axes (decode latency, tokens/answer) are "
        "excluded by design.",
        "",
        "## Built",
        "",
        "| table | question | n (cells) | caveat |",
        "| --- | --- | --- | --- |",
    ]
    for table in tables:
        question, caveat = SUMMARY.get(table.key, ("", ""))
        lines.append(f"| {table.key} | {question} | {counts.get(table.key, 0)} | {caveat} |")
    lines += [
        "",
        "## Defined but blocked / deferred",
        "",
        "| table | question | status |",
        "| --- | --- | --- |",
    ]
    for key, question, status in BLOCKED:
        lines.append(f"| {key} | {question} | {status} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for role, tag in DEFAULT_TAGS.items():
        parser.add_argument(f"--{role}-tag", default=tag, help=f"run_tag for the {role} source (default {tag})")
    parser.add_argument("--out", default=str(ROOT / "results" / "tables" / "mined"),
                        help="output dir for the mined CSV/markdown")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)
    tags = {role: getattr(args, f"{role}_tag") for role in DEFAULT_TAGS}

    tables, counts = build_mined(tags)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for table in tables:
        common.write_csv(table, out / f"{table.key}.csv")
    (out / "all_mined_tables.md").write_text(md.render_report(tables))
    write_summary(tables, counts, ROOT / "docs" / "generated" / "mined_tables.md")

    for table in tables:
        log.info("mine: %s (%d rows, n=%d cells)", table.key, len(table.rows), counts.get(table.key, 0))
    log.info("mine: wrote %d table(s) to %s + candidates summary", len(tables), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
