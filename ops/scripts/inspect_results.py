"""Inspect cached inference results: the document, gold pages, and both answers.

Purpose:
    Pick one or many cached cells for a generation task and dump a viewing packet
    you can open directly in VSCode: the source PDF, the rendered pages the model
    was fed, and an `info.md` with the question, gold pages, gold answer, model
    answer, every cached generate/judge field, and (if the judge phase ran) the
    verdict. Output goes to a gitignored `inspect/` folder at the repo root. It
    reads the prediction/result caches a run produced and never mutates them. One
    thing it cannot show: the judge's free-text rationale, since only the verdict
    and score are cached in `ResultRow`.

Pipeline role:
    A standalone debugging utility. Not part of generate/judge/build.

Arguments:
    `--generation TASK` (required), `--full`, `--run-tag`, and the selectors
    `--question`, `--doc`, `--representation`, `--condition`, `--incorrect-only`,
    `--abstained-only`, `--limit`, `--out`. See `--help`.

Examples:
    python -m ops.scripts.inspect_results --run-tag bf16-lowres --full \
        --generation G1_oracle_ladder --limit 5
    python -m ops.scripts.inspect_results --run-tag bf16-lowres --full \
        --generation G1_oracle_ladder --incorrect-only --limit 20
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import ROOT, ExperimentConfig  # noqa: E402
from data.loader import load_mmlongbench, resolve_pdf  # noqa: E402
from data.render import render_pdf  # noqa: E402
from experiments.engine.paths import experiment_paths  # noqa: E402
from pipeline.orchestrator import PredictionCache  # noqa: E402
from reporting.build import load_result_rows  # noqa: E402
from schema import PredictionRow, Question, ResultRow  # noqa: E402


def _cell_key(record: PredictionRow | ResultRow) -> tuple[str, str, str, str]:
    """Identity a prediction and its judged row share (independent of the judge)."""

    return (record.question_id, record.condition, record.representation, record.model_spec)


@dataclass(frozen=True)
class InspectItem:
    """One cached cell: the reasoner output, its question, and (if judged) the row."""

    prediction: PredictionRow
    question: Question | None
    row: ResultRow | None  # judged result, present only after the judge phase ran

    @property
    def judged(self) -> bool:
        return self.row is not None


def select_items(
    config: ExperimentConfig,
    task: str,
    *,
    question_id: str | None = None,
    doc_id: str | None = None,
    representation: str | None = None,
    condition: str | None = None,
    incorrect_only: bool = False,
    abstained_only: bool = False,
    limit: int | None = None,
) -> list[InspectItem]:
    """Load a task's cached cells, joined to questions and judged rows, filtered.

    Predictions are the durable source; a judged `ResultRow` is attached when
    `results.jsonl` exists. `incorrect_only`/`abstained_only` require the judge
    phase to have run (they filter on the row).
    """

    paths = experiment_paths(config, task)
    if not paths.predictions.exists():
        raise FileNotFoundError(
            f"no predictions for task {task!r} at {paths.predictions} "
            "(run generate first, and check --run-tag / --full)"
        )
    predictions = list(PredictionCache(paths.predictions))
    rows_by_cell: dict[tuple[str, str, str, str], ResultRow] = {}
    if paths.results.exists():
        for raw in load_result_rows(paths.results):
            row = ResultRow.from_dict(raw)
            rows_by_cell.setdefault(_cell_key(row), row)

    questions_by_id = {question.id: question for question in load_mmlongbench(data_dir=config.paths.data_dir)}

    items: list[InspectItem] = []
    for prediction in predictions:
        if question_id is not None and prediction.question_id != question_id:
            continue
        if doc_id is not None and prediction.doc_id != doc_id:
            continue
        if representation is not None and prediction.representation != representation:
            continue
        if condition is not None and prediction.condition != condition:
            continue
        row = rows_by_cell.get(_cell_key(prediction))
        if incorrect_only and (row is None or row.correct):
            continue
        if abstained_only and (row is None or not row.abstained):
            continue
        items.append(InspectItem(prediction, questions_by_id.get(prediction.question_id), row))

    items.sort(key=lambda item: (item.prediction.question_id, item.prediction.representation, item.prediction.condition))
    if limit is not None:
        items = items[:limit]
    return items


def item_slug(item: InspectItem) -> str:
    """A filesystem-safe, readable folder name for one cell."""

    prediction = item.prediction
    qid = prediction.question_id.replace(":", "_")
    return f"{qid}__{prediction.representation}__{prediction.condition}"


def _gold_pages_line(question: Question | None) -> str:
    """Render gold pages as both 0-based (used internally) and 1-based (source)."""

    if question is None:
        return "gold pages: (question not found in corpus)"
    zero_based = list(question.evidence_pages)
    raw = question.raw_fields.get("evidence_pages") if question.raw_fields else None
    suffix = f"; source (1-based): {raw}" if raw is not None else ""
    return f"gold pages (0-based): {zero_based}{suffix}"


def render_item_assets(item: InspectItem, dest: Path, config: ExperimentConfig, dpi: int = 144) -> list[str]:
    """Copy the source PDF and render the fed pages into `dest`. Return image names.

    Renders reuse the run's shared render cache (`config.paths.cache_dir`), so a
    page already rendered by a run is a cache hit, not a re-rasterize.
    """

    dest.mkdir(parents=True, exist_ok=True)
    question = item.question
    if question is None:
        return []
    pdf = resolve_pdf(question.doc_id, config.paths.data_dir)
    shutil.copy2(pdf, dest / pdf.name)
    fed_pages = item.prediction.page_indices or (0,)
    rendered = render_pdf(pdf, fed_pages, cache_dir=config.paths.cache_dir, dpi=dpi)
    image_names: list[str] = []
    for page in rendered:
        if page.image_path is None:
            continue
        name = Path(page.image_path).name
        shutil.copy2(page.image_path, dest / name)
        image_names.append(name)
    return image_names


def _field_lines(obj: object) -> list[str]:
    """One `- key: value` line per dataclass field, in declaration order."""

    return [f"- `{field.name}`: {getattr(obj, field.name)}" for field in fields(obj)]


def item_markdown(item: InspectItem, image_names: Iterable[str], image_prefix: str = "") -> str:
    """Build the markdown block for one cell (VSCode renders the local images).

    Dumps *every* field from the generate phase (`PredictionRow`) and the judge
    phase (`ResultRow`), plus the question context, so nothing is hidden.
    """

    prediction = item.prediction
    question = item.question
    row = item.row
    lines: list[str] = []
    lines.append(f"## {prediction.question_id}  ({prediction.representation} / {prediction.condition})")
    lines.append("")

    if question is not None:
        lines.append("**Question (from corpus):**")
        lines.append(f"- question: {question.question}")
        lines.append(f"- doc_id: {question.doc_id}")
        lines.append(f"- doc_type: {question.doc_type}")
        lines.append(f"- bin_label: {question.bin_label}")
        lines.append(f"- scan_label: {question.scan_label}")
        lines.append(f"- gold answer: {question.gold_answer}")
        lines.append(f"- answer_format: {question.answer_format}")
        lines.append(f"- {_gold_pages_line(question)}")
        lines.append(f"- is_unanswerable: {question.is_unanswerable}")
        lines.append(f"- evidence_sources: {list(question.evidence_sources)}")
        lines.append(f"- hop: {question.hop}")
        lines.append("")
    else:
        lines.append("**Question:** not found in the loaded corpus.")
        lines.append("")

    lines.append("**Generate fields (predictions.jsonl / PredictionRow):**")
    lines.extend(_field_lines(prediction))
    lines.append("")

    if row is not None:
        lines.append("**Judge fields (results.jsonl / ResultRow):**")
        lines.extend(_field_lines(row))
        lines.append("- _the judge's free-text rationale is not persisted; only the verdict/score above is cached._")
        lines.append("")
    else:
        lines.append("**Judge:** not run (no results.jsonl for this task).")
        lines.append("")

    for name in image_names:
        link = f"{image_prefix}{name}" if image_prefix else name
        lines.append(f"![{name}]({link})")
    lines.append("")
    return "\n".join(lines)


def write_item(item: InspectItem, out_root: Path, config: ExperimentConfig, dpi: int = 144) -> Path:
    """Write one cell into `out_root/<slug>/` with the PDF, page PNGs, and info.md."""

    dest = out_root / item_slug(item)
    image_names = render_item_assets(item, dest, config, dpi)
    (dest / "info.md").write_text(item_markdown(item, image_names))
    return dest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation", required=True, help="generation task, e.g. G1_oracle_ladder")
    parser.add_argument("--full", action="store_true", help="read the full-corpus cache (default: smoke)")
    parser.add_argument("--run-tag", help="run tag namespacing the cache tree")
    parser.add_argument("--question", help="filter to one question_id (e.g. mmlongbench:000123)")
    parser.add_argument("--doc", help="filter to one doc_id")
    parser.add_argument("--representation", help="filter to one rung: T/TL/TLV/V")
    parser.add_argument("--condition", help="filter to one conditioner, e.g. oracle")
    parser.add_argument("--incorrect-only", action="store_true", help="only judged-incorrect cells (needs judge phase)")
    parser.add_argument("--abstained-only", action="store_true", help="only abstained cells (needs judge phase)")
    parser.add_argument("--limit", type=int, help="cap the number of cells written")
    parser.add_argument("--out", type=Path, default=ROOT / "inspect", help="output dir (default: ./inspect)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(smoke=not args.full, run_tag=args.run_tag)
    items = select_items(
        config,
        args.generation,
        question_id=args.question,
        doc_id=args.doc,
        representation=args.representation,
        condition=args.condition,
        incorrect_only=args.incorrect_only,
        abstained_only=args.abstained_only,
        limit=args.limit,
    )
    if not items:
        print("no cells matched the filters")
        return 1
    for item in items:
        dest = write_item(item, args.out, config)
        print(f"wrote {dest}")
    print(f"\n{len(items)} cell(s) -> {args.out}  (open the info.md / PNGs in VSCode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
