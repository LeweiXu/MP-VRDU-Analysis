"""Pull a cached inference cell back together for eyeballing.

Purpose:
    Every number in the paper comes from one cell (question x page-selection x
    representation x model). When a result looks wrong you want to see it: the
    document pages the model was fed, the gold pages, the gold answer, the model's
    answer, and (if judged) the verdict. This module joins the cached
    `predictions.jsonl` (durable GPU output) and `results.jsonl` (judged rows) back
    to the `Question` + PDF and renders a viewing packet. It reads only caches plus
    the dataset and copies/renders files into a viewing dir; it never mutates the
    pipeline caches. One thing it cannot show: the judge's free-text rationale,
    since the orchestrator keeps only the verdict + score in `ResultRow`.

Pipeline role:
    Used by `scripts/inspect_results.py` (browse cells) and the gate-F2
    human-judging packet in `experiments/gates.py`.

Arguments:
    None. Import-only; callers pass an `ExperimentConfig`, a task name, and
    optional filters to `select_items`, then `write_item` / `write_packet`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

from config import ExperimentConfig
from data.loader import load_mmlongbench, resolve_pdf
from data.render import render_pdf
from experiments.paths import experiment_paths
from experiments.tables import load_result_rows
from pipeline.orchestrator import CachedPrediction, PredictionCache, ResultRow
from schema import Question


def _cell_key(record: CachedPrediction | ResultRow) -> tuple[str, str, str, str]:
    """Identity a prediction and its judged row share (independent of the judge)."""

    return (record.question_id, record.condition, record.representation, record.model_spec)


@dataclass(frozen=True)
class InspectItem:
    """One cached cell: the reasoner output, its question, and (if judged) the row."""

    prediction: CachedPrediction
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
        for row in load_result_rows(paths.results):
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


def cell_key(record: CachedPrediction | ResultRow) -> tuple[str, str, str, str]:
    """Public alias: the (question, condition, representation, model) cell identity."""

    return _cell_key(record)


def items_by_cell(config: ExperimentConfig, task: str) -> dict[tuple[str, str, str, str], InspectItem]:
    """All of a task's cells indexed by cell identity, for joining to a sheet."""

    return {_cell_key(item.prediction): item for item in select_items(config, task)}


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
    page generation already rendered is a cache hit, not a re-rasterize.
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

    Dumps *every* field from the generate phase (`CachedPrediction`) and the judge
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

    lines.append("**Generate fields (predictions.jsonl / CachedPrediction):**")
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


def write_packet(
    items: list[InspectItem],
    out_dir: Path,
    config: ExperimentConfig,
    *,
    dpi: int = 144,
    title: str = "Inference cells",
    packet_name: str = "view.md",
) -> Path:
    """Write many cells under `out_dir` plus one combined markdown packet.

    Each cell's images go in `out_dir/<slug>/`; the packet links them so a human
    can scroll one file in VSCode. Returns the packet path.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = [f"# {title}", "", f"{len(items)} cells.", ""]
    for item in items:
        slug = item_slug(item)
        image_names = render_item_assets(item, out_dir / slug, config, dpi)
        blocks.append(item_markdown(item, image_names, image_prefix=f"{slug}/"))
        blocks.append("---")
        blocks.append("")
    packet = out_dir / packet_name
    packet.write_text("\n".join(blocks))
    return packet
