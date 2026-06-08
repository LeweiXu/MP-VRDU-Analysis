"""Aggregate result JSONL into summary tables + breakdowns (Stage 7).

All metrics are recomputed from the per-question rows (not trusted from a run's
stdout), so the tables are a pure function of the JSONL on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from ..results import iter_results

# config fields surfaced as table columns (dotted paths into the meta config)
CONDITION_FIELDS = [
    "retrieval.method", "retrieval.top_k", "generation.modality",
    "generation.generator", "representation.parser", "representation.chunking",
]


def _get(d: dict, dotted: str, default=None):
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _split_meta_rows(path: str | Path):
    """Return (meta, scored_rows). Only per-question rows with a 'correct' field
    count as scored rows, so recall-only / summary files are ignored cleanly."""
    meta = None
    rows = []
    for r in iter_results(path):
        if r.get("kind") == "meta":
            meta = r
        elif "kind" in r:
            continue                  # recall_summary or other non-scored rows
        elif "correct" in r:
            rows.append(r)
    return meta, rows


def _binary_f1(gold: list[bool], pred: list[bool]) -> float:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_run(path: str | Path) -> Optional[dict]:
    """Summarise one result file: overall + by question-type + by evidence-source.

    Returns None for files with no scored question rows (e.g. recall-only files).
    """
    meta, rows = _split_meta_rows(path)
    if not rows:
        return None
    n = len(rows)
    correct = [bool(r.get("correct")) for r in rows]
    accuracy = _mean([float(c) for c in correct])
    f1 = _binary_f1([bool(r.get("gold_answerable")) for r in rows],
                    [not bool(r.get("pred_abstained")) for r in rows])
    recall = _mean([float(r["recall_at_k"]) for r in rows if "recall_at_k" in r])

    by_type: dict[str, dict] = {}
    for r in rows:
        t = r.get("question_type", "?")
        by_type.setdefault(t, {"n": 0, "correct": 0})
        by_type[t]["n"] += 1
        by_type[t]["correct"] += int(bool(r.get("correct")))
    for t, d in by_type.items():
        d["accuracy"] = d["correct"] / d["n"] if d["n"] else 0.0

    by_source: dict[str, dict] = {}
    for r in rows:
        for src in (r.get("evidence_sources") or ["(none)"]):
            by_source.setdefault(src, {"n": 0, "correct": 0})
            by_source[src]["n"] += 1
            by_source[src]["correct"] += int(bool(r.get("correct")))
    for s, d in by_source.items():
        d["accuracy"] = d["correct"] / d["n"] if d["n"] else 0.0

    cfg = (meta or {}).get("config", {})
    condition = {f: _get(cfg, f) for f in CONDITION_FIELDS}
    return {
        "path": str(path),
        "config_hash": (meta or {}).get("config_hash"),
        "name": cfg.get("name"),
        "condition": condition,
        "n": n,
        "accuracy": accuracy,
        "f1": f1,
        "mean_recall_at_k": recall,
        "by_question_type": by_type,
        "by_evidence_source": by_source,
    }


def aggregate_dir(results_dir: str | Path, pattern: str = "*.jsonl") -> list[dict]:
    """Summarise every result file in a directory, sorted by name then hash."""
    paths = sorted(Path(results_dir).glob(pattern))
    summaries = [s for s in (summarize_run(p) for p in paths) if s is not None]
    summaries.sort(key=lambda s: (s.get("name") or "", s.get("config_hash") or ""))
    return summaries


def to_markdown_table(summaries: Iterable[dict],
                      columns: Optional[list[str]] = None) -> str:
    """Render run summaries as a Markdown table (condition columns + metrics)."""
    summaries = list(summaries)
    cond_cols = columns or CONDITION_FIELDS
    headers = cond_cols + ["n", "accuracy", "f1", "recall@k"]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for s in summaries:
        cells = [str(s["condition"].get(c, "")) for c in cond_cols]
        cells += [str(s["n"]), f"{s['accuracy']:.3f}", f"{s['f1']:.3f}",
                  f"{s['mean_recall_at_k']:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
