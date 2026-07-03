"""MMLongBench-Doc loader normalising raw benchmark rows into study questions."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from config import DEFAULT_PATHS
from schema import Question


MMLONGBENCH_DIRNAME = "mmlongbench"


def find_mmlongbench_root(data_dir: Path | None = None) -> Path:
    """Return the directory containing MMLongBench `data/` and `documents/`."""

    root = Path(data_dir or DEFAULT_PATHS.data_dir)
    candidates = [root / MMLONGBENCH_DIRNAME, root]
    for candidate in candidates:
        if (candidate / "data").is_dir() and (candidate / "documents").is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not find MMLongBench-Doc under {root}; expected "
        "`mmlongbench/data/*.parquet` and `mmlongbench/documents/*.pdf`"
    )


def parquet_shards(dataset_root: Path) -> list[Path]:
    """Return sorted parquet shards from a staged MMLongBench root."""

    shards = sorted((dataset_root / "data").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards found under {dataset_root / 'data'}")
    return shards


def load_raw_mmlongbench(data_dir: Path | None = None, sample: int | None = None) -> list[dict[str, Any]]:
    """Load raw MMLongBench rows from staged parquet files."""

    import pandas as pd

    dataset_root = find_mmlongbench_root(data_dir)
    frames = [pd.read_parquet(shard) for shard in parquet_shards(dataset_root)]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if sample is not None:
        frame = frame.head(sample)
    return [normalise_raw_value(row) for row in frame.to_dict(orient="records")]


def normalise_raw_value(value: Any) -> Any:
    """Convert pandas/numpy scalars and missing values into plain Python values."""

    if isinstance(value, dict):
        return {str(key): normalise_raw_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise_raw_value(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return normalise_raw_value(value.tolist())
        except Exception:
            pass
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def parse_list_field(value: Any) -> list[Any]:
    """Parse MMLongBench fields stored as real lists or stringified lists."""

    value = normalise_raw_value(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.casefold() in {"none", "nan", "null"}:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            for parser in (ast.literal_eval, json.loads):
                try:
                    parsed = parser(stripped)
                    if isinstance(parsed, Iterable) and not isinstance(parsed, (str, bytes, dict)):
                        return list(parsed)
                    return [parsed]
                except Exception:
                    continue
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return list(value)
    return [value]


def parse_evidence_pages(value: Any) -> tuple[int, ...]:
    """Return zero-based evidence page indices from the one-based source field."""

    pages: list[int] = []
    for item in parse_list_field(value):
        try:
            source_page = int(item)
        except (TypeError, ValueError):
            continue
        pages.append(max(source_page - 1, 0))
    return tuple(dict.fromkeys(pages))


def parse_evidence_sources(value: Any) -> tuple[str, ...]:
    """Return evidence-source labels as strings."""

    return tuple(str(item) for item in parse_list_field(value))


def question_from_row(row: dict[str, Any], index: int) -> Question:
    """Normalise one raw MMLongBench row into a `Question`."""

    raw = normalise_raw_value(row)
    doc_id = str(raw.get("doc_id") or "")
    question = str(raw.get("question") or "")
    answer = str(raw.get("answer") or "")
    evidence_pages = parse_evidence_pages(raw.get("evidence_pages"))
    return Question(
        id=f"mmlongbench:{index:06d}",
        doc_id=doc_id,
        question=question,
        gold_answer=answer,
        answer_format=str(raw.get("answer_format") or ""),
        doc_type=str(raw.get("doc_type") or ""),
        evidence_pages=evidence_pages,
        evidence_sources=parse_evidence_sources(raw.get("evidence_sources")),
        hop="none",
        is_unanswerable=False,
        raw_fields=raw,
    )


def load_mmlongbench(data_dir: Path | None = None, sample: int | None = None) -> list[Question]:
    """Load staged MMLongBench-Doc rows as normalised questions."""

    rows = load_raw_mmlongbench(data_dir, sample)
    return [question_from_row(row, index) for index, row in enumerate(rows)]


def resolve_pdf(doc_id: str, data_dir: Path | None = None) -> Path:
    """Resolve a MMLongBench document id to a staged PDF path."""

    dataset_root = find_mmlongbench_root(data_dir)
    names = [doc_id]
    if not doc_id.lower().endswith(".pdf"):
        names.append(f"{doc_id}.pdf")
    for name in names:
        candidate = dataset_root / "documents" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not resolve MMLongBench PDF for doc_id={doc_id!r}")
