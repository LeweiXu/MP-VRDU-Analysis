"""Results writer/reader: one JSON object per question, JSONL on disk (Stage 0).

The filename encodes the config hash + a timestamp so runs never clobber each
other and any result file is traceable back to its exact config.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import RunConfig

RESULTS_DIR = Path("results")


def results_path(cfg: RunConfig, results_dir: str | Path = RESULTS_DIR,
                 timestamp: Optional[str] = None) -> Path:
    """Compute the JSONL path for a run: <name>__<hash>__<ts>.jsonl."""
    ts = timestamp or time.strftime("%Y%m%d-%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in cfg.name)
    return Path(results_dir) / f"{safe_name}__{cfg.hash()}__{ts}.jsonl"


class ResultsWriter:
    """Append-only JSONL writer. Writes a header line, then one row per question.

    Each line is flushed immediately so a crashed run still yields partial,
    valid JSONL up to the last completed question.
    """

    def __init__(self, path: str | Path, config: Optional[RunConfig] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._count = 0
        if config is not None:
            # A header row (kind="meta") records the full config inline.
            self.write({
                "kind": "meta",
                "config_hash": config.hash(),
                "config": config.to_dict(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, _is_row=False)

    def write(self, row: dict[str, Any], _is_row: bool = True) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        if _is_row:
            self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "ResultsWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_results(path: str | Path) -> list[dict[str, Any]]:
    """Read all rows (meta + per-question) back from a JSONL file."""
    return list(iter_results(path))


def iter_results(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Per-question rows only (drops the meta header)."""
    return [r for r in iter_results(path) if r.get("kind") != "meta"]
