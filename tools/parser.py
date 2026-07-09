"""PDF-parser layout-rich markdown text for the TL and TLV channels."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from config import DEFAULT_PATHS
from schema import Page

# The parser used when a run does not name one. The parser comparison varies this
# per run; T and V never use it.
DEFAULT_PARSER = "paddleocrvl"
PARSERS = ("paddleocrvl", "mineru", "unlimited")

# Hugging Face ids the isolated parser envs load. Passed to the worker so it does
# not need to import the project config.
PARSER_MODELS = {
    "paddleocrvl": "PaddlePaddle/PaddleOCR-VL",
    "mineru": "opendatalab/MinerU2.5-2509-1.2B",
    "unlimited": "baidu/Unlimited-OCR",
}

_WORKER = Path(__file__).with_name("parser_worker.py")


class ParserUnavailable(RuntimeError):
    """Raised when a parser's isolated env or backend cannot produce markdown.

    Distinct from `ParserCacheMiss`: a miss means nobody warmed the page yet,
    while this means the warm pass tried and could not run the parser (no env, a
    crashing backend). The driver logs it and TL/TLV cells then record a miss.
    """


class ParserCacheMiss(RuntimeError):
    """Raised when a page's parser markdown is not warmed on disk yet.

    The parser and the reasoner never share the GPU, so parser output only ever
    crosses to the reasoner through this disk cache: a run warms the cache in a
    pre-pass (in the parser's isolated env) before the reasoner loads. A miss at
    read time means that pre-pass has not run for this page.
    """


def _safe_stem(name: str) -> str:
    """Filesystem-safe, human-readable stem for a parser cache file."""

    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "document"


def _cache_file(page: Page, parser_tool: str, dpi: int) -> Path:
    """Disk path for one page's cached parser markdown."""

    stem = _safe_stem(Path(page.pdf_path).name)
    return DEFAULT_PATHS.cache_dir / "parser" / parser_tool / f"{stem}__dpi{dpi}__p{page.index:04d}.md"


def cached_markdown(page: Page, parser_tool: str = DEFAULT_PARSER, dpi: int = 144) -> str | None:
    """Return one page's cached parser markdown, or None on a miss."""

    path = _cache_file(page, parser_tool, dpi)
    try:
        return path.read_text() if path.exists() else None
    except OSError:
        return None


def write_markdown(page: Page, text: str, parser_tool: str = DEFAULT_PARSER, dpi: int = 144) -> None:
    """Persist one page's parser markdown (best effort; never raises)."""

    path = _cache_file(page, parser_tool, dpi)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    except OSError:
        pass


def parser_markdown(pages: Sequence[Page], parser_tool: str = DEFAULT_PARSER, dpi: int = 144) -> tuple[str, ...]:
    """Return per-page parser markdown, reading only from the warmed disk cache.

    Raises `ParserCacheMiss` for any page not yet warmed, so the reasoner path
    never triggers a parser model load.
    """

    out: list[str] = []
    for page in pages:
        text = cached_markdown(page, parser_tool, dpi)
        if text is None:
            raise ParserCacheMiss(
                f"no cached {parser_tool} markdown for {page.doc_id} page {page.index} "
                f"(dpi {dpi}); warm the parser cache first"
            )
        out.append(text)
    return tuple(out)


def parser_env_python(parser_tool: str) -> Path:
    """Resolve the Python interpreter for a parser's isolated env.

    Order: a per-parser override (`MPVRDU_PARSER_PYTHON_<TOOL>`), then a shared
    override (`MPVRDU_PARSER_PYTHON`, handy when one local env serves every
    parser), then the conventional layout `envs/parse-<tool>/bin/python`.
    """

    override = os.environ.get(f"MPVRDU_PARSER_PYTHON_{parser_tool.upper()}") or os.environ.get("MPVRDU_PARSER_PYTHON")
    if override:
        return Path(override)
    return DEFAULT_PATHS.env_dir / f"parse-{parser_tool}" / "bin" / "python"


def warm_parser_cache(pages: Sequence[Page], parser_tool: str = DEFAULT_PARSER, dpi: int = 144) -> None:
    """Run the parser over pages in its isolated env and write markdown to cache.

    The parser VLM is heavy and pinned to its own env, so it runs in a subprocess
    (never imported here) that writes each page's markdown to the same disk cache
    `parser_markdown` reads. Pages already cached are skipped, so a run warms each
    page once. Called only in the pre-pass with no reasoner resident, which is
    what keeps parser and reasoner off the GPU together. Raises `ParserUnavailable`
    if the env is missing or the worker fails to write some page.
    """

    if parser_tool not in PARSERS:
        raise ValueError(f"unknown parser {parser_tool!r}; expected one of {PARSERS}")

    missing = [page for page in pages if cached_markdown(page, parser_tool, dpi) is None]
    if not missing:
        return

    python = parser_env_python(parser_tool)
    if not Path(python).exists():
        raise ParserUnavailable(
            f"no parser env python for {parser_tool!r} at {python}; "
            f"set MPVRDU_PARSER_PYTHON_{parser_tool.upper()} (or MPVRDU_PARSER_PYTHON)"
        )

    jobs = []
    for page in missing:
        out = _cache_file(page, parser_tool, dpi)
        out.parent.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "pdf_path": str(page.pdf_path),
                "index": int(page.index),
                "doc_id": page.doc_id,
                "image_path": str(page.image_path) if page.image_path else None,
                "out_path": str(out),
            }
        )

    payload = json.dumps(
        {"parser_tool": parser_tool, "model_id": PARSER_MODELS[parser_tool], "dpi": int(dpi), "jobs": jobs}
    )
    proc = subprocess.run([str(python), str(_WORKER)], input=payload, text=True, capture_output=True)

    unwritten = [job["out_path"] for job in jobs if not Path(job["out_path"]).exists()]
    if unwritten:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise ParserUnavailable(
            f"{parser_tool}: warmed {len(jobs) - len(unwritten)}/{len(jobs)} pages, "
            f"{len(unwritten)} still missing (worker rc={proc.returncode}).\nstderr tail:\n{tail}"
        )
