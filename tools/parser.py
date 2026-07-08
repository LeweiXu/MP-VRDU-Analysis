"""PDF-parser layout-rich markdown text for the TL and TLV channels."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from config import DEFAULT_PATHS
from schema import Page

# The parser used when a run does not name one. The parser comparison varies this
# per run; T and V never use it.
DEFAULT_PARSER = "paddleocrvl"
PARSERS = ("paddleocrvl", "mineru", "unlimited")


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


def warm_parser_cache(pages: Sequence[Page], parser_tool: str = DEFAULT_PARSER, dpi: int = 144) -> None:
    """Run the parser over pages in its isolated env and write markdown to cache.

    The parser VLM is heavy and pinned to its own env, so this loads the backend
    lazily (never at import time) and is invoked only in the pre-pass, with no
    reasoner resident. The per-parser backend wiring is exercised on a GPU node.
    """

    if parser_tool not in PARSERS:
        raise ValueError(f"unknown parser {parser_tool!r}; expected one of {PARSERS}")
    raise NotImplementedError(
        f"parser backend for {parser_tool!r} runs in its isolated env on a GPU node; "
        "the per-parser runner is wired during the GPU bring-up step"
    )
