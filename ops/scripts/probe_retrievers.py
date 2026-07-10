"""Run every configured retrieval rung offline and report its real ranking."""

from __future__ import annotations

import json
import traceback

from config import DEFAULT_PATHS
from data.loader import load_mmlongbench, resolve_pdf
from data.render import pdf_page_count
from experiments.engine.paths import free_gpu
from retrievers.text import get_text_retriever
from retrievers.vision import get_vision_retriever


TEXT_METHODS = ("bm25", "bge-m3", "qwen3-embedding")
VISION_METHODS = ("colmodernvbert", "colqwen2.5", "colqwen3")


def main() -> int:
    """Probe all six methods and return nonzero unless every one ranks all pages."""

    question = next(
        question for question in load_mmlongbench(DEFAULT_PATHS.data_dir) if not question.is_unanswerable
    )
    page_count = pdf_page_count(resolve_pdf(question.doc_id, DEFAULT_PATHS.data_dir))
    cache_dir = DEFAULT_PATHS.cache_dir / "retriever-probe"
    failures = []

    for kind, names in (("text", TEXT_METHODS), ("vision", VISION_METHODS)):
        for name in names:
            retriever = None
            try:
                kwargs = {
                    "data_dir": DEFAULT_PATHS.data_dir,
                    "cache_dir": cache_dir,
                    "dpi": 200,
                }
                if kind == "text" and name != "bm25":
                    kwargs["allow_bm25_fallback"] = False
                if kind == "vision":
                    kwargs["allow_text_fallback"] = False
                factory = get_text_retriever if kind == "text" else get_vision_retriever
                retriever = factory(name, **kwargs)
                ranking = tuple(retriever.rank(question, page_count))
                if len(ranking) != page_count or set(ranking) != set(range(page_count)):
                    raise RuntimeError(f"invalid {name} ranking of {len(ranking)} pages for {page_count} inputs")
                print(json.dumps({"method": name, "modality": kind, "status": "ok", "top5": ranking[:5]}))
            except Exception as exc:
                failures.append(name)
                traceback.print_exc()
                print(json.dumps({"method": name, "modality": kind, "status": "error", "error": str(exc)}))
            finally:
                if retriever is not None and hasattr(retriever, "unload"):
                    retriever.unload()
                free_gpu()

    print(json.dumps({"passed": 6 - len(failures), "failed": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
