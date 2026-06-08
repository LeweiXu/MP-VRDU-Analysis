"""Repo-local cache configuration (imported before any HF/torch import).

By default every model/dataset download (HuggingFace hub, torch hub) goes into
``<repo>/.cache`` so nothing large lands in ``$HOME``. Resolution order:

1. If ``HF_HOME`` is already set, it is respected as-is (this is how Kaya batch
   jobs point at ``/group/<project>/hf_cache`` — see kaya_cheatsheet.md).
2. Else if ``MPVRDU_CACHE`` is set, caches go under it.
3. Else they go under ``<repo>/.cache``.

This module is imported for its side effects at the very top of
``mpvrdu/__init__.py``, BEFORE transformers / huggingface_hub are ever imported,
so the environment variables actually take effect.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_cache_env() -> Path:
    """Point HF/torch caches at a single root and return it."""
    cache_root = Path(os.environ.get("MPVRDU_CACHE", REPO_ROOT / ".cache"))
    # setdefault: never override an explicitly-set HF_HOME (Kaya /group case).
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    # sentence-transformers >=3 reads HF_HOME; older versions read this:
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME",
                          str(Path(os.environ["HF_HOME"]) / "sentence-transformers"))
    return cache_root


CACHE_ROOT = setup_cache_env()
HF_HOME = Path(os.environ["HF_HOME"])
