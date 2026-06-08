"""Stage 1 axis: evidence selection.

Baselines (no-retrieval floor, oracle ceiling) plus the Stage-4 retrievers
(BM25, TF-IDF, dense, ColPali/ColQwen, hybrid) — all behind the
`EvidenceSelector` interface the pipeline consumes.
"""

from .base import (EvidenceSelector, NoRetrieval, Oracle, Retriever,
                   RetrieverSelector, Selection, Unit, build_units)  # noqa: F401
from .factory import build_selector  # noqa: F401
from .evaluate import evaluate_retrieval  # noqa: F401
