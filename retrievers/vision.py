"""Vision page retrievers across cost rungs: ColModernVBERT, ColQwen2.5, and
ColQwen3."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from retrievers import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    Retriever,
    normalise_scores,
    rank_pages,
    render_document_pages,
    tokenize,
)
from schema import Question

COLMODERNVBERT_MODEL_ID = "ModernVBERT/colmodernvbert"
COLQWEN25_MODEL_ID = "vidore/colqwen2.5-v0.2"
COLQWEN3_MODEL_ID = "OpenSearch-AI/Ops-Colqwen3-4B"

# How many documents' page-image embeddings to keep on host memory at once. The
# retrieval pass is question-major and questions cluster by document, so a small
# window captures nearly all reuse while bounding host RAM.
PAGE_EMBED_CACHE_DOCS = 8


class ColVisionRetriever(Retriever):
    """Late-interaction page-image retriever (ColBERT-style multi-vector).

    Ranks rendered page images against the query with a ColPali-family model,
    loaded lazily so importing this module pulls no model. An injected `scorer`
    lets a smoke run rank without the heavy model; when neither the model nor a
    scorer is available it falls back to a deterministic text/order heuristic.
    """

    modality = "vision"
    model_id = ""
    # colpali_engine (model_cls, processor_cls) candidates for this rung, tried in
    # order. Same-family names only (naming-convention variants), never a different
    # architecture, so a wrong class raises rather than producing garbage rankings.
    model_classes: tuple[tuple[str, str], ...] = (("ColQwen2_5", "ColQwen2_5_Processor"),)

    def __init__(self, *, data_dir: Path | None = None, cache_dir: Path | None = None, dpi: int = 96,
                 model_id: str | None = None,
                 scorer: Callable[[Question, Sequence[Any]], Sequence[float]] | None = None,
                 allow_text_fallback: bool = True) -> None:
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.dpi = int(dpi)
        self.model_id = model_id or self.model_id
        self.scorer = scorer
        self.allow_text_fallback = bool(allow_text_fallback)
        self._model: Any | None = None
        self._processor: Any | None = None
        # Page-image embeddings are query-independent; keep the last few docs'
        # embeddings on CPU so the k-sweep and a document's questions reuse one
        # image forward pass instead of re-embedding every page per question.
        self._page_emb: "OrderedDict[tuple[str, int], Any]" = OrderedDict()
        # Retrieval cost (pivot 6.3): cumulative page-embed build time, last query time.
        self.index_build_s = 0.0
        self.last_query_s = 0.0

    def _load(self) -> tuple[Any, Any]:
        """Load the ColPali-family model/processor for `model_id` (GPU node).

        Dispatches to the rung's `model_classes` in colpali_engine. A rung whose
        class is absent in the installed colpali_engine raises (rather than silently
        loading a wrong architecture); the retrieval side-artifact catches that and
        records an honest miss for that method.
        """

        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        import colpali_engine.models as cem
        import torch

        cuda = bool(torch.cuda.is_available())
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if cuda else torch.float32,
            "device_map": "cuda:0" if cuda else "cpu",
        }
        errors: list[str] = []
        for model_cls_name, proc_cls_name in self.model_classes:
            model_cls = getattr(cem, model_cls_name, None)
            proc_cls = getattr(cem, proc_cls_name, None)
            if model_cls is None or proc_cls is None:
                errors.append(f"{model_cls_name}/{proc_cls_name}: not in colpali_engine")
                continue
            try:
                self._model = model_cls.from_pretrained(self.model_id, **kwargs).eval()
                self._processor = proc_cls.from_pretrained(self.model_id)
                return self._model, self._processor
            except Exception as exc:  # noqa: BLE001 - try the next candidate name
                errors.append(f"{model_cls_name}/{proc_cls_name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"no colpali_engine class loaded {self.model_id!r} ({self.name}); tried " + "; ".join(errors)
        )

    def unload(self) -> None:
        """Drop the weights/processor so they free the GPU."""

        self._model = None
        self._processor = None

    def _page_embeddings(self, question: Question, pages: Sequence[Any]) -> Any:
        """Return the document's page-image embeddings, cached on CPU per doc."""

        key = (question.doc_id, len(pages))
        cached = self._page_emb.get(key)
        if cached is not None:
            self._page_emb.move_to_end(key)
            return cached
        import torch
        from PIL import Image

        model, processor = self._load()
        images = [Image.open(page.image_path).convert("RGB") for page in pages if page.image_path]
        if len(images) != len(pages):
            raise ValueError("all pages need image_path for vision retrieval")
        start = perf_counter()
        batch_images = processor.process_images(images).to(model.device)
        with torch.no_grad():
            embeddings = model(**batch_images)
        embeddings = embeddings.to("cpu")
        self.index_build_s += perf_counter() - start
        self._page_emb[key] = embeddings
        while len(self._page_emb) > PAGE_EMBED_CACHE_DOCS:
            self._page_emb.popitem(last=False)
        return embeddings

    def _model_scores(self, question: Question, pages: Sequence[Any]) -> list[float]:
        import torch

        model, processor = self._load()
        # Page-embed build is timed inside _page_embeddings (index cost); the query
        # timer below covers only the query encode + multi-vector scoring.
        image_embeddings = self._page_embeddings(question, pages).to(model.device)
        start = perf_counter()
        batch_queries = processor.process_queries([question.question]).to(model.device)
        with torch.no_grad():
            query_embeddings = model(**batch_queries)
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)
        self.last_query_s = perf_counter() - start
        try:
            row = scores[0].tolist()
        except Exception:
            row = scores.tolist()[0]
        return [float(s) for s in row]

    def _fallback_scores(self, question: Question, pages: Sequence[Any]) -> list[float]:
        query = set(tokenize(question.question))
        return [float(len(query.intersection(tokenize(p.text)))) + 1.0 / (1 + int(p.index)) for p in pages]

    def rank(self, question: Question, page_count: int) -> tuple[int, ...]:
        pages = render_document_pages(question, page_count, data_dir=self.data_dir,
                                      cache_dir=self.cache_dir, dpi=self.dpi, render_images=True)
        if not pages:
            self.last_query_s = 0.0
            return ()
        try:
            if self.scorer is not None:
                start = perf_counter()
                raw = list(self.scorer(question, pages))
                self.last_query_s = perf_counter() - start
            else:
                raw = self._model_scores(question, pages)  # sets last_query_s
        except Exception:
            if not self.allow_text_fallback:
                raise
            start = perf_counter()
            raw = self._fallback_scores(question, pages)
            self.last_query_s = perf_counter() - start
        return rank_pages(normalise_scores(raw), page_count)

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        return self.rank(question, page_count)[: int(k)]


class ColModernVbertRetriever(ColVisionRetriever):
    """Cheap vision rung: ColModernVBERT (~250M)."""

    name = "colmodernvbert"
    model_id = COLMODERNVBERT_MODEL_ID
    model_classes = (
        ("ColModernVBert", "ColModernVBertProcessor"),
        ("ColModernVBERT", "ColModernVBERTProcessor"),
        ("ColModernBert", "ColModernBertProcessor"),
    )


class ColQwen25Retriever(ColVisionRetriever):
    """Mid vision rung: ColQwen2.5 (reuses existing caches)."""

    name = "colqwen2.5"
    model_id = COLQWEN25_MODEL_ID
    model_classes = (("ColQwen2_5", "ColQwen2_5_Processor"),)


class ColQwen3Retriever(ColVisionRetriever):
    """Expensive vision rung: ColQwen3-4B (ViDoRe SOTA-class)."""

    name = "colqwen3"
    model_id = COLQWEN3_MODEL_ID
    model_classes = (
        ("ColQwen3", "ColQwen3Processor"),
        ("ColQwen3", "ColQwen3_Processor"),
    )


VISION_RETRIEVERS = {
    "colmodernvbert": ColModernVbertRetriever,
    "colqwen2.5": ColQwen25Retriever,
    "colqwen3": ColQwen3Retriever,
}


def get_vision_retriever(name: str, **kwargs: Any) -> Retriever:
    """Return a vision retriever by cost-rung name."""

    if name not in VISION_RETRIEVERS:
        raise KeyError(f"unknown vision retriever {name!r}; use one of {tuple(VISION_RETRIEVERS)}")
    return VISION_RETRIEVERS[name](**kwargs)
