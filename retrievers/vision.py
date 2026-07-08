"""Vision page retrievers across cost rungs: ColModernVBERT, ColQwen2.5, and
ColQwen3."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
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


class ColVisionRetriever(Retriever):
    """Late-interaction page-image retriever (ColBERT-style multi-vector).

    Ranks rendered page images against the query with a ColPali-family model,
    loaded lazily so importing this module pulls no model. An injected `scorer`
    lets a smoke run rank without the heavy model; when neither the model nor a
    scorer is available it falls back to a deterministic text/order heuristic.
    """

    modality = "vision"
    model_id = ""

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

    def _load(self) -> tuple[Any, Any]:
        """Load the ColPali-family model/processor for `model_id` (GPU node)."""

        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        import torch
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        cuda = bool(torch.cuda.is_available())
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if cuda else torch.float32,
            "device_map": "cuda:0" if cuda else "cpu",
        }
        # The exact model/processor class per repo is confirmed during GPU
        # bring-up; ColQwen2.5 is the one already validated here.
        self._model = ColQwen2_5.from_pretrained(self.model_id, **kwargs).eval()
        self._processor = ColQwen2_5_Processor.from_pretrained(self.model_id)
        return self._model, self._processor

    def unload(self) -> None:
        """Drop the weights/processor so they free the GPU."""

        self._model = None
        self._processor = None

    def _model_scores(self, question: Question, pages: Sequence[Any]) -> list[float]:
        import torch
        from PIL import Image

        model, processor = self._load()
        images = [Image.open(page.image_path).convert("RGB") for page in pages if page.image_path]
        if len(images) != len(pages):
            raise ValueError("all pages need image_path for vision retrieval")
        batch_images = processor.process_images(images).to(model.device)
        batch_queries = processor.process_queries([question.question]).to(model.device)
        with torch.no_grad():
            image_embeddings = model(**batch_images)
            query_embeddings = model(**batch_queries)
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)
        try:
            row = scores[0].tolist()
        except Exception:
            row = scores.tolist()[0]
        return [float(s) for s in row]

    def _fallback_scores(self, question: Question, pages: Sequence[Any]) -> list[float]:
        query = set(tokenize(question.question))
        return [float(len(query.intersection(tokenize(p.text)))) + 1.0 / (1 + int(p.index)) for p in pages]

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        pages = render_document_pages(question, page_count, data_dir=self.data_dir,
                                      cache_dir=self.cache_dir, dpi=self.dpi, render_images=True)
        if not pages:
            return ()
        try:
            raw = list(self.scorer(question, pages)) if self.scorer is not None else self._model_scores(question, pages)
        except Exception:
            if not self.allow_text_fallback:
                raise
            raw = self._fallback_scores(question, pages)
        return rank_pages(normalise_scores(raw), k)


class ColModernVbertRetriever(ColVisionRetriever):
    """Cheap vision rung: ColModernVBERT (~250M)."""

    name = "colmodernvbert"
    model_id = COLMODERNVBERT_MODEL_ID


class ColQwen25Retriever(ColVisionRetriever):
    """Mid vision rung: ColQwen2.5 (reuses existing caches)."""

    name = "colqwen2.5"
    model_id = COLQWEN25_MODEL_ID


class ColQwen3Retriever(ColVisionRetriever):
    """Expensive vision rung: ColQwen3-4B (ViDoRe SOTA-class)."""

    name = "colqwen3"
    model_id = COLQWEN3_MODEL_ID


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
