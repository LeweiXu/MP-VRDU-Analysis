"""Stage-4 retriever implementations behind the `Retriever` interface.

Build order (plan Stage 4): BM25 -> TF-IDF -> dense -> ColPali -> ColQwen.
Sparse retrievers are pure-CPU and fully local. Dense caches embeddings per doc.
Visual retrievers (ColPali/ColQwen) consume PAGE IMAGES and NEVER the text index
(enforced via modality="visual" -> build_units renders images). Heavy deps are
lazy-imported so the light env imports this module fine.
"""

from __future__ import annotations

import re
from typing import Optional

from ..logging_utils import get_logger
from .base import Retriever, Unit

log = get_logger("retrieve")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _topk(scores, ids, k):
    import numpy as np

    if len(ids) == 0:
        return []
    order = np.argsort(np.asarray(scores))[::-1][:k]
    return [(ids[i], float(scores[i])) for i in order]


def _free_cuda() -> None:
    """Best-effort GPU memory release after unloading a model."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class BM25Retriever(Retriever):
    name = "bm25"
    modality = "text"

    def index(self, units: list[Unit], doc_id: Optional[str] = None) -> None:
        from rank_bm25 import BM25Okapi

        self.ids = [u.unit_id for u in units]
        corpus = [tokenize(u.text or "") for u in units]
        # BM25Okapi needs non-empty corpus; guard empty docs.
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None
        self._corpus_empty = not any(corpus)

    def retrieve(self, query: str, k: int):
        if self._corpus_empty or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        return _topk(scores, self.ids, k)


class TFIDFRetriever(Retriever):
    name = "tfidf"
    modality = "text"

    def index(self, units: list[Unit], doc_id: Optional[str] = None) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.ids = [u.unit_id for u in units]
        texts = [u.text or "" for u in units]
        self._vec = TfidfVectorizer(stop_words="english")
        try:
            self._matrix = self._vec.fit_transform(texts)
            self._ok = True
        except ValueError:
            # empty vocabulary (all-empty/stopword pages)
            self._ok = False

    def retrieve(self, query: str, k: int):
        if not self._ok:
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        qv = self._vec.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        return _topk(sims, self.ids, k)


class DenseRetriever(Retriever):
    """Single-vector dense text retrieval (SentenceTransformers).

    Embeddings are cached per doc_id so re-selecting questions over the same
    document doesn't re-encode (plan Stage 4: cache embeddings).
    """

    name = "dense"
    modality = "text"

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_id = model_id
        self._model = None
        self._cache: dict[str, object] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading dense encoder %s", self.model_id)
            self._model = SentenceTransformer(self.model_id)
        return self._model

    def index(self, units: list[Unit], doc_id: Optional[str] = None) -> None:
        self.ids = [u.unit_id for u in units]
        texts = [u.text or "" for u in units]
        if doc_id is not None and doc_id in self._cache:
            self._emb = self._cache[doc_id]
            return
        emb = self.model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False) if texts else None
        self._emb = emb
        if doc_id is not None:
            self._cache[doc_id] = emb

    def retrieve(self, query: str, k: int):
        if self._emb is None or len(self.ids) == 0:
            return []
        q = self.model.encode([query], normalize_embeddings=True,
                              show_progress_bar=False)[0]
        sims = self._emb @ q  # cosine (normalised)
        return _topk(sims, self.ids, k)

    def unload(self) -> None:
        self._model = None
        self._cache.clear()
        _free_cuda()


class _VisualLateInteraction(Retriever):
    """Shared ColPali/ColQwen late-interaction logic (lazy GPU)."""

    modality = "visual"

    def __init__(self, model_id: str, model_cls_name: str, processor_cls_name: str,
                 device: Optional[str] = None, batch_size: int = 2):
        self.model_id = model_id
        self._model_cls_name = model_cls_name
        self._processor_cls_name = processor_cls_name
        self.device = device
        # encode pages in small batches: the underlying VLM computes full-vocab
        # logits per image, so a whole long doc in one batch OOMs a 12GB card.
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._cache: dict[str, object] = {}

    def _ensure_model(self):
        if self._model is not None:
            return
        import colpali_engine.models as cem
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_cls = getattr(cem, self._model_cls_name)
        proc_cls = getattr(cem, self._processor_cls_name)
        log.info("loading visual retriever %s (%s) on %s",
                 self.model_id, self._model_cls_name, device)
        self._model = model_cls.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=device).eval()
        self._processor = proc_cls.from_pretrained(self.model_id)

    def index(self, units: list[Unit], doc_id: Optional[str] = None) -> None:
        self.ids = [u.unit_id for u in units]
        if doc_id is not None and doc_id in self._cache:
            self._doc_emb = self._cache[doc_id]
            return
        self._ensure_model()
        import torch
        from PIL import Image

        # multi-vector embeddings per page (variable length) -> keep as a list.
        # Encode in small batches so per-call logits stay within GPU memory.
        embs: list = []
        for i in range(0, len(units), self.batch_size):
            chunk = units[i:i + self.batch_size]
            images = [Image.open(u.image_path).convert("RGB") for u in chunk]
            batch = self._processor.process_images(images).to(self._model.device)
            with torch.no_grad():
                out = self._model(**batch)          # [b, seq, dim]
            for j in range(out.shape[0]):
                embs.append(out[j])                 # keep on GPU (compact)
        self._doc_emb = embs
        if doc_id is not None:
            self._cache[doc_id] = embs

    def retrieve(self, query: str, k: int):
        if len(self.ids) == 0:
            return []
        self._ensure_model()
        import torch

        batch = self._processor.process_queries([query]).to(self._model.device)
        with torch.no_grad():
            q_emb = self._model(**batch)            # [1, q_seq, dim]
        scores = self._processor.score_multi_vector(q_emb, self._doc_emb)[0]
        return _topk(scores.float().cpu().numpy(), self.ids, k)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._cache.clear()
        _free_cuda()


class ColPaliRetriever(_VisualLateInteraction):
    name = "colpali"

    def __init__(self, model_id: str = "vidore/colpali-v1.3", device=None):
        super().__init__(model_id, "ColPali", "ColPaliProcessor", device)


class ColQwenRetriever(_VisualLateInteraction):
    name = "colqwen"

    def __init__(self, model_id: str = "vidore/colqwen2.5-v0.2", device=None):
        # colpali_engine exposes ColQwen2_5 / ColQwen2_5_Processor for 2.5.
        super().__init__(model_id, "ColQwen2_5", "ColQwen2_5_Processor", device)


def build_retriever(method: str, cfg) -> Retriever:
    """cfg is a RetrievalConfig. Construct a single (non-hybrid) retriever."""
    if method == "bm25":
        return BM25Retriever()
    if method == "tfidf":
        return TFIDFRetriever()
    if method == "dense":
        return DenseRetriever(
            model_id=cfg.embedding_model or "sentence-transformers/all-MiniLM-L6-v2")
    if method == "colpali":
        return ColPaliRetriever(model_id=cfg.visual_model or "vidore/colpali-v1.3")
    if method == "colqwen":
        return ColQwenRetriever(model_id=cfg.visual_model or "vidore/colqwen2.5-v0.2")
    raise ValueError(f"not a single-retriever method: {method!r}")
