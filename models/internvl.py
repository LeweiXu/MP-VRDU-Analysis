"""InternVL reasoner backend."""

from __future__ import annotations

import os
import time
from io import BytesIO
from typing import Any

from models.payload import IMAGE_PLACEHOLDER, ModelInput
from pipeline.reasoner import Reasoner
from schema import ImagePart, Prediction, Question


MODEL_IDS: dict[str, str] = {
    "internvl3-8b-local": "OpenGVLab/InternVL3-8B",
}

from config import DEFAULT_PROMPT_MODE, PROMPT_MODES

PROMPT_TEMPLATE_VERSION = "internvl3-v1"
PROMPT_HEADER = "You are answering a question about a document."
PROMPT_BODY = "Question:\n{question}\n\nDocument evidence:\n{context}\n\nAnswer:"
DEFAULT_INSTRUCTION = PROMPT_MODES[DEFAULT_PROMPT_MODE]

# InternVL encodes a 448px tile as (448 / 14) ** 2 = 1024 patches, then pixel-unshuffles
# by 0.5, so one tile costs (448 / 14 * 0.5) ** 2 = 256 vision tokens. A page becomes
# one or more tiles (see `dynamic_preprocess`), so the per-page cost is that times the
# tile count rather than a fixed figure.
INTERNVL_PATCH_SIZE = 14
INTERNVL_DOWNSAMPLE_RATIO = 0.5
INTERNVL_TOKENS_PER_TILE = int((448 // INTERNVL_PATCH_SIZE * INTERNVL_DOWNSAMPLE_RATIO) ** 2)


def model_id_for_spec(spec: str) -> str:
    """Return the Hugging Face model id for an InternVL model spec."""

    try:
        return MODEL_IDS[spec]
    except KeyError as exc:
        raise ValueError(f"unsupported InternVL spec {spec!r}") from exc


def hf_cache_dir_from_env() -> str | None:
    """Return the configured Hugging Face cache directory."""

    for name in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def render_prompt(
    question: Question, model_input: ModelInput, instruction: str | None = None
) -> tuple[str, tuple[ImagePart, ...]]:
    """Render the prompt with a chosen instruction preamble (full context).

    `instruction` is the abstention/guidance preamble; None uses the default
    (abstention-targeted) instruction, and an empty string gives no guidance.
    """

    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    context, images = model_input.to_local_prompt()
    context = context.strip() or "(no document evidence was provided)"
    header = PROMPT_HEADER if not instruction.strip() else f"{PROMPT_HEADER}\n{instruction.strip()}"
    text = f"{header}\n\n" + PROMPT_BODY.format(question=question.question.strip(), context=context)
    return text, images


def _count_text_tokens(tokenizer: Any, text: str) -> int:
    """Best-effort tokenizer count with whitespace fallback."""

    try:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return len(input_ids[0]) if input_ids and isinstance(input_ids[0], list) else len(input_ids)
    except Exception:
        return max(1, len(text.split()))


# Tile budget per resolution preset. These are NOT pixel-matched to the Qwen budgets
# the presets are written as: a tile covers 448**2 px, so matching Qwen's pixel budget
# buys 2-4 tiles, and at that count the closest grid to a page's ~0.77 aspect ratio is
# still 1x1 — the squashed square this whole path exists to avoid. Six tiles is the
# first budget that reaches 2x3, the natural portrait grid, so `med` is anchored there
# and the ladder is built around it. The consequence is that InternVL sees more visual
# tokens per page than Qwen at the same preset name (~1536 vs ~450 at med), so read the
# ladder as a fidelity ladder, not a cost-matched one, and compare cost via the
# recorded `total_visual_tokens` rather than by preset name.
TILES_BY_PIXEL_BUDGET: tuple[tuple[int, int], ...] = (
    (313600, 3),   # low
    (501760, 6),   # med (deployment default)
    (752640, 9),   # high
)
DEFAULT_MAX_TILES = 6


def tiles_for_budget(max_pixels: int | None, *, image_size: int) -> int:
    """Tile budget for a per-page pixel budget, via `TILES_BY_PIXEL_BUDGET`.

    An unrecognised budget falls to the nearest preset rather than a made-up tile
    count, so a new resolution preset degrades to the closest known rung instead of
    silently reverting to a single squashed tile.
    """

    if not max_pixels:
        return DEFAULT_MAX_TILES
    budget = int(max_pixels)
    return min(TILES_BY_PIXEL_BUDGET, key=lambda pair: abs(pair[0] - budget))[1]


def _closest_aspect_ratio(aspect_ratio: float, candidates: list[tuple[int, int]]) -> tuple[int, int]:
    """The (cols, rows) tiling whose aspect ratio is nearest the image's."""

    return min(candidates, key=lambda r: (abs(aspect_ratio - r[0] / r[1]), -(r[0] * r[1])))


def dynamic_preprocess(image: Any, *, image_size: int, max_tiles: int, use_thumbnail: bool = True) -> list[Any]:
    """Split a page into aspect-preserving tiles, the way InternVL expects to see it.

    A document page is tall and narrow; squashing it into one square tile costs both
    resolution and geometry, which is most of what makes page text legible. So pick
    the (cols, rows) tiling closest to the page's own aspect ratio within the tile
    budget, resize to exactly that grid, and cut it up. A multi-tile page also gets a
    whole-page thumbnail appended so the model keeps global layout alongside the
    detail crops, which is InternVL's own convention.
    """

    width, height = image.size
    aspect_ratio = width / height if height else 1.0
    candidates = sorted(
        {(cols, rows) for n in range(1, max_tiles + 1) for cols in range(1, n + 1)
         for rows in range(1, n + 1) if cols * rows <= max_tiles},
        key=lambda r: r[0] * r[1],
    )
    cols, rows = _closest_aspect_ratio(aspect_ratio, candidates)
    resized = image.resize((image_size * cols, image_size * rows))
    tiles = [
        resized.crop((c * image_size, r * image_size, (c + 1) * image_size, (r + 1) * image_size))
        for r in range(rows)
        for c in range(cols)
    ]
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _to_tensor(tile: Any) -> Any:
    """Normalize one 448px tile into an InternVL-compatible tensor."""

    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform(tile).unsqueeze(0).to(torch.bfloat16)


def _open_image(part: ImagePart) -> Any:
    from PIL import Image

    source = part.image_path if part.image_path else BytesIO(part.read_bytes())
    return Image.open(source).convert("RGB")


def _image_tensors(
    parts: tuple[ImagePart, ...], *, image_size: int, max_tiles: int, device: Any
) -> tuple[Any, list[int]]:
    """Tiled pixel values plus the per-image tile counts (`None, []` for text-only).

    The tile counts are what `chat()` needs to bind each `<image>` placeholder to its
    own run of tiles, so they must stay aligned with the order of `parts`.
    """

    if not parts:
        return None, []
    import torch

    tensors: list[Any] = []
    counts: list[int] = []
    for part in parts:
        tiles = dynamic_preprocess(_open_image(part), image_size=image_size, max_tiles=max_tiles)
        tensors += [_to_tensor(tile) for tile in tiles]
        counts.append(len(tiles))
    pixel_values = torch.cat(tensors, dim=0)
    return (pixel_values.to(device) if device is not None else pixel_values), counts


class InternVLBackend(Reasoner):
    """InternVL local backend using the model's `chat()` helper."""

    #: Recorded per cell; a subclass with its own prompt assembly overrides both
    #: `render` and this id so its cells are distinguishable in metadata.
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION

    def __init__(
        self,
        spec: str,
        *,
        model_id: str | None = None,
        max_new_tokens: int = 64,
        image_size: int = 448,
        max_pixels: int | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.spec = spec
        self.model_id = model_id or model_id_for_spec(spec)
        self.max_new_tokens = int(max_new_tokens)
        self.image_size = int(image_size)
        # Read fresh on every answer(): the driver reuses one loaded backend across
        # a resolution sweep and rebinds this attribute per resolution.
        self.max_pixels = int(max_pixels) if max_pixels is not None else None
        self._tokenizer = tokenizer
        self._model = model
        self.local_files_only = (
            any(os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
            if local_files_only is None
            else bool(local_files_only)
        )
        self.cache_dir = hf_cache_dir_from_env()

    def _load_components(self) -> tuple[Any, Any]:
        """Load and cache the tokenizer/model pair."""

        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        import torch
        from transformers import AutoModel, AutoTokenizer

        common_kwargs = {"trust_remote_code": True, "local_files_only": self.local_files_only}
        if self.cache_dir:
            common_kwargs["cache_dir"] = self.cache_dir
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False, **common_kwargs)
        model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
            **common_kwargs,
        ).eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def free(self) -> None:
        """Drop the loaded model/tokenizer so they release GPU memory."""

        self._model = None
        self._tokenizer = None

    def render(self, question: Question, model_input: ModelInput):
        """Assemble this backend's prompt; the per-model override point."""

        return render_prompt(question, model_input, self.prompt_instruction)

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        tokenizer, model = self._load_components()
        prompt, images = self.render(question, model_input)
        device = getattr(model, "device", None)
        max_tiles = tiles_for_budget(self.max_pixels, image_size=self.image_size)
        pixel_values, tile_counts = _image_tensors(
            images, image_size=self.image_size, max_tiles=max_tiles, device=device
        )
        generation_config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        text_prompt = prompt.replace(IMAGE_PLACEHOLDER, "<image>")
        num_patches_list = tile_counts or None

        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        try:
            answer = model.chat(
                tokenizer,
                pixel_values,
                text_prompt,
                generation_config,
                num_patches_list=num_patches_list,
                return_history=False,
            )
        except TypeError:
            answer = model.chat(tokenizer, pixel_values, text_prompt, generation_config)
        if cuda:
            torch.cuda.synchronize()
        latency_s = time.perf_counter() - start
        peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if cuda else 0

        answer_text = str(answer[0] if isinstance(answer, tuple) else answer).strip()
        total_text_tokens = _count_text_tokens(tokenizer, prompt.replace(IMAGE_PLACEHOLDER, ""))
        return Prediction(
            text=answer_text,
            model_spec=self.spec,
            total_text_tokens=total_text_tokens,
            total_visual_tokens=sum(tile_counts) * INTERNVL_TOKENS_PER_TILE,
            text_tokens_fed=total_text_tokens,  # no cap: everything is fed
            output_tokens=max(1, len(answer_text.split())),
            latency_s=latency_s,
            # chat() does not expose a prefill/decode boundary, so the split is
            # left at zero and only the end-to-end latency is recorded.
            prefill_latency_s=0.0,
            decode_latency_s=0.0,
            peak_vram_bytes=peak_vram_bytes,
            metadata={
                "backend": "hf-transformers-internvl-chat",
                "model_id": self.model_id,
                "prompt_template_version": self.prompt_template_version,
                "max_new_tokens": self.max_new_tokens,
                # No output_truncated here: chat() returns text only, so
                # output_tokens is a whitespace estimate and comparing it to the
                # budget would misreport. Absent = unmeasured (like the prefill
                # split), never false.
                "max_pixels": self.max_pixels,
                "max_tiles": max_tiles,
                "tiles_per_image": tile_counts,
                "n_image_parts": len(images),
                "local_files_only": self.local_files_only,
                "cache_dir": self.cache_dir,
            },
        )
