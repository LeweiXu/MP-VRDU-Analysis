"""Runs one PDF parser inside its own isolated env and writes per-page markdown.

Invoked as a subprocess by `tools.parser.warm_parser_cache`; reads a JSON job on
stdin and writes each page's markdown to its `out_path`. Backends import lazily,
and nothing from the project is imported, so a minimal parser env can run it.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _page_image(job: dict, dpi: int) -> str:
    """Return a path to the page image, rendering from the PDF if needed."""

    image_path = job.get("image_path")
    if image_path and Path(image_path).exists():
        return image_path
    import fitz  # PyMuPDF

    doc = fitz.open(job["pdf_path"])
    try:
        page = doc[int(job["index"])]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        out = Path(job["out_path"]).with_suffix(".render.png")
        pix.save(str(out))
    finally:
        doc.close()
    return str(out)


# -- paddleocrvl -------------------------------------------------------------

_PADDLE_VL = None
PADDLE_PIPELINE_VERSION = "v1"  # PaddleOCR-VL-0.9B


def _paddleocrvl(image_path: str, model_id: str) -> str:
    """Run the full PaddleOCR-VL-0.9B page parser.

    ``pipeline_version="v1"`` explicitly selects the original 0.9B pipeline.
    There is deliberately no basic OCR or PP-Structure fallback: selecting
    ``paddleocrvl`` must either use the registered model or fail visibly.
    """

    del model_id  # The Paddle pipeline registry selects its own v1 model bundle.
    global _PADDLE_VL
    if _PADDLE_VL is None:
        from paddleocr import PaddleOCRVL

        _PADDLE_VL = PaddleOCRVL(pipeline_version=PADDLE_PIPELINE_VERSION)
    result = _PADDLE_VL.predict(image_path)
    parts = []
    for res in result:
        # paddleocr 3.7 exposes the page markdown as a computed `.markdown`
        # property (a dict with `markdown_texts`); `.get("markdown")` is not a
        # stored key and returns None, so prefer the property and fall back to the
        # mapping for older layouts.
        md = getattr(res, "markdown", None)
        if md is None and hasattr(res, "get"):
            md = res.get("markdown")
        if isinstance(md, dict):
            md = md.get("markdown_texts") or md.get("text") or md.get("markdown")
        if md:
            parts.append(str(md))
    return "\n\n".join(parts).strip()


# -- transformers VLM parsers (mineru / unlimited) ---------------------------

_HF = {}

_OCR_PROMPT = "Convert this document page to clean Markdown. Output only the Markdown."


def _load_vlm(model_id: str):
    """Load a chat-style OCR VLM with the correct auto-class + processor (mineru).

    MinerU 2.5 is a Qwen2-VL (`Qwen2VLConfig`) served through the standard vision
    auto-classes. `AutoModelForCausalLM` rejects it ("Unrecognized configuration
    class"), so load through the vision auto-classes, newest first, falling back
    across transformers versions. (Unlimited-OCR is NOT loaded here — its custom
    config maps to `AutoModel`, see `_load_unlimited`.)
    """

    import transformers
    from transformers import AutoProcessor

    if model_id in _HF:
        return _HF[model_id]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_kwargs = dict(trust_remote_code=True, torch_dtype="auto", device_map="auto")
    errors = []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **load_kwargs)
            _HF[model_id] = (processor, model)
            return _HF[model_id]
        except Exception as exc:  # noqa: BLE001 - try the next vision auto-class
            errors.append(f"{cls_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"could not load VLM {model_id!r} via a vision auto-class; tried: " + " | ".join(errors))


def _hf_vlm_markdown(image_path: str, model_id: str) -> str:
    """Transformers image->markdown for the chat-style VLM parser backend (mineru).

    Runs a single OCR-to-markdown generation on one page. The stack is pinned in
    its own Kaya env (parse-mineru) and exercised there, not locally.
    """

    import torch
    from PIL import Image

    processor, model = _load_vlm(model_id)

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _OCR_PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    trimmed = generated[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


# -- Unlimited-OCR -----------------------------------------------------------
# Baidu Unlimited-OCR is a DeepSeek-OCR-style model: its custom `UnlimitedOCRConfig`
# is registered (via auto_map) to `AutoModel`, not the image-text-to-text classes,
# and it has its own `model.infer(...)` OCR entry point instead of `.generate()` +
# a chat template. The "gundam" args below (base_size/image_size/crop_mode) are the
# upstream-recommended document-parsing config.

_UNLIMITED_PROMPT = "<image>document parsing."


def _load_unlimited(model_id: str):
    """Load Unlimited-OCR via `AutoModel` + `AutoTokenizer` (trust_remote_code)."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    if model_id in _HF:
        return _HF[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16,
    )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    _HF[model_id] = (tokenizer, model)
    return _HF[model_id]


def _unlimited_markdown(image_path: str, model_id: str) -> str:
    """Unlimited-OCR page->markdown via `model.infer(...)`.

    `infer` returns None in normal mode but writes `{output_path}/result.md` when
    `save_results=True`; we point it at a throwaway dir and read that file back
    (falling back to `eval_mode=True`, which returns the same text, if the file is
    missing).
    """

    import tempfile

    tokenizer, model = _load_unlimited(model_id)
    with tempfile.TemporaryDirectory() as out_dir:
        model.infer(
            tokenizer,
            prompt=_UNLIMITED_PROMPT,
            image_file=image_path,
            output_path=out_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=True,
        )
        result = Path(out_dir) / "result.md"
        if result.exists():
            return result.read_text(encoding="utf-8").strip()
    text = model.infer(
        tokenizer, prompt=_UNLIMITED_PROMPT, image_file=image_path, output_path="",
        base_size=1024, image_size=640, crop_mode=True, max_length=32768,
        no_repeat_ngram_size=35, ngram_window=128, eval_mode=True,
    )
    return (text or "").strip()


def _markdown(parser_tool: str, model_id: str, image_path: str) -> str:
    if parser_tool == "paddleocrvl":
        return _paddleocrvl(image_path, model_id)
    if parser_tool == "unlimited":
        return _unlimited_markdown(image_path, model_id)
    return _hf_vlm_markdown(image_path, model_id)


def main() -> int:
    job_spec = json.loads(sys.stdin.read())
    parser_tool = job_spec["parser_tool"]
    model_id = job_spec["model_id"]
    dpi = int(job_spec.get("dpi", 200))
    written = 0
    for job in job_spec["jobs"]:
        try:
            image_path = _page_image(job, dpi)
            text = _markdown(parser_tool, model_id, image_path)
            Path(job["out_path"]).write_text(text)
            written += 1
        except Exception:  # noqa: BLE001 - one page failing must not sink the batch
            print(f"parser_worker: FAILED {job.get('doc_id')} p{job.get('index')}", file=sys.stderr)
            traceback.print_exc()
    print(f"parser_worker: {parser_tool} wrote {written}/{len(job_spec['jobs'])} pages", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
