"""Runs one PDF parser inside its own isolated env and writes per-page markdown.

Invoked as a subprocess by `tools.parser.warm_parser_cache`; reads a JSON job on
stdin and writes each page's markdown to its `out_path`. Backends import lazily,
and nothing from the project is imported, so a minimal parser env can run it.
"""

from __future__ import annotations

import json
import os
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

_PADDLE = None


def _paddle_basic(image_path: str) -> str:
    """Read-order text join from PaddleOCR det+rec (offline-friendly floor)."""

    global _PADDLE
    if _PADDLE is None:
        from paddleocr import PaddleOCR

        _PADDLE = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )
    result = _PADDLE.predict(image_path)
    if not result:
        return ""
    res = result[0]
    texts = res.get("rec_texts") if hasattr(res, "get") else None
    return "\n".join(t for t in (texts or []) if t).strip()


def _paddle_structure(image_path: str) -> str:
    """PP-StructureV3 layout-aware markdown (needs the layout models present)."""

    from paddleocr import PPStructureV3

    pipeline = PPStructureV3()
    result = pipeline.predict(image_path)
    parts = []
    for res in result:
        md = res.get("markdown") if hasattr(res, "get") else getattr(res, "markdown", None)
        if isinstance(md, dict):
            md = md.get("markdown_texts") or md.get("text")
        if md:
            parts.append(str(md))
    return "\n\n".join(parts).strip()


def _paddle_vl(image_path: str, model_id: str) -> str:
    """PaddleOCR-VL end-to-end VLM parser markdown (richer envs only)."""

    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL()
    result = pipeline.predict(image_path)
    parts = []
    for res in result:
        md = res.get("markdown") if hasattr(res, "get") else getattr(res, "markdown", None)
        if isinstance(md, dict):
            md = md.get("markdown_texts") or md.get("text")
        if md:
            parts.append(str(md))
    return "\n\n".join(parts).strip()


def _paddleocrvl(image_path: str, model_id: str) -> str:
    """Best available paddle markdown. The richer VLM / structure paths need
    extra models, so they are gated behind MPVRDU_PADDLE_RICH; the det+rec floor
    runs anywhere the base PaddleOCR weights are staged."""

    if os.environ.get("MPVRDU_PADDLE_RICH"):
        for attempt in (lambda p: _paddle_vl(p, model_id), _paddle_structure):
            try:
                text = attempt(image_path)
                if text:
                    return text
            except Exception:  # noqa: BLE001 - cascade to the offline floor
                traceback.print_exc()
    return _paddle_basic(image_path)


# -- transformers VLM parsers (mineru / unlimited) ---------------------------

_HF = {}

_OCR_PROMPT = "Convert this document page to clean Markdown. Output only the Markdown."


def _hf_vlm_markdown(image_path: str, model_id: str) -> str:
    """Generic transformers image->markdown for the VLM parser backends.

    MinerU 2.5 and Unlimited-OCR are both VLMs loaded via transformers; this runs
    a single OCR-to-markdown generation. Verified on the Kaya parser envs (each
    heavy stack is pinned there); not exercised locally.
    """

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    if model_id not in _HF:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )
        _HF[model_id] = (processor, model)
    processor, model = _HF[model_id]

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _OCR_PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    trimmed = generated[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def _markdown(parser_tool: str, model_id: str, image_path: str) -> str:
    if parser_tool == "paddleocrvl":
        return _paddleocrvl(image_path, model_id)
    return _hf_vlm_markdown(image_path, model_id)


def main() -> int:
    job_spec = json.loads(sys.stdin.read())
    parser_tool = job_spec["parser_tool"]
    model_id = job_spec["model_id"]
    dpi = int(job_spec.get("dpi", 144))
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
