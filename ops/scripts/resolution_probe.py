#!/usr/bin/env python
# kaya: target=gpu offline=true
"""Find the highest visual-resolution preset that fits a V100 at the worst case.

Sweeps the named resolution presets on the V rung (image-only, so the result is
parser-independent) at the worst-case page count and reports, per preset, the
peak per-GPU VRAM and whether it OOMs. The highest preset that runs without OOM
becomes the single fixed deployment resolution used by every table except the
scientific resolution sweep.

Why V-rung image-only: with the input-token cap gone, vision-token volume
(resolution x page count) is the binding VRAM constraint. An image-only probe is
the honest floor and does not depend on the parser text path.

Run on Kaya:
    python -m ops.kaya.kaya run --target gpu \
        --gres gpu:v100:2 --time 00:30:00 ops/scripts/resolution_probe.py

The 8B primary reasoner needs 2 V100s (bf16 weights ~16GB), so request 2 GPUs;
the probe mirrors production loading (device_map=auto, per-GPU headroom reserve,
memory-efficient SDPA kernel). Results land in results/probes/resolution_probe.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Preset -> per-page pixel cap (tokens_per_page * 28 * 28). Mirrors
# config.VISUAL_RESOLUTION_PRESETS; hardcoded fallback keeps the probe runnable
# even if config imports drag in unstaged deps.
try:
    from config import VISUAL_RESOLUTION_PRESETS as PRESETS  # type: ignore
except Exception:
    PRESETS = {
        "full": 1280 * 28 * 28,
        "high": 768 * 28 * 28,
        "med": 512 * 28 * 28,
        "low": 320 * 28 * 28,
        "min": 224 * 28 * 28,
    }

# The 16GB deployment-target line (one V100). A preset whose worst per-GPU peak
# stays under this is "deployable on a modest 16GB target"; the operational
# verdict is the highest preset that simply does not OOM on the granted GPUs.
TARGET_GIB = 16.0


def build_pages(n: int, width: int, height: int, out_dir: Path):
    """Write n RGB noise PNG pages (sized above the 'full' cap so smart_resize
    downscales each to the preset's budget) and return their paths.

    Feeding file paths mirrors the production image block exactly
    (`{"type": "image", "image": str(path), "max_pixels": ...}`), so the probe
    exercises the same qwen_vl_utils path the real reasoner uses."""
    import numpy as np
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        arr = rng.integers(0, 256, size=(height, width, 3), dtype="uint8")
        p = out_dir / f"page_{i:02d}.png"
        Image.fromarray(arr, mode="RGB").save(p)
        paths.append(str(p))
    return paths


def messages_for(pages, max_pixels: int):
    """V-rung, image-only message: all pages + one short question."""
    content = [{"type": "image", "image": p, "max_pixels": int(max_pixels)} for p in pages]
    content.append({"type": "text", "text": "Summarize what these pages contain in one sentence."})
    return [{"role": "user", "content": content}]


def max_memory_map(torch):
    """Reserve ~5GiB/GPU headroom for activations when sharding, matching
    production loading. Single-GPU returns None (use the GPU fully)."""
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return None
    mapping = {}
    for i in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        mapping[i] = f"{max(4, int(total_gib - 5))}GiB"
    return mapping


def sdpa_context(torch):
    """Prefer the memory-efficient SDPA kernel (Volta has no FlashAttention-2)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        from contextlib import nullcontext

        return nullcontext()
    return sdpa_kernel(
        [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
    )


def vision_token_count(inputs) -> int | None:
    """Best-effort: vision tokens from image_grid_thw / merge_size**2."""
    grid = None
    if hasattr(inputs, "get"):
        grid = inputs.get("image_grid_thw")
    if grid is None:
        return None
    try:
        import torch

        prod = int(torch.prod(grid, dim=1).sum().item())
        # Qwen merges 2x2 patches into one token by default.
        return prod // 4
    except Exception:
        return None


def load_model(model_id: str):
    import torch
    import transformers
    from transformers import AutoProcessor

    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForImageTextToText")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_kwargs = {"torch_dtype": "auto", "device_map": "auto", "trust_remote_code": True}
    mm = max_memory_map(torch)
    if mm is not None:
        load_kwargs["max_memory"] = mm
    model = model_cls.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return processor, model


def run_preset(name, max_pixels, pages, processor, model, max_new_tokens):
    import torch
    from qwen_vl_utils import process_vision_info

    n_dev = torch.cuda.device_count()
    for i in range(n_dev):
        torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.empty_cache()

    messages = messages_for(pages, max_pixels)
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    device = getattr(model, "device", None)
    if device is not None and hasattr(inputs, "to"):
        inputs = inputs.to(device)

    vis_tokens = vision_token_count(inputs)
    input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else None

    result = {
        "preset": name, "max_pixels": int(max_pixels), "pages": len(pages),
        "vision_tokens": vis_tokens, "input_len": input_len,
    }
    try:
        start = time.perf_counter()
        with sdpa_context(torch), torch.inference_mode():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        result["latency_s"] = round(time.perf_counter() - start, 3)
        peaks = [torch.cuda.max_memory_allocated(i) / (1024**3) for i in range(n_dev)]
        result["status"] = "ok"
        result["per_gpu_peak_gib"] = [round(p, 2) for p in peaks]
        result["max_peak_gib"] = round(max(peaks), 2)
        result["fits_16gib"] = max(peaks) <= TARGET_GIB
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        result["status"] = "oom"
        result["error"] = str(exc)[:200]
        torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--pages", type=int, default=10, help="worst-case page count")
    ap.add_argument("--page-width", type=int, default=1240)
    ap.add_argument("--page-height", type=int, default=1754)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--presets", nargs="*", default=None, help="subset; default all")
    ap.add_argument("--out", default="results/probes/resolution_probe.json")
    args = ap.parse_args()

    import torch

    names = args.presets or list(PRESETS.keys())
    # Ascending pixel budget: small first, so an OOM localizes the boundary.
    names = sorted(names, key=lambda n: PRESETS[n])

    print(f"[probe] model={args.model_id} gpus={torch.cuda.device_count()} "
          f"pages={args.pages} presets={names}", flush=True)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"[probe] gpu{i}: {p.name} {p.total_memory/(1024**3):.1f}GiB", flush=True)

    pages = build_pages(
        args.pages, args.page_width, args.page_height,
        REPO_ROOT / "results" / "probes" / "pages",
    )
    processor, model = load_model(args.model_id)

    results = []
    for name in names:
        r = run_preset(name, PRESETS[name], pages, processor, model, args.max_new_tokens)
        results.append(r)
        print(f"[probe] {name:>5}: {r.get('status')} "
              f"peak={r.get('max_peak_gib')}GiB vis_tok={r.get('vision_tokens')} "
              f"lat={r.get('latency_s')}s", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    fit = [r for r in ok if r.get("fits_16gib")]
    highest_ok = max(ok, key=lambda r: r["max_pixels"])["preset"] if ok else None
    highest_fit = max(fit, key=lambda r: r["max_pixels"])["preset"] if fit else None

    verdict = {
        "model_id": args.model_id,
        "pages": args.pages,
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_properties(0).name if torch.cuda.device_count() else None,
        "target_gib": TARGET_GIB,
        "highest_preset_no_oom": highest_ok,
        "highest_preset_under_16gib": highest_fit,
        "deployment_resolution": highest_fit or highest_ok,
        "results": results,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2))
    print(f"\n[probe] VERDICT deployment_resolution={verdict['deployment_resolution']} "
          f"(no-oom={highest_ok}, under-16GiB={highest_fit})", flush=True)
    print(f"[probe] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
