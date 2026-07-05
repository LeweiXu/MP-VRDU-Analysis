"""Probe: can Qwen3-VL-8B run on a single 16GB V100, and how?

Purpose:
    A one-off feasibility probe (not part of the pipeline) that loads
    Qwen3-VL-8B on ONE V100 in several memory regimes and reports whether each
    fits, its peak VRAM, and per-generation latency for a text-only and an
    image+text prompt. It answers "can the 8B run on a single 16GB GPU, and at
    what cost" so the choice between 1x and 2x V100 for the grid is evidence-based.

    Regimes tried, in order (each guarded so one failure still reports the rest):
      1. 4-bit NF4 (bitsandbytes)   - weights ~5-6GB, no CPU offload
      2. 8-bit    (bitsandbytes)    - weights ~8-9GB, no CPU offload
      3. bf16, single GPU, no offload - the ~16GB baseline, expected to OOM
    The separate `cli.experiments --phase generate --gres gpu:v100:1` job covers the 4th regime
    (bf16 with device_map="auto" CPU offload), so this probe skips it.

Pipeline role:
    None. Throwaway operational probe. Reads only the offline HF cache; writes
    nothing except stdout (the SLURM log) and a peak-memory summary.

CLI:
    `python -m kaya.kaya submit --gres gpu:v100:1 --time 00:30:00 --mem 48G \
        --job-name qwen8b-1gpu-probe scripts/single_gpu_probe.py`

Arguments:
    None. Resource overrides go to `kaya.kaya submit`, before the script path.
"""

# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=qwen8b-1gpu-probe

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 602_112  # 768*28*28, the 8B per-page cap from config.MAX_PIXELS_BY_SIZE
MAX_NEW_TOKENS = 32
PROMPT = "What was the total revenue in 2024? Answer concisely."


def _make_test_image(path: Path) -> Path:
    """Write a small synthetic 'document page' so the vision path is exercised."""

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1280, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 80), "ACME Corp - Annual Report 2024", fill="black")
    draw.text((60, 160), "Total revenue 2024: $12.3M", fill="black")
    draw.text((60, 240), "Total revenue 2023: $9.8M", fill="black")
    image.save(path)
    return path


def _gib(num_bytes: float) -> float:
    return round(num_bytes / (1024**3), 2)


def _generate(model: Any, processor: Any, messages: list[dict], label: str) -> None:
    """Run one generation and print latency + peak VRAM."""

    import torch
    from qwen_vl_utils import process_vision_info

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda:0")

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    trimmed = out[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    peak = torch.cuda.max_memory_allocated()
    print(f"    [{label}] latency={latency:6.1f}s  peak_vram={_gib(peak):5.2f}GiB  answer={answer!r}", flush=True)


def _try_regime(label: str, image_path: Path, *, quant: str | None) -> None:
    """Load 8B under one memory regime and run text + image generations."""

    import torch
    import transformers
    from transformers import AutoProcessor

    print(f"\n=== regime: {label} ===", flush=True)
    total = torch.cuda.mem_get_info("cuda:0")[1]
    print(f"    gpu total={_gib(total)}GiB", flush=True)
    torch.cuda.reset_peak_memory_stats()
    model = None
    try:
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "device_map": {"": 0},  # force single GPU, no CPU/disk offload
            "torch_dtype": torch.bfloat16,
        }
        if quant == "4bit":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quant == "8bit":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)

        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration")
        load_start = time.perf_counter()
        model = model_cls.from_pretrained(MODEL_ID, **kwargs)
        model.eval()
        load_s = time.perf_counter() - load_start
        weights_peak = torch.cuda.max_memory_allocated()
        print(f"    LOADED  load_time={load_s:5.1f}s  weights_peak_vram={_gib(weights_peak)}GiB", flush=True)

        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True, trust_remote_code=True)

        text_msg = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
        _generate(model, processor, text_msg, "text-only")

        image_msg = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path), "max_pixels": MAX_PIXELS},
                {"type": "text", "text": PROMPT},
            ],
        }]
        _generate(model, processor, image_msg, "image+text")
        print(f"    RESULT {label}: FITS", flush=True)
    except Exception as exc:  # noqa: BLE001 - probe reports every failure mode
        kind = type(exc).__name__
        oom = "out of memory" in str(exc).lower() or "CUDA out of memory" in str(exc)
        print(f"    RESULT {label}: FAILED ({'OOM' if oom else kind}): {str(exc)[:200]}", flush=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("NO GPU", flush=True)
        return 1
    print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0), flush=True)
    try:
        import bitsandbytes

        print("bitsandbytes", bitsandbytes.__version__, flush=True)
    except Exception as exc:  # noqa: BLE001
        print("bitsandbytes import failed:", exc, flush=True)

    image_path = _make_test_image(Path("/tmp") / "probe_page.png")

    _try_regime("4bit-nf4", image_path, quant="4bit")
    _try_regime("8bit", image_path, quant="8bit")
    _try_regime("bf16-no-offload", image_path, quant=None)

    print("\nprobe done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
