"""Probe: which SDPA backend/dtype avoids the O(seq^2) attention OOM on a V100?

Purpose:
    The 4-bit smoke OOM'd allocating 105 GiB inside `scaled_dot_product_attention`
    on a long text sequence (a dense page's serialized bbox-layout JSON is ~30k
    tokens). The math kernel materializes the full [heads, seq, seq] score matrix;
    the memory-efficient kernel would not. This probe finds, on the actual V100,
    which (compute dtype x forced SDPA backend) combination lets a long-sequence
    generate run without OOM, so the reasoner backend can force it.

Pipeline role:
    Throwaway operational probe. Loads 4-bit 8B, builds a ~14k-token text prompt,
    and runs generate under several forced-backend / dtype combinations, reporting
    OK/peak-VRAM or the failure per combination.

CLI:
    `python -m kaya.kaya submit --gres gpu:v100:1 --time 00:30:00 --mem 48G \
        --job-name attn-probe kaya/attn_probe.py`

Arguments:
    None. SLURM resource overrides go to `kaya.kaya submit` before the script
    path; the probe itself takes no command-line arguments.
"""

# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=attn-probe

from __future__ import annotations

import gc
import time
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
N_WORDS = 14000  # ~14k tokens: math score matrix would be tens of GiB -> OOM


def _load(compute_dtype: Any) -> tuple[Any, Any]:
    import torch
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        quantization_config=quant,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True, trust_remote_code=True)
    return model, processor


def _run(model: Any, processor: Any, label: str, backends: list | None) -> None:
    import torch

    text = "word " * N_WORDS
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], return_tensors="pt").to("cuda:0")
    seq = int(inputs["input_ids"].shape[1])

    ctx: Any
    if backends is None:
        from contextlib import nullcontext

        ctx = nullcontext()
    else:
        from torch.nn.attention import sdpa_kernel

        ctx = sdpa_kernel(backends)

    torch.cuda.reset_peak_memory_stats()
    try:
        start = time.perf_counter()
        with ctx, torch.inference_mode():
            model.generate(**inputs, max_new_tokens=8, do_sample=False)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"    [{label}] seq={seq} OK  {time.perf_counter()-start:5.1f}s  peak={peak:5.2f}GiB", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"    [{label}] seq={seq} FAIL {type(exc).__name__}: {str(exc)[:130]}", flush=True)


def main() -> int:
    import torch
    from torch.nn.attention import SDPBackend

    print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0), flush=True)

    combos = [
        ("default", None),
        ("efficient-only", [SDPBackend.EFFICIENT_ATTENTION]),
        ("efficient+math", [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]),
    ]
    for dtype, dtype_name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16")]:
        print(f"\n=== compute dtype: {dtype_name} ===", flush=True)
        model, processor = _load(dtype)
        for label, backends in combos:
            _run(model, processor, f"{dtype_name}/{label}", backends)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    print("\nattn probe done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
