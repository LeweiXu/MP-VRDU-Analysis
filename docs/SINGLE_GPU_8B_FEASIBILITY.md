# Can Qwen3-VL-8B run on a single 16GB V100?

Written 2026-07-05. Evidence from probe job `1003970` (`scripts/single_gpu_probe.py`)
on one `Tesla V100-PCIE-16GB` (15.77 GiB usable), transformers 4.57.6, torch
2.7.0+cu126, bitsandbytes 0.49.2.

## Short answer

**Yes, but only if you quantize (or CPU-offload).** Full bf16 does not fit; 4-bit
and 8-bit both fit comfortably with correct answers. The real win of going to one
GPU is not memory or speed, it is **queue time**: 1-GPU jobs backfill in minutes,
while a 2-GPU node took an overnight wait. The real cost is **fidelity**:
quantized weights are not the pre-registered bf16 the paper's numbers are defined
on.

## What the probe measured

Each regime loads the real 8B onto one GPU (`device_map={"":0}`, no offload),
then runs a text-only and an image+text generation (32 new tokens, 8B pixel cap
602,112).

| Regime | Fits 16GB? | Weights VRAM | Peak VRAM | Load | text gen | image+text gen | Answer (image) |
|---|---|---|---|---|---|---|---|
| **4-bit NF4** (bnb) | ✅ yes | 7.13 GiB | 7.13 GiB | 31s | 2.4s | 32.4s | `$12.3M` ✓ correct |
| **8-bit** (bnb) | ✅ yes | 9.42 GiB | 10.16 GiB | 23s | 1.3s | 31.8s | `$13.3M` ✗ off by a digit |
| **bf16, no offload** | ❌ **OOM** | ~16 GiB | — | — | — | — | (0 MiB free at load) |

The synthetic test page said "Total revenue 2024: $12.3M". 4-bit read it right;
8-bit misread one digit. That is a single anecdotal cell, not a benchmark, but it
is a concrete reminder that quantization changes outputs.

Not tested empirically (cancelled to avoid racing t1-full's cache file), but
well understood:
- **bf16 + CPU offload** (`device_map="auto"` on 1 GPU): the exact weights fit by
  spilling ~half the layers to CPU RAM. It runs, but every token crosses the
  PCIe/CPU boundary, so generation is CPU-bound and slow (expect several times the
  ~30s/image-cell above). Needs ~24-48 GiB host RAM for the offloaded shards.
- **2×V100 bf16** (what the grid uses now): 32 GiB combined, exact weights sharded
  by accelerate. The fidelity reference. Downside is scheduling a whole free
  2-GPU node.

## Why image cells take ~30s either way

The V100 is Volta (sm_70), which has no FlashAttention-2, so attention falls back
to the O(seq²) SDPA math kernel. That, plus per-matmul dequantization overhead in
bitsandbytes, dominates an image cell. Text-only cells are fast (1-2s). So the
throughput bottleneck on a V100 is the vision path and the attention kernel, not
memory, and 4-bit does not make it faster than bf16 per token (often slightly
slower). Quantization buys you *fit on one GPU*, not speed.

Rough grid cost implication: with images only on the `TLV` and `V` rungs, the
100-q/domain subset (309 Q) is ~618 image cells. At ~30s each plus fast text
cells, that is on the order of 5-8h single-threaded, before multi-page oracle
cells (which attach several images and cost more). Full corpus (1091 Q) is ~4x
that. Memory is solved; wall-clock is the thing to watch.

## Ways to make single-GPU feasible, ranked

1. **4-bit NF4 (recommended for iteration / as an appendix row).** Most headroom
   (7 GiB used of 15.77), correct on the probe, schedules fastest. Best for smoke
   tests, plumbing, and a possible "does the doc-type frontier survive 4-bit?"
   appendix. Tie results to the bitsandbytes version for reproducibility.
2. **8-bit.** Closer to bf16 fidelity than 4-bit, still one GPU (10 GiB). A
   fidelity-vs-schedulability compromise. Validate against a bf16 slice before
   trusting any main number (the probe already showed an 8-bit miss).
3. **bf16 + CPU offload on 1 GPU.** Exact pre-registered weights, one GPU, but
   slow. Use only if you must have bf16 on a single GPU and can spend the time;
   request ~48 GiB host RAM.
4. **vLLM with a pre-quantized AWQ/GPTQ checkpoint.** Would be faster than bnb
   HF generation, but needs (a) confirmed Qwen3-VL support in the pinned vLLM
   0.9.2, and (b) an available AWQ checkpoint. More setup; worth a look only if
   throughput becomes the blocker.
5. **Tighter `max_pixels` / fewer oracle pages.** Orthogonal. Cuts the vision
   sequence (already capped at 602k for 8B) and helps latency, but does **not**
   shrink the weights, so it cannot make bf16 fit on its own. Useful alongside the
   options above.

## Recommendation for this project

- **Keep the pre-registered main runs (Table 1 / the F1 gate) on 2×V100 bf16.**
  That is the exact setup in `docs/USER_GUIDE.md` §6; the frontier numbers should not
  be defined on quantized weights. The current `t1-full` (1001899) is doing this.
- **Use 1×V100 4-bit for fast iteration** (smoke, plumbing, memory-fix checks) and
  optionally as an explicit **quantization-sensitivity appendix** row. This is
  also the pressure valve when 2-GPU nodes are hard to get: the probe backfilled
  in ~2 minutes vs the overnight wait for a 2-GPU node.
- If 2-GPU nodes stay scarce and time is tight, **8-bit on 1×V100** is the
  fallback for main-ish runs, but only after checking it against a bf16 slice on a
  few cells per bin.

## To actually run the pipeline quantized (not done yet)

The probe is standalone. To run real experiments in 4-bit, `LocalVLMBackend`
(`models/local_vlm.py`) needs an optional `quantization` argument threaded through
`get_reasoner(spec, ...)` and set from config (e.g. `config.quantization in
{None, "4bit", "8bit"}`), passing a `BitsAndBytesConfig` into `from_pretrained`.
That is a small additive change behind the frozen `Reasoner` ABC (the cache key
would need `quantization` added so quantized rows do not collide with bf16 rows).
Not implemented here since the main runs stay bf16; say the word if you want the
4-bit path wired in for an appendix.

## Repro

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --gres gpu:v100:1 --time 00:30:00 \
  --mem 48G --job-name qwen8b-1gpu-probe scripts/single_gpu_probe.py
# then: kaya.kaya pull ; see logs/qwen8b-1gpu-probe_<id>.out
```

(`bitsandbytes==0.49.2` was installed into the remote `envs/mpvrdu` for this; it
persists because `envs/` is excluded from the rsync push.)
