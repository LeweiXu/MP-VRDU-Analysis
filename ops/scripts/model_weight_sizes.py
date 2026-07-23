"""Write annotations/model_weights.csv: per model_spec weight-only memory.

Reads safetensors tensor metadata (shapes and dtypes, never the weights themselves)
and records how many bytes each reasoner's parameters occupy. Run it by hand when a
model spec is added; the table build reads the CSV, never the network.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import ROOT  # noqa: E402
from models.internvl import MODEL_IDS as INTERNVL_IDS  # noqa: E402
from models.llama_vision import MODEL_IDS as LLAMA_IDS  # noqa: E402
from models.qwen3vl import MODEL_IDS as QWEN_IDS  # noqa: E402

OUT = ROOT / "annotations" / "model_weights.csv"

# bitsandbytes replaces the transformer's 2D Linear weights and leaves everything else
# (embeddings, lm_head, norms, biases) in the compute dtype. This matches the default
# skip behaviour for the configs in models/qwen3vl.py::_quantization_config, which set
# no llm_int8_skip_modules.
_LINEAR = re.compile(
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|mlp\.\w*fc\d?|attn\.\w*proj)\.weight$"
)
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "U8": 1, "I8": 1}

# NF4 packs two params per byte. Double quant stores one absmax per 64-param block as
# int8, plus one fp32 second-level scale per 256 blocks.
_NF4_BLOCK = 64
_NF4_DOUBLE_QUANT_BLOCK = 256


def _tensors(repo_id: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Every tensor's dtype and shape, from the hub's safetensors headers."""

    from huggingface_hub import get_safetensors_metadata

    meta = get_safetensors_metadata(repo_id)
    out: dict[str, tuple[str, tuple[int, ...]]] = {}
    for file_meta in meta.files_metadata.values():
        for name, info in file_meta.tensors.items():
            out[name] = (str(info.dtype), tuple(info.shape))
    return out


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _sizes(repo_id: str) -> tuple[int, int, int, int]:
    """Return (bf16 bytes, 8bit bytes, 4bit bytes, quantizable params)."""

    exact = 0
    quantizable = 0
    other_bytes = 0
    for name, (dtype, shape) in _tensors(repo_id).items():
        numel = _numel(shape)
        width = _DTYPE_BYTES.get(dtype, 2)
        exact += numel * width
        if len(shape) == 2 and _LINEAR.search(name):
            quantizable += numel
        else:
            other_bytes += numel * width
    int8 = quantizable + other_bytes
    nf4 = int(
        quantizable * 0.5
        + quantizable / _NF4_BLOCK
        + quantizable / (_NF4_BLOCK * _NF4_DOUBLE_QUANT_BLOCK) * 4
        + other_bytes
    )
    return exact, int8, nf4, quantizable


def main() -> int:
    specs = {**QWEN_IDS, **INTERNVL_IDS, **LLAMA_IDS}
    rows: list[dict[str, object]] = []
    for spec, repo_id in sorted(specs.items()):
        try:
            exact, int8, nf4, quantizable = _sizes(repo_id)
        except Exception as exc:  # noqa: BLE001 - a missing repo must not sink the rest
            print(f"skip {spec}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        rows.append({"model_spec": spec, "hf_repo": repo_id,
                     "weights_bytes": exact, "method": "exact", "quantizable_params": quantizable})
        # Only the Qwen family was ever run quantized (models/__init__ passes no
        # quantization to the InternVL backend), so only it gets the derived rows.
        if spec in QWEN_IDS:
            rows.append({"model_spec": f"{spec}-8bit", "hf_repo": repo_id,
                         "weights_bytes": int8, "method": "derived", "quantizable_params": quantizable})
            rows.append({"model_spec": f"{spec}-4bit", "hf_repo": repo_id,
                         "weights_bytes": nf4, "method": "derived", "quantizable_params": quantizable})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_spec", "hf_repo", "weights_bytes",
                                                    "method", "quantizable_params"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")
    for row in rows:
        print(f"  {row['model_spec']:<28} {int(row['weights_bytes']) / 1e9:6.2f} GB  ({row['method']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
