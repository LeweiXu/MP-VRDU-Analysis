# Environments

Four isolated conda envs. The core env runs the reasoners, retrievers, judges,
and the T-rung text; each parser gets its own env because their VLM stacks pin
transformers/torch (or PaddlePaddle) versions that will not co-exist with the
core reasoner. Parsers only ever cross to the reasoner through the disk cache, so
they never share VRAM or an env.

| Env | Framework | Requirements | What runs there |
|---|---|---|---|
| `core` | torch | `core.txt` | Qwen3-VL, InternVL, retrievers, judges, PyMuPDF (T) |
| `parse-paddleocrvl` | **PaddlePaddle** | `parse-paddleocrvl.txt` | PaddleOCR-VL-0.9B page parsing |
| `parse-mineru` | torch | `parse-mineru.txt` | MinerU 2.5 (transformers VLM backend) |
| `parse-unlimited` | torch | `parse-unlimited.txt` | Baidu Unlimited-OCR (3B MoE) |

The dependency files above are machine-agnostic. The only thing that changes per
machine is the **framework build** (CUDA wheel index + torch/paddle version),
because the three target GPUs are different architectures. `setup_env.py` holds
this matrix and applies it; the three configurations are:

## The three machine configurations

| Machine | GPU (arch) | CUDA | torch index | flash-attn |
|---|---|---|---|---|
| **kaya** | V100 (sm_70) | 12.6 | `whl/cu126` | no (Volta has no FA2) |
| **local** | RTX 5070 (sm_120) | 12.8 | `whl/cu128` | no (nvcc absent) |
| **supervisor** | H100 (sm_90) | 12.6 | `whl/cu126` | yes |

Per-env framework build (same across machines except the CUDA index):

| Env | Build | Version |
|---|---|---|
| `core` | torch + torchvision | 2.7.0 / 0.22.0 (local: 2.8.0 / 0.23.0 for sm_120) |
| `parse-mineru` | torch + torchvision | 2.7.0 / 0.22.0 |
| `parse-unlimited` | torch + torchvision | 2.10.0 / 0.25.0 (upstream-tested) |
| `parse-paddleocrvl` | paddlepaddle-gpu | 3.0.0 from `paddlepaddle.org.cn/packages/stable/cu126` |

Only **kaya** is built and tested right now. local/supervisor are specified but
not installed yet.

## Build

```bash
# one env
python -m ops.kaya.kaya run ops/scripts/setup_env.py -- --machine kaya --env core
# all four
python -m ops.kaya.kaya run ops/scripts/setup_env.py -- --machine kaya --env all
```

`setup_env.py` creates the conda prefix, installs the framework from the right
index, installs the env's requirements, and runs `pip check`. Model and dataset
downloads are **not** here: they live in `prestage.py`.
