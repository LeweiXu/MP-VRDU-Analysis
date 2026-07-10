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
because the deployment GPUs have different architectures. `setup_env.py` holds
this matrix and applies it; the two configurations are:

## Machine configurations

| Machine | GPU (arch) | CUDA | torch index | flash-attn |
|---|---|---|---|---|
| **V100** | V100 (sm_70) | 12.6 | `whl/cu126` | no (Volta has no FA2) |
| **H100** | H100 (sm_90) | 12.6 | `whl/cu126` | yes |

Per-env framework build (same across machines except the CUDA index):

| Env | Build | Version |
|---|---|---|
| `core` | torch + torchvision | 2.7.0 / 0.22.0 |
| `parse-mineru` | torch + torchvision | 2.7.0 / 0.22.0 |
| `parse-unlimited` | torch + torchvision | 2.10.0 / 0.25.0 (upstream-tested) |
| `parse-paddleocrvl` | paddlepaddle-gpu | 3.3.1 from `paddlepaddle.org.cn/packages/stable/cu126` |

Kaya uses `--machine V100`. The H100 supervisor uses `--machine H100 --local`.

## Build

```bash
# one env
python -m ops.kaya.kaya run ops/scripts/setup_env.py -- --machine V100 --env core
# all four
python -m ops.kaya.kaya run ops/scripts/setup_env.py -- --machine V100 --env all
```

`setup_env.py` creates the conda prefix, installs the framework from the right
index, installs the env's requirements, and runs `pip check`. A failed env gets
one clean retry, and all requested env results are reported together. Model and
dataset downloads are not here: they live in `prestage.py`.

## Everything lives in the project directory

Nothing installs to `$HOME` or a shared system location. Every heavy artifact is
kept under the project root, gitignored, and rsync-excluded from Kaya so each
machine keeps its own copy:

| Artifact | Location | How |
|---|---|---|
| conda environments | `envs/<name>/` | `conda create -p envs/<name>` |
| model + parser weights | `.cache/` (HF hub layout) | `HF_HOME`/`HF_HUB_CACHE` |
| MinerU aux models | `.cache/modelscope/` | `MODELSCOPE_CACHE`, `MINERU_MODEL_SOURCE=huggingface` |
| PaddleOCR-VL models | `.cache/paddle*` (from HF) | `PADDLE_PDX_MODEL_SOURCE=huggingface` |
| datasets | `.data/` | prestage staging |
| conda pkg cache | `.cache/conda-pkgs/` | `CONDA_PKGS_DIRS` |
| pip wheel cache | `.cache/pip/` | `PIP_CACHE_DIR` |
| torch/triton/inductor | `.cache/{torch,triton,inductor}/` | matching env vars |
| anything XDG | `.cache/xdg/` | `XDG_CACHE_HOME` |

The Kaya run wrapper (`ops/kaya/kaya.py::artifact_exports`) exports all of these
before any remote command, so `setup_env.py` and `prestage.py` inherit them.
`.gitignore` and `rsync_excludes` both cover `envs/`, `.cache/`, `.data/`,
`results/`, `logs/`, so none of it is committed or synced.
