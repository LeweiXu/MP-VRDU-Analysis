"""Smoke-test CUDA visibility inside the Kaya conda environment.

Purpose:
    Runs as a minimal SLURM GPU job to confirm that the configured conda
    environment can import torch and see an allocated CUDA device.

Pipeline role:
    Operational sanity check before model-heavy stages. A failure means the Kaya
    partition/GRES/module/environment configuration must be fixed before M3.

CLI:
    `python -m kaya.kaya submit scripts/gpu_test.py`

Arguments:
    None. SLURM resource overrides, if needed, are supplied to `kaya.kaya
    submit` before the script path, not to this script.
"""

# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=mpvrdu_gpu_test

from __future__ import annotations


def main() -> int:
    import torch

    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
    return 0 if torch.cuda.is_available() else 1


if __name__ == "__main__":
    raise SystemExit(main())
