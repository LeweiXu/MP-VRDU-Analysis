"""Smoke-test CUDA visibility inside the Kaya conda environment."""

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
