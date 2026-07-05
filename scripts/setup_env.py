"""Build or update the MP-VRDU conda environment on Kaya.

Purpose:
    Runs on the Kaya login node outside the project environment to create the
    configured conda prefix and install `requirements.txt` using the configured
    PyTorch/CUDA wheel index.

Pipeline role:
    Must be run after source sync and before prestaging or GPU jobs whenever
    requirements change. It leaves the environment at `<remote_root>/envs/mpvrdu`
    and prints the installed torch/CUDA versions.

CLI:
    `python -m kaya.kaya run scripts/setup_env.py -- [--skip-pip-check]`

Arguments:
    --skip-pip-check: do not run `python -m pip check` after installation.
"""

# kaya: target=login
# kaya: env=false
# kaya: offline=false

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from kaya.kaya import load_config


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    """Run one setup command with streaming output."""

    print("[setup_env]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pip-check", action="store_true", help="skip `python -m pip check`")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(ROOT / "kaya/config.json")
    env = Path(config.remote_path("env"))
    env_parent = env.parent
    cache = Path(config.remote_path("cache"))
    requirements = Path(config.remote_root) / "requirements.txt"

    env_parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    if not (env / "conda-meta").is_dir():
        run(["conda", "create", "-p", str(env), f"python={config.raw['python_version']}", "-y"])

    python = ["conda", "run", "-p", str(env), "python"]
    run([*python, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    run(
        [
            *python,
            "-m",
            "pip",
            "install",
            "--extra-index-url",
            config.raw["pip"]["torch_index_url"],
            "-r",
            str(requirements),
        ]
    )
    if not args.skip_pip_check:
        run([*python, "-m", "pip", "check"])
    run(
        [
            *python,
            "-c",
            "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
