"""Build the MP-VRDU conda environments for one target machine.

Creates the isolated env partition (core reasoning env + one env per parser) at
`<root>/envs/<name>`, installs the framework build (torch or PaddlePaddle) from
the machine's CUDA wheel index, installs the env's requirements from
`docs/requirements/`, and runs `pip check`. Model and dataset downloads are not
here; they live in `prestage.py`.

Run one env or all four on Kaya (via the login-node runner):
    python -m ops.kaya.kaya run ops/scripts/setup_env.py -- --machine kaya --env core
    python -m ops.kaya.kaya run ops/scripts/setup_env.py -- --machine kaya --env all

Or run it directly on the machine you are on (e.g. the H100 supervisor), building
the envs inside this checkout at `envs/<name>` where the rest of the code looks
for them:
    python -m ops.scripts.setup_env --machine supervisor --env all --local
"""

# kaya: target=login
# kaya: env=false
# kaya: offline=false

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from ops.kaya.kaya import load_config

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "docs" / "requirements"

TORCH_INDEX = "https://download.pytorch.org/whl/{cuda}"
PADDLE_INDEX = "https://www.paddlepaddle.org.cn/packages/stable/{cuda}/"

# Per-env framework build + requirements file. torch/torchvision default here and
# may be overridden per machine (below); the parser envs pin their own stacks.
ENVS: dict[str, dict] = {
    "core": {"req": "core.txt", "framework": "torch", "torch": "2.7.0", "torchvision": "0.22.0"},
    "parse-mineru": {"req": "parse-mineru.txt", "framework": "torch", "torch": "2.7.0", "torchvision": "0.22.0"},
    "parse-unlimited": {"req": "parse-unlimited.txt", "framework": "torch", "torch": "2.10.0", "torchvision": "0.25.0"},
    # paddle 3.0.0 crashes PaddleOCR-VL's static predictor with a PIR strides
    # error (InvalidArgument); 3.3.1 fixes it. Same version we run locally.
    "parse-paddleocrvl": {"req": "parse-paddleocrvl.txt", "framework": "paddle", "paddle": "3.3.1"},
}

# The three machine configurations. Only the CUDA wheel index and a few
# framework versions differ; the requirements files are shared.
MACHINES: dict[str, dict] = {
    "kaya": {"cuda": "cu126", "flash_attn": False, "torch_overrides": {}},
    "local": {"cuda": "cu128", "flash_attn": False,
              "torch_overrides": {"core": ("2.8.0", "0.23.0"), "parse-mineru": ("2.8.0", "0.23.0")}},
    "supervisor": {"cuda": "cu126", "flash_attn": True, "torch_overrides": {}},
}


def run(command: list[str]) -> None:
    print("[setup_env]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def env_prefix(config, name: str, *, local: bool = False) -> Path:
    """Where env `name` is built: this checkout's `envs/` when local, else the
    Kaya remote root. Local matches `config.DEFAULT_PATHS.env_dir`, which is where
    the parser worker and reasoner resolve their interpreters."""

    root = ROOT if local else Path(config.remote_root)
    return root / "envs" / name


def install_framework(python: list[str], machine_cfg: dict, env_name: str, env_cfg: dict) -> None:
    """Install torch (from the PyTorch index) or paddlepaddle (from Paddle's)."""
    cuda = machine_cfg["cuda"]
    if env_cfg["framework"] == "paddle":
        run([*python, "-m", "pip", "install", "-i", PADDLE_INDEX.format(cuda=cuda),
             f"paddlepaddle-gpu=={env_cfg['paddle']}"])
        return
    torch_v, tv_v = machine_cfg["torch_overrides"].get(
        env_name, (env_cfg["torch"], env_cfg["torchvision"]))
    run([*python, "-m", "pip", "install", "--index-url", TORCH_INDEX.format(cuda=cuda),
         f"torch=={torch_v}", f"torchvision=={tv_v}"])


def build_env(config, machine: str, env_name: str, python_version: str, *, skip_pip_check: bool,
              local: bool = False) -> None:
    machine_cfg = MACHINES[machine]
    env_cfg = ENVS[env_name]
    prefix = env_prefix(config, env_name, local=local)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[setup_env] ==== {env_name} on {machine} ({machine_cfg['cuda']}) ====", flush=True)
    if not (prefix / "conda-meta").is_dir():
        run(["conda", "create", "-p", str(prefix), f"python={python_version}", "-y"])
    python = ["conda", "run", "--no-capture-output", "-p", str(prefix), "python"]

    run([*python, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    install_framework(python, machine_cfg, env_name, env_cfg)
    run([*python, "-m", "pip", "install", "-r", str(REQUIREMENTS / env_cfg["req"])])
    if machine_cfg["flash_attn"] and env_cfg["framework"] == "torch":
        # H100 (sm_90) has FlashAttention-2; V100/Blackwell-local do not build it.
        # Best-effort: the reasoner falls back to the memory-efficient SDPA kernel
        # if this wheel can't build, so a failed build must not sink the whole env.
        try:
            run([*python, "-m", "pip", "install", "flash-attn", "--no-build-isolation"])
        except subprocess.CalledProcessError as exc:
            print(f"[setup_env] WARNING: flash-attn build failed ({exc}); "
                  "continuing without it (the reasoner uses SDPA).", flush=True)
    if not skip_pip_check:
        run([*python, "-m", "pip", "check"])
    # Report the framework version. Runs on the GPU-less login node, where paddle
    # (and any GPU C-extension) can't fully import for lack of libcuda.so.1, so
    # fall back to installed-package metadata rather than failing the build.
    pkg = "paddlepaddle-gpu" if env_cfg["framework"] == "paddle" else "torch"
    run([*python, "-c",
         "import importlib.metadata as M\n"
         "try:\n"
         f"    import {'paddle' if pkg=='paddlepaddle-gpu' else 'torch'} as fw; "
         "    print(fw.__name__, fw.__version__)\n"
         "except Exception as e:\n"
         f"    print('{pkg}', M.version('{pkg}'), '(binary load deferred to GPU node:', type(e).__name__, ')')\n"])
    print(f"[setup_env] {env_name} ready at {prefix}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", choices=sorted(MACHINES), default="kaya")
    parser.add_argument("--env", choices=[*sorted(ENVS), "all"], default="all")
    parser.add_argument("--python", help="python version for new envs (default from config)")
    parser.add_argument("--skip-pip-check", action="store_true")
    parser.add_argument("--local", action="store_true",
                        help="build envs in this checkout's envs/ (run directly on the target machine, "
                             "e.g. the H100 supervisor) instead of the Kaya remote root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(ROOT / "ops" / "kaya" / "config.json")
    python_version = args.python or str(config.raw.get("python_version", "3.11"))
    names = list(ENVS) if args.env == "all" else [args.env]
    for name in names:
        build_env(config, args.machine, name, python_version,
                  skip_pip_check=args.skip_pip_check, local=args.local)
    print(f"\n[setup_env] done: {', '.join(names)} on {args.machine}"
          f"{' (local checkout)' if args.local else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
