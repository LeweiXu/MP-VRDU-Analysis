"""Build and validate the isolated conda environments for a V100 or H100 machine.

Each environment gets one clean retry after a failed build, then the command reports
all passes and failures together.
"""

# kaya: target=login
# kaya: env=false
# kaya: offline=false

from __future__ import annotations

import argparse
import os
import shutil
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

# The two deployment machine configurations. Only the CUDA wheel index and
# flash-attn support differ; the requirements files are shared.
MACHINES: dict[str, dict] = {
    "V100": {"cuda": "cu126", "flash_attn": False, "torch_overrides": {}},
    "H100": {"cuda": "cu126", "flash_attn": True, "torch_overrides": {}},
}


def run(command: list[str]) -> None:
    print("[setup_env]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def prepare_package_cache_env(root: Path) -> Path:
    """Point conda and pip caches inside the target checkout."""

    cache = root / ".cache"
    locations = {
        "CONDA_PKGS_DIRS": cache / "conda-pkgs",
        "PIP_CACHE_DIR": cache / "pip",
        "XDG_CACHE_HOME": cache / "xdg",
    }
    for name, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    return locations["CONDA_PKGS_DIRS"]


def remove_env_prefix(prefix: Path) -> None:
    """Remove an environment prefix left incomplete by a failed build."""

    if prefix.exists():
        print(f"[setup_env] removing incomplete env prefix {prefix}", flush=True)
        shutil.rmtree(prefix)


def prefix_is_usable(prefix: Path) -> bool:
    """Return whether an existing conda prefix can run its Python."""

    if not (prefix / "conda-meta").is_dir():
        return False
    result = subprocess.run(
        ["conda", "run", "-p", str(prefix), "python", "-c", "import sys; print(sys.executable)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def clear_conda_package_cache(cache_dir: Path) -> None:
    """Clear shared conda packages before retrying a failed environment."""

    print(f"[setup_env] clearing conda package cache {cache_dir}", flush=True)
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


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
    if prefix.exists() and not prefix_is_usable(prefix):
        remove_env_prefix(prefix)
    if not prefix.exists():
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
    parser.add_argument("--machine", choices=sorted(MACHINES), default="V100")
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
    root = ROOT if args.local else Path(config.remote_root)
    package_cache = prepare_package_cache_env(root)
    passed: list[str] = []
    failed: dict[str, str] = {}
    for name in names:
        prefix = env_prefix(config, name, local=args.local)
        for attempt in (1, 2):
            try:
                build_env(config, args.machine, name, python_version,
                          skip_pip_check=args.skip_pip_check, local=args.local)
            except Exception as exc:
                remove_env_prefix(prefix)
                if attempt == 1:
                    print(f"[setup_env] {name} failed: {exc}", flush=True)
                    clear_conda_package_cache(package_cache)
                    print(f"[setup_env] retrying {name} from a clean prefix", flush=True)
                    continue
                failed[name] = str(exc)
                print(f"[setup_env] FAILED {name}: {exc}", flush=True)
            else:
                passed.append(name)
            break

    location = "local checkout" if args.local else str(root)
    print(f"\n[setup_env] summary for {args.machine} at {location}", flush=True)
    print(f"[setup_env] passed: {', '.join(passed) or '(none)'}", flush=True)
    if failed:
        for name, reason in failed.items():
            print(f"[setup_env] failed: {name}: {reason}", flush=True)
        return 1
    print("[setup_env] all requested environments are ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
