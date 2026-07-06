"""Shared plumbing for the three experiment roles (generate / judge / build).

Purpose:
    Holds what the role modules all need but none of them owns: logging setup,
    the per-task cache/table path layout, and the phase-status artifact. Keeping
    it here is what lets the driver, the task files, and the reporting module
    stay strictly about their one role.

Pipeline role:
    A leaf module (imports only config). `experiment_paths(config, name)` maps a
    generation-task name to its cache files under
    `results/cache/<mode>[/<run_tag>]/<name>/` and its tables under
    `results/tables/<mode>[-<run_tag>]/`; `--run-tag` is already folded into
    `config.paths.cache_dir` (see `ExperimentConfig.__post_init__`).

Arguments:
    None. Import-only.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import ExperimentConfig


log = logging.getLogger("mpvrdu.experiments")


def configure_logging(verbose: bool) -> None:
    """Send `mpvrdu.*` logs to stdout at DEBUG (verbose) or INFO level.

    `force=True` replaces any handler a previous call installed, and the stdout
    StreamHandler flushes per record so lines show up promptly in a SLURM log
    even when a later cell crashes. Call it once from an entry point before a run.
    """

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("mpvrdu").setLevel(logging.DEBUG if verbose else logging.INFO)


def answer_preview(text: str, limit: int = 160) -> str:
    """One-line, length-capped preview of an answer for logs."""

    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def free_gpu() -> None:
    """Best-effort release of freed CUDA memory back to the driver.

    Python-drops of model objects return their tensors to torch's caching
    allocator, but not to the driver until `empty_cache`. Calling this after each
    heavyweight stage (parser, retriever, reasoner) is what lets the next stage
    have the whole GPU on a 16GB V100. Never raises.
    """

    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def mode(config: ExperimentConfig) -> str:
    """Return the cache-partition name for this config."""

    return "smoke" if config.smoke else "full"


@dataclass(frozen=True)
class ExperimentPaths:
    """Per-generation-task cache/side/table locations, all root-relative."""

    root: Path
    predictions: Path
    generate_results: Path
    results: Path
    side_dir: Path
    table_dir: Path


def experiment_paths(config: ExperimentConfig, name: str) -> ExperimentPaths:
    """Resolve the cache/table paths for one generation task (by name)."""

    # `config.paths.cache_dir` already carries any `run_tag` (see
    # ExperimentConfig.__post_init__), so predictions/renders/side records isolate
    # automatically. The table dir lives under results_dir, so tag it here too.
    table_partition = mode(config) if config.run_tag is None else f"{mode(config)}-{config.run_tag}"
    root = config.paths.cache_dir / mode(config) / name
    return ExperimentPaths(
        root=root,
        predictions=root / "predictions.jsonl",
        generate_results=root / "generate_results.jsonl",
        results=root / "results.jsonl",
        side_dir=root,
        table_dir=config.paths.results_dir / "tables" / table_partition,
    )


@dataclass(frozen=True)
class ExperimentRunStatus:
    """Outcome of one generation task's phase inside a grouped run."""

    experiment: str
    phase: str
    status: str
    path: Path
    error_type: str = ""
    error: str = ""


def write_phase_status(
    config: ExperimentConfig,
    name: str,
    *,
    phase: str,
    status: str,
    error: BaseException | None = None,
) -> ExperimentRunStatus:
    """Write one per-task phase status JSON artifact and return its summary."""

    paths = experiment_paths(config, name)
    paths.root.mkdir(parents=True, exist_ok=True)
    path = paths.root / f"{phase}_status.json"
    payload = {
        "experiment": name,
        "phase": phase,
        "status": status,
        "mode": mode(config),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": str(paths.predictions),
        "generate_results": str(paths.generate_results),
        "results": str(paths.results),
    }
    error_type = ""
    error_text = ""
    if error is not None:
        error_type = type(error).__name__
        error_text = str(error)
        payload.update(
            {
                "error_type": error_type,
                "error": error_text,
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
        )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return ExperimentRunStatus(name, phase, status, path, error_type, error_text)
