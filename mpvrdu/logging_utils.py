"""Structured logging + seed control (Stage 0)."""

from __future__ import annotations

import logging
import os
import random
import sys

_CONFIGURED = False


def get_logger(name: str = "mpvrdu", level: int | None = None) -> logging.Logger:
    """Return a process-wide logger with a consistent format.

    Level can be overridden by the MPVRDU_LOGLEVEL env var (e.g. DEBUG).
    """
    global _CONFIGURED
    if not _CONFIGURED:
        env_level = os.environ.get("MPVRDU_LOGLEVEL", "INFO").upper()
        root_level = level if level is not None else getattr(logging, env_level, logging.INFO)
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("mpvrdu")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(root_level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("mpvrdu") else f"mpvrdu.{name}")


def set_seed(seed: int) -> None:
    """Fix RNG seeds for reproducibility. Torch/numpy seeded if importable."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
