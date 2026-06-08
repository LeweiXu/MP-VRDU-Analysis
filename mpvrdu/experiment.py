"""Experiment suites: expand a compact axes spec into many RunConfigs.

A suite YAML has two parts:

    defaults:        # a partial RunConfig applied to every run (incl. the
                     # generation block — the ONE place to edit for Kaya)
      data: {name: mmlongbench-doc, slice: full}
      generation: {generator: local_small_vlm, model_id: Qwen/Qwen2.5-VL-3B-Instruct}
      ...
    substudies:
      A_retrieval:
        axes:                              # cross-producted
          retrieval.method:    [bm25, dense, colpali]
          generation.modality: [image, text]
          retrieval.top_k:     [4]

Each axes combination is deep-merged onto `defaults`, given an auto name, and
validated. One suite -> a list of (substudy, RunConfig). Switching the whole grid
to Kaya is a single edit to `defaults.generation` (generator + model_id).
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import yaml

from .config import RunConfig, dict_to_config


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _short(value: Any) -> str:
    return str(value).replace("/", "-").replace(" ", "")


def expand_suite(suite: dict) -> list[tuple[str, RunConfig]]:
    """Expand a loaded suite dict into [(substudy, RunConfig), ...]."""
    defaults = suite.get("defaults", {})
    substudies = suite.get("substudies", {})
    runs: list[tuple[str, RunConfig]] = []

    for sub_name, spec in substudies.items():
        axes = spec.get("axes", {})
        keys = list(axes)
        value_lists = [axes[k] for k in keys]
        for combo in itertools.product(*value_lists):
            override: dict = {}
            for k, v in zip(keys, combo):
                _set_dotted(override, k, v)
            merged = _deep_merge(defaults, override)
            # auto name: substudy + each varying axis's value
            name_bits = [sub_name] + [_short(v) for v in combo]
            merged["name"] = "__".join(name_bits)
            runs.append((sub_name, dict_to_config(merged)))
    return runs


def load_suite(path: str | Path) -> list[tuple[str, RunConfig]]:
    with Path(path).open() as fh:
        suite = yaml.safe_load(fh)
    return expand_suite(suite)
