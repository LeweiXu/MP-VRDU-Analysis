"""Generator factory: config -> Generator instance.

Kept separate from base.py so importing the interface never drags in the VLM
wrappers (which lazy-import torch only when actually constructed).
"""

from __future__ import annotations

from typing import Optional

from ..config import GenerationConfig
from ..data.dataset import Dataset
from .base import Generator
from .mock import MockGenerator


def build_generator(cfg: GenerationConfig,
                    dataset: Optional[Dataset] = None) -> Generator:
    if cfg.generator == "mock":
        gold_lookup = {}
        if dataset is not None:
            gold_lookup = {q.qid: q.answer for q in dataset.questions}
        return MockGenerator(mode=cfg.mock_mode, gold_lookup=gold_lookup)

    if cfg.generator == "local_small_vlm":
        from .vlm import LocalSmallVLM

        return LocalSmallVLM(
            model_id=cfg.model_id or "Qwen/Qwen2.5-VL-3B-Instruct",
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
            device_map=cfg.device_map, load_in_4bit=cfg.load_in_4bit,
            max_pixels=cfg.max_pixels, min_pixels=cfg.min_pixels)

    if cfg.generator == "kaya_vlm":
        from .vlm import KayaVLM

        return KayaVLM(
            model_id=cfg.model_id or "Qwen/Qwen2.5-VL-7B-Instruct",
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
            device_map=cfg.device_map, load_in_4bit=cfg.load_in_4bit,
            max_pixels=cfg.max_pixels, min_pixels=cfg.min_pixels)

    raise ValueError(f"unknown generator {cfg.generator!r}")
