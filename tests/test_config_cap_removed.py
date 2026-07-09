"""The input-token cap is gone entirely; resolution presets remain the only
per-page vision knob."""

from __future__ import annotations

import importlib


def test_no_input_token_cap_symbols() -> None:
    config = importlib.import_module("config")
    for gone in ("max_input_tokens", "MAX_INPUT_TOKENS_BY_SIZE",
                 "max_input_tokens_for_spec"):
        assert not hasattr(config, gone), f"config still exposes removed cap: {gone}"


def test_resolution_presets_present() -> None:
    config = importlib.import_module("config")
    presets = getattr(config, "VISUAL_RESOLUTION_PRESETS", None)
    assert presets, "config must keep VISUAL_RESOLUTION_PRESETS"
    # cost-ordered presets; values are per-page pixel caps.
    assert set(presets) == {"low", "med", "high"}
