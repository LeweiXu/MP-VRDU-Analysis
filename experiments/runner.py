"""Expand experiment configs into cached pipeline cells.

Purpose:
    Reserved for the full runner that will turn an `ExperimentConfig` into the
    matrix of questions, conditions, representations, model specs, and policy
    settings needed for the paper tables.

Pipeline role:
    Section-2 stages will use this module as the scale-up path after the MVP
    CLIs prove each component. It should coordinate cache-resumable sweeps
    without changing the frozen orchestrator interfaces.

Arguments:
    None. This module is import-only until a future CLI delegates to it.
"""
