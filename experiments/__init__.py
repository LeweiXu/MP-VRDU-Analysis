"""Experiment orchestration package.

Purpose:
    Groups smoke-corpus selection, future experiment expansion, and future table
    aggregation code under one import namespace.

Pipeline role:
    `experiments.smoke` is active in the MVP; `experiments.runner` and
    `experiments.tables` become the full-run and reporting surfaces later.

Arguments:
    None. This package initializer is import-only.
"""
