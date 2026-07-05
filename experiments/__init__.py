"""Experiment orchestration package.

Purpose:
    Groups the study by *role*, not by paper table: generation (the only GPU
    work), judging, and table building are separate modules so generation can run
    on Kaya while judging and aggregation stay local.

Pipeline role:
    `experiments.generation` defines the generation tasks (G1..G6) and runs them
    on a GPU; `experiments.judge` scores their cached predictions locally;
    `experiments.build` routes each task's judged rows into the eight table CSVs
    (+ a combined markdown). `experiments.paths` holds the shared cache layout;
    `experiments.tables` holds the pure per-table aggregation functions;
    `experiments.corpus`/`experiments.smoke` resolve the question set.

Arguments:
    None. This package initializer is import-only.
"""
