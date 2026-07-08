"""Experiment orchestration package.

Purpose:
    Groups the study by *role*, not by paper table: generation (the only GPU
    work), judging, and table building are separate modules so generation can run
    on Kaya while judging and aggregation stay local.

Pipeline role:
    Task definitions live one-per-file in `experiments/G*_*.py` (subclassing
    `experiments.base.GenerationTask`); `experiments.registry` collects them.
    `experiments.driver` is the generate (GPU) + judge (local) engine;
    `reporting.build` routes each task's judged rows into the eight table
    CSVs (+ a combined markdown) via the `reporting.tables` builders.
    `experiments.paths` holds the shared cache layout;
    `experiments.corpus`/`experiments.smoke` resolve the question set. The
    runnable entry points are the thin wrappers `cli/{generate,judge,build}.py`.

Arguments:
    None. This package initializer is import-only.
"""
