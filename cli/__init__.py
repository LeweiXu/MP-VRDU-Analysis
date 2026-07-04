"""CLI package for MP-VRDU operational entry points.

Purpose:
    Groups runnable commands under `python -m cli.<name>` so probes, experiment
    runs, and table building can share the importable project code without shell
    wrappers.

Pipeline role:
    `cli.run_probe` checks feasibility, `cli.experiments` runs the paper-table
    experiments (generate/judge/build), and `cli.build_tables` aggregates cached
    rows into CSVs directly.

Arguments:
    None. This package initializer is import-only; arguments live in the
    individual CLI modules.
"""
