"""CLI package for MP-VRDU operational entry points.

Purpose:
    Groups runnable commands under `python -m cli.<name>` so probes, experiment
    runs, and table building can share the importable project code without shell
    wrappers.

Pipeline role:
    `cli.run_probe` checks feasibility, `cli.run_experiment` exercises cached
    pipeline cells, and `cli.build_tables` will aggregate result CSVs.

Arguments:
    None. This package initializer is import-only; arguments live in the
    individual CLI modules.
"""
