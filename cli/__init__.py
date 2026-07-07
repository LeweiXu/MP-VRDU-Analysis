"""CLI package for MP-VRDU operational entry points.

Purpose:
    Groups runnable commands under `python -m cli.<name>` so probes, experiment
    runs, and table building can share the importable project code without shell
    wrappers.

Pipeline role:
    `cli` holds only the three experiment roles: `cli.generate` (GPU), `cli.judge`,
    and `cli.build`. Table building lives in the `reporting` package and the
    Section-2 gates in the `gates` package (`python -m gates`); standalone utilities
    (feasibility probes `scripts.run_probe`, inspection, annotation) live under
    `scripts/`.

Arguments:
    None. This package initializer is import-only; arguments live in the
    individual CLI modules.
"""
