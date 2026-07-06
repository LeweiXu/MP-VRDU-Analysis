"""CLI package for MP-VRDU operational entry points.

Purpose:
    Groups runnable commands under `python -m cli.<name>` so probes, experiment
    runs, and table building can share the importable project code without shell
    wrappers.

Pipeline role:
    `cli.run_probe` checks feasibility and `cli.gates` evaluates Section-2 gate
    artifacts. The paper-table experiments moved to the `experiments` package,
    split by role: `cli.generate` (GPU), `cli.judge`, and `cli.build` — thin
    wrappers over the `experiments` package.

Arguments:
    None. This package initializer is import-only; arguments live in the
    individual CLI modules.
"""
