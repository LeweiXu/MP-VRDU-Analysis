"""CLI package for MP-VRDU operational entry points.

Purpose:
    Groups runnable commands under `python -m cli.<name>` so probes, experiment
    runs, and table building can share the importable project code without shell
    wrappers.

Pipeline role:
    `cli` holds only the three experiment roles, thin wrappers over the
    `experiments` package: `cli.generate` (GPU), `cli.judge`, and `cli.build`.
    Standalone utilities (feasibility probes `scripts.run_probe`, Section-2 gates
    `scripts.gates`, inspection, annotation) live under `scripts/`.

Arguments:
    None. This package initializer is import-only; arguments live in the
    individual CLI modules.
"""
