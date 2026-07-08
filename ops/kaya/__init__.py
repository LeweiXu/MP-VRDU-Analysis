"""Kaya HPC operational tooling package.

Purpose:
    Groups scripts and helpers that operate the two-machine local/Kaya workflow:
    sync, environment setup, prestaging, probes, GPU smoke tests, and SLURM job
    submission.

Pipeline role:
    Keeps cluster-specific mechanics under `kaya/` so the core pipeline remains
    root-relative and runnable in either local or remote mirrors.

Arguments:
    None. This package initializer is import-only.
"""
