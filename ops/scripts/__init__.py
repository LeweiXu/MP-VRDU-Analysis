"""Operational scripts package: staging, environment setup, and probes.

Purpose:
    Groups the standalone operational utilities that are not part of the
    experiment pipeline but support it: dataset/model staging (`download_hf`,
    `prestage`), Kaya environment setup (`setup_env`), GPU and resolution probes
    (`gpu_test`, `resolution_probe`), and dataset profiling (`profile_datasets`,
    `dataset_stats`, `kaya_status`). Each is runnable with
    `python -m ops.scripts.<name>` locally or dispatched to Kaya via the runner.

Pipeline role:
    None at import time. This package exists so the scripts are importable as
    modules; the experiment code never depends on running them.

Arguments:
    None. This is the package marker; each script defines its own command-line
    arguments in its own module docstring.
"""
