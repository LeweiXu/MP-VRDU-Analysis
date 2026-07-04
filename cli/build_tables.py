"""Build paper-table CSVs from cached prediction and score rows.

Purpose:
    This module is reserved for the reporting CLI that turns `results/cache/`
    rows into the eight v3 table shapes. It keeps table building separate from
    expensive model execution so cached runs can be re-aggregated locally.

Pipeline role:
    Stage M5/F-stage work will call `experiments.tables` builders from here.
    The file is currently a documented placeholder; it intentionally performs no
    work until those table builders are implemented.

CLI:
    Planned command form is `python -m cli.build_tables ...`.

Arguments:
    No arguments are implemented yet. Future arguments should be documented in
    this module docstring when the parser is added.
"""
