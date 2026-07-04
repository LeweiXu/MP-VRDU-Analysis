"""Experiment orchestration package.

Purpose:
    Groups the per-table experiments and the machinery that runs them under one
    import namespace. Each paper table is one reusable `Experiment` (`T1_headline`
    … `T8_scale`) that serves both the smoke and full runs.

Pipeline role:
    `experiments.base` holds the `Experiment` contract; `experiments.registry`
    maps names/groups to instances; `experiments.driver` runs them in two phases
    (generate on GPU, judge/build anywhere); `experiments.tables` holds the shared
    table-building primitives; `experiments.corpus`/`experiments.smoke` resolve
    the question set.

Arguments:
    None. This package initializer is import-only.
"""
