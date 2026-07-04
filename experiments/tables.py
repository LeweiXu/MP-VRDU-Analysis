"""Aggregate cached result rows into the paper's CSV table shapes.

Purpose:
    Reserved for builders for Tables 1-8: doc-type frontiers, analytical slices,
    family and dataset replications, evidence composition, matched-vs-cross
    retrieval, routing policies, and scale sanity.

Pipeline role:
    `cli.build_tables` will call this module after experiment runs have filled
    `results/cache/`. Keeping aggregation here lets tables be rebuilt without
    rerunning models.

Arguments:
    None. This module is import-only until Stage M5 implements the builders.
"""
